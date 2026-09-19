"""Document-QA dataset contract for fine-tuning: the pinned CORD-v2 receipt sample, validation, seeded
page-disjoint splitting, BYOD loaders and JSONL export.

The default dataset is **real**: CORD-v2 (Park et al., 2019; NAVER Clova; CC BY 4.0) — photographed Indonesian
receipts whose every printed word carries a pixel box and a field category (`total.total_price`,
`sub_total.tax_price`, `menu.nm`, …). LayoutLM reads words and boxes, never pixels, so this module fetches
**only the `ground_truth` column** of the two pinned Hub parquet files (the 100-receipt `test` and
`validation` shards): the parquet footer and the one column chunk are read over HTTP range requests through
`pyarrow` (about 0.4 MB of the 476 MB the two files hold with their images). Each file is pinned three
ways — the immutable dataset revision in the URL, the byte size and SHA-256 the Hub declares for the file
(checked against the file metadata before any byte is read), and a SHA-256 of the decoded column (checked
after reading) — and refused on any mismatch. Questions are templated from the field categories: a category
that occurs on exactly one line of a receipt becomes one question whose gold answer is that line's value
words, so every answer is a contiguous span of the page's OCR words by construction.

A record is ``{id, page_id, question, words, boxes, image_size, answer_start, answer_end, answers}`` — the
page's words with one pixel xyxy box each and the inclusive word indices of the gold span; ``answers`` holds
the span text. Every question on the same page lands in the same split.
"""

from __future__ import annotations

import hashlib
import io
import json
import random
import re
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .pipeline import MAX_QUESTION_CHARS, MODEL_ID, _check_document, _check_question

CORPUS_NAME = "CORD-v2"
CORPUS_REPO = "naver-clova-ix/cord-v2"
CORPUS_REVISION = "7f0115a4b758a71d6473b8d085751692da2fef98"
CORPUS_RELEASE = "Hugging Face Hub dataset revision 7f0115a4 (2022-07-19)"
CORPUS_LICENSE = "CC BY 4.0 (Park et al. 2019; NAVER Clova; huggingface.co/datasets/naver-clova-ix/cord-v2)"
CORPUS_COLUMN = "ground_truth"
# The two shards read: path, size and SHA-256 as the Hub declares them for the LFS object (the whole file is
# never downloaded, so these are checked against the file metadata), the row count, and the SHA-256 of the
# decoded `ground_truth` column (canonical JSON list of the row strings), checked after the read.
CORPUS_FILES: dict[str, dict[str, Any]] = {
    "test": {
        "path": "data/test-00000-of-00001-9c204eb3f4e11791.parquet",
        "bytes": 234_202_795,
        "sha256": "51c65f1788faff392abe2a0b55b023eb23e9be551c509138eaa3a832514224e7",
        "rows": 100,
        "column_sha256": "b499e58aa298e5242b5222b7e333e92e6affb79a192def149d0a0aac3e41d9f1",
    },
    "validation": {
        "path": "data/validation-00000-of-00001-cc3c5779fe22e8ca.parquet",
        "bytes": 242_080_800,
        "sha256": "0d0f6dac11fdcc549de2746aa9f53136a3bc22a2a1aff2b0b847f7622ad60c15",
        "rows": 100,
        "column_sha256": "adf8303ec0295af1ef11a63dd0b72453e95399cfd2cd76e3d494c284730bdef9",
    },
}
DEFAULT_CACHE_DIR = Path("weights") / "cord-v2"
# One question per field category that occurs on exactly one line of the receipt; the gold answer is that
# line's words. `menu.nm` (item names) occurs on most receipts several times, so it asks for the *first*
# item — the topmost `menu.nm` line — and is skipped when that item's name spans more than one line.
QUESTION_TEMPLATES: dict[str, str] = {
    "total.total_price": "What is the total amount?",
    "sub_total.subtotal_price": "What is the subtotal?",
    "sub_total.tax_price": "What is the tax amount?",
    "sub_total.service_price": "What is the service charge?",
    "sub_total.discount_price": "What is the discount amount?",
    "total.cashprice": "How much cash was paid?",
    "total.changeprice": "How much change was given?",
    "total.creditcardprice": "How much was paid by card?",
    "total.menuqty_cnt": "How many items were bought?",
    "menu.nm": "What is the name of the first item?",
}
FIRST_ITEM_CATEGORY = "menu.nm"
SAMPLE_SEED = 42
SAMPLE_SPLIT = {
    "train": 119,
    "validation": 30,
    "test": 50,
}  # receipts (pages), not questions; 199 unique pages
MIN_RECORDS = 8
MAX_RECORDS = 20_000
_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def column_digest(rows: Sequence[str]) -> str:
    """SHA-256 of the decoded column as a canonical JSON list of its row strings."""
    return _sha256_bytes(json.dumps(list(rows), ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


class _HttpRangeFile(io.RawIOBase):
    """A seekable read-only view of one HTTPS object served with `Range` requests (what `pyarrow` needs to
    read a parquet footer and a single column chunk without downloading the file)."""

    def __init__(self, url: str, size: int) -> None:
        self.url, self.size, self.pos = url, size, 0
        self.fetched = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        base = {0: 0, 1: self.pos, 2: self.size}[whence]
        self.pos = max(0, base + offset)
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        if n <= 0 or self.pos >= self.size:
            return b""
        end = min(self.size, self.pos + n) - 1
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(request, timeout=180) as response:  # noqa: S310 (pinned https URL)
            if response.status != 206:
                raise ValueError(f"{self.url}: server ignored the Range request (HTTP {response.status})")
            data = response.read()
        self.fetched += len(data)
        self.pos += len(data)
        return data

    def readinto(self, buffer: Any) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


def _hub_column(split: str, spec: Mapping[str, Any]) -> list[str]:
    """Read the pinned shard's `ground_truth` column from the Hub: the file's declared size and LFS SHA-256
    are checked against the pins first, then only the parquet footer and that column are fetched."""
    import pyarrow.parquet as pq
    from huggingface_hub import get_hf_file_metadata, hf_hub_url

    url = hf_hub_url(CORPUS_REPO, spec["path"], repo_type="dataset", revision=CORPUS_REVISION)
    metadata = get_hf_file_metadata(url)
    declared = (metadata.etag or "").strip('"')
    if metadata.size != spec["bytes"] or declared != spec["sha256"]:
        raise ValueError(
            f"{split} shard: the Hub declares {metadata.size} bytes / sha256 {declared[:16]}…, "
            f"pinned {spec['bytes']} / {spec['sha256'][:16]}…"
        )
    handle = _HttpRangeFile(url, spec["bytes"])
    table = pq.ParquetFile(handle).read(columns=[CORPUS_COLUMN])
    return [str(value) for value in table.column(CORPUS_COLUMN).to_pylist()]


def fetch_corpus(
    *,
    cache_dir: str | Path | None = None,
    fetcher: Callable[[str, Mapping[str, Any]], Sequence[str]] | None = None,
) -> dict[str, list[str]]:
    """Return the pinned shards' `ground_truth` rows per split from the cache or the Hub, digest-verified."""
    cache = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)
    out: dict[str, list[str]] = {}
    for split, spec in CORPUS_FILES.items():
        local = cache / f"{split}.{CORPUS_COLUMN}.json"
        rows: list[str] | None = None
        if local.is_file():
            cached = json.loads(local.read_text(encoding="utf-8"))
            if isinstance(cached, list) and column_digest(cached) == spec["column_sha256"]:
                rows = [str(value) for value in cached]
        if rows is None:
            rows = [
                str(value)
                for value in (fetcher(split, spec) if fetcher is not None else _hub_column(split, spec))
            ]
            if len(rows) != spec["rows"] or column_digest(rows) != spec["column_sha256"]:
                raise ValueError(
                    f"{split} shard: fetched {len(rows)} rows with column sha256 "
                    f"{column_digest(rows)[:16]}…, "
                    f"pinned {spec['rows']} / {spec['column_sha256'][:16]}…"
                )
            local.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        out[split] = rows
    return out


def _quad_to_box(quad: Mapping[str, Any], width: int, height: int) -> list[float] | None:
    xs = [float(quad[k]) for k in ("x1", "x2", "x3", "x4")]
    ys = [float(quad[k]) for k in ("y1", "y2", "y3", "y4")]
    x0, y0 = max(0.0, min(xs)), max(0.0, min(ys))
    x1, y1 = min(float(width), max(xs)), min(float(height), max(ys))
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def page_from_ground_truth(ground_truth: str | Mapping[str, Any], page_id: str) -> dict[str, Any]:
    """One CORD `ground_truth` JSON → the page's words, boxes, size and the labelled lines over them.

    Words are concatenated line by line in the annotation's order (`valid_line`), every word box is the
    axis-aligned hull of its quadrilateral clipped to the page, and words with an empty text or a degenerate
    box are dropped. Each line records its field `category`, `group_id`, its inclusive word span and the span
    of its **value** words (`is_key` 0 — a printed key such as `TOTAL` is part of the line but not of the
    answer); `value_start` is `None` when the value words are not one contiguous run."""
    data = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    size = data["meta"]["image_size"]
    width, height = int(size["width"]), int(size["height"])
    words: list[str] = []
    boxes: list[list[float]] = []
    lines: list[dict[str, Any]] = []
    for line in data.get("valid_line", []):
        start = len(words)
        values: list[int] = []
        for word in line.get("words", []):
            text = " ".join(str(word.get("text", "")).split())
            box = _quad_to_box(word["quad"], width, height) if text else None
            if box is None:
                continue
            if not int(word.get("is_key", 0)):
                values.append(len(words))
            words.append(text)
            boxes.append(box)
        if len(words) > start:
            contiguous = bool(values) and values == list(range(values[0], values[-1] + 1))
            lines.append(
                {
                    "category": str(line.get("category", "")),
                    "group_id": int(line.get("group_id", -1)),
                    "start": start,
                    "end": len(words) - 1,
                    "value_start": values[0] if contiguous else None,
                    "value_end": values[-1] if contiguous else None,
                }
            )
    return {"id": page_id, "words": words, "boxes": boxes, "image_size": [width, height], "lines": lines}


def read_corpus(rows: Mapping[str, Sequence[str]]) -> dict[str, list[dict[str, Any]]]:
    """The fetched shards as pages, ids `<split>-<row>`."""
    out: dict[str, list[dict[str, Any]]] = {}
    for split, values in rows.items():
        out[split] = [
            page_from_ground_truth(value, f"{split}-{index:03d}") for index, value in enumerate(values)
        ]
    return out


def questions_for_page(page: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The templated questions a page supports: one per category on exactly one line, plus the first item;
    the gold answer is the line's contiguous run of value words (lines without one are skipped)."""
    by_category: dict[str, list[dict[str, Any]]] = {}
    for line in page["lines"]:
        by_category.setdefault(line["category"], []).append(line)
    out = []
    for category, question in QUESTION_TEMPLATES.items():
        candidates = by_category.get(category, [])
        if not candidates:
            continue
        if category == FIRST_ITEM_CATEGORY:
            topmost = min(candidates, key=lambda line: (page["boxes"][line["start"]][1], line["start"]))
            same_group = [line for line in candidates if line["group_id"] == topmost["group_id"]]
            if len(same_group) != 1:
                continue
            line = topmost
        elif len(candidates) == 1:
            line = candidates[0]
        else:
            continue
        if line.get("value_start") is None:
            continue
        start, end = int(line["value_start"]), int(line["value_end"])
        out.append(
            {
                "page_id": page["id"],
                "question": question,
                "field": category,
                "words": list(page["words"]),
                "boxes": [list(box) for box in page["boxes"]],
                "image_size": list(page["image_size"]),
                "answer_start": start,
                "answer_end": end,
                "answers": [" ".join(page["words"][start : end + 1])],
            }
        )
    return out


def build_sample_dataset(
    pages: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    seed: int = SAMPLE_SEED,
    sizes: Mapping[str, int] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Pool every fetched page that supports at least one question, drop any page whose lower-cased word
    sequence repeats an earlier page (CORD-v2 holds a few identical receipts across its shards), shuffle the
    pages with `seed`, cut **by page** into `sizes` (train / validation / test receipts) and expand each page
    into its questions — so no page, and no question, is shared between splits."""
    sizes = dict(sizes or SAMPLE_SPLIT)
    pool = [dict(page) for split in sorted(pages) for page in pages[split]]
    seen: set[tuple[str, ...]] = set()
    supported = []
    for page in pool:
        key = _page_key(page)
        if key in seen or not questions_for_page(page):
            continue
        seen.add(key)
        supported.append(page)
    needed = sum(sizes.values())
    if len(supported) < needed:
        raise ValueError(f"{len(supported)} pages support a question; the split sizes need {needed}")
    random.Random(seed).shuffle(supported)
    out: dict[str, list[dict[str, Any]]] = {}
    offset = 0
    for name in ("train", "validation", "test"):
        chosen = supported[offset : offset + sizes[name]]
        offset += sizes[name]
        records = [record for page in chosen for record in questions_for_page(page)]
        out[name] = [{"id": f"{name}-{index:04d}", **record} for index, record in enumerate(records)]
    return out


def fetch_sample_dataset(
    *,
    cache_dir: str | Path | None = None,
    fetcher: Callable[[str, Mapping[str, Any]], Sequence[str]] | None = None,
    seed: int = SAMPLE_SEED,
    sizes: Mapping[str, int] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """The tutorial splits from the pinned shards."""
    return build_sample_dataset(
        read_corpus(fetch_corpus(cache_dir=cache_dir, fetcher=fetcher)), seed=seed, sizes=sizes
    )


def _check_record(record: Any, index: int) -> dict[str, Any]:
    label = f"records[{index}]"
    if not isinstance(record, Mapping):
        raise ValueError(
            f"{label} must be a mapping with id/question/words/boxes/image_size/answer_start/answer_end"
        )
    for key in ("id", "question", "words", "boxes", "image_size", "answer_start", "answer_end"):
        if key not in record:
            raise ValueError(f"{label} is missing {key!r}")
    rid = record["id"]
    if not isinstance(rid, str) or not _ID_RE.match(rid):
        raise ValueError(f"{label}: id must match {_ID_RE.pattern}")
    try:
        words, _grid, size = _check_document(record["words"], record["boxes"], record["image_size"])
        question = _check_question(record["question"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: {exc}") from exc
    start, end = record["answer_start"], record["answer_end"]
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (start, end)):
        raise ValueError(f"{label}: answer_start and answer_end must be ints")
    if not 0 <= start <= end < len(words):
        raise ValueError(f"{label}: answer span {start}..{end} is not inside the {len(words)} words")
    text = " ".join(words[start : end + 1])
    answers = record.get("answers", [text])
    if isinstance(answers, str) or not isinstance(answers, Sequence) or not answers or answers[0] != text:
        raise ValueError(f"{label}: answers[0] must be the span text {text[:40]!r}")
    item = {
        "id": rid,
        "page_id": str(record.get("page_id", rid)),
        "question": question,
        "words": words,
        "boxes": [[float(v) for v in box] for box in record["boxes"]],
        "image_size": list(size),
        "answer_start": start,
        "answer_end": end,
        "answers": [str(a) for a in answers],
    }
    if "field" in record:
        item["field"] = str(record["field"])
    return item


def validate_dataset(
    records: Sequence[Mapping[str, Any]], *, min_records: int = MIN_RECORDS, max_records: int = MAX_RECORDS
) -> dict[str, Any]:
    """Structural validation of a document-QA dataset; raises ValueError before any model import."""
    if isinstance(records, Mapping) or not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError(
            "records must be a list of {id, question, words, boxes, image_size, answer_start, answer_end}"
        )
    if not min_records <= len(records) <= max_records:
        raise ValueError(f"{len(records)} records; {min_records}..{max_records} are required")
    checked = []
    ids: set[str] = set()
    pages: set[str] = set()
    for index, record in enumerate(records):
        item = _check_record(record, index)
        if item["id"] in ids:
            raise ValueError(f"duplicate id {item['id']!r}")
        ids.add(item["id"])
        pages.add(item["page_id"])
        checked.append(item)
    return {
        "records": checked,
        "n_records": len(checked),
        "unique_pages": len(pages),
        "words_per_page": {
            "min": min(len(r["words"]) for r in checked),
            "max": max(len(r["words"]) for r in checked),
        },
        "answer_words": {
            "min": min(r["answer_end"] - r["answer_start"] + 1 for r in checked),
            "max": max(r["answer_end"] - r["answer_start"] + 1 for r in checked),
        },
        "question_chars": {
            "min": min(len(r["question"]) for r in checked),
            "max": max(len(r["question"]) for r in checked),
        },
        "max_question_chars": MAX_QUESTION_CHARS,
        "digest": dataset_digest(checked),
        "model_id": MODEL_ID,
    }


def dataset_digest(records: Sequence[Mapping[str, Any]]) -> str:
    payload = [
        [r["id"], r["question"], r["words"], r["boxes"], r["image_size"], r["answer_start"], r["answer_end"]]
        for r in records
    ]
    return _sha256_bytes(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def gold_texts(record: Mapping[str, Any]) -> list[str]:
    """The accepted answers of a record (the gold span text)."""
    return [str(a) for a in record["answers"]]


def _page_key(record: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(w).lower() for w in record["words"])


def check_split_disjoint(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Assert no page id and no page content (its lower-cased word sequence) appears in two splits."""
    seen_ids: dict[str, str] = {}
    seen_words: dict[tuple[str, ...], str] = {}
    for name, records in splits.items():
        for record in records:
            page_id = str(record.get("page_id", record["id"]))
            if page_id in seen_ids and seen_ids[page_id] != name:
                raise ValueError(f"page {page_id!r} appears in both {seen_ids[page_id]} and {name}")
            seen_ids[page_id] = name
            key = _page_key(record)
            if key in seen_words and seen_words[key] != name:
                raise ValueError(
                    f"a page's words ({' '.join(key[:6])!r}…) appear in both {seen_words[key]} and {name}"
                )
            seen_words[key] = name
    return {name: len(records) for name, records in splits.items()}


def split_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    val_fraction: float = 0.15,
    test_fraction: float = 0.2,
    seed: int = 0,
) -> dict[str, list[dict[str, Any]]]:
    """Seeded split of a BYOD dataset into train/validation/test **by page**: every question on the same page
    lands in the same split, so a test page is never seen in training."""
    if not (0.0 <= val_fraction < 1.0 and 0.0 < test_fraction < 1.0 and val_fraction + test_fraction < 1.0):
        raise ValueError("fractions must satisfy 0 <= val < 1, 0 < test < 1, val + test < 1")
    checked = validate_dataset(records)["records"]
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in checked:
        groups.setdefault(record["page_id"], []).append(record)
    order = list(groups.values())
    random.Random(seed).shuffle(order)
    n_test = max(1, round(len(checked) * test_fraction))
    n_val = round(len(checked) * val_fraction)
    splits: dict[str, list[dict[str, Any]]] = {"test": [], "validation": [], "train": []}
    for group in order:
        if len(splits["test"]) < n_test:
            splits["test"].extend(group)
        elif len(splits["validation"]) < n_val:
            splits["validation"].extend(group)
        else:
            splits["train"].extend(group)
    if len(splits["train"]) < MIN_RECORDS:
        raise ValueError(
            f"split leaves {len(splits['train'])} training records; at least {MIN_RECORDS} are required"
        )
    return splits


def load_byod_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Read records from a JSON array or a JSONL file of
    ``{id, page_id, question, words, boxes, image_size, answer_start, answer_end}`` objects."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"dataset not found: {file_path}")
    suffix = file_path.suffix.lower()
    text = file_path.read_text(encoding="utf-8")
    if suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if suffix == ".json":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("JSON dataset must be an array of records")
        return data
    raise ValueError("BYOD datasets must be .json or .jsonl")


def write_dataset_jsonl(records: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    """One record per line in the shape `load_byod_dataset` reads back."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = (
        "id",
        "page_id",
        "question",
        "words",
        "boxes",
        "image_size",
        "answer_start",
        "answer_end",
        "answers",
    )
    with open(out, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps({k: record[k] for k in keys if k in record}, ensure_ascii=False) + "\n")
    return out
