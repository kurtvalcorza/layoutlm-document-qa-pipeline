"""Official-partition CORD v2 loading against the frozen branch manifests.

The committed ``data_manifest.json`` / ``split_manifest.json`` (produced by
``freeze_receipt_assets.py``) are the single source of truth for shard pins, receipt IDs, per-row
image/pixel/annotation hashes, roles and integrity exclusions. At run time every shard is verified by
full-file SHA-256 and every row is re-hashed and compared with the manifest; roles and exclusions are
recomputed and must equal the frozen split. References are written to a separate evaluator-owned
directory and never enter prediction inputs.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from receipt_common import (
    FIELDS,
    ContractError,
    IntegrityError,
    digest,
    download_pinned,
    file_key,
    read_jsonl,
    safe_receipt_id,
    write_json,
    write_jsonl,
)
from receipt_fields import amount, box_union, key_text

DATASET_ID = "naver-clova-ix/cord-v2"
ROLE_SALT = "receipt-capstone-v1|42|"
CATEGORY = {
    "total_amount": "total.total_price",
    "subtotal_amount": "sub_total.subtotal_price",
    "tax_amount": "sub_total.tax_price",
    "service_charge": "sub_total.service_price",
}
# An explicit, versioned compatibility alias (README 'subtotal' vs pinned JSON 'sub_total').
ALIASES = {"subtotal": "sub_total"}
REFERENCE_PARSER_ID = "cord-reference-parser-v1"
NOT_TOTAL = (
    "total.cashprice",
    "total.changeprice",
    "total.creditcardprice",
    "total.emoneyprice",
    "sub_total.discount_price",
)


def category_alias(value: str) -> str:
    parts = value.split(".")
    parts[0] = ALIASES.get(parts[0], parts[0])
    return ".".join(parts)


def pixel_sha256(image: Any) -> str:
    """Hash of EXIF-oriented RGB pixels, identical to the freezer's definition."""
    return hashlib.sha256(
        json.dumps(list(image.size), separators=(",", ":")).encode() + image.tobytes()
    ).hexdigest()


def assign_roles(ids_by_split: dict[str, list[str]], salt: str = ROLE_SALT) -> dict[str, str]:
    """Deterministic validation split by ascending SHA-256 of salt + id, before exclusions."""
    if set(ids_by_split) != {"train", "validation", "test"}:
        raise ContractError("Official train/validation/test partitions are required")
    all_ids = [i for ids in ids_by_split.values() for i in ids]
    if len(all_ids) != len(set(all_ids)):
        raise ContractError("Receipt IDs collide across source partitions")
    order = sorted(ids_by_split["validation"], key=lambda i: hashlib.sha256((salt + i).encode()).hexdigest())
    half = len(order) // 2
    roles = {i: "train" for i in ids_by_split["train"]}
    roles.update({i: "test" for i in ids_by_split["test"]})
    roles.update({i: "validation_model" if j < half else "validation_policy" for j, i in enumerate(order)})
    return roles


def integrity_cohort(rows: list[dict[str, Any]], roles: dict[str, str]) -> tuple[list[str], list[dict]]:
    """Exact pixel-hash families: cross-role families are quarantined wholesale; within a role the
    lexically first valid ID is kept. Mirrors the freezer's rule so the frozen split can be checked."""
    groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if row.get("pixel_sha256"):
            groups[row["pixel_sha256"]].append(row["receipt_id"])
    status = {r["receipt_id"]: r.get("source_error") for r in rows}
    excluded = {i: reason for i, reason in status.items() if reason}
    for family in groups.values():
        if len(family) < 2:
            continue
        family = sorted(family)
        cross = len({roles[i] for i in family}) > 1
        for i in family:
            if cross:
                excluded[i] = "cross_role_exact_duplicate"
            elif i != family[0]:
                excluded[i] = "within_role_exact_duplicate"
    keep = sorted(i for i in status if i not in excluded)
    return keep, [{"receipt_id": i, "reason": r} for i, r in sorted(excluded.items())]


def orientation_matrix(orientation: int, width: int, height: int) -> list[list[int]]:
    """Affine map from original pixel coordinates to EXIF-transposed coordinates."""
    matrices = {
        1: [[1, 0, 0], [0, 1, 0]],
        2: [[-1, 0, width], [0, 1, 0]],
        3: [[-1, 0, width], [0, -1, height]],
        4: [[1, 0, 0], [0, -1, height]],
        5: [[0, 1, 0], [1, 0, 0]],
        6: [[0, -1, height], [1, 0, 0]],
        7: [[0, -1, height], [-1, 0, width]],
        8: [[0, 1, 0], [-1, 0, width]],
    }
    if orientation not in matrices:
        raise ContractError("Unsupported EXIF orientation")
    return matrices[orientation]


def transform_box(box: list[float], matrix: list[list[int]]) -> list[float]:
    points = [(box[x], box[y]) for x in (0, 2) for y in (1, 3)]
    mapped = [
        (
            matrix[0][0] * x + matrix[0][1] * y + matrix[0][2],
            matrix[1][0] * x + matrix[1][1] * y + matrix[1][2],
        )
        for x, y in points
    ]
    return [
        min(p[0] for p in mapped),
        min(p[1] for p in mapped),
        max(p[0] for p in mapped),
        max(p[1] for p in mapped),
    ]


def inverse_matrix(matrix: list[list[int]]) -> list[list[int]]:
    """Inverse of an orientation matrix (integer rotation/reflection plus translation)."""
    (a, b, tx), (c, d, ty) = matrix
    det = a * d - b * c
    ia, ib, ic, id_ = d * det, -b * det, -c * det, a * det  # det is +-1
    return [[ia, ib, -(ia * tx + ib * ty)], [ic, id_, -(ic * tx + id_ * ty)]]


def inspect_image(
    data: bytes, max_bytes: int = 20 * 1024**2, max_pixels: int = 20_000_000, max_side: int = 10_000
) -> tuple[Any, dict[str, Any]]:
    """Validate before decoding, then return EXIF-oriented RGB and the coordinate map."""
    import warnings

    from PIL import Image, ImageOps

    if len(data) > max_bytes:
        raise ContractError("Image exceeds 20 MiB")
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(data)) as image:
            if image.format not in ("JPEG", "PNG") or getattr(image, "n_frames", 1) != 1:
                raise ContractError("Only single-frame JPEG/PNG images are supported")
            width, height = image.size
            if width * height > max_pixels or max(width, height) > max_side or min(width, height) < 1:
                raise ContractError("Image exceeds 20 MP / 10,000-pixel side ceiling")
            orientation = int(image.getexif().get(274, 1))
            matrix = orientation_matrix(orientation, width, height)
            image.load()
            result = ImageOps.exif_transpose(image).convert("RGB")
            fmt = image.format
    return result, {
        "image_sha256": hashlib.sha256(data).hexdigest(),
        "pixel_sha256": pixel_sha256(result),
        "original_size": [width, height],
        "image_size": list(result.size),
        "format": fmt,
        "orientation": {"exif": orientation, "matrix": matrix, "original_size": [width, height]},
    }


def _word_box(word: dict[str, Any], original_size: list[int], matrix: list[list[int]]) -> list[float]:
    quad = word["quad"]
    xs = [float(quad[f"x{k}"]) for k in range(1, 5)]
    ys = [float(quad[f"y{k}"]) for k in range(1, 5)]
    width, height = original_size
    # Clip to the page: some published quads touch or exceed the border by a pixel.
    box = [max(0.0, min(xs)), max(0.0, min(ys)), min(float(width), max(xs)), min(float(height), max(ys))]
    if not (box[0] < box[2] and box[1] < box[3]):
        raise ContractError("Reference word box is empty after clipping to the image")
    return transform_box(box, matrix)


def _values_agree(values: list[dict[str, Any]], policy: str, currency: str) -> tuple[str | None, str]:
    parsed = [amount(v["text"], policy, currency) for v in values]
    if any(p["parse_status"] != "ok" for p in parsed):
        return None, "unparseable_source_value"
    distinct = {p["normalized_amount"] for p in parsed}
    if len(distinct) != 1:
        return None, "multiple_incompatible_values"
    return distinct.pop(), "ok"


def references(
    ground_truth: str | dict[str, Any],
    image_info: dict[str, Any],
    policy: str = "cord_mixed_v1",
    currency: str = "unspecified",
) -> dict[str, Any]:
    """Prediction-blind evaluator references with explicit reference states.

    States: ``present_usable`` (unique canonical value agreed by the labelled value words and, where
    present, ``gt_parse``), ``not_annotated`` (never zero or absent), ``ambiguous_reference`` and
    ``invalid_reference``. Several lines of one category are usable only when every value word group
    normalises to the same amount (the value is unique, the evidence region is not; boxes are kept).
    """
    data = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    ref = {
        field: {
            "state": "not_annotated",
            "raw_text": None,
            "normalized_amount": None,
            "value_boxes": [],
            "source_category": CATEGORY[field],
        }
        for field in FIELDS
    }
    meta_size = data.get("meta", {}).get("image_size", {})
    if [meta_size.get("width"), meta_size.get("height")] != image_info["original_size"]:
        return {
            "fields": {
                f: {**v, "state": "invalid_reference", "reason": "annotation_image_size_mismatch"}
                for f, v in ref.items()
            },
            "diagnostic_tokens": [],
            "key_phrases": [],
            "errors": ["annotation_image_size_mismatch"],
            "parser_id": REFERENCE_PARSER_ID,
        }
    words_out, source_values, key_phrases, errors = [], defaultdict(list), [], []
    matrix = image_info["orientation"]["matrix"]
    for line_index, line in enumerate(data.get("valid_line", [])):
        category = category_alias(str(line.get("category", "")))
        values, boxes, keys, invalid = [], [], [], False
        for word_index, word in enumerate(line.get("words", [])):
            try:
                text = str(word["text"])
                if not text.strip():
                    continue
                box = _word_box(word, image_info["original_size"], matrix)
                row = (
                    int(word.get("row_id", line_index))
                    if str(word.get("row_id", "")).lstrip("-").isdigit()
                    else line_index
                )
                words_out.append({"text": text, "box": box, "row": row})
                if int(word.get("is_key", 0)):
                    keys.append(key_text(text))
                else:
                    values.append(text)
                    boxes.append(box)
            except (KeyError, ValueError, TypeError, ContractError):
                invalid = True
                errors.append(f"invalid_word:{line_index}:{word_index}")
        if keys and any(keys):
            key_phrases.append({"category": category, "phrase": " ".join(k for k in keys if k)})
        source_values[category].append(
            {"text": " ".join(values), "box": box_union(boxes) if boxes else None, "invalid": invalid}
        )
    parsed_source = data.get("gt_parse", {}) or {}
    for field, category in CATEGORY.items():
        group, leaf = category.split(".")
        group_value = parsed_source.get(group)
        alias_value = parsed_source.get("subtotal") if group == "sub_total" else None
        if group_value is not None and alias_value is not None and group_value != alias_value:
            ref[field].update(state="ambiguous_reference", reason="sub_total_alias_conflict")
            continue
        group_value = group_value if group_value is not None else (alias_value or {})
        parsed_value = group_value.get(leaf) if isinstance(group_value, dict) else None
        lines = source_values.get(category, [])
        if not lines and parsed_value is None:
            continue
        entry = ref[field]
        entry["source_values"] = [line["text"] for line in lines]
        entry["parsed_source_value"] = parsed_value
        if not lines:
            entry.update(state="ambiguous_reference", reason="gt_parse_without_value_words")
            continue
        if any(line["invalid"] or line["box"] is None for line in lines):
            entry.update(state="invalid_reference", reason="invalid_or_missing_value_words")
            continue
        value, reason = _values_agree(lines, policy, currency)
        if value is None:
            entry.update(state="ambiguous_reference", reason=reason)
            continue
        if parsed_value is not None:
            parsed_items = parsed_value if isinstance(parsed_value, list) else [parsed_value]
            if not all(isinstance(p, str) for p in parsed_items):
                entry.update(state="ambiguous_reference", reason="structured_gt_parse_value")
                continue
            p_value, p_reason = _values_agree([{"text": p} for p in parsed_items], policy, currency)
            if p_value != value:
                entry.update(state="ambiguous_reference", reason="gt_parse_value_word_disagreement")
                continue
        entry.update(
            state="present_usable",
            raw_text=lines[0]["text"],
            normalized_amount=value,
            value_boxes=[line["box"] for line in lines],
            n_lines=len(lines),
            currency=amount(lines[0]["text"], policy, currency)["currency"],
        )
    # Reference-text diagnostic input: natural page words (labels included), no categories/is_key.
    rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for word in words_out:
        rows[word["row"]].append(word)
    ordered = sorted(
        rows.values(), key=lambda ws: (min(w["box"][1] for w in ws), min(w["box"][0] for w in ws))
    )
    diagnostic = []
    for row_index, row_words in enumerate(ordered):
        for word_index, word in enumerate(sorted(row_words, key=lambda w: (w["box"][0], w["box"][1]))):
            diagnostic.append(
                {
                    "id": len(diagnostic),
                    "text": word["text"],
                    "box": word["box"],
                    "line": [0, 0, row_index],
                    "word": word_index,
                    "confidence": None,
                }
            )
    return {
        "fields": ref,
        "diagnostic_tokens": diagnostic,
        "key_phrases": key_phrases,
        "errors": errors,
        "parser_id": REFERENCE_PARSER_ID,
    }


def _frozen_rows(manifests: dict[str, Any]) -> dict[str, dict[str, Any]]:
    split_rows = {r["receipt_id"]: r for r in manifests["split"]["rows"]}
    rows = {}
    for row in manifests["data"]["rows"]:
        rid = safe_receipt_id(row["receipt_id"])
        if split_rows.get(rid, {}).get("role") != row["role"]:
            raise IntegrityError(f"Data and split manifests disagree for {rid}")
        rows[rid] = {**row, "included": split_rows[rid]["included"]}
    if len(rows) != len(split_rows):
        raise IntegrityError("Data and split manifests list different receipts")
    return rows


def check_frozen_split(manifests: dict[str, Any]) -> dict[str, Any]:
    """Recompute roles and exclusions from manifest facts; they must equal the frozen split."""
    rows = list(_frozen_rows(manifests).values())
    ids = {
        s: [r["receipt_id"] for r in rows if r["official_split"] == s]
        for s in ("train", "validation", "test")
    }
    roles = assign_roles(ids, manifests["split"]["salt"])
    keep, exclusions = integrity_cohort(rows, roles)
    frozen_keep = sorted(r["receipt_id"] for r in rows if r["included"])
    if any(roles[r["receipt_id"]] != r["role"] for r in rows) or keep != frozen_keep:
        raise IntegrityError("Recomputed roles/exclusions differ from the frozen split manifest")
    return {
        "roles": roles,
        "keep": keep,
        "exclusions": exclusions,
        "role_counts": dict(Counter(roles[i] for i in keep)),
    }


def prepare_cord(
    inputs_dir: Path,
    references_dir: Path,
    image_store: Path,
    cache: Path,
    manifests: dict[str, Any],
    *,
    allowed_hosts: tuple[str, ...] = ("huggingface.co",),
    shard_filter: set[str] | None = None,
) -> dict[str, Any]:
    """Verify every pinned shard and row, store images content-addressed, split inputs/references.

    ``shard_filter`` restricts which official splits are read; it exists only for clearly labelled
    reduced CPU fixtures and is never used by the canonical notebook path.
    """
    import pyarrow.parquet as pq

    frozen = _frozen_rows(manifests)
    split_check = check_frozen_split(manifests)
    image_store.mkdir(parents=True, exist_ok=True)
    cohort, evaluator, downloads, seen = [], [], [], set()
    by_source = {(r["source_path"], r["source_row"]): r for r in frozen.values()}
    for pin in manifests["data"]["files"]:
        if shard_filter is not None and pin["official_split"] not in shard_filter:
            continue
        path = download_pinned(
            pin["url"],
            cache / Path(pin["path"]).name,
            {"sha256": pin["sha256"], "bytes": pin["bytes"]},
            allowed_hosts,
            max_bytes=600 * 1024**2,
        )
        downloads.append(
            {
                "path": pin["path"],
                "official_split": pin["official_split"],
                "bytes": pin["bytes"],
                "publisher_sha256": pin["publisher_sha256"],
                "locally_verified_sha256": pin["sha256"],
                "verification": "complete_file_bytes_sha256",
            }
        )
        row_index = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1, columns=["image", "ground_truth"]):
            example = batch.to_pylist()[0]
            frozen_row = by_source.get((pin["path"], row_index))
            row_index += 1
            if frozen_row is None:
                raise IntegrityError(f"Unlisted source row {pin['path']}#{row_index - 1}")
            rid = frozen_row["receipt_id"]
            seen.add(rid)
            raw = example["image"]["bytes"]
            gt = example["ground_truth"]
            if hashlib.sha256(raw).hexdigest() != frozen_row["image_sha256"]:
                raise IntegrityError(f"Image bytes changed for {rid}")
            if hashlib.sha256(gt.encode("utf-8")).hexdigest() != frozen_row["annotation_sha256"]:
                raise IntegrityError(f"Annotation bytes changed for {rid}")
            if not frozen_row["included"]:
                continue
            image, info = inspect_image(raw)
            if info["pixel_sha256"] != frozen_row["pixel_sha256"]:
                raise IntegrityError(f"Decoded pixels changed for {rid}")
            stored = image_store / f"{info['image_sha256']}.{'png' if info['format'] == 'PNG' else 'jpg'}"
            if not stored.exists():
                tmp = stored.with_suffix(".part")
                tmp.write_bytes(raw)
                tmp.replace(stored)
            del image
            cohort.append(
                {
                    "receipt_id": rid,
                    "role": frozen_row["role"],
                    "official_split": frozen_row["official_split"],
                    "image_file": stored.name,
                    **{
                        k: info[k]
                        for k in (
                            "image_sha256",
                            "pixel_sha256",
                            "original_size",
                            "image_size",
                            "format",
                            "orientation",
                        )
                    },
                }
            )
            evaluator.append({"receipt_id": rid, "role": frozen_row["role"], **references(gt, info)})
    expected = {
        rid for rid, r in frozen.items() if shard_filter is None or r["official_split"] in shard_filter
    }
    if seen != expected:
        raise IntegrityError("Source rows differ from the frozen manifest")
    cohort.sort(key=lambda r: r["receipt_id"])
    evaluator.sort(key=lambda r: r["receipt_id"])
    counts = Counter(r["role"] for r in cohort)
    states = {f: dict(sorted(Counter(r["fields"][f]["state"] for r in evaluator).items())) for f in FIELDS}
    states_by_role = {
        role: {
            f: dict(sorted(Counter(r["fields"][f]["state"] for r in evaluator if r["role"] == role).items()))
            for f in FIELDS
        }
        for role in sorted(counts)
    }
    reasons = Counter(
        r["fields"][f].get("reason") for r in evaluator for f in FIELDS if r["fields"][f].get("reason")
    )
    inputs_dir.mkdir(parents=True, exist_ok=True)
    references_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(inputs_dir / "cohort.jsonl", cohort)
    write_json(
        inputs_dir / "data_card.json",
        {
            "dataset_id": manifests["data"].get("dataset_id", DATASET_ID),
            "revision": manifests["data"]["revision"],
            "license": "CC-BY-4.0",
            "population": "Indonesian receipts (CORD v2 public sample); not Philippine business records",
            "files": downloads,
            "nominal_counts": manifests["data"]["nominal_counts"],
            "frozen_role_counts": dict(sorted(counts.items())),
            "exclusions": split_check["exclusions"],
            "cohort_digest": digest(cohort),
            "split_manifest_digest": digest(manifests["split"]),
            "reduced_fixture": shard_filter is not None,
            "near_duplicate_audit": "not performed; exact-pixel audit is not template-disjointness",
            "pretraining_overlap": "unknown",
        },
    )
    write_jsonl(references_dir / "references.jsonl", evaluator)
    write_json(
        references_dir / "reference_audit.json",
        {
            "parser_id": REFERENCE_PARSER_ID,
            "field_reference_states": states,
            "field_reference_states_by_role": states_by_role,
            "reason_counts": dict(sorted(reasons.items())),
            "annotation_errors": sum(len(r["errors"]) for r in evaluator),
            "reference_digest": digest(evaluator),
            "independent_human_review_count": 0,
            "label": "published-source references, not independently verified financial truth",
        },
    )
    return {"role_counts": dict(counts), "states": states}


class ReferenceReader:
    """Evaluator-owned reference access with a purpose log (spec section 5.1)."""

    PURPOSES = (
        "training_supervision",
        "keyword_development",
        "model_selection",
        "policy_selection",
        "final_evaluation",
        "reference_text_diagnostic",
        "recoverability_diagnostic",
        "source_audit",
        "byod_training_supervision",
    )

    def __init__(self, path: Path, log_path: Path) -> None:
        self.rows = {r["receipt_id"]: r for r in read_jsonl(path)}
        self.log_path = log_path

    def get(self, roles: tuple[str, ...], purpose: str) -> dict[str, dict[str, Any]]:
        if purpose not in self.PURPOSES:
            raise ContractError(f"Undeclared reference purpose {purpose}")
        selected = {i: r for i, r in self.rows.items() if r["role"] in roles}
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps({"purpose": purpose, "roles": list(roles), "count": len(selected)}) + "\n")
        return selected


def cohort_image(image_store: Path, record: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Read one stored image and re-verify its byte and pixel hashes before use."""
    data = (image_store / record["image_file"]).read_bytes()
    image, info = inspect_image(data)
    if info["image_sha256"] != record["image_sha256"] or info["pixel_sha256"] != record["pixel_sha256"]:
        raise IntegrityError(f"Stored image changed for {record['receipt_id']}")
    return image, info


def image_name(receipt_id: str) -> str:
    return file_key(receipt_id)
