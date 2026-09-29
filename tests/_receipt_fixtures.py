"""Shared CPU fixtures for the receipt capstone tests.

Everything here is a labelled test double: synthetic rendered receipts, recorded OCR tokens and a tiny
randomly initialised LayoutLM that reuses the repository's real tokenizer files. None of it is
pretrained-model, Tesseract or CORD evidence.
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

CARRIED = (
    "receipt_common.py",
    "receipt_fields.py",
    "receipt_data.py",
    "receipt_ocr.py",
    "receipt_models.py",
    "receipt_training.py",
    "receipt_metrics.py",
    "receipt_policy.py",
    "receipt_artifact.py",
    "receipt_byod.py",
)
TOKENIZER_FILES = (
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def document(lines: list[list[str]], confidence: float | None = 90.0, width: int = 600) -> dict:
    """An OCR document with one token per word, line identifiers and monotone geometry."""
    tokens = []
    for li, words in enumerate(lines):
        x = 10
        for wi, word in enumerate(words):
            w = 12 * len(word)
            tokens.append(
                {
                    "id": len(tokens),
                    "text": word,
                    "box": [x, 20 + 30 * li, x + w, 40 + 30 * li],
                    "line": [1, 1, li + 1],
                    "word": wi + 1,
                    "confidence": confidence,
                }
            )
            x += w + 400 // max(1, len(words))
    return {
        "receipt_id": "doc",
        "image_size": [max([width] + [t["box"][2] + 10 for t in tokens]), 60 + 30 * len(lines)],
        "tokens": tokens,
        "ocr_status": "ok",
        "input_source": "actual_ocr",
    }


def render(lines: list[list[str]], width: int = 520):
    """Rendered synthetic receipt image and the exact word tokens drawn on it (test-double OCR)."""
    from PIL import Image, ImageDraw

    height = 40 + 34 * len(lines)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    tokens = []
    for li, words in enumerate(lines):
        y = 20 + 34 * li
        positions = (
            [20]
            if len(words) == 1
            else [20 + int(k * (width - 140) / (len(words) - 1)) for k in range(len(words))]
        )
        for wi, (word, x) in enumerate(zip(words, positions, strict=True)):
            draw.text((x, y), word, fill="black")
            x0, y0, x1, y1 = draw.textbbox((x, y), word)
            tokens.append(
                {
                    "id": len(tokens),
                    "text": word,
                    "box": [x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)],
                    "line": [1, 1, li + 1],
                    "word": wi + 1,
                    "confidence": 91.0,
                }
            )
    return image, tokens


def fmt(n: int) -> str:
    s = f"{n:,}".replace(",", ".")
    return s


def receipt_content(k: int, with_tax: bool = True) -> dict:
    sub = 10_000 + 1_000 * k
    tax = sub // 10 if with_tax else 0
    service = 500 if k % 3 == 0 else 0
    total = sub + tax + service
    cash = ((total // 50_000) + 1) * 50_000
    lines = [["STORE", f"N{k:03d}"], ["ITEM", fmt(sub)], ["SUBTOTAL", fmt(sub)]]
    if with_tax:
        lines.append(["TAX", fmt(tax)])
    if service:
        lines.append(["SERVICE", fmt(service)])
    lines += [["TOTAL", fmt(total)], ["CASH", fmt(cash)], ["CHANGE", fmt(cash - total)]]
    categories = {1: "menu.price", 2: "sub_total.subtotal_price"}
    return {
        "lines": lines,
        "sub": sub,
        "tax": tax if with_tax else None,
        "service": service or None,
        "total": total,
        "categories": categories,
    }


def ground_truth(
    lines: list[list[str]], tokens: list[dict], size: tuple[int, int], shuffle_total: int | None = None
) -> str:
    """CORD-v2-like annotation for a rendered receipt (value words is_key=0, labels is_key=1)."""
    cat = {
        "ITEM": "menu.price",
        "SUBTOTAL": "sub_total.subtotal_price",
        "TAX": "sub_total.tax_price",
        "SERVICE": "sub_total.service_price",
        "TOTAL": "total.total_price",
        "CASH": "total.cashprice",
        "CHANGE": "total.changeprice",
        "STORE": "menu.nm",
    }
    by_line: dict[int, list[dict]] = {}
    for t in tokens:
        by_line.setdefault(t["line"][2], []).append(t)
    valid, parse = [], {"menu": [], "sub_total": {}, "total": {}}
    for li, words in enumerate(lines, 1):
        category = cat[words[0]]
        entries = []
        for wi, t in enumerate(by_line[li]):
            text = t["text"]
            if shuffle_total is not None and category == "total.total_price" and wi == 1:
                text = fmt(shuffle_total)
            x0, y0, x1, y1 = t["box"]
            entries.append(
                {
                    "quad": {"x1": x0, "y1": y0, "x2": x1, "y2": y0, "x3": x1, "y3": y1, "x4": x0, "y4": y1},
                    "is_key": 1 if wi == 0 and category != "menu.nm" else 0,
                    "row_id": li,
                    "text": text,
                }
            )
        valid.append({"category": category, "group_id": li, "words": entries})
        group, leaf = category.split(".")
        value = entries[-1]["text"]
        if group in ("sub_total", "total"):
            parse[group][leaf] = value
    return json.dumps(
        {
            "gt_parse": parse,
            "meta": {
                "version": "2.0.0",
                "split": "x",
                "image_id": 0,
                "image_size": {"width": size[0], "height": size[1]},
            },
            "valid_line": valid,
            "roi": {},
            "repeating_symbol": [],
            "dontcare": [],
        }
    )


def make_dataset(base: Path, counts: dict[str, int] | None = None, shuffle_test_totals: bool = False) -> dict:
    """Synthetic CORD-like shards + frozen manifests + test-double OCR table in ``base``."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import receipt_data as data

    counts = counts or {"train": 14, "validation": 8, "test": 6}
    cache = base / "cache" / "cord"
    cache.mkdir(parents=True, exist_ok=True)
    files, rows, table = [], [], {}
    k = 0
    for split, n in counts.items():
        images, gts = [], []
        for i in range(n):
            content = receipt_content(k, with_tax=(k % 4 != 1))
            image, tokens = render(content["lines"])
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            raw = buffer.getvalue()
            shuffled = (content["total"] + 7_000) if (shuffle_test_totals and split == "test") else None
            gt = ground_truth(content["lines"], tokens, image.size, shuffled)
            images.append({"bytes": raw, "path": None})
            gts.append(gt)
            decoded, info = data.inspect_image(raw)
            table[info["pixel_sha256"]] = tokens
            rows.append(
                {
                    "receipt_id": f"cord-v2:{split}:{i:04d}",
                    "official_split": split,
                    "source_path": f"data/{split}-00000-of-00001.parquet",
                    "source_row": i,
                    "source_error": None,
                    "image_sha256": sha(raw),
                    "pixel_sha256": info["pixel_sha256"],
                    "annotation_sha256": sha(gt.encode()),
                    "image_bytes": len(raw),
                    "image_format": "PNG",
                    "exif_orientation": 1,
                    "original_size": list(image.size),
                    "oriented_size": list(image.size),
                }
            )
            k += 1
        name = f"{split}-00000-of-00001.parquet"
        pq.write_table(pa.table({"image": images, "ground_truth": gts}), cache / name)
        payload = (cache / name).read_bytes()
        files.append(
            {
                "path": f"data/{name}",
                "official_split": split,
                "bytes": len(payload),
                "sha256": sha(payload),
                "publisher_sha256": sha(payload),
                "url": f"https://huggingface.co/datasets/test/fixture/resolve/{'0' * 40}/data/{name}",
            }
        )
    ids = {s: [r["receipt_id"] for r in rows if r["official_split"] == s] for s in counts}
    roles = data.assign_roles(ids)
    keep, _ = data.integrity_cohort(rows, roles)
    for r in rows:
        r["role"] = roles[r["receipt_id"]]
    split_rows = [
        {
            "receipt_id": r["receipt_id"],
            "official_split": r["official_split"],
            "role": r["role"],
            "included": r["receipt_id"] in keep,
            "exclusion_reason": None,
        }
        for r in rows
    ]
    manifests = {
        "data": {
            "schema": "org.dimer.receipt-source.v1",
            "dataset_id": "test/fixture",
            "revision": "0" * 40,
            "license": "synthetic",
            "files": files,
            "rows": rows,
            "nominal_counts": counts,
        },
        "split": {"schema": "org.dimer.receipt-splits.v1", "salt": data.ROLE_SALT, "rows": split_rows},
    }
    (base / "ocr_double.json").write_text(json.dumps(table), encoding="utf-8")
    return {"manifests": manifests, "cache": base / "cache", "ocr_table": table}


def tiny_snapshot(base: Path) -> tuple[Path, dict]:
    """Tiny randomly initialised LayoutLM QA model (12 layers so the last-4 rule is exercised)."""
    import torch
    from transformers import LayoutLMConfig, LayoutLMForQuestionAnswering

    target = base / "tiny-model"
    target.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    config = LayoutLMConfig(
        vocab_size=50265,
        hidden_size=16,
        num_hidden_layers=12,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=514,
        type_vocab_size=1,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        max_2d_position_embeddings=1024,
        layer_norm_eps=1e-5,
        tokenizer_class="RobertaTokenizer",
    )
    model = LayoutLMForQuestionAnswering(config)
    model.save_pretrained(target, safe_serialization=True)
    for name in TOKENIZER_FILES:
        shutil.copyfile(ROOT / "weights" / "layoutlm-document-qa" / name, target / name)
    files = [
        {"path": p.name, "bytes": p.stat().st_size, "sha256": sha(p.read_bytes())}
        for p in sorted(target.iterdir())
        if p.is_file()
    ]
    manifest = {
        "modelId": "test/tiny-random-layoutlm",
        "revision": "0" * 40,
        "files": files,
        "note": "tiny random test double",
    }
    (base / "tiny-model-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return target, manifest


def carrier(base: Path, manifests: dict) -> Path:
    """Lay out a run directory exactly like the notebook carrier (flat files, capstone.py)."""
    root = base / "run"
    root.mkdir(parents=True, exist_ok=True)
    for name in CARRIED:
        shutil.copyfile(TOOLS / name, root / name)
    shutil.copyfile(TOOLS / "receipt_capstone.py", root / "capstone.py")
    for name in ("model", "ocr"):
        shutil.copyfile(
            ROOT / "tutorials" / "receipt_intelligence" / f"{name}_manifest.json",
            root / f"{name}_manifest.json",
        )
    for name, value in manifests.items():
        (root / f"{name}_manifest.json").write_text(json.dumps(value), encoding="utf-8")
    return root
