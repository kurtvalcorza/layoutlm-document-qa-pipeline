"""Official-partition CORD image loader; references never enter prediction objects."""
from __future__ import annotations

import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from receipt_common import (ContractError, FIELDS, IntegrityError, digest, download_pinned,
                            file_hash, safe_id, write_json, write_jsonl)
from receipt_fields import amount, box_union

DATASET_ID = "naver-clova-ix/cord-v2"
DATASET_REVISION = "7f0115a4b758a71d6473b8d085751692da2fef98"
SOURCE_FILES = [
    {"split": "train", "path": "data/train-00000-of-00004-b4aaeceff1d90ecb.parquet",
     "sha256": "da3994eee1bf9bd3c57f0d53a72c3a6812c8696c5ba26245987949ddf73483cc", "bytes": None},
    {"split": "train", "path": "data/train-00001-of-00004-7dbbe248962764c5.parquet",
     "sha256": "cce4def16a0d6a6c75f80be712f7494c56c318a8829b712f5c62650155c9e58e", "bytes": None},
    {"split": "train", "path": "data/train-00002-of-00004-688fe1305a55e5cc.parquet",
     "sha256": "591e2db8fe8b1d364b054f46e8c375b7f00e72578914ba46a573b6858162cab2", "bytes": None},
    {"split": "train", "path": "data/train-00003-of-00004-2d0cd200555ed7fd.parquet",
     "sha256": "1ffd9de8d6fbcee7630fd4cdfedff05b9b7fabc0fae4fc17557c5fe7cf178748", "bytes": None},
    {"split": "validation", "path": "data/validation-00000-of-00001-cc3c5779fe22e8ca.parquet",
     "sha256": "0d0f6dac11fdcc549de2746aa9f53136a3bc22a2a1aff2b0b847f7622ad60c15", "bytes": 242080800},
    {"split": "test", "path": "data/test-00000-of-00001-9c204eb3f4e11791.parquet",
     "sha256": "51c65f1788faff392abe2a0b55b023eb23e9be551c509138eaa3a832514224e7", "bytes": 234202795},
]
CATEGORY = {"total_amount": "total.total_price", "subtotal_amount": "sub_total.subtotal_price",
            "tax_amount": "sub_total.tax_price", "service_charge": "sub_total.service_price"}
# An explicit, versioned compatibility alias, not fuzzy field-name matching.
ALIASES = {"subtotal": "sub_total"}


def category_alias(value: str) -> str:
    parts = value.split(".")
    parts[0] = ALIASES.get(parts[0], parts[0])
    return ".".join(parts)


def assign_roles(ids_by_split: dict[str, list[str]]) -> dict[str, str]:
    if set(ids_by_split) != {"train", "validation", "test"}:
        raise ContractError("Official train/validation/test partitions are required")
    all_ids = [i for ids in ids_by_split.values() for i in ids]
    if len(all_ids) != len(set(all_ids)):
        raise ContractError("Receipt IDs collide across source partitions")
    order = sorted(ids_by_split["validation"], key=lambda i: hashlib.sha256(
        ("receipt-capstone-v1|42|" + i).encode()).hexdigest())
    if len(order) != 100:
        raise ContractError("Canonical CORD v2 validation must have 100 rows before exclusions")
    roles = {i: "train" for i in ids_by_split["train"]}
    roles.update({i: "test" for i in ids_by_split["test"]})
    roles.update({i: "validation_model" if j < 50 else "validation_policy" for j, i in enumerate(order)})
    return roles


def integrity_cohort(rows: list[dict[str, Any]], roles: dict[str, str]) -> tuple[list[str], list[dict[str, Any]]]:
    """Union exact byte/pixel families; quarantine cross-role families wholesale."""
    parents = {r["receipt_id"]: r["receipt_id"] for r in rows}
    def find(i: str) -> str:
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    seen: dict[tuple[str, str], str] = {}
    for row in rows:
        i = row["receipt_id"]
        for key in ("image_sha256", "pixel_sha256"):
            value = row.get(key)
            if not value:
                continue
            signature = key, value
            if signature in seen:
                parents[find(i)] = find(seen[signature])
            seen[signature] = i
    groups: dict[str, list[str]] = defaultdict(list)
    by_id = {r["receipt_id"]: r for r in rows}
    for i in parents:
        groups[find(i)].append(i)
    keep, excluded = [], []
    for family in groups.values():
        family = sorted(family)
        if len({roles[i] for i in family}) > 1:
            excluded.extend({"receipt_id": i, "reason": "cross_role_exact_duplicate", "family": family}
                            for i in family)
        else:
            valid = [i for i in family if by_id[i].get("source_status") == "ok"]
            canonical = valid[0] if valid else None
            for i in family:
                if i == canonical:
                    keep.append(i)
                else:
                    excluded.append({"receipt_id": i, "reason": "within_role_duplicate" if i in valid
                                     else by_id[i].get("source_status", "invalid_source"), "family": family})
    return sorted(keep), sorted(excluded, key=lambda r: r["receipt_id"])


def orientation_matrix(orientation: int, width: int, height: int) -> list[list[int]]:
    matrices = {
        1: [[1, 0, 0], [0, 1, 0]], 2: [[-1, 0, width], [0, 1, 0]],
        3: [[-1, 0, width], [0, -1, height]], 4: [[1, 0, 0], [0, -1, height]],
        5: [[0, 1, 0], [1, 0, 0]], 6: [[0, -1, height], [1, 0, 0]],
        7: [[0, -1, height], [-1, 0, width]], 8: [[0, 1, 0], [-1, 0, width]],
    }
    if orientation not in matrices:
        raise ContractError("Unsupported EXIF orientation")
    return matrices[orientation]


def transform_box(box: list[float], matrix: list[list[int]]) -> list[float]:
    points = [(box[x], box[y]) for x in (0, 2) for y in (1, 3)]
    mapped = [(matrix[0][0] * x + matrix[0][1] * y + matrix[0][2],
               matrix[1][0] * x + matrix[1][1] * y + matrix[1][2]) for x, y in points]
    return [min(p[0] for p in mapped), min(p[1] for p in mapped),
            max(p[0] for p in mapped), max(p[1] for p in mapped)]


def inspect_image(data: bytes) -> tuple[Any, dict[str, Any]]:
    """Return EXIF-oriented RGB and the original-to-oriented coordinate map."""
    import warnings
    from PIL import Image, ImageOps
    if len(data) > 20 * 1024**2:
        raise ContractError("Image exceeds 20 MiB")
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(data)) as image:
            if image.format not in ("JPEG", "PNG") or getattr(image, "n_frames", 1) != 1:
                raise ContractError("Only single-frame JPEG/PNG images are supported")
            width, height = image.size
            if width * height > 20_000_000 or max(width, height) > 10000 or min(width, height) < 1:
                raise ContractError("Image exceeds 20 MP / 10,000-pixel side ceiling")
            orientation = int(image.getexif().get(274, 1))
            matrix = orientation_matrix(orientation, width, height)
            image.load()
            result = ImageOps.exif_transpose(image).convert("RGB")
            fmt = image.format
    pixel_hash = hashlib.sha256(str(result.size).encode() + b"RGB\0" + result.tobytes()).hexdigest()
    return result, {"image_sha256": hashlib.sha256(data).hexdigest(), "pixel_sha256": pixel_hash,
                    "original_size": [width, height], "image_size": list(result.size), "format": fmt,
                    "orientation": {"exif": orientation, "matrix": matrix, "original_size": [width, height]}}


def _word_box(word: dict[str, Any], original_size: list[int], matrix: list[list[int]]) -> list[float]:
    quad = word["quad"]
    xs = [float(quad[f"x{k}"]) for k in range(1, 5)]
    ys = [float(quad[f"y{k}"]) for k in range(1, 5)]
    box = [min(xs), min(ys), max(xs), max(ys)]
    width, height = original_size
    if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
        raise ContractError("Reference word box is outside the original image")
    return transform_box(box, matrix)


def references(ground_truth: str | dict[str, Any], image_info: dict[str, Any],
               policy: str = "cord_v1", currency: str = "unspecified") -> dict[str, Any]:
    """Parse references prediction-blind; retain unknown/ambiguous states explicitly."""
    data = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    ref = {field: {"state": "not_annotated", "raw_text": None, "normalized_amount": None,
                   "value_box": None} for field in FIELDS}
    diagnostic_tokens = []
    source_values: dict[str, list[dict[str, Any]]] = defaultdict(list)
    errors = []
    meta_size = data.get("meta", {}).get("image_size", {})
    if [meta_size.get("width"), meta_size.get("height")] != image_info["original_size"]:
        return {"fields": {f: {**v, "state": "invalid_reference"} for f, v in ref.items()},
                "diagnostic_tokens": [], "errors": ["annotation_image_size_mismatch"]}
    for line_index, line in enumerate(data.get("valid_line", [])):
        value_words, value_boxes = [], []
        invalid = False
        for word_index, word in enumerate(line.get("words", [])):
            try:
                text = str(word["text"])
                box = _word_box(word, image_info["original_size"], image_info["orientation"]["matrix"])
                if not text.strip():
                    continue
                diagnostic_tokens.append({"id": len(diagnostic_tokens), "text": text, "box": box,
                                          "line": [0, 0, line_index], "word": word_index, "confidence": None})
                if not int(word.get("is_key", 0)):
                    value_words.append(text)
                    value_boxes.append(box)
            except (KeyError, ValueError, TypeError):
                invalid = True
                errors.append(f"invalid_word_geometry_or_text:{line_index}:{word_index}")
        category = category_alias(str(line.get("category", "")))
        source_values[category].append({"text": " ".join(value_words),
                                        "box": box_union(value_boxes) if value_boxes else None,
                                        "invalid": invalid})
    parsed_source = data.get("gt_parse", {})
    for field, category in CATEGORY.items():
        group, leaf = category.split(".")
        group_value = parsed_source.get(group)
        alias_group = parsed_source.get("subtotal") if group == "sub_total" else None
        if group_value is not None and alias_group is not None and group_value != alias_group:
            ref[field]["state"] = "ambiguous_reference"
            continue
        if group_value is None:
            group_value = alias_group or {}
        parsed_value = group_value.get(leaf) if isinstance(group_value, dict) else None
        lines = source_values.get(category, [])
        if not lines and parsed_value is None:
            continue
        if len(lines) != 1 or isinstance(parsed_value, (dict, list)):
            ref[field].update(state="ambiguous_reference", source_values=lines, parsed_value=parsed_value)
            continue
        line = lines[0]
        normalized = amount(line["text"], policy, currency)
        reference_state = "present_usable"
        if line["invalid"] or line["box"] is None:
            reference_state = "invalid_reference"
        elif normalized["parse_status"] != "ok":
            reference_state = "ambiguous_reference"
        if parsed_value is not None:
            parsed_normalized = amount(str(parsed_value), policy, currency)
            if (parsed_normalized["parse_status"] != "ok" or
                    parsed_normalized["normalized_amount"] != normalized["normalized_amount"]):
                reference_state = "ambiguous_reference"
        ref[field] = {"state": reference_state, "raw_text": line["text"],
                      "normalized_amount": normalized["normalized_amount"] if reference_state == "present_usable" else None,
                      "value_box": line["box"], "source_category": category,
                      "parsed_source_value": parsed_value, "currency": normalized["currency"]}
    return {"fields": ref, "diagnostic_tokens": diagnostic_tokens, "errors": errors}


def prepare_cord(destination: Path, cache: Path) -> dict[str, Any]:
    """Download all official shards, stream images, audit, then freeze roles/cohort."""
    import pyarrow.parquet as pq
    images_dir = destination / "images"
    images_dir.mkdir(parents=True)
    rows, evaluator_rows, downloads = [], [], []
    ids_by_split: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    offset: Counter[str] = Counter()
    for pin in SOURCE_FILES:
        path = download_pinned(
            f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{DATASET_REVISION}/{pin['path']}",
            cache / Path(pin["path"]).name, pin, ("huggingface.co",), max_bytes=600 * 1024**2)
        downloads.append({**pin, "downloaded_bytes": path.stat().st_size,
                          "locally_computed_sha256": file_hash(path), "verification": "complete_file_bytes"})
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1, columns=["image", "ground_truth"]):
            example = batch.to_pylist()[0]
            split = pin["split"]
            row_index = offset[split]
            offset[split] += 1
            i = safe_id(f"cord-{split}-{row_index:04d}")
            ids_by_split[split].append(i)
            record = {"receipt_id": i, "source_split": split, "source_row": row_index,
                      "source_path": pin["path"], "source_status": "ok"}
            try:
                image_bytes = example["image"]["bytes"]
                if not isinstance(image_bytes, bytes):
                    raise ContractError("Pinned CORD image column must carry bytes")
                image, info = inspect_image(image_bytes)
                record.update(info)
                extension = ".png" if info["format"] == "PNG" else ".jpg"
                relative_image = f"images/{i}{extension}"
                (destination / relative_image).write_bytes(image_bytes)
                record["image_path"] = relative_image
                del image
                gt = example["ground_truth"]
                record["annotation_sha256"] = hashlib.sha256(gt.encode("utf-8")).hexdigest()
                reference = references(gt, info)
                evaluator_rows.append({"receipt_id": i, **reference})
            except (ContractError, OSError, ValueError, TypeError, KeyError) as error:
                record["source_status"] = f"invalid_source:{type(error).__name__}"
                record["source_error"] = str(error)
            rows.append(record)
    if dict(offset) != {"train": 800, "validation": 100, "test": 100}:
        raise IntegrityError(f"Unexpected official source counts: {dict(offset)}")
    roles = assign_roles(ids_by_split)
    keep, exclusions = integrity_cohort(rows, roles)
    keep_set = set(keep)
    cohort = [{**r, "role": roles[r["receipt_id"]]} for r in rows if r["receipt_id"] in keep_set]
    evaluator_rows = [r for r in evaluator_rows if r["receipt_id"] in keep_set]
    # Metadata-only source audit. No model predictions are available here.
    counts = Counter(r["role"] for r in cohort)
    frequencies = {field: dict(Counter(r["fields"][field]["state"] for r in evaluator_rows)) for field in FIELDS}
    annotation_families: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if row.get("annotation_sha256"):
            annotation_families[row["annotation_sha256"]].append(row["receipt_id"])
    audit = {"source_counts": dict(offset), "role_counts": dict(counts), "cohort_count": len(cohort),
             "exclusions": exclusions, "field_reference_states": frequencies,
             "annotation_duplicate_groups": [v for v in annotation_families.values() if len(v) > 1],
             "near_duplicate_audit": "not_performed", "independent_human_review_count": 0,
             "pretraining_overlap": "unknown", "merchant_template_disjointness": "not_established",
             "label": "published-source exploratory tutorial, not independently verified financial truth"}
    write_jsonl(destination / "source_records.jsonl", rows)
    write_jsonl(destination / "cohort.jsonl", cohort)
    (destination / "references").mkdir()
    write_jsonl(destination / "references" / "references.jsonl", evaluator_rows)
    manifest = {"dataset_id": DATASET_ID, "revision": DATASET_REVISION, "files": downloads,
                "license": "CC-BY-4.0", "population": "Indonesian receipts",
                "cohort_digest": digest(cohort), "role_digest": digest(roles), "roles": roles,
                "reference_digest": digest(evaluator_rows)}
    write_json(destination / "data_manifest.json", manifest)
    write_json(destination / "dataset_audit.json", audit)
    return audit
