"""Optional bring-your-own-data intake (spec section 18): inference and adaptation/evaluation modes.

Receipts stay in the hosted runtime. The ZIP is validated before any OCR or training: safe paths,
limits, declared roles, disjoint groups, unique images, supported number/currency policies, and
checked value references with boxes in original image coordinates for adaptation mode.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from receipt_artifact import safe_extract
from receipt_common import (
    FIELDS,
    ContractError,
    digest,
    read_json,
    read_jsonl,
    safe_receipt_id,
    safe_relative,
    write_json,
    write_jsonl,
)
from receipt_data import inspect_image, transform_box
from receipt_fields import amount, valid_box

BYOD_SCHEMA = "org.dimer.receipt-byod.v1"
BYOD_POLICIES = ("dot_decimal_comma_grouping", "comma_decimal_dot_grouping")
ADAPT_ROLES = ("train", "validation_model", "validation_policy", "test", "inference")
MAX_IMAGES = 1000
PRIVACY_NOTICE = (
    "Process only receipts you are authorised to process in this hosted runtime. Remove "
    "unnecessary personal or payment details first. Nothing is sent to an OCR/LLM API; review "
    "every export before sharing. Trained weights are not guaranteed privacy-free."
)


def validate_manifest(manifest: dict[str, Any], mode: str) -> dict[str, Any]:
    if manifest.get("schema") != BYOD_SCHEMA:
        raise ContractError(f"manifest.json must declare schema {BYOD_SCHEMA}")
    if manifest.get("authorized") is not True:
        raise ContractError("manifest.json must set authorized=true for receipts you may process")
    if manifest.get("number_format_policy") not in BYOD_POLICIES:
        raise ContractError(f"number_format_policy must be one of {BYOD_POLICIES}")
    currency = manifest.get("currency")
    if not isinstance(currency, str) or not (
        currency == "unspecified" or (len(currency) == 3 and currency.isupper())
    ):
        raise ContractError("currency must be a three-letter code such as PHP, or 'unspecified'")
    receipts = manifest.get("receipts")
    if not isinstance(receipts, list) or not 1 <= len(receipts) <= MAX_IMAGES:
        raise ContractError(f"receipts must list 1..{MAX_IMAGES} entries")
    ids, images = set(), set()
    for entry in receipts:
        rid = safe_receipt_id(entry.get("receipt_id", ""))
        image = str(safe_relative(entry.get("image", "")))
        if rid in ids or image in images:
            raise ContractError(f"Duplicate receipt id or image path: {rid}")
        if not image.lower().endswith((".png", ".jpg", ".jpeg")):
            raise ContractError("Only JPEG/PNG receipt images are accepted")
        ids.add(rid)
        images.add(image)
        if mode == "adapt":
            if entry.get("role") not in ADAPT_ROLES:
                raise ContractError(f"{rid}: role must be one of {ADAPT_ROLES}")
            if not isinstance(entry.get("group_id"), str) or not entry["group_id"]:
                raise ContractError(f"{rid}: adaptation mode needs a group_id (merchant/batch) per receipt")
        elif entry.get("role", "inference") != "inference":
            raise ContractError("Inference mode accepts only role 'inference'")
    if mode == "adapt":
        roles_by_group = defaultdict(set)
        for entry in receipts:
            roles_by_group[entry["group_id"]].add(entry["role"])
        crossing = sorted(g for g, roles in roles_by_group.items() if len(roles) > 1)
        if crossing:
            raise ContractError(f"Groups appear in more than one role: {crossing[:5]}")
        counts = Counter(e["role"] for e in receipts)
        missing = [r for r in ("train", "validation_model", "validation_policy", "test") if not counts[r]]
        if missing:
            raise ContractError(f"Adaptation/evaluation mode requires nonempty roles; missing {missing}")
    return {"policy": manifest["number_format_policy"], "currency": currency, "count": len(receipts)}


def _annotation(entry: dict[str, Any], info: dict[str, Any], policy: str, currency: str) -> dict[str, Any]:
    fields = entry.get("fields")
    if not isinstance(fields, dict) or set(fields) - set(FIELDS):
        raise ContractError(f"{entry.get('receipt_id')}: annotation fields must be a subset of {FIELDS}")
    out = {}
    width, height = info["original_size"]
    for field in FIELDS:
        item = fields.get(field, {"status": "not_annotated"})
        status = item.get("status")
        if status == "not_annotated":
            out[field] = {
                "state": "not_annotated",
                "raw_text": None,
                "normalized_amount": None,
                "value_boxes": [],
            }
            continue
        if status != "present":
            raise ContractError(f"{entry['receipt_id']}.{field}: status must be 'present' or 'not_annotated'")
        parsed = amount(item.get("raw_value"), policy, currency)
        if parsed["parse_status"] != "ok":
            raise ContractError(f"{entry['receipt_id']}.{field}: raw_value is not a supported amount")
        if "canonical_value" in item and item["canonical_value"] != parsed["normalized_amount"]:
            raise ContractError(f"{entry['receipt_id']}.{field}: canonical_value contradicts raw_value")
        boxes = item.get("value_boxes")
        if not isinstance(boxes, list) or not boxes or not all(valid_box(b, width, height) for b in boxes):
            raise ContractError(
                f"{entry['receipt_id']}.{field}: value_boxes must be valid original-image xyxy boxes"
            )
        if not isinstance(item.get("provenance"), str) or not item["provenance"]:
            raise ContractError(f"{entry['receipt_id']}.{field}: annotation provenance is required")
        matrix = info["orientation"]["matrix"]
        out[field] = {
            "state": "present_usable",
            "raw_text": item["raw_value"],
            "normalized_amount": parsed["normalized_amount"],
            "value_boxes": [transform_box([float(v) for v in b], matrix) for b in boxes],
            "provenance": item["provenance"],
            "currency": currency,
        }
    return out


def prepare_byod(
    archive: Path, mode: str, inputs_dir: Path, references_dir: Path, image_store: Path, workdir: Path
) -> dict[str, Any]:
    """Validate and convert a BYOD ZIP into the same cohort/reference files as CORD preparation."""
    if mode not in ("inference", "adapt"):
        raise ContractError("BYOD_MODE must be 'inference' or 'adapt'")
    extracted = workdir / "byod_extracted"
    names = safe_extract(archive, extracted, allowed_suffixes=(".json", ".jsonl", ".png", ".jpg", ".jpeg"))
    if "manifest.json" not in names:
        raise ContractError("The BYOD ZIP needs manifest.json at its root")
    manifest = read_json(extracted / "manifest.json")
    contract = validate_manifest(manifest, mode)
    declared = {"manifest.json"} | {e["image"] for e in manifest["receipts"]}
    if mode == "adapt":
        declared.add("annotations.jsonl")
    if set(names) != declared:
        raise ContractError(f"Undeclared or missing archive members: {sorted(set(names) ^ declared)[:5]}")
    annotations = {}
    if mode == "adapt":
        for row in read_jsonl(extracted / "annotations.jsonl"):
            rid = safe_receipt_id(row.get("receipt_id", ""))
            if rid in annotations:
                raise ContractError(f"Duplicate annotation for {rid}")
            annotations[rid] = row
    image_store.mkdir(parents=True, exist_ok=True)
    cohort, evaluator, hashes = [], [], {}
    for entry in manifest["receipts"]:
        data = (extracted / entry["image"]).read_bytes()
        try:
            _, info = inspect_image(data)
        except ContractError as error:
            raise ContractError(f"{entry['receipt_id']}: {error}") from error
        if info["pixel_sha256"] in hashes:
            raise ContractError(f"{entry['receipt_id']} duplicates {hashes[info['pixel_sha256']]}")
        hashes[info["pixel_sha256"]] = entry["receipt_id"]
        stored = image_store / f"{info['image_sha256']}.{'png' if info['format'] == 'PNG' else 'jpg'}"
        stored.write_bytes(data)
        role = entry.get("role", "inference")
        cohort.append(
            {
                "receipt_id": entry["receipt_id"],
                "role": role,
                "official_split": "byod",
                "group_id": hashlib.sha256(str(entry.get("group_id", "")).encode()).hexdigest()[:16],
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
        if mode == "adapt" and role != "inference":
            if entry["receipt_id"] not in annotations:
                raise ContractError(f"Missing annotations for {entry['receipt_id']}")
            fields = _annotation(
                annotations[entry["receipt_id"]], info, contract["policy"], contract["currency"]
            )
            evaluator.append(
                {
                    "receipt_id": entry["receipt_id"],
                    "role": role,
                    "fields": fields,
                    "diagnostic_tokens": [],
                    "key_phrases": [],
                    "errors": [],
                    "parser_id": "byod-v1",
                }
            )
    extra = set(annotations) - {e["receipt_id"] for e in manifest["receipts"]}
    if extra:
        raise ContractError(f"Annotations reference undeclared receipts: {sorted(extra)[:5]}")
    counts = Counter(r["role"] for r in cohort)
    write_jsonl(inputs_dir / "cohort.jsonl", sorted(cohort, key=lambda r: r["receipt_id"]))
    write_json(
        inputs_dir / "data_card.json",
        {
            "dataset_id": "byod",
            "mode": mode,
            "population": "user-supplied authorised receipts",
            "number_format_policy": contract["policy"],
            "currency": contract["currency"],
            "frozen_role_counts": dict(counts),
            "cohort_digest": digest(cohort),
            "privacy": PRIVACY_NOTICE,
            "small_sample": len(cohort) < 200,
            "reduced_fixture": False,
        },
    )
    write_jsonl(references_dir / "references.jsonl", sorted(evaluator, key=lambda r: r["receipt_id"]))
    write_json(
        references_dir / "reference_audit.json",
        {
            "parser_id": "byod-v1",
            "field_reference_states": {
                f: dict(Counter(r["fields"][f]["state"] for r in evaluator)) for f in FIELDS
            },
            "independent_human_review_count": 0,
            "label": "user-supplied references; not verified by the notebook",
        },
    )
    return {"mode": mode, "counts": dict(counts), **contract}


def example_manifest() -> dict[str, Any]:
    """A documentation example (not data)."""
    return {
        "schema": BYOD_SCHEMA,
        "authorized": True,
        "number_format_policy": "dot_decimal_comma_grouping",
        "currency": "PHP",
        "processing_scope": "capstone exercise in this hosted runtime",
        "receipts": [
            {"receipt_id": "r001", "image": "images/r001.png", "role": "train", "group_id": "store-a"}
        ],
    }
