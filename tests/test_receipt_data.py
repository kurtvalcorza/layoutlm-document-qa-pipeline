"""Frozen-manifest loading, roles, duplicates, orientation and reference states (spec sections 4-6)."""

import copy
import io
import json
from pathlib import Path

import pytest
import receipt_data as data
from receipt_common import ContractError, IntegrityError, read_jsonl

from _receipt_fixtures import ROOT, TOOLS, make_dataset

MANIFESTS = ROOT / "tutorials" / "receipt_intelligence"


def committed():
    return {
        "data": json.loads((MANIFESTS / "data_manifest.json").read_text(encoding="utf-8")),
        "split": json.loads((MANIFESTS / "split_manifest.json").read_text(encoding="utf-8")),
    }


def test_committed_split_is_reproduced_from_manifest_facts():
    result = data.check_frozen_split(committed())
    assert result["role_counts"] == {
        "train": 798,
        "validation_model": 50,
        "validation_policy": 50,
        "test": 100,
    }
    assert {e["receipt_id"] for e in result["exclusions"]} == {"cord-v2:train:0109", "cord-v2:train:0285"}


def test_tampered_split_manifest_is_refused():
    manifests = committed()
    row = next(r for r in manifests["split"]["rows"] if r["role"] == "validation_model")
    row["role"] = "validation_policy"
    with pytest.raises(IntegrityError):
        data.check_frozen_split(manifests)


def test_all_three_image_partitions_are_pinned_by_full_file_digest():
    files = committed()["data"]["files"]
    assert sorted({f["official_split"] for f in files}) == ["test", "train", "validation"]
    assert all(
        f["bytes"] and len(f["sha256"]) == 64 and f["url"].startswith("https://huggingface.co/")
        for f in files
    )
    assert all("7f0115a4b758a71d6473b8d085751692da2fef98" in f["url"] for f in files)


def test_role_assignment_is_deterministic_and_salted():
    ids = {"train": ["a"], "validation": [f"v{i}" for i in range(10)], "test": ["t"]}
    first, second = data.assign_roles(ids), data.assign_roles(ids)
    assert first == second
    assert sum(1 for v in first.values() if v == "validation_model") == 5
    salts = {
        tuple(sorted(k for k, v in data.assign_roles(ids, salt=f"s{i}|").items() if v == "validation_model"))
        for i in range(5)
    }
    assert len(salts) > 1  # the salt participates in the order
    with pytest.raises(ContractError):
        data.assign_roles({"train": ["x"], "validation": ["x"], "test": []})


def test_cross_role_duplicates_are_quarantined_wholesale():
    rows = [
        {"receipt_id": "a", "pixel_sha256": "p"},
        {"receipt_id": "b", "pixel_sha256": "p"},
        {"receipt_id": "c", "pixel_sha256": "q"},
        {"receipt_id": "d", "pixel_sha256": "q"},
    ]
    keep, excluded = data.integrity_cohort(rows, {"a": "train", "b": "test", "c": "train", "d": "train"})
    assert keep == ["c"]
    assert {e["receipt_id"]: e["reason"] for e in excluded} == {
        "a": "cross_role_exact_duplicate",
        "b": "cross_role_exact_duplicate",
        "d": "within_role_exact_duplicate",
    }


@pytest.mark.parametrize("orientation", range(1, 9))
def test_orientation_matrix_matches_pillow_and_inverts(orientation):
    from PIL import Image, ImageOps

    image = Image.new("RGB", (7, 4), "white")
    image.putpixel((1, 2), (255, 0, 0))
    exif = Image.Exif()
    exif[274] = orientation
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", exif=exif, quality=100)
    with Image.open(io.BytesIO(buffer.getvalue())) as opened:
        oriented = ImageOps.exif_transpose(opened)
    matrix = data.orientation_matrix(orientation, 7, 4)
    box = data.transform_box([1, 2, 2, 3], matrix)
    assert oriented.size == tuple(int(v) for v in data.transform_box([0, 0, 7, 4], matrix)[2:])
    back = data.transform_box(box, data.inverse_matrix(matrix))
    assert back == [1, 2, 2, 3]


def _gt(lines, parse, size=(100, 100)):
    valid = []
    for category, words in lines:
        valid.append(
            {
                "category": category,
                "words": [
                    {
                        "quad": {
                            "x1": x,
                            "y1": 10,
                            "x2": x + 9,
                            "y2": 10,
                            "x3": x + 9,
                            "y3": 20,
                            "x4": x,
                            "y4": 20,
                        },
                        "is_key": key,
                        "row_id": 1,
                        "text": text,
                    }
                    for text, key, x in words
                ],
            }
        )
    return {
        "gt_parse": parse,
        "meta": {"image_size": {"width": size[0], "height": size[1]}},
        "valid_line": valid,
    }


INFO = {"original_size": [100, 100], "orientation": {"matrix": data.orientation_matrix(1, 100, 100)}}


def test_reference_states_and_not_annotated_is_not_zero():
    gt = _gt(
        [
            ("total.total_price", [("TOTAL", 1, 0), ("75.000", 0, 40)]),
            ("sub_total.subtotal_price", [("70.000", 0, 40)]),
        ],
        {"total": {"total_price": "75.000"}, "sub_total": {"subtotal_price": "70.000"}},
    )
    ref = data.references(gt, INFO)["fields"]
    assert (
        ref["total_amount"]["state"] == "present_usable"
        and ref["total_amount"]["normalized_amount"] == "75000"
    )
    assert ref["total_amount"]["raw_text"] == "75.000"  # key words never enter the value
    assert ref["tax_amount"] == {**ref["tax_amount"], "state": "not_annotated", "normalized_amount": None}


def test_subtotal_alias_and_gt_parse_disagreement():
    alias = _gt(
        [("subtotal.subtotal_price", [("70.000", 0, 40)])], {"subtotal": {"subtotal_price": "70.000"}}
    )
    assert data.references(alias, INFO)["fields"]["subtotal_amount"]["state"] == "present_usable"
    disagree = _gt([("total.total_price", [("75.000", 0, 40)])], {"total": {"total_price": "76.000"}})
    assert data.references(disagree, INFO)["fields"]["total_amount"]["state"] == "ambiguous_reference"


def test_multiple_lines_same_value_usable_but_different_values_ambiguous():
    same = _gt([("total.total_price", [("75.000", 0, 40)]), ("total.total_price", [("75,000", 0, 60)])], {})
    ref = data.references(same, INFO)["fields"]["total_amount"]
    assert ref["state"] == "present_usable" and len(ref["value_boxes"]) == 2
    diff = _gt([("total.total_price", [("75.000", 0, 40)]), ("total.total_price", [("80.000", 0, 60)])], {})
    assert data.references(diff, INFO)["fields"]["total_amount"]["state"] == "ambiguous_reference"


def test_image_size_mismatch_invalidates_all_references():
    gt = _gt([("total.total_price", [("75.000", 0, 40)])], {}, size=(99, 100))
    assert {v["state"] for v in data.references(gt, INFO)["fields"].values()} == {"invalid_reference"}


def test_diagnostic_tokens_carry_no_categories_or_key_flags():
    gt = _gt([("total.total_price", [("TOTAL", 1, 0), ("75.000", 0, 40)])], {})
    tokens = data.references(gt, INFO)["diagnostic_tokens"]
    assert [t["text"] for t in tokens] == ["TOTAL", "75.000"]
    assert all(set(t) == {"id", "text", "box", "line", "word", "confidence"} for t in tokens)


def test_host_sample_loader_is_never_used():
    for path in TOOLS.glob("receipt_*.py"):
        text = path.read_text(encoding="utf-8")
        assert "fetch_sample_dataset" not in text and "layoutlm_document_qa_pipeline" not in text, path.name


def test_prepare_verifies_rows_and_separates_references(tmp_path: Path):
    ds = make_dataset(tmp_path, {"train": 3, "validation": 2, "test": 2})
    summary = data.prepare_cord(
        tmp_path / "inputs", tmp_path / "refs", tmp_path / "images", ds["cache"] / "cord", ds["manifests"]
    )
    assert summary["role_counts"] == {"train": 3, "validation_model": 1, "validation_policy": 1, "test": 2}
    cohort = read_jsonl(tmp_path / "inputs" / "cohort.jsonl")
    assert all("fields" not in r and "gt_parse" not in json.dumps(r) for r in cohort)
    refs = read_jsonl(tmp_path / "refs" / "references.jsonl")
    assert {r["receipt_id"] for r in refs} == {r["receipt_id"] for r in cohort}


def test_prepare_refuses_changed_annotation_bytes(tmp_path: Path):
    ds = make_dataset(tmp_path, {"train": 2, "validation": 2, "test": 1})
    manifests = copy.deepcopy(ds["manifests"])
    manifests["data"]["rows"][0]["annotation_sha256"] = "0" * 64
    with pytest.raises(IntegrityError):
        data.prepare_cord(tmp_path / "i", tmp_path / "r", tmp_path / "im", ds["cache"] / "cord", manifests)
