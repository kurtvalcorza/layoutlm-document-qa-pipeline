"""Bundle verification, safe archives and BYOD refusals (spec sections 15 and 18)."""

import io
import json
import stat
import zipfile
from pathlib import Path

import pytest
import receipt_artifact as art
import receipt_byod as byod
from receipt_common import ContractError, IntegrityError, spreadsheet_safe

from _receipt_fixtures import render

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

BASE = {"model_id": "m", "revision": "r", "weight_sha256": "w", "tokenizer_sha256": {"tokenizer.json": "t"}}
PARTS = {
    name: {"name": name}
    for name in (
        "model_config",
        "field_schema",
        "questions",
        "preprocessing",
        "ocr_manifest",
        "number_format_policy",
        "keyword_rules",
        "review_policy",
        "runtime_manifest",
    )
}


def bundle(tmp_path: Path) -> Path:
    tensors = {"qa_outputs.weight": torch.ones(2, 3), "qa_outputs.bias": torch.zeros(2)}
    art.export_bundle(tmp_path / "b", tensors, PARTS, BASE, {"selected_epoch": 1})
    return tmp_path / "b"


def test_bundle_round_trip_and_contents(tmp_path: Path):
    root = bundle(tmp_path)
    manifest = art.verify_bundle(
        root, expected_base=BASE, allowed_tensors={"qa_outputs.weight", "qa_outputs.bias"}
    )
    assert manifest["contains"].startswith("trained tensor subset")
    assert sorted(art.load_tensors(root)) == ["qa_outputs.bias", "qa_outputs.weight"]


@pytest.mark.parametrize("attack", ["extra_file", "altered_policy", "base", "tensor_set", "pinned_digest"])
def test_bundle_refusals(tmp_path: Path, attack):
    root = bundle(tmp_path)
    kwargs = {"expected_base": BASE, "allowed_tensors": {"qa_outputs.weight", "qa_outputs.bias"}}
    if attack == "extra_file":
        (root / "evil.py").write_text("print(1)")
    elif attack == "altered_policy":
        (root / "review_policy.json").write_text(json.dumps({"threshold": 0}))
    elif attack == "base":
        kwargs["expected_base"] = {**BASE, "revision": "other"}
    elif attack == "tensor_set":
        kwargs["allowed_tensors"] = {"qa_outputs.weight"}
    else:
        kwargs["expected"] = {"review_policy.json": "0" * 64}
    with pytest.raises(IntegrityError):
        art.verify_bundle(root, **kwargs)


def _zip(entries, *, symlink=None):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        for name, data in entries.items():
            z.writestr(name, data)
        if symlink:
            info = zipfile.ZipInfo(symlink)
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            z.writestr(info, "/etc/passwd")
    return buffer.getvalue()


@pytest.mark.parametrize(
    "entries,symlink",
    [
        ({"../x.json": "{}"}, None),
        ({"/abs.json": "{}"}, None),
        ({"a.json": "{}"}, "link.json"),
        ({"a.exe": "x"}, None),
    ],
)
def test_safe_extract_refuses_unsafe_archives(tmp_path: Path, entries, symlink):
    path = tmp_path / "a.zip"
    path.write_bytes(_zip(entries, symlink=symlink))
    with pytest.raises(ContractError):
        art.safe_extract(path, tmp_path / "out", allowed_suffixes=(".json", ".png"))


def test_safe_extract_refuses_bombs_and_duplicates(tmp_path: Path):
    bomb = tmp_path / "bomb.zip"
    with zipfile.ZipFile(bomb, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("a.json", "0" * 5_000_000)
    with pytest.raises(ContractError, match="ratio"):
        art.safe_extract(bomb, tmp_path / "o1")
    dup = tmp_path / "dup.zip"
    with zipfile.ZipFile(dup, "w") as z:
        z.writestr("a.json", "{}")
        z.writestr("A.json", "{}")
    with pytest.raises(ContractError, match="Duplicate"):
        art.safe_extract(dup, tmp_path / "o2")


def _png(lines):
    image, _ = render(lines)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue(), image.size


def byod_zip(tmp_path: Path, mode="adapt", **overrides) -> Path:
    receipts, files, annotations = [], {}, []
    roles = ["train"] * 3 + ["validation_model", "validation_policy", "test"]
    for i, role in enumerate(roles):
        data, size = _png([["TOTAL", f"{i + 1},250.00"]])
        files[f"images/r{i}.png"] = data
        receipts.append(
            {"receipt_id": f"r{i}", "image": f"images/r{i}.png", "role": role, "group_id": f"g{i}"}
        )
        annotations.append(
            {
                "receipt_id": f"r{i}",
                "fields": {
                    "total_amount": {
                        "status": "present",
                        "raw_value": f"{i + 1},250.00",
                        "value_boxes": [[100, 10, 200, 40]],
                        "provenance": "checked by the owner",
                    }
                },
            }
        )
    manifest = {
        "schema": byod.BYOD_SCHEMA,
        "authorized": True,
        "number_format_policy": "dot_decimal_comma_grouping",
        "currency": "PHP",
        "receipts": receipts
        if mode == "adapt"
        else [{k: v for k, v in r.items() if k not in ("role", "group_id")} for r in receipts],
    }
    manifest.update(overrides)
    files["manifest.json"] = json.dumps(manifest)
    if mode == "adapt":
        files["annotations.jsonl"] = "\n".join(json.dumps(a) for a in annotations)
    path = tmp_path / "byod.zip"
    path.write_bytes(_zip(files))
    return path


def test_byod_adapt_converts_to_cohort_and_references(tmp_path: Path):
    summary = byod.prepare_byod(
        byod_zip(tmp_path), "adapt", tmp_path / "in", tmp_path / "ref", tmp_path / "img", tmp_path / "work"
    )
    assert summary["counts"] == {"train": 3, "validation_model": 1, "validation_policy": 1, "test": 1}
    refs = [json.loads(line) for line in (tmp_path / "ref" / "references.jsonl").read_text().splitlines()]
    assert refs[0]["fields"]["total_amount"]["normalized_amount"] == "1250"  # ".00" is not a distinct value
    assert refs[0]["fields"]["tax_amount"]["state"] == "not_annotated"


def test_byod_inference_mode(tmp_path: Path):
    summary = byod.prepare_byod(
        byod_zip(tmp_path, mode="inference"),
        "inference",
        tmp_path / "in",
        tmp_path / "ref",
        tmp_path / "img",
        tmp_path / "work",
    )
    assert summary["counts"] == {"inference": 6}


@pytest.mark.parametrize(
    "override,match",
    [
        ({"authorized": False}, "authorized"),
        ({"number_format_policy": "cord_mixed_v1"}, "number_format_policy"),
        ({"currency": "peso"}, "currency"),
        ({"schema": "x"}, "schema"),
    ],
)
def test_byod_manifest_refusals(tmp_path: Path, override, match):
    with pytest.raises(ContractError, match=match):
        byod.prepare_byod(
            byod_zip(tmp_path, **override),
            "adapt",
            tmp_path / "in",
            tmp_path / "ref",
            tmp_path / "img",
            tmp_path / "work",
        )


def test_byod_group_crossing_roles_is_refused():
    receipts = [
        {"receipt_id": "a", "image": "a.png", "role": "train", "group_id": "g"},
        {"receipt_id": "b", "image": "b.png", "role": "test", "group_id": "g"},
    ]
    manifest = {
        "schema": byod.BYOD_SCHEMA,
        "authorized": True,
        "number_format_policy": "dot_decimal_comma_grouping",
        "currency": "PHP",
        "receipts": receipts,
    }
    with pytest.raises(ContractError, match="more than one role"):
        byod.validate_manifest(manifest, "adapt")


def test_byod_missing_roles_and_contradictory_values_are_refused(tmp_path: Path):
    manifest = {
        "schema": byod.BYOD_SCHEMA,
        "authorized": True,
        "number_format_policy": "dot_decimal_comma_grouping",
        "currency": "PHP",
        "receipts": [{"receipt_id": "a", "image": "a.png", "role": "train", "group_id": "g"}],
    }
    with pytest.raises(ContractError, match="missing"):
        byod.validate_manifest(manifest, "adapt")
    info = {"original_size": [100, 100], "orientation": {"matrix": [[1, 0, 0], [0, 1, 0]]}}
    entry = {
        "receipt_id": "a",
        "fields": {
            "total_amount": {
                "status": "present",
                "raw_value": "1,250.00",
                "canonical_value": "125000",
                "value_boxes": [[1, 1, 5, 5]],
                "provenance": "x",
            }
        },
    }
    with pytest.raises(ContractError, match="contradicts"):
        byod._annotation(entry, info, "dot_decimal_comma_grouping", "PHP")
    entry["fields"]["total_amount"].update(canonical_value="1250", value_boxes=[[1, 1, 500, 5]])
    with pytest.raises(ContractError, match="value_boxes"):
        byod._annotation(entry, info, "dot_decimal_comma_grouping", "PHP")


def test_spreadsheet_formula_strings_are_neutralised_but_numbers_are_not():
    assert spreadsheet_safe("=HYPERLINK(1)") == "'=HYPERLINK(1)" and spreadsheet_safe("@x") == "'@x"
    assert spreadsheet_safe(-1.5) == "-1.5" and spreadsheet_safe(None) == ""
