"""Full stage chain on CPU with labelled test doubles, plus stage ownership and gold non-interference.

Synthetic receipts, recorded OCR tokens and a tiny random LayoutLM: this proves the stages connect,
refuse tampering and keep references out of every fit/selection step. It is not performance evidence.
"""

import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from _receipt_fixtures import carrier, make_dataset, tiny_snapshot

pytest.importorskip("torch")
pytest.importorskip("matplotlib")

from receipt_common import IntegrityError, StageStore, read_json, read_jsonl  # noqa: E402

ALL = (
    "prepare",
    "references",
    "ocr_runtime",
    "model",
    "ocr",
    "keywords",
    "rules",
    "frozen",
    "train",
    "select_policy",
    "freeze",
    "evaluate",
    "diagnose",
    "export",
    "replay",
    "report",
)
UNTIL_FREEZE = ALL[: ALL.index("freeze") + 1]


def build_run(base: Path, stages, shuffle_test_totals=False) -> Path:
    ds = make_dataset(base, shuffle_test_totals=shuffle_test_totals)
    snapshot, _ = tiny_snapshot(base)
    root = carrier(base, ds["manifests"])
    config = {
        "run_id": "fixture",
        "source": "cord",
        "mode": "canonical",
        "cache_dir": str(ds["cache"]),
        "device": "cpu",
        "number_format_policy": "cord_mixed_v1",
        "currency": "unspecified",
        "model_override": {
            "snapshot": str(snapshot),
            "manifest": str(base / "tiny-model-manifest.json"),
            "random": True,
        },
        "ocr_test_double": str(base / "ocr_double.json"),
        "ocr_workers": 2,
    }
    (root / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
    for stage in stages:
        run(root, "--stage", stage)
    return root


def run(root: Path, *args, check=True):
    done = subprocess.run(
        [sys.executable, str(root / "capstone.py"), "--root", str(root), *args],
        capture_output=True,
        text=True,
    )
    if check and done.returncode:
        raise AssertionError(f"{args} failed:\n{done.stdout[-2000:]}\n{done.stderr[-4000:]}")
    return done


@pytest.fixture(scope="module")
def full_run(tmp_path_factory) -> Path:
    return build_run(tmp_path_factory.mktemp("chain"), ALL)


def test_chain_produces_required_outputs(full_run: Path):
    out = full_run / "outputs"
    for name in (
        "records.csv",
        "field_metrics.json",
        "system_comparison.csv",
        "review_policy.json",
        "review_metrics.json",
        "failure_summary.csv",
        "training_history.json",
        "selection_record.json",
        "run_manifest.json",
        "summary.md",
        "results.zip",
        "replay_report.json",
    ):
        assert (out / name).is_file(), name
    assert read_json(out / "replay_report.json")["all_parity"] is True
    history = read_json(out / "training_history.json")
    assert [h["epoch"] for h in history["history"]] == [1, 2, 3, 4] and 1 <= history["selected_epoch"] <= 4
    assert history["optimizer_steps"] > 0 and history["changed_tensors"] > 0


def test_every_frozen_receipt_has_a_record_for_every_system(full_run: Path):
    lines = read_jsonl(full_run / "stages" / "export" / "predictions.jsonl")
    cohort = [
        r
        for r in read_jsonl(full_run / "stages" / "prepare" / "cohort.jsonl")
        if r["role"] in ("validation_model", "validation_policy", "test")
    ]
    assert len(lines) == 3 * len(cohort)
    for record in lines:
        assert record["schema"] == "org.dimer.receipt-intelligence.prediction.v1"
        assert set(record["fields"]) == {"total_amount", "subtotal_amount", "tax_amount", "service_charge"}
        assert record["input_source"] == "actual_ocr" and record["total_review"]["state"] in (
            "needs_review",
            "total_unflagged",
        )
        assert "reference" not in json.dumps(record["fields"])
        assert record["arithmetic_check"]["state"] == "not_evaluable"


def test_rule_baseline_recovers_synthetic_totals(full_run: Path):
    metrics = read_json(full_run / "outputs" / "field_metrics.json")
    assert metrics["systems"]["rules_baseline"]["total_em"] == 1.0  # synthetic layout matches the rules


def test_evidence_zip_and_bundle_hold_no_images_or_references(full_run: Path):
    with zipfile.ZipFile(full_run / "outputs" / "results.zip") as archive:
        names = archive.namelist()
    assert not [n for n in names if n.endswith((".png", ".jpg", ".parquet"))]
    bundle = full_run / "stages" / "export" / "receipt_intelligence_artifact"
    text = "".join(p.read_text(encoding="utf-8", errors="ignore") for p in bundle.glob("*.json"))
    assert "normalized_amount" not in text and "value_boxes" not in text


def test_test_references_are_read_only_after_freeze(full_run: Path):
    log = read_jsonl(full_run / "outputs" / "reference_access_log.jsonl")
    purposes = [(e["purpose"], tuple(e["roles"])) for e in log]
    first_test = min(i for i, (_, roles) in enumerate(purposes) if "test" in roles)
    assert all(
        p in ("final_evaluation", "reference_text_diagnostic") for p, roles in purposes if "test" in roles
    )
    assert all("test" not in roles for _, roles in purposes[:first_test])
    assert ("policy_selection", ("validation_policy",)) in purposes[:first_test]


@pytest.mark.parametrize("target", ["adapter", "policy"])
def test_evaluation_refuses_changes_after_freeze(full_run: Path, tmp_path: Path, target):
    copy = tmp_path / "run"
    shutil.copytree(full_run, copy)
    if target == "adapter":
        path = copy / "stages" / "train" / "adapter.safetensors"
        path.write_bytes(path.read_bytes()[:-1] + b"\x00")
    else:
        path = copy / "stages" / "select_policy" / "review_policy.json"
        path.write_text(
            path.read_text(encoding="utf-8").replace('"policy_feasible"', '"policy_feasible_"'),
            encoding="utf-8",
        )
    done = run(copy, "--stage", "evaluate", check=False)
    assert done.returncode != 0 and "IntegrityError" in done.stderr


def test_activity_writes_separately_and_leaves_canonical_policy(full_run: Path):
    before = (full_run / "stages" / "select_policy" / "review_policy.json").read_bytes()
    run(full_run, "--activity-target", "0.9")
    assert (full_run / "outputs" / "activity" / "target_0.90" / "activity_comparison.csv").is_file()
    assert (full_run / "stages" / "select_policy" / "review_policy.json").read_bytes() == before
    run(full_run, "--stage", "report")  # the canonical chain still verifies


def test_changed_test_labels_do_not_change_any_fitted_or_selected_state(full_run: Path, tmp_path: Path):
    other = build_run(tmp_path, UNTIL_FREEZE, shuffle_test_totals=True)
    for relative in (
        "train/adapter.safetensors",
        "keywords/keyword_rules.json",
        "select_policy/review_policy.json",
        "rules/predictions_rules.jsonl",
        "frozen/predictions_frozen.jsonl",
        "train/predictions_adapted.jsonl",
    ):
        assert (other / "stages" / relative).read_bytes() == (full_run / "stages" / relative).read_bytes(), (
            relative
        )
    a = read_json(full_run / "stages" / "freeze" / "selection_record.json")
    b = read_json(other / "stages" / "freeze" / "selection_record.json")
    for key in ("frozen_at", "record_digest"):
        a.pop(key), b.pop(key)
    assert a == b
    refs_a = {r["receipt_id"]: r for r in read_jsonl(full_run / "stages" / "references" / "references.jsonl")}
    refs_b = {r["receipt_id"]: r for r in read_jsonl(other / "stages" / "references" / "references.jsonl")}
    changed = [
        i for i in refs_a if refs_a[i]["fields"]["total_amount"] != refs_b[i]["fields"]["total_amount"]
    ]
    assert changed and all(refs_a[i]["role"] == "test" for i in changed)


def test_stage_store_failed_retry_leaves_no_stale_success(tmp_path: Path):
    graph = {"a": [], "b": ["a"]}
    store = StageStore(tmp_path, "code", {"c": 1}, graph)
    with store.stage("a") as out:
        (out / "x.txt").write_text("1")
    with store.stage("b") as out:
        (out / "y.txt").write_text("2")
    store.verify("b")
    with pytest.raises(RuntimeError), store.stage("a") as out:
        (out / "x.txt").write_text("changed")
        raise RuntimeError("simulated failure")
    assert not (tmp_path / "a").exists() and not (tmp_path / "b").exists()
    with pytest.raises((IntegrityError, FileNotFoundError)):
        store.verify("b")


def test_stage_store_detects_altered_outputs_and_changed_configuration(tmp_path: Path):
    store = StageStore(tmp_path, "code", {"c": 1}, {"a": []})
    with store.stage("a") as out:
        (out / "x.txt").write_text("1")
    (tmp_path / "a" / "x.txt").write_text("2")
    with pytest.raises(IntegrityError):
        store.verify("a")
    with pytest.raises(IntegrityError):
        StageStore(tmp_path, "code", {"c": 2}, {"a": []}).verify("a")
