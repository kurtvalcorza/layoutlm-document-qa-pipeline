"""Regression tests for the 2026-10-02 review of the document QA workshop notebook (DQA-M1..M4, DQA-m1..m5).

The notebook's own cells (or the functions defined in them) run here against small stand-ins: no model
weights, no GPU, no network. These tests check the review's acceptance criteria; they are not execution
evidence for the notebook in a hosted runtime.
"""

import ast
import contextlib
import csv
import io
import json
import random
import re
import sys
import types
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "tutorials"
    / "DIMER_Document_QA_LayoutLM_vs_Pix2Struct_Workshop.ipynb"
)
FIELDS = {
    "total.total_price": "What is the total amount?",
    "sub_total.tax_price": "What is the tax amount?",
    "menu.nm": "What is the name of the first item?",
}


def _notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _cells():
    return [(c["id"], c["cell_type"], "".join(c["source"])) for c in _notebook()["cells"]]


def _src(cid):
    return next(s for i, _, s in _cells() if i == cid)


def _order():
    return [i for i, _, _ in _cells()]


def _defs(cid, *names):
    """Source of the named top-level functions / assignments of one cell, in cell order."""
    source = _src(cid)
    out = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            out.append(ast.get_source_segment(source, node))
        elif isinstance(node, (ast.Assign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id in names for t in targets):
                out.append(ast.get_source_segment(source, node))
    return "\n".join(out)


def _set_param(source, name, value):
    lines = source.split("\n")
    hits = [k for k, line in enumerate(lines) if line.startswith(f"{name} = ")]
    assert len(hits) == 1, name
    lines[hits[0]] = f"{name} = {value!r}"
    return "\n".join(lines)


def _run(source, ns):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(source, "<cell>", "exec"), ns)
    return out.getvalue()


@pytest.fixture
def shown(monkeypatch):
    """A stand-in IPython.display that records what the notebook displays."""
    calls = []
    disp = types.ModuleType("IPython.display")
    disp.display = lambda *objs, **_: calls.extend(objs)
    ipython = types.ModuleType("IPython")
    ipython.display = disp
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", disp)
    return calls


def _metrics_ns(**extra):
    ns = {
        "np": np,
        "re": re,
        "csv": csv,
        "defaultdict": defaultdict,
        "json": json,
        "Path": Path,
        "Image": Image,
        "ImageDraw": ImageDraw,
        "random": random,
        "QUESTION_TEMPLATES": FIELDS,
        "SAMPLE_SEED": 42,
    }
    _run(_src("ef5d4ed3"), ns)  # normalize_answer, anls, exact_match, answer_in_ocr, corpus_metrics
    _run(_defs("b1722b90", "write_csv"), ns)
    _run(_defs("aada1835", "subset_coverage", "print_field_deltas"), ns)
    ns.update(extra)
    return ns


def _records(n=6):
    words = ["Coffee", "3.00", "TAX", "0.30", "TOTAL", "3.30"]
    boxes = [
        [10, 10, 60, 20],
        [100, 10, 140, 20],
        [10, 30, 40, 40],
        [100, 30, 140, 40],
        [10, 50, 50, 60],
        [100, 50, 140, 60],
    ]
    gold = {"total.total_price": "3.30", "sub_total.tax_price": "0.30", "menu.nm": "Coffee"}
    out = []
    for k in range(n):
        field = list(FIELDS)[k % 3]
        out.append(
            {
                "id": f"test-{k:04d}",
                "page_id": f"test-{k // 3:03d}",
                "field": field,
                "question": FIELDS[field],
                "words": words,
                "boxes": boxes,
                "image_size": [200, 100],
                "answers": [gold[field]],
            }
        )
    return out


# --------------------------------------------------------------------------------------- DQA-m1
@pytest.mark.parametrize("value", [0, -3, 51, True])
def test_config_refuses_out_of_range_robustness_records(value):
    with pytest.raises(ValueError, match="ROBUSTNESS_MAX_RECORDS"):
        _run(_set_param(_src("5f810ac8"), "ROBUSTNESS_MAX_RECORDS", value), {})


def test_config_default_is_accepted():
    _run(_src("5f810ac8"), {})


def test_pix_answer_reads_the_generation_budget_at_call_time():
    import inspect

    import torch

    seen = {}

    class Proc:
        class image_processor:  # noqa: N801
            def __call__(self, *a, **k):
                return types.SimpleNamespace(to=lambda device: {"flattened_patches": torch.zeros(1, 2, 3)})

        image_processor = image_processor()
        tokenizer = types.SimpleNamespace(batch_decode=lambda g, skip_special_tokens: ["4.00"])

    class Model:
        def generate(self, **kw):
            seen["budget"] = kw["max_new_tokens"]
            return torch.zeros(1, 3, dtype=torch.long)

    ns = {
        "MAX_NEW_TOKENS": 32,
        "torch": torch,
        "DEVICE": "cpu",
        "FONT_BYTES": b"",
        "pix_processor": Proc(),
        "pix_model": Model(),
    }
    _run(_defs("c598d7a7", "pix_answer"), ns)
    assert inspect.signature(ns["pix_answer"]).parameters["max_new_tokens"].default is None
    ns["MAX_NEW_TOKENS"] = 8
    ns["pix_answer"](Image.new("RGB", (64, 64)), "What is the total?")
    assert seen["budget"] == 8


# --------------------------------------------------------------------------------------- DQA-m2
def test_six_rejection_probes_run_and_are_exported():
    ns = {
        "Image": Image,
        "MAX_NEW_TOKENS": 32,
        "layout_model": object(),
        "layout_tokenizer": object(),
        "pix_model": object(),
        "pix_processor": object(),
    }
    _run(_defs("9f0db1aa", "layout_answer", "rejection_probe", "REJECTION_PROBES"), ns)
    _run(_defs("c598d7a7", "pix_answer", "_blank_page", "REJECTION_PROBES"), ns)
    probes = ns["REJECTION_PROBES"]
    assert len(probes) == 6
    assert all(p["refused"] is True and p["message"] for p in probes)
    assert {p["check"].split(":")[0] for p in probes} == {"LayoutLM", "Pix2Struct"}
    exports = _src("b1722b90")
    assert '"rejection_probes":REJECTION_PROBES' in exports
    assert "empty question is rejected" not in exports


def test_rejection_probe_raises_when_an_invalid_input_is_accepted():
    ns = {}
    _run(_defs("9f0db1aa", "rejection_probe"), ns)
    with pytest.raises(RuntimeError, match="not refused"):
        ns["rejection_probe"]("accepting check", lambda: None)


# --------------------------------------------------------------------------------------- DQA-M4
def test_layout_answer_after_release_says_how_to_reload():
    ns = {}
    _run(_defs("9f0db1aa", "layout_answer"), ns)
    with pytest.raises(RuntimeError, match="reloads it"):
        ns["layout_answer"]("What is the total?", ["TOTAL"], [[0, 0, 10, 10]], (100, 100))


def test_layout_experiment_rows_carry_the_field_and_print_coverage():
    recs = _records(6)
    canonical = [
        {"record_id": r["id"], "answer": r["answers"][0], "anls": 1.0, "exact_match": True} for r in recs
    ]
    ns = _metrics_ns(
        test_records=recs,
        layout_rows=canonical,
        ROBUSTNESS_MAX_RECORDS=6,
        RUN_MODALITY_ROBUSTNESS=True,
        layout_answer=lambda q, w, b, s: {"answer": "3.00"},
    )
    out = _run(_src("aada1835"), ns)
    assert all("field" in row for row in ns["layout_no_layout_rows"])
    assert "'pages': 2" in out
    assert "menu.nm" in out and "total.total_price" in out


@pytest.mark.parametrize("change", ["zero", "shuffle", "coarse"])
def test_activity_runs_after_release_and_leaves_canonical_outputs(tmp_path, change):
    recs = _records(6)
    canonical = [{"record_id": r["id"], "answer": r["answers"][0], "anls": 1.0} for r in recs]
    life = []

    def layout_answer(question, words, boxes, image_size):
        assert life[-1] == "load", "the activity must load LayoutLM itself"
        for x0, y0, x1, y1 in boxes:
            assert 0 <= x0 <= x1 <= image_size[0] and 0 <= y0 <= y1 <= image_size[1]
        return {"answer": "3.00"}

    canonical_file = tmp_path / "predictions.csv"
    canonical_file.write_text("canonical\n", encoding="utf-8")
    ns = _metrics_ns(
        test_records=recs,
        layout_rows=canonical,
        OUTPUT_DIR=str(tmp_path),
        layout_answer=layout_answer,
        load_layoutlm=lambda: life.append("load"),
        release_layoutlm=lambda: life.append("release"),
    )
    source = _set_param(
        _set_param(_src("dqa-activity"), "RUN_LAYOUT_ACTIVITY", True), "ACTIVITY_LAYOUT_CHANGE", change
    )
    out = _run(_set_param(source, "ACTIVITY_MAX_RECORDS", 6), ns)
    assert life == ["load", "release"]
    assert "total.total_price" in out and "'pages': 2" in out
    rows = list(csv.DictReader((tmp_path / "activity" / f"layout_{change}.csv").open(encoding="utf-8")))
    assert len(rows) == 6 and {r["change"] for r in rows} == {change}
    assert canonical_file.read_text(encoding="utf-8") == "canonical\n"


def test_activity_is_off_by_default_and_refuses_unknown_change():
    ns = _metrics_ns(test_records=_records(3), layout_rows=[], load_layoutlm=lambda: pytest.fail("loaded"))
    assert "Activity not run" in _run(_src("dqa-activity"), ns)
    source = _set_param(
        _set_param(_src("dqa-activity"), "RUN_LAYOUT_ACTIVITY", True), "ACTIVITY_LAYOUT_CHANGE", "rotate"
    )
    with pytest.raises(ValueError, match="ACTIVITY_LAYOUT_CHANGE"):
        _run(source, _metrics_ns(test_records=_records(3), layout_rows=[]))


def test_try_it_markdown_is_followed_by_the_activity_cell():
    order = _order()
    assert order[order.index("guided-01") + 1] == "dqa-activity"
    text = _src("guided-01")
    assert "RUN_LAYOUT_ACTIVITY = True" in text and "reloads LayoutLM" in text
    assert "Use the existing LayoutLM modality experiment" not in text


# --------------------------------------------------------------------------------------- DQA-M3
def test_a_receipt_is_shown_before_any_model_result(shown):
    order = _order()
    assert order.index("dqa-preview") < order.index("de054504")
    rec = _records(1)[0]
    page = {
        "id": rec["page_id"],
        "image": Image.new("RGB", (900, 1200), "white"),
        "words": rec["words"],
        "boxes": rec["boxes"],
    }
    out = _run(
        _src("dqa-preview"),
        {"test_records": [rec], "test_pages": {rec["page_id"]: page}, "ImageDraw": ImageDraw},
    )
    assert len(shown) == 1 and isinstance(shown[0], Image.Image) and shown[0].width == 700
    assert rec["question"] in out


def _comparison(tmp_path):
    recs = _records(6)
    pages = {
        r["page_id"]: {
            "id": r["page_id"],
            "image": Image.new("RGB", (200, 100), "white"),
            "boxes": r["boxes"],
            "words": r["words"],
        }
        for r in recs
    }
    layout_rows, pix_rows, agreement = [], [], []
    for k, r in enumerate(recs):
        l_ok, p_ok = k % 2 == 0, k % 3 == 0
        layout_rows.append(
            {
                "record_id": r["id"],
                "answer": r["answers"][0] if l_ok else "3.00",
                "exact_match": l_ok,
                "anls": 1.0 if l_ok else 0.0,
                "span_score": 0.1 * (k + 1),
                "answer_union_box": r["boxes"][-1],
            }
        )
        pix_answer = r["answers"][0] if p_ok else "3,30 total"
        pix_rows.append(
            {
                "record_id": r["id"],
                "field": r["field"],
                "gold_answer": r["answers"][0],
                "answer": pix_answer,
                "exact_match": p_ok,
                "anls": 1.0 if p_ok else 0.4,
                "answer_in_ocr": p_ok,
                "truncated": False,
            }
        )
        cat = (
            "both_exact"
            if l_ok and p_ok
            else "layoutlm_only"
            if l_ok
            else "pix2struct_only"
            if p_ok
            else "neither_exact"
        )
        agreement.append(
            {
                "record_id": r["id"],
                "page_id": r["page_id"],
                "question": r["question"],
                "gold_answer": r["answers"][0],
                "layoutlm_answer": layout_rows[-1]["answer"],
                "layoutlm_anls": layout_rows[-1]["anls"],
                "pix2struct_answer": pix_answer,
                "pix2struct_anls": pix_rows[-1]["anls"],
                "agreement_category": cat,
            }
        )
    return {
        "test_pages": pages,
        "layout_rows": layout_rows,
        "pix_rows": pix_rows,
        "agreement_rows": agreement,
        "layout_by_id": {r["record_id"]: r for r in layout_rows},
        "pix_by_id": {r["record_id"]: r for r in pix_rows},
        "OUTPUT_DIR": str(tmp_path),
        "Image": Image,
        "ImageDraw": ImageDraw,
        "Path": Path,
    }


def test_panels_are_displayed_and_include_the_diagnostic_cases(tmp_path, shown):
    ns = _comparison(tmp_path)
    out = _run(_src("52ae9160"), ns)
    cats = [e["agreement_category"] for e in ns["selected_examples"]]
    assert {"pix2struct_not_in_ocr", "layoutlm_confident_wrong"} <= set(cats)
    assert len(shown) == len(cats) >= 4
    assert len(list((tmp_path / "examples").glob("*.png"))) == len(cats)
    confident = next(
        e for e in ns["selected_examples"] if e["agreement_category"] == "layoutlm_confident_wrong"
    )
    assert confident["record_id"] == "test-0005"  # the highest span score among LayoutLM's wrong answers
    assert "pix2struct_not_in_ocr: test-0001" in out


def test_answers_not_in_ocr_are_listed(tmp_path):
    out = _run(_src("dqa-offpage"), _comparison(tmp_path))
    assert "not found as a contiguous OCR span: 4 of 6" in out
    assert "test-0001" in out and "3,30 total" in out


# --------------------------------------------------------------------------------------- DQA-M1 / M2
def _byod_ns(tmp_path, life=None):
    life = [] if life is None else life

    def layout_answer(q, words, boxes, size):
        return {
            "answer": words[-1],
            "start": len(words) - 1,
            "end": len(words) - 1,
            "score": 0.9,
            "n_windows": 1,
            "answer_union_box": boxes[-1],
        }

    def pix_answer(image, q):
        return {"answer": "standin answer", "new_tokens": 3, "truncated": False}

    return _metrics_ns(
        OUTPUT_DIR=str(tmp_path / "out" / "document_qa_comparison"),
        USE_BYOD=False,
        BYOD_PATH="",
        layout_answer=layout_answer,
        pix_answer=pix_answer,
        load_layoutlm=lambda: life.append("load_layoutlm"),
        release_layoutlm=lambda: life.append("release_layoutlm"),
        load_pix2struct=lambda: life.append("load_pix2struct"),
        release_pix2struct=lambda: life.append("release_pix2struct"),
    )


def _byod_record(rid, file="p1.png", answers=("3.30",)):
    r = {
        "id": rid,
        "page_id": file.split(".")[0],
        "file": file,
        "question": "What is the total amount?",
        "words": ["TOTAL", "3.30"],
        "boxes": [[10, 50, 50, 60], [100, 50, 140, 60]],
    }
    if answers is not None:
        r["answers"] = list(answers)
    return r


def _write_byod(root, records, pages=None):
    (root / "pages").mkdir(parents=True)
    for name, size in (pages or {"p1.png": (200, 100), "p2.png": (200, 100)}).items():
        Image.new("RGB", size, "white").save(root / "pages" / name)
    (root / "records.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return root


BAD_BYOD = {
    "missing page file": (
        [_byod_record("a"), _byod_record("b9", file="absent.png")],
        None,
        "'b9'",
        "not found",
    ),
    "malformed box": (
        [_byod_record("a"), {**_byod_record("b2"), "boxes": [[1, 2, 3], [100, 50, 140, 60]]}],
        None,
        "'b2'",
        "four numbers",
    ),
    "box outside page": (
        [_byod_record("a"), {**_byod_record("b2"), "boxes": [[1, 2, 3, 4], [100, 50, 999, 60]]}],
        None,
        "'b2'",
        "outside",
    ),
    "blank question": (
        [_byod_record("a"), {**_byod_record("b2"), "question": "   "}],
        None,
        "'b2'",
        "question",
    ),
    "page wider than 4096": (
        [_byod_record("a"), _byod_record("b2", file="wide.png")],
        {"p1.png": (200, 100), "wide.png": (5000, 100)},
        "'b2'",
        "16..4096",
    ),
    "duplicate id": ([_byod_record("a"), _byod_record("a", file="p2.png")], None, "'a'", "duplicate"),
    "answers not a list": (
        [_byod_record("a"), {**_byod_record("b2"), "answers": "3.30"}],
        None,
        "'b2'",
        "answers",
    ),
    "mixed labels": ([_byod_record("a"), _byod_record("b2", answers=None)], None, "'b2'", "every record"),
    "empty OCR word": (
        [_byod_record("a"), {**_byod_record("b2"), "words": ["", "3.30"]}],
        None,
        "'b2'",
        "non-empty",
    ),
    "missing key": (
        [_byod_record("a"), {k: v for k, v in _byod_record("b2").items() if k != "boxes"}],
        None,
        "'b2'",
        "boxes",
    ),
}


@pytest.mark.parametrize("case", list(BAD_BYOD))
def test_byod_refusals_name_the_record_before_any_model_loads(tmp_path, case):
    records, pages, rid, rule = BAD_BYOD[case]
    life = []
    ns = _byod_ns(tmp_path, life)
    ns.update(USE_BYOD=True, BYOD_PATH=str(_write_byod(tmp_path / "ds", records, pages)))
    with pytest.raises(ValueError) as err:
        _run(_src("77f434e8"), ns)
    assert rid in str(err.value) and rule in str(err.value)
    assert life == []


def test_byod_refuses_more_than_100_pages(tmp_path):
    records = [_byod_record(f"q{k}", file=f"p{k}.png") for k in range(101)]
    pages = {f"p{k}.png": (200, 100) for k in range(101)}
    ns = _byod_ns(tmp_path)
    ns.update(USE_BYOD=True, BYOD_PATH=str(_write_byod(tmp_path / "ds", records, pages)))
    with pytest.raises(ValueError, match="101 page images"):
        _run(_src("77f434e8"), ns)


def _csv_rows(path):
    return list(csv.DictReader(path.open(encoding="utf-8")))


def test_byod_labelled_run_shows_and_exports_answers(tmp_path):
    life = []
    ns = _byod_ns(tmp_path, life)
    out_dir = Path(ns["OUTPUT_DIR"])
    out_dir.mkdir(parents=True)
    (out_dir / "predictions.csv").write_text("canonical\n", encoding="utf-8")
    root = _write_byod(tmp_path / "ds", [_byod_record("q1"), _byod_record("q2", file="p2.png")])
    ns.update(USE_BYOD=True, BYOD_PATH=str(root))
    out = _run(_src("77f434e8"), ns)
    assert life == ["load_layoutlm", "release_layoutlm", "load_pix2struct", "release_pix2struct"]
    assert "q1" in out and "q2" in out and "standin answer" in out and '"measured"' in out
    rows = _csv_rows(out_dir / "byod" / "predictions.csv")
    assert len(rows) == 4 and {r["model"] for r in rows} == {"LayoutLM", "Pix2Struct"}
    assert {r["anls"] for r in rows if r["model"] == "LayoutLM"} == {"1.0"}
    assert {r["exact_match"] for r in rows if r["model"] == "Pix2Struct"} == {"False"}
    metrics = json.loads((out_dir / "byod" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["LayoutLM"]["anls"] == 1.0 and metrics["evaluation_verdict"] == "measured"
    assert (out_dir / "predictions.csv").read_text(encoding="utf-8") == "canonical\n"


def test_byod_unlabelled_run_is_not_measurable_but_keeps_answers(tmp_path):
    ns = _byod_ns(tmp_path)
    root = _write_byod(tmp_path / "ds", [_byod_record("u1", answers=None)])
    ns.update(USE_BYOD=True, BYOD_PATH=str(root))
    out = _run(_src("77f434e8"), ns)
    assert "not-measurable" in out and "u1" in out and "standin answer" in out
    rows = _csv_rows(Path(ns["OUTPUT_DIR"]) / "byod" / "predictions.csv")
    assert [r["answer"] for r in rows] == ["3.30", "standin answer"]
    assert {r["anls"] for r in rows} == {""}


def test_byod_accepts_the_documented_zip_and_refuses_escaping_members(tmp_path):
    root = _write_byod(tmp_path / "ds", [_byod_record("z1")])
    archive = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for p in root.rglob("*"):
            if p.is_file():
                z.write(p, "dataset/" + p.relative_to(root).as_posix())
    ns = _byod_ns(tmp_path)
    ns.update(USE_BYOD=True, BYOD_PATH=str(archive))
    assert "z1" in _run(_src("77f434e8"), ns)
    slip = tmp_path / "slip.zip"
    with zipfile.ZipFile(slip, "w") as z:
        z.writestr("../escape.txt", "x")
        z.writestr("records.jsonl", json.dumps(_byod_record("z1")) + "\n")
    ns = _byod_ns(tmp_path)
    ns.update(USE_BYOD=True, BYOD_PATH=str(slip))
    with pytest.raises(ValueError, match="outside the dataset folder"):
        _run(_src("77f434e8"), ns)
    assert not (tmp_path / "out" / "escape.txt").exists()


def test_byod_section_states_limits_layouts_and_data_locality():
    text = _src("291a71b6")
    for limit in ("1..500", "1..100", "16..4096", "1..256", "1..2000", "`.zip`"):
        assert limit in text, limit
    assert "never leave this notebook runtime" in text
    assert "byod/predictions.csv" in text


# --------------------------------------------------------------------------------------- DQA-m3 / m4 / m5
def test_runtime_orientation_and_cpu_warning():
    opening = _src("guided-00")
    assert "2.1 GB" in opening and "152 seconds" in opening and "T4" in opening
    assert "No GPU detected" in _src("7984f79d")


def test_each_experiment_output_is_followed_by_what_to_notice():
    order = _order()
    for cid, note in (
        ("aada1835", "dqa-notice-13"),
        ("b1a10dd3", "dqa-notice-18"),
        ("6206457c", "dqa-notice-23"),
    ):
        assert order[order.index(cid) + 1] == note
        assert _src(note).startswith("> **What to notice.**")
        assert "subset_coverage(" in _src(cid)


def test_metric_explanation_glossary_and_terminology():
    metrics = _src("f3f4db23")
    assert "edit distance" in metrics and "0.875" in metrics and "Exact Match = 0" in metrics
    glossary = _src("guided-05")
    for term in ("OCR", "Word box", "Span", "Exact Match", "Token window", "Patch", "Uncalibrated score"):
        assert f"**{term}**" in glossary, term
    assert "Workshop" not in _src("8ad583a7")


def test_the_review_edit_is_recorded_in_the_notebook_metadata():
    revisions = _notebook()["metadata"]["dimer"]["review_revisions"]
    entry = next(r for r in revisions if r["date"] == "2026-10-02")
    assert entry["reviewed_revision"].startswith("07e4c08")
    assert entry["hosted_execution"] == "pending"
    assert "clean_runtime_evidence" not in _notebook()["metadata"]["dimer"]
