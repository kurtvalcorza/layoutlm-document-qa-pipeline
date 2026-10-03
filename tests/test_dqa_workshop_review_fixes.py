"""Regression tests for the 2026-10-02 review of the document QA workshop notebook (DQA-M1..M4, DQA-m1..m5).

Since 2026-10-03 the notebook carries its model and evaluation code as tools/document_qa_workshop.py and runs
it in an isolated environment, so most checks call that stage file directly; the notebook cells that remain
(controls, activity controls, displays) are executed here with stand-ins. No model weights, no GPU, no
network. These tests check the review's acceptance criteria; they are not execution evidence for the
notebook in a hosted runtime.
"""

import contextlib
import csv
import importlib
import inspect
import io
import json
import sys
import types
import zipfile
from pathlib import Path

import document_qa_workshop
import pytest
from PIL import Image

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


def _call(function, *args, **kwargs):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        result = function(*args, **kwargs)
    return result, out.getvalue()


@pytest.fixture
def dqa(tmp_path):
    """A freshly loaded stage module with its output and state folders under tmp_path."""
    module = importlib.reload(document_qa_workshop)
    module.OUTPUT_DIR = str(tmp_path / "out" / "document_qa_comparison")
    module.WORK_DIR = str(tmp_path / "work")
    module.QUESTION_TEMPLATES = FIELDS
    yield module
    importlib.reload(document_qa_workshop)


@pytest.fixture
def shown(monkeypatch):
    """A stand-in IPython.display that records what the notebook displays."""
    calls = []
    disp = types.ModuleType("IPython.display")
    disp.display = lambda *objs, **_: calls.extend(objs)
    disp.Image = lambda filename: ("image", filename)
    disp.Markdown = lambda text: ("markdown", text)
    ipython = types.ModuleType("IPython")
    ipython.display = disp
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", disp)
    return calls


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


@pytest.mark.parametrize("value", [0, -3, 51, True])
def test_stage_file_refuses_out_of_range_robustness_records(dqa, value):
    dqa.ROBUSTNESS_MAX_RECORDS = value
    with pytest.raises(ValueError, match="ROBUSTNESS_MAX_RECORDS"):
        dqa.validate_controls()


def test_config_default_is_accepted():
    _run(_src("5f810ac8"), {})


def test_pix_answer_reads_the_generation_budget_at_call_time(dqa):
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

    dqa.torch, dqa.FONT_BYTES, dqa.pix_processor, dqa.pix_model = torch, b"", Proc(), Model()
    assert inspect.signature(dqa.pix_answer).parameters["max_new_tokens"].default is None
    dqa.MAX_NEW_TOKENS = 8
    dqa.pix_answer(Image.new("RGB", (64, 64)), "What is the total?")
    assert seen["budget"] == 8


# --------------------------------------------------------------------------------------- DQA-m2
def test_six_rejection_probes_run_and_are_exported(dqa):
    dqa.layout_model = dqa.layout_tokenizer = dqa.pix_model = dqa.pix_processor = object()
    probes = dqa.layoutlm_rejection_probes() + dqa.pix2struct_rejection_probes()
    assert len(probes) == 6
    assert all(p["refused"] is True and p["message"] for p in probes)
    assert {p["check"].split(":")[0] for p in probes} == {"LayoutLM", "Pix2Struct"}
    exports = inspect.getsource(dqa.export_all)
    assert '"rejection_probes":lc["rejection_probes"]+pc["rejection_probes"]' in exports
    assert "empty question is rejected" not in exports


def test_rejection_probe_raises_when_an_invalid_input_is_accepted(dqa):
    with pytest.raises(RuntimeError, match="not refused"):
        dqa.rejection_probe("accepting check", lambda: None)


# --------------------------------------------------------------------------------------- DQA-M4
def test_layout_answer_without_a_loaded_model_says_how_to_reload(dqa):
    with pytest.raises(RuntimeError, match="reloads it"):
        dqa.layout_answer("What is the total?", ["TOTAL"], [[0, 0, 10, 10]], (100, 100))


def test_layout_experiment_rows_carry_the_field_and_print_coverage(dqa, monkeypatch):
    recs = _records(6)
    canonical = [
        {"record_id": r["id"], "answer": r["answers"][0], "anls": 1.0, "exact_match": True} for r in recs
    ]
    dqa.ROBUSTNESS_MAX_RECORDS = 6
    monkeypatch.setattr(dqa, "layout_answer", lambda q, w, b, s: {"answer": "3.00"})
    rows, out = _call(dqa.layout_no_layout, recs, canonical)
    assert all("field" in row for row in rows)
    assert "'pages': 2" in out
    assert "menu.nm" in out and "total.total_price" in out


@pytest.mark.parametrize("change", ["zero", "shuffle", "coarse"])
def test_activity_runs_and_leaves_canonical_outputs(dqa, monkeypatch, change):
    recs = _records(6)
    canonical = [{"record_id": r["id"], "answer": r["answers"][0], "anls": 1.0} for r in recs]
    life = []

    def layout_answer(question, words, boxes, image_size):
        assert life[-1] == "load", "the activity must load LayoutLM itself"
        for x0, y0, x1, y1 in boxes:
            assert 0 <= x0 <= x1 <= image_size[0] and 0 <= y0 <= y1 <= image_size[1]
        return {"answer": "3.00"}

    out_dir = Path(dqa.OUTPUT_DIR)
    out_dir.mkdir(parents=True)
    (out_dir / "predictions.csv").write_text("canonical\n", encoding="utf-8")
    monkeypatch.setattr(dqa, "layout_answer", layout_answer)
    monkeypatch.setattr(dqa, "load_layoutlm", lambda: life.append("load"))
    monkeypatch.setattr(dqa, "release_layoutlm", lambda: life.append("release"))
    _, out = _call(dqa.run_activity, recs, canonical, change, 6)
    assert life == ["load", "release"]
    assert "total.total_price" in out and "'pages': 2" in out
    rows = list(csv.DictReader((out_dir / "activity" / f"layout_{change}.csv").open(encoding="utf-8")))
    assert len(rows) == 6 and {r["change"] for r in rows} == {change}
    assert (out_dir / "predictions.csv").read_text(encoding="utf-8") == "canonical\n"


def test_activity_cell_is_off_by_default_and_refuses_unknown_change():
    def run_stage(*_):
        pytest.fail("the activity stage ran")

    ns = {"run_stage": run_stage, "state": lambda name: {"test_records": _records(3)}, "show_source": print}
    assert "Activity not run" in _run(_src("dqa-activity"), dict(ns))
    source = _set_param(
        _set_param(_src("dqa-activity"), "RUN_LAYOUT_ACTIVITY", True), "ACTIVITY_LAYOUT_CHANGE", "rotate"
    )
    with pytest.raises(ValueError, match="ACTIVITY_LAYOUT_CHANGE"):
        _run(source, dict(ns))
    on = _set_param(_src("dqa-activity"), "RUN_LAYOUT_ACTIVITY", True)
    source = _set_param(on, "ACTIVITY_MAX_RECORDS", 4)
    with pytest.raises(ValueError, match=r"1\.\.3"):
        _run(source, dict(ns))


def test_activity_cell_runs_the_activity_stage_with_its_controls():
    calls = []
    ns = {
        "run_stage": lambda *a: calls.append(a),
        "state": lambda name: {"test_records": _records(6)},
        "show_source": lambda *names: None,
    }
    on = _set_param(_src("dqa-activity"), "RUN_LAYOUT_ACTIVITY", True)
    source = _set_param(on, "ACTIVITY_MAX_RECORDS", 5)
    _run(_set_param(source, "ACTIVITY_LAYOUT_CHANGE", "coarse"), ns)
    assert calls == [("activity", "--change", "coarse", "--max-records", "5")]


def test_activity_stage_refuses_out_of_range_controls(dqa):
    with pytest.raises(ValueError, match="ACTIVITY_LAYOUT_CHANGE"):
        dqa.run_activity(_records(3), [], "rotate", 2)
    with pytest.raises(ValueError, match="ACTIVITY_MAX_RECORDS"):
        dqa.run_activity(_records(3), [], "zero", 4)


def test_try_it_markdown_is_followed_by_the_activity_cell():
    order = _order()
    assert order[order.index("guided-01") + 1] == "dqa-activity"
    text = _src("guided-01")
    assert "RUN_LAYOUT_ACTIVITY = True" in text and "reloads LayoutLM" in text
    assert "Use the existing LayoutLM modality experiment" not in text


# --------------------------------------------------------------------------------------- DQA-M3
def test_a_receipt_is_shown_before_any_model_result(dqa, tmp_path, shown):
    order = _order()
    assert order.index("dqa-preview") < order.index("de054504")
    rec = _records(1)[0]
    page = {
        "id": rec["page_id"],
        "image": Image.new("RGB", (900, 1200), "white"),
        "words": rec["words"],
        "boxes": rec["boxes"],
    }
    path = tmp_path / "preview.png"
    preview, out = _call(dqa.draw_preview, [rec], {rec["page_id"]: page}, path)
    assert preview.width == 700 and Image.open(path).width == 700
    assert rec["question"] in out
    calls = []
    ns = {"run_stage": lambda *a: calls.append(a), "state": lambda name: {"path": str(path)}}
    _run("from IPython.display import Image as ShowImage, display\n" + _src("dqa-preview"), ns)
    assert calls == [("preview",)] and shown == [("image", str(path))]


def _comparison():
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
    layout_rows, pix_rows = [], []
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
        pix_rows.append(
            {
                "record_id": r["id"],
                "field": r["field"],
                "gold_answer": r["answers"][0],
                "answer": r["answers"][0] if p_ok else "3,30 total",
                "exact_match": p_ok,
                "anls": 1.0 if p_ok else 0.4,
                "answer_in_ocr": p_ok,
                "truncated": False,
            }
        )
    return recs, pages, layout_rows, pix_rows


def test_panels_include_the_diagnostic_cases_and_are_displayed(dqa, tmp_path, shown):
    recs, pages, layout_rows, pix_rows = _comparison()
    agreement_rows, _ = dqa.agreement(recs, layout_rows, pix_rows)
    examples = dqa.select_examples(agreement_rows, layout_rows, pix_rows)
    cats = [e["agreement_category"] for e in examples]
    assert {"pix2struct_not_in_ocr", "layoutlm_confident_wrong"} <= set(cats) and len(cats) >= 4
    confident = next(e for e in examples if e["agreement_category"] == "layoutlm_confident_wrong")
    assert confident["record_id"] == "test-0005"  # the highest span score among LayoutLM's wrong answers
    off_page = next(e for e in examples if e["agreement_category"] == "pix2struct_not_in_ocr")
    assert off_page["record_id"] == "test-0001"
    by_l = {r["record_id"]: r for r in layout_rows}
    by_p = {r["record_id"]: r for r in pix_rows}
    panel = dqa.draw_qa_panel(examples[0], pages, by_l, by_p)
    assert panel.height == 100 + 218
    listed = [{"category": c, "record_id": f"test-{k:04d}", "path": f"p{k}.png"} for k, c in enumerate(cats)]
    out = _run(
        "from IPython.display import Image as ShowImage, display\n" + _src("52ae9160"),
        {"run_stage": lambda *a: None, "state": lambda name: listed},
    )
    assert len(shown) == len(cats)
    assert f"{cats[0]}: test-0000 (p0.png)" in out


def test_answers_not_in_ocr_are_listed(dqa):
    _, _, _, pix_rows = _comparison()
    _, out = _call(dqa.print_off_page, pix_rows)
    assert "not found as a contiguous OCR span: 4 of 6" in out
    assert "test-0001" in out and "3,30 total" in out


# --------------------------------------------------------------------------------------- DQA-M1 / M2
def _byod_module(dqa, monkeypatch, life=None):
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

    monkeypatch.setattr(dqa, "layout_answer", layout_answer)
    monkeypatch.setattr(dqa, "pix_answer", pix_answer)
    monkeypatch.setattr(dqa, "load_layoutlm", lambda: life.append("load_layoutlm"))
    monkeypatch.setattr(dqa, "release_layoutlm", lambda: life.append("release_layoutlm"))
    monkeypatch.setattr(dqa, "load_pix2struct", lambda: life.append("load_pix2struct"))
    monkeypatch.setattr(dqa, "release_pix2struct", lambda: life.append("release_pix2struct"))
    return dqa


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
def test_byod_refusals_name_the_record_before_any_model_loads(dqa, monkeypatch, tmp_path, case):
    records, pages, rid, rule = BAD_BYOD[case]
    life = []
    _byod_module(dqa, monkeypatch, life)
    dqa.BYOD_PATH = str(_write_byod(tmp_path / "ds", records, pages))
    with pytest.raises(ValueError) as err:
        _call(dqa.run_byod)
    assert rid in str(err.value) and rule in str(err.value)
    assert life == []


def test_byod_refuses_more_than_100_pages(dqa, monkeypatch, tmp_path):
    records = [_byod_record(f"q{k}", file=f"p{k}.png") for k in range(101)]
    pages = {f"p{k}.png": (200, 100) for k in range(101)}
    _byod_module(dqa, monkeypatch)
    dqa.BYOD_PATH = str(_write_byod(tmp_path / "ds", records, pages))
    with pytest.raises(ValueError, match="101 page images"):
        _call(dqa.run_byod)


def _csv_rows(path):
    return list(csv.DictReader(path.open(encoding="utf-8")))


def test_byod_labelled_run_shows_and_exports_answers(dqa, monkeypatch, tmp_path):
    life = []
    _byod_module(dqa, monkeypatch, life)
    out_dir = Path(dqa.OUTPUT_DIR)
    out_dir.mkdir(parents=True)
    (out_dir / "predictions.csv").write_text("canonical\n", encoding="utf-8")
    dqa.BYOD_PATH = str(_write_byod(tmp_path / "ds", [_byod_record("q1"), _byod_record("q2", file="p2.png")]))
    _, out = _call(dqa.run_byod)
    assert life == ["load_layoutlm", "release_layoutlm", "load_pix2struct", "release_pix2struct"]
    assert "q1" in out and "q2" in out and "standin answer" in out and '"measured"' in out
    rows = _csv_rows(out_dir / "byod" / "predictions.csv")
    assert len(rows) == 4 and {r["model"] for r in rows} == {"LayoutLM", "Pix2Struct"}
    assert {r["anls"] for r in rows if r["model"] == "LayoutLM"} == {"1.0"}
    assert {r["exact_match"] for r in rows if r["model"] == "Pix2Struct"} == {"False"}
    metrics = json.loads((out_dir / "byod" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["LayoutLM"]["anls"] == 1.0 and metrics["evaluation_verdict"] == "measured"
    assert (out_dir / "predictions.csv").read_text(encoding="utf-8") == "canonical\n"


def test_byod_unlabelled_run_is_not_measurable_but_keeps_answers(dqa, monkeypatch, tmp_path):
    _byod_module(dqa, monkeypatch)
    dqa.BYOD_PATH = str(_write_byod(tmp_path / "ds", [_byod_record("u1", answers=None)]))
    _, out = _call(dqa.run_byod)
    assert "not-measurable" in out and "u1" in out and "standin answer" in out
    rows = _csv_rows(Path(dqa.OUTPUT_DIR) / "byod" / "predictions.csv")
    assert [r["answer"] for r in rows] == ["3.30", "standin answer"]
    assert {r["anls"] for r in rows} == {""}


def test_byod_accepts_the_documented_zip_and_refuses_escaping_members(dqa, monkeypatch, tmp_path):
    root = _write_byod(tmp_path / "ds", [_byod_record("z1")])
    archive = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for p in root.rglob("*"):
            if p.is_file():
                z.write(p, "dataset/" + p.relative_to(root).as_posix())
    _byod_module(dqa, monkeypatch)
    dqa.BYOD_PATH = str(archive)
    assert "z1" in _call(dqa.run_byod)[1]
    slip = tmp_path / "slip.zip"
    with zipfile.ZipFile(slip, "w") as z:
        z.writestr("../escape.txt", "x")
        z.writestr("records.jsonl", json.dumps(_byod_record("z1")) + "\n")
    dqa.BYOD_PATH = str(slip)
    with pytest.raises(ValueError, match="outside the dataset folder"):
        _call(dqa.run_byod)
    assert not (tmp_path / "out" / "escape.txt").exists()


def test_byod_cell_runs_the_stage_only_when_enabled(tmp_path):
    calls = []
    metrics = tmp_path / "out" / "byod" / "metrics.json"
    metrics.parent.mkdir(parents=True)
    metrics.write_text('{"evaluation_verdict": "measured"}', encoding="utf-8")
    ns = {
        "run_stage": lambda *a: calls.append(a),
        "read_json": lambda p: json.loads(Path(p).read_text(encoding="utf-8")),
        "Path": Path,
        "OUTPUT_DIR": str(tmp_path / "out"),
        "USE_BYOD": False,
    }
    assert "BYOD disabled" in _run(_src("77f434e8"), ns)
    assert calls == []
    ns["USE_BYOD"] = True
    _run(_src("77f434e8"), ns)
    assert calls == [("byod",)] and ns["byod_result"]["metrics"]["evaluation_verdict"] == "measured"


def test_byod_section_states_limits_layouts_and_data_locality():
    text = _src("291a71b6")
    for limit in ("1..500", "1..100", "16..4096", "1..256", "1..2000", "`.zip`"):
        assert limit in text, limit
    assert "never leave this notebook runtime" in text
    assert "byod/predictions.csv" in text


# --------------------------------------------------------------------------------------- DQA-m3 / m4 / m5
def test_runtime_orientation_and_cpu_warning(dqa):
    opening = _src("guided-00")
    assert "2.1 GB" in opening and "152 seconds" in opening and "T4" in opening
    assert "No GPU detected" in _src("7984f79d")
    assert "No GPU detected" in inspect.getsource(dqa.stage_environment)


def test_each_experiment_output_is_followed_by_what_to_notice(dqa):
    order = _order()
    for cid, note, stage, function in (
        ("aada1835", "dqa-notice-13", "layoutlm-no-layout", dqa.layout_no_layout),
        ("b1a10dd3", "dqa-notice-18", "pix2struct-degraded", dqa.pix_degraded),
        ("6206457c", "dqa-notice-23", "paraphrase", dqa.stage_paraphrase),
    ):
        assert order[order.index(cid) + 1] == note
        assert _src(note).startswith("> **What to notice.**")
        assert f'run_stage("{stage}")' in _src(cid)
        assert "subset_coverage(" in inspect.getsource(function)


def test_metric_explanation_glossary_and_terminology(dqa):
    metrics = _src("f3f4db23")
    assert "edit distance" in metrics and "0.875" in metrics and "Exact Match = 0" in metrics
    glossary = _src("guided-05")
    for term in ("OCR", "Word box", "Span", "Exact Match", "Token window", "Patch", "Uncalibrated score"):
        assert f"**{term}**" in glossary, term
    assert _src("8ad583a7").strip().endswith('run_stage("summary")')
    assert "Workshop" not in inspect.getsource(dqa.stage_summary)


def test_the_review_edit_is_recorded_in_the_notebook_metadata():
    revisions = _notebook()["metadata"]["dimer"]["review_revisions"]
    entry = next(r for r in revisions if r["date"] == "2026-10-02")
    assert entry["reviewed_revision"].startswith("07e4c08")
    assert entry["hosted_execution"] == "pending"
    assert "clean_runtime_evidence" not in _notebook()["metadata"]["dimer"]
