"""Regression tests for the 2026-10-12 review round of the document QA workshop (DQA2-M1, DQA2-m1..m6).

Static checks of the notebook and stage-file checks with stand-ins (no model weights, no GPU, no network).
They check the review's acceptance criteria; they are not execution evidence for a hosted runtime.
"""

import ast
import contextlib
import importlib
import io
import json
import re
from pathlib import Path

import document_qa_workshop
import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "DIMER_Document_QA_LayoutLM_vs_Pix2Struct_Workshop.ipynb"


def _notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _src(cid):
    return next("".join(c["source"]) for c in _notebook()["cells"] if c["id"] == cid)


def _markdown():
    return "\n".join("".join(c["source"]) for c in _notebook()["cells"] if c["cell_type"] == "markdown")


def _run(source, ns):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(source, "<cell>", "exec"), ns)
    return out.getvalue()


@pytest.fixture
def dqa(tmp_path):
    module = importlib.reload(document_qa_workshop)
    module.OUTPUT_DIR = str(tmp_path / "out" / "document_qa_comparison")
    module.WORK_DIR = str(tmp_path / "work")
    yield module
    importlib.reload(document_qa_workshop)


# --------------------------------------------------------------------------------------- DQA2-M1 (option A)
@pytest.mark.parametrize("cid", ["974ba2f6", "ed23cbed", "0c461246", "e51c806d", "f8e545d9"])
def test_the_annotated_words_and_the_upper_bound_are_disclosed(cid):
    text = _src(cid).lower()
    assert "annotat" in text and ("upper bound" in text or "perfect ocr" in text), cid


def test_the_opening_says_the_words_are_not_ocr_engine_output_and_qualifies_the_gap():
    text = _src("974ba2f6")
    assert "not the output of an OCR engine" in text
    assert "0.843 against 0.801" in text and "`cf98b6e`" in text
    assert "What happens if OCR misses the answer?" not in _src("dqa-preview-md")
    assert "a reading the OCR disagrees with" not in _src("dqa-offpage-md")


# --------------------------------------------------------------------------------------- DQA2-m1
def test_section_9_labels_the_anls_variant_and_scoring_is_unchanged(dqa):
    text = _src("f3f4db23")
    assert "variant of the official DocVQA ANLS" in text and "`anls('46', ['6'])` is 0.5 here" in text
    assert dqa.anls("46", ["6"]) == 0.5  # the score is labelled, not changed
    assert dqa.normalize_answer("1,250.00") == "1 250 00"


# --------------------------------------------------------------------------------------- DQA2-m2
def test_quoted_runtimes_name_their_revision():
    for cid, figure in (("guided-00", "143 seconds"), ("guided-01", "0.033 s")):
        sentence = next(s for s in re.split(r"(?<=\.)\s+", _src(cid)) if figure in s)
        assert "`cf98b6e`" in sentence and "not re-run" in sentence, cid
    assert "2026-09-26), answering" not in _src("guided-00")


# --------------------------------------------------------------------------------------- DQA2-m3
def test_only_input_refused_is_a_refusal(dqa, monkeypatch, tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "use_byod": False, "byod_path": "", "max_new_tokens": 32, "run_paraphrase_experiment": True,
        "paraphrase_max_records": 20, "run_unanswerable_probe": True, "unanswerable_pages": 5,
        "run_modality_robustness": True, "robustness_max_records": 20,
        "output_dir": str(tmp_path / "out"), "work_dir": str(tmp_path / "work"),
    }), encoding="utf-8")

    def refuse(args):
        raise dqa.InputRefused("records.jsonl line 2 (id 'b2'): box 1 lies outside the page.")

    def crash(args):
        raise ValueError("a genuine defect")

    monkeypatch.setitem(dqa.STAGES, "refuse-test", refuse)
    monkeypatch.setitem(dqa.STAGES, "crash-test", crash)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = dqa.main(["--config", str(config), "--stage", "refuse-test"])
    assert code == dqa.REFUSAL_EXIT == 65
    assert out.getvalue() == ("Refused: records.jsonl line 2 (id 'b2'): box 1 lies outside the page. "
                              "Correct it, then run the cell again.\n")
    with pytest.raises(ValueError, match="genuine defect"):
        dqa.main(["--config", str(config), "--stage", "crash-test"])
    assert not (tmp_path / "work" / "state" / "refusal.txt").exists()


def test_byod_input_errors_are_input_refused_and_clear_an_earlier_result(dqa, tmp_path):
    previous = Path(dqa.OUTPUT_DIR) / "byod"
    previous.mkdir(parents=True)
    (previous / "metrics.json").write_text("{}", encoding="utf-8")
    (tmp_path / "empty").mkdir()
    dqa.BYOD_PATH = str(tmp_path / "empty")
    with pytest.raises(dqa.InputRefused, match="records.jsonl and a pages/ folder"):
        dqa.run_byod()
    assert not previous.exists()
    with pytest.raises(dqa.InputRefused, match="ACTIVITY_LAYOUT_CHANGE"):
        dqa.run_activity([], [], "rotate", 1)
    dqa.ROBUSTNESS_MAX_RECORDS = 0
    with pytest.raises(dqa.InputRefused, match="ROBUSTNESS_MAX_RECORDS"):
        dqa.validate_controls()


class _Process:
    def __init__(self, lines, code):
        self.stdout = iter(lines)
        self.returncode = None
        self._code = code

    def wait(self):
        self.returncode = self._code


def _run_stage(tmp_path, lines, code):
    """The notebook's own run_stage(), taken from the runtime cell, with a stand-in subprocess."""
    tree = ast.parse(_src("7984f79d"))
    func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_stage")
    import os
    import types

    ns = {
        "json": json, "os": os, "WORK_ROOT": tmp_path, "PYTHON": "python", "STAGE_SCRIPT": "stage.py",
        "ENV": {},
        "subprocess": types.SimpleNamespace(Popen=lambda *a, **k: _Process(lines, code), PIPE=-1, STDOUT=-2),
        "USE_BYOD": True, "BYOD_PATH": "x", "MAX_NEW_TOKENS": 32, "RUN_PARAPHRASE_EXPERIMENT": True,
        "PARAPHRASE_MAX_RECORDS": 20, "RUN_UNANSWERABLE_PROBE": True, "UNANSWERABLE_PAGES": 5,
        "RUN_MODALITY_ROBUSTNESS": True, "ROBUSTNESS_MAX_RECORDS": 20, "OUTPUT_DIR": "out",
        "WORK_DIR": "work",
    }
    exec(compile(ast.Module(body=[func], type_ignores=[]), "<run_stage>", "exec"), ns)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        result = ns["run_stage"]("byod")
    return result, out.getvalue()


def test_run_stage_shows_a_refusal_once_without_raising(tmp_path):
    line = ("Refused: records.jsonl line 2 (id 'b2'): missing required keys ['boxes']. "
            "Correct it, then run the cell again.\n")
    result, out = _run_stage(tmp_path, [line], 65)
    assert result is None and out == line and "Traceback" not in out


def test_run_stage_raises_on_a_genuine_failure_without_repeating_the_output(tmp_path):
    lines = ["Traceback (most recent call last):\n", "RuntimeError: shard SHA-256 mismatch\n"]
    with pytest.raises(RuntimeError, match=r"byod failed with exit 1.*full log"):
        _run_stage(tmp_path, lines, 1)
    assert (tmp_path / "logs" / "byod.log").read_text(encoding="utf-8") == "".join(lines)


def test_byod_cell_keeps_no_result_after_a_refusal(tmp_path):
    ns = {
        "run_stage": lambda *a: None,
        "read_json": lambda p: json.loads(Path(p).read_text(encoding="utf-8")),
        "Path": Path, "OUTPUT_DIR": str(tmp_path / "out"), "USE_BYOD": True,
    }
    _run(_src("77f434e8"), ns)
    assert ns["byod_result"] is None


def test_activity_refusal_runs_nothing():
    source = _src("dqa-activity").replace("RUN_LAYOUT_ACTIVITY = False", "RUN_LAYOUT_ACTIVITY = True", 1)
    source = source.replace('ACTIVITY_LAYOUT_CHANGE = "shuffle"', 'ACTIVITY_LAYOUT_CHANGE = "rotate"', 1)
    calls = []
    ns = {"run_stage": lambda *a: calls.append(a), "show_source": lambda *a: calls.append(a),
          "state": lambda name: {"test_records": [{}] * 3}}
    out = _run(source, ns)
    assert calls == [] and out.count("Refused:") == 1


# --------------------------------------------------------------------------------------- DQA2-m4
def test_page_ids_and_the_pooled_split_are_explained():
    split = _src("3775d6d3")
    assert "CORD's test and validation shards combined" in split and "`validation-037`" in split
    assert "names the CORD shard it came from" in _src("dqa-preview-md")


# --------------------------------------------------------------------------------------- DQA2-m5
def test_training_data_licences_and_references():
    models = _src("6661fbcd")
    assert "fine-tuned by Impira on SQuAD 2.0 [2] and DocVQA [3]" in models
    assert "Licence: MIT" in models and "Apache-2.0" in models and "no overlap" in models
    text = _markdown()
    refs = text.split("\n# References\n", 1)[1]
    entries = re.findall(r"^\[(\d+)\] ", refs, re.M)
    assert entries == [str(n) for n in range(1, 9)]
    body = text.split("\n# References\n", 1)[0]
    for n in entries:
        assert f"[{n}]" in body, f"reference [{n}] is never cited"
    assert all("https://" in line for line in refs.strip().split("\n\n"))


# --------------------------------------------------------------------------------------- DQA2-m6
def test_metadata_kind_profile_and_no_stale_pending():
    nb = _notebook()
    dimer = nb["metadata"]["dimer"]
    assert dimer["notebook_profile"] == "TASK-INFERENCE" and "profile" not in dimer
    assert dimer["notebook_kind"] == "WORKSHOP" and dimer["notebook_mode"] == "WORKSHOP"
    assert dimer["notebook_spec"] == "2.1"  # declared version kept in this round
    assert not [r for r in dimer["review_revisions"] if r["hosted_execution"] == "pending"]
    assert dimer["review_revisions"][-1]["date"] == "2026-10-12"
    first_code = next(i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code")
    opening = "".join(nb["cells"][0]["source"])
    assert first_code > 0 and "> **⚠️ Notice.** For research, training and evaluation only." in opening
    assert "**Notebook kind:** `WORKSHOP`" in opening
