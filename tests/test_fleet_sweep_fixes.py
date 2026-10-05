"""Regression tests for the 2026-10-05 fleet-sweep fixes (SWP-R restart guard, SWP-G guided layer and the
repository-specific SWP-A / SWP-F / SWP-B fixes recorded in docs/reviews/2026-10-05-fleet-sweep/).

Every test needs only CI's dependencies. The notebooks' own cell sources are executed with stand-ins; no model, no
network and no torch are needed.
"""
# ruff: noqa: E501

from __future__ import annotations

import functools
import hashlib
import importlib.util
import json
import re
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ['layoutlm_document_qa_colab']
LOCK = ROOT / 'tutorials/requirements-colab.lock.txt'
MIN_PREDICT = {'layoutlm_document_qa_colab': 5}


@functools.cache
def _nb_text(name: str) -> str:
    return (ROOT / "tutorials" / f"{name}.ipynb").read_text(encoding="utf-8")


def _nb(name: str) -> dict:
    return json.loads(_nb_text(name))


def _code_cells(notebook: dict) -> list[dict]:
    return [c for c in notebook["cells"] if c["cell_type"] == "code"]


def _cell(notebook: dict, marker: str) -> str:
    found = [c["source"] for c in _code_cells(notebook) if marker in c["source"]]
    assert len(found) == 1, f"expected one code cell containing {marker!r}, found {len(found)}"
    return found[0]


def _build():
    spec = importlib.util.spec_from_file_location("_sweep_build_notebook", ROOT / "tools" / "build_notebook.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    return build


# --- SWP-R: no in-kernel install, no restart, idempotent Section 1 (shared by every notebook) -------------------------


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_swp_r_nothing_is_pip_installed_into_the_kernel_and_no_restart_is_requested(name):
    notebook = _nb(name)
    code = "\n".join(c["source"] for c in _code_cells(notebook))
    assert "pip install" not in code and "'-m', 'pip'" not in code
    assert "restart the runtime" not in json.dumps(notebook).lower()
    kernel = [c for c in _code_cells(notebook) if "# dimer: kernel cell" in c["source"]]
    assert len(kernel) == 1, "exactly one cell may run in the kernel"
    source = kernel[0]["source"]
    for needed in ("'--require-hashes', '--only-binary', ':all:'", "'--managed-python'", "UV_SHA256", "LOCK_SHA256"):
        assert needed in source
    # The worker gets a clean interpreter environment and a non-interactive matplotlib backend.
    for needed in ('MPLBACKEND="Agg"', '"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"'):
        assert needed in source
    assert notebook["metadata"]["dimer"]["environment"].startswith("isolated hash-locked uv environment")


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_swp_r_carried_lock_is_the_committed_lock_and_pins_every_runtime_pin(name):
    source = _cell(_nb(name), "# dimer: kernel cell")
    lock_text = LOCK.read_text(encoding="utf-8")
    digest = re.search(r"^LOCK_SHA256 = '([0-9a-f]{64})'$", source, re.M).group(1)
    assert digest == hashlib.sha256(lock_text.encode("utf-8")).hexdigest()
    assert f"LOCK_TEXT = r'''{lock_text}'''" in source
    build = _build()
    build.check_lock(build._pins(ROOT), lock_text)  # raises SystemExit on any drift


class _Shell:
    def __init__(self) -> None:
        self.input_transformers_cleanup: list = []


def test_swp_r_section_1_is_idempotent_and_keeps_the_live_worker(tmp_path, monkeypatch, capsys):
    """Re-running the Section 1 cell reuses the matching environment (no download) and keeps the live worker, so the
    variables later cells created survive and the cells after it are not stranded."""
    source = _cell(_nb(NOTEBOOKS[0]), "# dimer: kernel cell")
    lock_sha = re.search(r"^LOCK_SHA256 = '([0-9a-f]{64})'$", source, re.M).group(1)
    env = tmp_path / "env"
    (env / "bin").mkdir(parents=True)
    (env / "bin" / "python").symlink_to(sys.executable)  # stand-in interpreter for the isolated environment
    (env / ".dimer-lock-sha256").write_text(lock_sha + "\n", encoding="utf-8")
    monkeypatch.setenv("DIMER_ISOLATED_ENV", str(env))
    monkeypatch.delenv("DIMER_NOTEBOOK_CI_PREINSTALLED", raising=False)
    shell = _Shell()
    ipython = types.ModuleType("IPython")
    ipython.get_ipython = lambda: shell
    ipython_display = types.ModuleType("IPython.display")
    ipython_display.display = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", ipython_display)

    def no_download(*args, **kwargs):
        raise AssertionError("a matching environment must be reused, not downloaded again")

    monkeypatch.setattr("urllib.request.urlopen", no_download)
    namespace: dict = {"__name__": "__main__"}
    exec(compile(source, "<section 1>", "exec"), namespace)
    runtime = namespace["_DIMER_ISOLATED_RUNTIME"]
    try:
        assert "'reused': True" in capsys.readouterr().out
        runtime.run("learner_value = 41 + 1\n")
        exec(compile(source, "<section 1 again>", "exec"), namespace)  # the learner re-runs Section 1 on its own
        assert namespace["_DIMER_ISOLATED_RUNTIME"] is runtime and runtime.alive()
        assert [t.__name__ for t in shell.input_transformers_cleanup] == ["_route_to_isolated_runtime"]
        runtime.run("print('value', learner_value)\n")
        assert "value 42" in capsys.readouterr().out
        assert namespace["_route_to_isolated_runtime"](["x = 1\n"]) == ["_DIMER_ISOLATED_RUNTIME.run('x = 1\\n')\n"]
        assert namespace["_route_to_isolated_runtime"]([source]) == [source]  # the kernel cell itself stays in the kernel
        with pytest.raises(RuntimeError, match="ZeroDivisionError"):
            runtime.run("1 / 0\n")
    finally:
        runtime.close()


# --- SWP-G: the guided layer and infrastructure labelling (shared) ----------------------------------------------------


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_swp_g_guided_layer_is_present(name):
    notebook = _nb(name)
    markdown = "\n".join(c["source"] for c in notebook["cells"] if c["cell_type"] == "markdown")
    for heading in (
        "**Who this notebook is for.**",
        "**Input → Model → Output.**",
        "**How to use this notebook.**",
        "**Roadmap:**",
        "## Troubleshooting",
        "## Glossary",
        "## Conclusion (your notes)",
        "## Change one thing (next experiments)",
    ):
        assert heading in markdown, heading
    assert markdown.count("**Predict:**") >= MIN_PREDICT[name]
    assert markdown.count("<details><summary>Check your reasoning</summary>") >= MIN_PREDICT[name]
    assert "Run all completes in one pass" in markdown


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_swp_g_infrastructure_cells_are_labelled_and_collapsed(name):
    cells = _code_cells(_nb(name))
    infra = [c for c in cells if c["metadata"].get("cellView") == "form"]
    assert any("# dimer: kernel cell" in c["source"] for c in infra)
    assert any(c["metadata"].get("dimer", {}).get("embedded_module") for c in infra)
    assert any(c["source"].startswith("# @title Infrastructure: stage and digest-verify") for c in infra)
    learner = [c for c in cells if c["metadata"].get("cellView") != "form"]
    assert learner and all("# @title Infrastructure" not in c["source"] for c in learner)


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_swp_g_no_template_placeholders_leak(name):
    notebook = _nb(name)
    text = "\n".join(
        c["source"] for c in notebook["cells"] if not c.get("metadata", {}).get("dimer", {}).get("embedded_module")
    )
    for leftover in ("{{", "{MODEL_ID}", "{stem}", "@P:"):
        assert leftover not in text, leftover


def _colab(monkeypatch, upload) -> None:
    google = types.ModuleType("google")
    google.__path__ = []
    colab_mod = types.ModuleType("google.colab")
    files = types.ModuleType("google.colab.files")
    files.upload = upload
    colab_mod.files = files
    google.colab = colab_mod
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab_mod)
    monkeypatch.setitem(sys.modules, "google.colab.files", files)


def _no_colab(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "google.colab", None)  # import fails as it does on Kaggle / Jupyter


NB = NOTEBOOKS[0]


def _learner_code() -> str:
    return "\n".join(c["source"] for c in _code_cells(_nb(NB)) if not c["metadata"].get("dimer", {}).get("embedded_module"))


# --- SWP-A: quality outcomes are recorded verdicts, never asserts -------------------------------------------------


def test_swp_a_no_quality_assert_remains():
    code = _learner_code()
    assert "assert frozen_test['anls'] > baseline_last['anls']" not in code
    assert "assert adapted_test['anls'] > frozen_test['anls']" not in code
    quality = [line for line in code.splitlines() if line.strip().startswith("assert ") and re.search(r"anls|exact_match", line)]
    assert quality == []
    # Contract integrity stays a hard check.
    assert "assert parity['identical_answers'] == parity['of']" in code


def test_swp_a_negative_results_are_recorded_and_do_not_stop_the_notebook():
    s6 = _cell(_nb(NB), "frozen_test = pipe.evaluate(test_records)")
    frozen_line = next(line for line in s6.splitlines() if line.startswith("frozen_verdict = "))
    s8 = _cell(_nb(NB), "adapted_test = pipe.evaluate(test_records)")
    verdict_block = s8[s8.index("delta_anls = ") : s8.index("for metric, row in comparison.items():")]
    for frozen, baseline, adapted, expected in ((0.3, 0.4, 0.2, "worse"), (0.8, 0.4, 0.8, "no gain"), (0.84, 0.41, 0.94, "improved")):
        ns = {"frozen_test": {"anls": frozen}, "baseline_last": {"anls": baseline}, "adapted_test": {"anls": adapted}, "comparison": {}}
        exec(frozen_line, ns)
        exec(verdict_block, ns)
        assert ns["comparison"]["verdicts"]["adapted_vs_frozen_test_anls"] == expected
        assert ns["comparison"]["verdicts"]["frozen_vs_last_number_baseline"] == ("above the last-number baseline" if frozen > baseline else "not above the last-number baseline")
    # The verdicts reach the exported report and result.
    assert s8.index("comparison['verdicts'] = ") < s8.index("json.dump(evaluation_report_payload")


# --- SWP-F: Sections 6 and 7 always start from the frozen model --------------------------------------------------


def test_swp_f_frozen_pipeline_reloads_an_adapted_pipeline(capsys):
    s6 = _cell(_nb(NB), "def frozen_pipeline():")
    helper = s6[s6.index("def frozen_pipeline():") : s6.index("\n\n\nfrozen_pipeline()")]
    loads = []

    class Stand:
        @staticmethod
        def from_pretrained(weights_dir):
            loads.append(weights_dir)
            return types.SimpleNamespace(adapter=None)

    ns = {"pipe": types.SimpleNamespace(adapter={"best_epoch": 5}), "LayoutLMDocumentQAPipeline": Stand, "WEIGHTS_DIR": "w"}
    exec(helper, ns)
    ns["frozen_pipeline"]()
    assert loads == ["w"] and ns["pipe"].adapter is None
    assert "Reloaded the frozen model" in capsys.readouterr().out
    ns["frozen_pipeline"]()
    assert loads == ["w"]  # a frozen pipeline is kept


def test_swp_f_scoring_and_training_cells_call_frozen_pipeline_first():
    s6 = _cell(_nb(NB), "frozen_test = pipe.evaluate(test_records)")
    assert s6.index("\nfrozen_pipeline()\n") < s6.index("frozen_test = pipe.evaluate(")
    s7 = _cell(_nb(NB), "adapt_result = pipe.adapt(")
    assert s7.index("frozen_pipeline()") < s7.index("adapt_result = pipe.adapt(")


# --- SWP-B: BYOD path, guarded upload, refusals that name the file ------------------------------------------------


def _byod(monkeypatch, path: str, tmp_path) -> dict:
    from layoutlm_document_qa_pipeline.samples import load_byod_dataset

    source = _cell(_nb(NB), "BYOD_PATH = ''")
    block = source[source.index("def byod_file(") : source.index("    splits = split_dataset(records, seed=SPLIT_SEED)")]
    block = block.replace("os.makedirs('outputs', exist_ok=True)\n", "")
    monkeypatch.chdir(tmp_path)
    ns = {"Path": Path, "json": json, "USE_BYOD": True, "BYOD_PATH": path, "load_byod_dataset": load_byod_dataset}
    exec(compile(block, "<section 4 BYOD>", "exec"), ns)
    return ns


def test_swp_b_byod_path_works_outside_colab(monkeypatch, tmp_path):
    _no_colab(monkeypatch)
    data = tmp_path / "pages.jsonl"
    data.write_text(json.dumps({"id": "a"}) + "\n" + json.dumps({"id": "b"}) + "\n", encoding="utf-8")
    ns = _byod(monkeypatch, str(data), tmp_path)
    assert ns["file_name"] == "pages.jsonl" and [r["id"] for r in ns["records"]] == ["a", "b"]


def test_swp_b_refusals_name_the_file_and_the_rule(monkeypatch, tmp_path):
    _no_colab(monkeypatch)
    with pytest.raises(FileNotFoundError, match="BYOD_PATH .*missing.json.* is not a file"):
        _byod(monkeypatch, str(tmp_path / "missing.json"), tmp_path)
    with pytest.raises(RuntimeError, match="BYOD_PATH is empty and this runtime has no Colab upload dialog"):
        _byod(monkeypatch, "", tmp_path)
    broken = tmp_path / "broken.jsonl"
    broken.write_text('{"id": "a"}\n{not json\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"^broken\.jsonl line 2: "):
        _byod(monkeypatch, str(broken), tmp_path)
    wrong = tmp_path / "pages.csv"
    wrong.write_text("id\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"^pages\.csv: BYOD datasets must be \.json or \.jsonl"):
        _byod(monkeypatch, str(wrong), tmp_path)


def test_swp_b_cancelled_or_multiple_uploads_are_refused_and_one_upload_works(monkeypatch, tmp_path):
    _colab(monkeypatch, lambda: {})
    with pytest.raises(RuntimeError, match="got 0 .*upload cancelled or empty"):
        _byod(monkeypatch, "", tmp_path)
    _colab(monkeypatch, lambda: {"a.json": b"[]", "b.json": b"[]"})
    with pytest.raises(RuntimeError, match="got 2"):
        _byod(monkeypatch, "", tmp_path)
    _colab(monkeypatch, lambda: {"mine.json": b'[{"id": "x"}]'})
    ns = _byod(monkeypatch, "", tmp_path)
    assert ns["file_name"] == "mine.json" and ns["records"] == [{"id": "x"}]


def test_swp_g_checkpoint_answers_agree_with_the_recorded_run():
    """Numbers quoted in the worked answers are the recorded 2026-09-19 Kaggle T4 run's (docs/release-verification.md)."""
    record = (ROOT / "docs" / "release-verification.md").read_text(encoding="utf-8")
    markdown = "\n".join(c["source"] for c in _nb(NB)["cells"] if c["cell_type"] == "markdown")
    for fact in ("last_number: 0.406", "keyword_lookup: 0.635", "frozen: 0.843", "adapted: 0.94", "total.menuqty_cnt: {n: 12, frozen: 0.17, adapted: 0.83}", "identical_answers: 8, of: 8", "0.819 → 0.870 → 0.908 → 0.894 → 0.925 → 0.960 → 0.956"):
        assert fact in record, fact
    for quoted in ("last-number 0.406", "keyword lookup 0.635", "frozen 0.843", "adapted 0.94", "0.17 → 0.83", "8 of 8", "0.819 → 0.870 → 0.908 → 0.894 → 0.925 → 0.960 → 0.956"):
        assert quoted in markdown, quoted
