"""Regression tests for the Notebook Review Framework v1 findings on `tutorials/layoutlm_document_qa_colab.ipynb`
(review PR #13: LDQ-M1..M3, LDQ-m1..m5).

The notebook's own cells are executed from the committed JSON in a namespace of the package's public API and inert
stand-ins (an injected-runner pipeline, a fake `google.colab`, synthetic receipts). Nothing here loads the pinned
checkpoint, so the whole file runs under CI's install line.
"""
# ruff: noqa: E501  -- assertion messages and cell sources are kept on one line

from __future__ import annotations

import ast
import contextlib
import importlib.util
import json
import re
import sys
import types
from pathlib import Path

import pytest

# Windows conda trap (fleet note, bioclip2 row 6): import torch before any NumPy linear algebra in this process.
with contextlib.suppress(ImportError):
    import torch  # noqa: F401

import layoutlm_document_qa_pipeline as ldq  # noqa: E402
from layoutlm_document_qa_pipeline import (  # noqa: E402
    LayoutLMDocumentQAPipeline,
    byod_record_limits,
    check_split_disjoint,
    load_byod_dataset,
    page_from_ground_truth,
    questions_for_page,
    split_dataset,
    split_minimums,
    write_dataset_jsonl,
)

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "layoutlm_document_qa_colab.ipynb"


def _cells():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]


def _source(cell) -> str:
    return "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]


def _code_after(heading: str) -> str:
    cells = _cells()
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "markdown" and heading in _source(cell):
            for nxt in cells[i + 1 :]:
                if nxt["cell_type"] == "code":
                    return _source(nxt)
    raise AssertionError(f"no code cell after {heading!r}")


def _markdown() -> str:
    return "\n".join(_source(c) for c in _cells() if c["cell_type"] == "markdown")


def _quad(x0, y0, x1, y1):
    return {"x1": x0, "y1": y0, "x2": x1, "y2": y0, "x3": x1, "y3": y1, "x4": x0, "y4": y1}


def _line(category, group_id, y, tokens):
    words, x = [], 20
    for text, is_key in tokens:
        for piece in text.split():
            words.append({"quad": _quad(x, y, x + 12 * len(piece), y + 16), "is_key": is_key, "text": piece})
            x += 12 * len(piece) + 8
    return {"words": words, "category": category, "group_id": group_id}


def _page(index: int) -> dict:
    total = 11_000 + 500 * index
    lines = [
        _line("menu.nm", 1, 40, [(f"ITEM{index}", 0)]),
        _line("sub_total.subtotal_price", 5, 120, [("SUBTOTAL", 1), (f"{total - 1000:,}", 0)]),
        _line("total.total_price", 9, 180, [("TOTAL", 1), (f"{total:,}", 0)]),
    ]
    return page_from_ground_truth({"meta": {"image_size": {"width": 400, "height": 300}}, "valid_line": lines}, f"page{index:03d}")


def _one_per_page(n: int) -> list[dict]:
    """n records, one question per page (the case the stated BYOD minimum is about)."""
    return [{"id": f"q{i:03d}", **questions_for_page(_page(i))[0]} for i in range(n)]


def _all_questions(n_pages: int) -> list[dict]:
    out = []
    for i in range(n_pages):
        for q in questions_for_page(_page(i)):
            out.append({"id": f"p{i:03d}-{len(out):03d}", **q})
    return out


def _fake_colab(monkeypatch, uploads: list[dict]):
    queue = list(uploads)
    files = types.ModuleType("google.colab.files")
    files.upload = lambda: queue.pop(0)
    colab = types.ModuleType("google.colab")
    colab.files = files
    google = types.ModuleType("google")
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setitem(sys.modules, "google.colab.files", files)


def _namespace() -> dict:
    ns: dict = {name: getattr(ldq, name) for name in ldq.__all__}
    ns.update({"os": __import__("os"), "Path": Path, "json": json})
    return ns


def _section4(monkeypatch, tmp_path, *, byod_path: str = "") -> dict:
    """Execute Section 4 verbatim with USE_BYOD = True (form literals substituted) in a namespace of the package API."""
    monkeypatch.chdir(tmp_path)
    source = _code_after("## 4. CORD-v2 receipt corpus, validation and split")
    source = source.replace("USE_BYOD = False", "USE_BYOD = True", 1).replace("BYOD_PATH = ''", f"BYOD_PATH = {byod_path!r}", 1)
    ns = _namespace()
    exec(compile(source, "<section 4>", "exec"), ns)
    return ns


# --- LDQ-m1: the BYOD minimum, refusals that name the split, JSON line numbers, BYOD_PATH ------------------------


def test_byod_minimum_is_twelve_records_and_matches_the_prose():
    assert byod_record_limits() == (12, 20_000) and split_minimums() == {"train": 8, "validation": 2, "test": 2}
    markdown = _markdown()
    assert "**at least 12 records with one question per page**" in markdown
    assert "a dataset needs 8..20,000 records" not in markdown


def test_twelve_and_twenty_one_question_pages_pass_and_eleven_is_refused_naming_the_split():
    splits = split_dataset(_one_per_page(12))
    assert {k: len(v) for k, v in splits.items()} == {"test": 2, "validation": 2, "train": 8} and check_split_disjoint(splits)
    assert sum(len(v) for v in split_dataset(_one_per_page(20)).values()) == 20  # the review's failing 20-record case
    with pytest.raises(ValueError, match=r"the train split would hold 7 records \(at least 8 are required\): 11 records on 11 pages.*7/2/2; a dataset needs at least 12 records with one question per page"):
        split_dataset(_one_per_page(11))
    with pytest.raises(ValueError, match=r"the train split would hold 2 records \(at least 8 are required\): 3 records on 3 pages.*2/0/1.*at least 12 records"):
        split_dataset(_one_per_page(3))


def test_multi_question_pages_move_whole_and_the_refusal_says_so():
    records = _all_questions(4)  # 3 questions per page: 12 records on 4 pages
    with pytest.raises(ValueError, match=r"split would hold .*12 records on 4 pages.*a page never straddles two splits"):
        split_dataset(records)


def test_jsonl_and_json_parse_errors_name_the_line(tmp_path):
    path = write_dataset_jsonl(_one_per_page(3), tmp_path / "data.jsonl")
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join([lines[0], lines[1], lines[2][:-5], ""]), encoding="utf-8")
    with pytest.raises(ValueError, match=r"data\.jsonl, line 3: not valid JSON"):
        load_byod_dataset(path)
    bad = tmp_path / "data.json"
    bad.write_text("[\n  {\"id\": \"a\"},\n  {oops}\n]", encoding="utf-8")
    with pytest.raises(ValueError, match=r"data\.json, line 3: not valid JSON"):
        load_byod_dataset(bad)
    (tmp_path / "data.csv").write_text("id", encoding="utf-8")
    with pytest.raises(ValueError, match=".json or .jsonl"):
        load_byod_dataset(tmp_path / "data.csv")


def test_byod_path_runs_section4_without_colab(monkeypatch, tmp_path, capsys):
    monkeypatch.setitem(sys.modules, "google.colab", None)  # `from google.colab import files` would raise ImportError
    path = write_dataset_jsonl(_one_per_page(20), tmp_path / "mine.jsonl")
    ns = _section4(monkeypatch, tmp_path, byod_path=str(path))
    assert sum(len(v) for v in ns["splits"].values()) == 20 and ns["data_source"] == "BYOD (mine.jsonl)"
    assert ns["byod"]["records"] == 20 and len(ns["byod"]["sha256"]) == 64
    out = capsys.readouterr().out
    assert "'byod_minimum_records': 12" in out and "'column_sha256': None" in out
    with pytest.raises(ValueError, match=r"the train split would hold"):
        _section4(monkeypatch, tmp_path, byod_path=str(write_dataset_jsonl(_one_per_page(11), tmp_path / "small.jsonl")))


def test_cancelled_upload_no_colab_and_missing_path_are_actionable(monkeypatch, tmp_path):
    _fake_colab(monkeypatch, [{}])
    with pytest.raises(ValueError, match=r"Upload exactly one .json or .jsonl file \(received 0\).*BYOD_PATH"):
        _section4(monkeypatch, tmp_path)
    monkeypatch.setitem(sys.modules, "google.colab", None)
    with pytest.raises(RuntimeError, match="needs Google Colab.*BYOD_PATH"):
        _section4(monkeypatch, tmp_path)
    with pytest.raises(FileNotFoundError, match="is not a file in this runtime"):
        _section4(monkeypatch, tmp_path, byod_path=str(tmp_path / "missing.jsonl"))


def test_an_uploaded_file_goes_through_section4(monkeypatch, tmp_path):
    payload = write_dataset_jsonl(_one_per_page(13), tmp_path / "up.jsonl").read_bytes()
    _fake_colab(monkeypatch, [{"up.jsonl": payload}])
    ns = _section4(monkeypatch, tmp_path)
    assert ns["byod"]["file"] == "up.jsonl" and sum(len(v) for v in ns["splits"].values()) == 13


# --- LDQ-M2 / LDQ-M3: every pass starts from the pretrained model ------------------------------------------------


def _injected(answer_index, adapter=None):
    def runner(question, words, grid_boxes):
        i = answer_index(question, words)
        return {"start": i, "end": i, "score": 0.5, "n_windows": 1}

    pipe = LayoutLMDocumentQAPipeline(runner, "cpu", "float32", "injected", lambda q, w: 1)
    pipe.adapter = adapter
    return pipe


def test_adapt_refuses_an_already_adapted_pipeline():
    pipe = _injected(lambda q, w: 0, adapter={"best_epoch": 1})
    with pytest.raises(ValueError, match="already adapted.*from_pretrained"):
        pipe.adapt(_one_per_page(8))


def test_sections_5_to_7_reset_and_section_6_refuses_an_adapted_model():
    for heading in ("## 5. Fit check", "## 6. Baselines", "## 7. Bounded fine-tuning"):
        assert "reset_to_pretrained()" in _code_after(heading), heading
    section6 = _code_after("## 6. Baselines")
    assert section6.index("reset_to_pretrained()") < section6.index("pipe.evaluate(")
    assert "if frozen_test['adapted']:" in section6
    section7 = _code_after("## 7. Bounded fine-tuning")
    assert section7.index("reset_to_pretrained()") < section7.index("pipe.adapt(")
    parity = _code_after("## 9. Re-read the invoice")
    assert "raise RuntimeError(f'Reload parity failed: {parity}." in parity and "re-run from Section 7" in parity


def test_reset_to_pretrained_reloads_only_an_adapted_pipeline():
    source = _code_after("## 5. Fit check")
    block = source[source.index("def reset_to_pretrained():") : source.index("reset_to_pretrained()\n")]
    loads: list = []

    class _Loader:
        @staticmethod
        def from_pretrained(weights_dir):
            loads.append(weights_dir)
            return _injected(lambda q, w: 0)

    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None))
    ns = {"gc": __import__("gc"), "torch": fake_torch, "LayoutLMDocumentQAPipeline": _Loader, "WEIGHTS_DIR": "w", "pipe": _injected(lambda q, w: 0)}
    exec(compile(block, "<reset>", "exec"), ns)
    ns["reset_to_pretrained"]()
    assert loads == []
    ns["pipe"] = _injected(lambda q, w: 0, adapter={"best_epoch": 2})
    ns["reset_to_pretrained"]()
    assert loads == ["w"] and ns["pipe"].adapter is None


def test_experiments_and_byod_name_their_run_after_scope():
    markdown = _markdown()
    assert "re-run from that cell" not in markdown
    experiments = markdown[markdown.index("**Optional experiments") : markdown.index("## Troubleshooting")]
    for line in [ln for ln in experiments.splitlines() if ln.startswith("- **") and "Section 11" not in ln]:
        assert re.search(r"Section \d", line) and ("Run after" in line or "run only that cell" in line), line
    assert "**Predict → Change one thing → Run → Observe → Explain**" in markdown
    assert "select the Section 7 cell and choose **Runtime → Run after**" in markdown
    assert "select that cell and choose **Runtime → Run after**" in markdown  # the BYOD declaration


# --- LDQ-m4: results are recorded, regressions and disagreements shown ------------------------------------------


def _run_sections_6_and_8(monkeypatch, tmp_path, frozen, adapted) -> dict:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir(exist_ok=True)
    records = _all_questions(6)
    ns = _namespace()
    ns.update({"collections": __import__("collections"), "time": __import__("time"), "pipe": frozen, "test_records": records, "val_records": records[:4], "reset_to_pretrained": lambda: None})
    exec(compile(_code_after("## 6. Baselines"), "<section 6>", "exec"), ns)
    ns.update({"pipe": adapted, "USE_BYOD": True, "byod": {"file": "x.jsonl"}, "EPOCHS": 6, "LEARNING_RATE": 3e-5, "TRAINABLE_ENCODER_LAYERS": 4, "data_source": "BYOD (x.jsonl)", "dataset_manifests": {"test": {"digest": "d"}}, "disjoint": {"test": len(records)}, "pages": {"test": 6}, "fit": {}, "adapt_seconds": 1.0, "adapt_result": {"best_epoch": 3, "history": [], "trainable_names": []}})
    exec(compile(_code_after("## 8. Held-out evaluation"), "<section 8>", "exec"), ns)
    return ns


def test_a_field_that_loses_score_is_reported_and_nothing_is_asserted(monkeypatch, tmp_path, capsys):
    # frozen: the last word (the total) for every question; adapted: the last word except for totals (first word)
    frozen = _injected(lambda q, w: len(w) - 1)
    adapted = _injected(lambda q, w: 0 if "total amount" in q else len(w) - 1, adapter={"best_epoch": 3})
    ns = _run_sections_6_and_8(monkeypatch, tmp_path, frozen, adapted)
    assert ns["adapted_beats_frozen"] is False and "did not help here" in ns["reading"]
    lost = [f for f, row in ns["comparison"]["by_field"].items() if row["adapted"] < row["frozen"]]
    assert ns["worse_fields"] == lost and lost
    assert len(ns["disagreements"]) == 6 and "'questions_where_the_models_disagree': 6" in capsys.readouterr().out
    report = json.loads((tmp_path / "outputs" / "layoutlm_document_qa_evaluation_report.json").read_text(encoding="utf-8"))
    assert report["outcomes"]["adapted_beats_frozen"] is False and len(report["disagreements"]) == 6 and len(report["run_history"]) == 1
    assert all("too_few_to_read" in row for row in report["comparison"]["by_field"].values())


def test_no_learner_cell_asserts_a_result():
    for cell in _cells():
        source = _source(cell)
        if cell["cell_type"] != "code" or "dimer" in cell.get("metadata", {}) or "# dimer: kernel cell" in source:
            continue
        assert not re.search(r"^\s*assert ", source, re.M), source[:120]


def test_expected_values_quote_the_recorded_run_and_state_the_regression():
    markdown = _markdown()
    assert "**0.819 → 0.952**" in markdown and "recorded Kaggle Tesla T4 run" in markdown and "**Run-to-run spread.**" in markdown
    assert "**the discount field went down, 1.00 → 0.75**" in markdown
    assert "to about 0.96" not in markdown
    verification = (ROOT / "docs" / "release-verification.md").read_text(encoding="utf-8")
    assert "≈ 0.82 → ≈ 0.94" not in verification and "0.819 → 0.952" in verification


# --- LDQ-m2: single braces ---------------------------------------------------------------------------------------


def test_markdown_shows_single_braces():
    markdown = _markdown()
    assert "{{" not in markdown and "}}" not in markdown
    assert "`[A-Za-z0-9_.:-]{1,64}`" in markdown and "`{id, page_id, question, words, boxes, image_size, answer_start, answer_end}`" in markdown


# --- LDQ-M1 / LDQ-m3: isolated runtime, status --------------------------------------------------------------------


def test_exactly_two_kernel_cells_and_no_restart_text():
    kernel = [c for c in _cells() if c["cell_type"] == "code" and "# dimer: kernel cell" in _source(c)]
    assert len(kernel) == 2
    install = _source(kernel[0])
    assert "--require-hashes" in install and "--managed-python" in install and "LOCK_SHA256" in install and "MANAGED_PYTHON = '3.12.12'" in install
    markdown = _markdown()
    assert "Restart the runtime" not in markdown and "its restart" not in markdown
    meta = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["metadata"]["dimer"]
    assert meta["notebook_spec"] == "2.2" and meta["generated_from"]["generator"] == "build_notebook.py/2.1"
    status = (ROOT / "STATUS.md").read_text(encoding="utf-8")
    assert "Current status: **Candidate**" in status and "manual restart" in status
    verification = (ROOT / "docs" / "release-verification.md").read_text(encoding="utf-8")
    assert "**Completed in two passes, not promotion evidence**" in verification and "(1 restart after install cell)" not in verification
    assert "restart after the install is\n   expected" not in verification


def _load_build():
    spec = importlib.util.spec_from_file_location("build_notebook_under_test", ROOT / "tools" / "build_notebook.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lock_pins_every_direct_dependency_with_hashes():
    build = _load_build()
    lock = (ROOT / "tutorials" / "requirements-colab.lock.txt").read_text(encoding="utf-8")
    build.check_lock(build._pins(ROOT), lock)
    assert len(build.lock_packages(lock)) == 48


@pytest.mark.parametrize("real_google", [False, True])
def test_worker_colab_stubs_have_specs(monkeypatch, real_google):
    """Colab only: accelerate calls importlib.util.find_spec("google.colab"), which raised on a spec-less stub."""
    router = [_source(c) for c in _cells() if c["cell_type"] == "code"][1]
    worker = next(
        node.value.value
        for node in ast.parse(router).body
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "_WORKER_SOURCE"
    )
    start = worker.index('if os.environ.get("DIMER_KERNEL_IS_COLAB") == "1":')
    shim = worker[start : worker.index('_main = types.ModuleType("__main__")', start)]
    fake_google = types.ModuleType("google")
    fake_google.__path__ = []
    monkeypatch.setitem(sys.modules, "google", fake_google if real_google else None)
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    monkeypatch.delitem(sys.modules, "google.colab.files", raising=False)
    monkeypatch.setenv("DIMER_KERNEL_IS_COLAB", "1")
    try:
        exec(compile(shim, "worker-colab-shim", "exec"), {"os": __import__("os"), "sys": sys, "types": types, "_send": None, "_recv": None})
        for name in ("google.colab", "google.colab.files"):
            spec = importlib.util.find_spec(name)
            assert spec is not None and spec.name == name
        assert sys.modules["google.colab"].__path__ == [] and callable(sys.modules["google.colab.files"].upload)
        if not real_google:
            assert importlib.util.find_spec("google") is not None
    finally:
        for name in ("google", "google.colab", "google.colab.files"):
            sys.modules.pop(name, None)  # monkeypatch then restores whatever was there before


def test_generator_refuses_a_top_level_name_bound_differently_in_two_modules():
    build = _load_build()
    build.check_name_collisions({"a.py": "X = 1\n", "b.py": "X = 1\n"})
    with pytest.raises(SystemExit):
        build.check_name_collisions({"a.py": "X = 1\n", "b.py": "X = 2\n"})


# --- LDQ-m5: guided layer and infrastructure labels ---------------------------------------------------------------


def test_guided_layer_and_infrastructure_labels():
    markdown = _markdown()
    for marker, least in (("**Predict before running:**", 7), ("**What to notice:**", 7), ("<summary>Check your reasoning</summary>", 8), ("> **Infrastructure.**", 3)):
        assert markdown.count(marker) >= least, marker
    for marker in ("**Who this is for.**", "**Input → Model → Output.**", "**How to use this notebook.**", "**Roadmap:**", "## 10. Try it — questions the page cannot answer", "## 11. Your turn — change one thing", "## Troubleshooting", "## Glossary", "## Conclusion (your notes)"):
        assert marker in markdown, marker
    code = [c for c in _cells() if c["cell_type"] == "code"]
    setup = code[:7]  # install, router, runtime record, three carried modules, model staging
    assert all(c["metadata"].get("cellView") == "form" for c in setup)
    assert all(_source(c).startswith("# @title Infrastructure: ") for c in setup)
    assert "cellView" not in code[7]["metadata"]  # the learning path starts in Section 4
    assert "UNSUPPORTED_QUESTIONS = [" in _code_after("## 10. Try it")
    registry = (ROOT / "tutorials" / "README.md").read_text(encoding="utf-8")
    for item in ("GDL1", "GDL2", "GDL3", "GDL4", "GDL6", "GDL7", "GDL9", "GDL10", "GDL11", "GDL13", "GDL14"):
        assert f"| {item} " in registry, item
