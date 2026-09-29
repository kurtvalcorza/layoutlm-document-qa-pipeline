"""Generated-notebook contract: parity, carried sources, stage order, guided layer, standalone rules."""

import ast
import importlib.util
import json
import re

import pytest

from _receipt_fixtures import ROOT, TOOLS

NOTEBOOK = ROOT / "tutorials" / "DIMER_Small_Business_Receipt_Intelligence_Capstone.ipynb"


def load_generator():
    spec = importlib.util.spec_from_file_location(
        "build_receipt_capstone", TOOLS / "build_receipt_capstone.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def source(cell):
    return "".join(cell["source"])


def test_generator_parity():
    content = json.dumps(load_generator().build(), indent=1, ensure_ascii=False) + "\n"
    same = NOTEBOOK.read_text(encoding="utf-8") == content  # a bool: never ask pytest to diff 1 MB strings
    assert same, "notebook is stale; run: python tools/build_receipt_capstone.py"


def carried(notebook):
    cell = next(c for c in notebook["cells"] if c["metadata"].get("dimer", {}).get("embedded_sources"))
    tree = ast.parse(source(cell))
    values = {
        t.targets[0].id: ast.literal_eval(t.value)
        for t in tree.body
        if isinstance(t, ast.Assign)
        and isinstance(t.targets[0], ast.Name)
        and t.targets[0].id in ("CARRIED_FILES", "CARRIED_HASHES", "NOTEBOOK_REVISION")
    }
    return values


def test_carried_sources_equal_repository_and_hashes_verify(notebook):
    import hashlib

    values = carried(notebook)
    files, hashes = values["CARRIED_FILES"], values["CARRIED_HASHES"]
    same = files["capstone.py"] == (TOOLS / "receipt_capstone.py").read_text(encoding="utf-8")
    assert same, "capstone.py drifted"
    for name in load_generator().MODULES:
        same = files[name] == (TOOLS / name).read_text(encoding="utf-8")
        assert same, f"{name} drifted"
    for name in ("data", "split", "model", "ocr"):
        repo = json.loads(
            (ROOT / "tutorials" / "receipt_intelligence" / f"{name}_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        assert json.loads(files[f"{name}_manifest.json"]) == repo
    same = files["requirements.txt"] == (TOOLS / "receipt-requirements.lock").read_text(encoding="utf-8")
    assert same, "lock drifted"
    assert all(hashlib.sha256(files[n].encode()).hexdigest() == h for n, h in hashes.items())


def test_revision_is_consistent(notebook):
    import receipt_capstone

    assert (
        carried(notebook)["NOTEBOOK_REVISION"]
        == load_generator().NOTEBOOK_REVISION
        == receipt_capstone.NOTEBOOK_REVISION
    )
    assert notebook["metadata"]["dimer"]["notebook_revision"] == receipt_capstone.NOTEBOOK_REVISION


def test_canonical_stage_order(notebook):
    import receipt_capstone

    calls = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code" and "Optional BYOD run" not in source(cell):
            calls += re.findall(r"run_stage\('([a-z_]+)'\)", source(cell))
    assert calls == [
        "ocr_runtime",
        "model",
        "prepare",
        "references",
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
    ]
    graph = receipt_capstone.CANONICAL_GRAPH
    assert set(calls) == set(graph)
    assert all(calls.index(dep) < calls.index(stage) for stage in graph for dep in graph[stage])


def test_standalone_and_no_prompts(notebook):
    cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    code = "\n".join(source(c) for c in cells if not c["metadata"].get("dimer", {}).get("embedded_sources"))
    carried_code = "\n".join(v for k, v in carried(notebook)["CARRIED_FILES"].items() if k.endswith(".py"))
    for forbidden in (
        "git clone",
        "pip install -e",
        "layoutlm_document_qa_pipeline",
        "files.upload(",
        "input(",
        "getpass",
        "apt-get install",
        "sys.path.insert",
    ):
        found = forbidden in code  # bools only: never ask pytest to render megabyte strings
        assert not found, forbidden
    for forbidden in (
        "sys.path.insert",
        "layoutlm_document_qa_pipeline",
        "subprocess.run(['apt",
        "trust_remote_code=True",
    ):
        found = forbidden in carried_code
        assert not found, forbidden
    assert "USE_BYOD = False" in code and "BYOD_AUTHORIZED = False" in code


def test_guided_layer_is_complete(notebook):
    text = "\n".join(source(c) for c in notebook["cells"] if c["cell_type"] == "markdown")
    for number in range(16):
        assert re.search(rf"^## {number}\. ", text, re.M), f"section {number}"
    assert text.count("**Predict") >= 10 and text.count("<details><summary>Show a worked answer") >= 9
    for phrase in (
        "### Glossary",
        "AI Assistance Disclosure",
        "not required submissions",
        "Indonesian",
        "not_annotated",
        "refer all",
        "reference_text_diagnostic",
        "https://doi.org/10.1109/TIT.1970.1054406",
        "https://doi.org/10.1145/3394486.3403172",
    ):
        assert phrase in text, phrase
    meta = notebook["metadata"]["dimer"]
    assert (meta["profile"], meta["pedagogical_mode"], meta["notebook_spec"], meta["release_status"]) == (
        "E2E",
        "GUIDED",
        "2.2",
        "Candidate",
    )
