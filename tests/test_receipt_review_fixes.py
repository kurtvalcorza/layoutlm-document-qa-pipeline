"""Regression tests for the 2026-10-02 Notebook Review Framework v1 findings (RC-B1, RC-M1, RC-m1..m5).

Each test is the acceptance check written in the review, made executable. CPU only, no model download.
See docs/reviews/2026-10-02-notebook-review/.
"""

import ast
import importlib.util
import json
import re

import pytest

from _receipt_fixtures import ROOT, TOOLS

NOTEBOOK = ROOT / "tutorials" / "DIMER_Small_Business_Receipt_Intelligence_Capstone.ipynb"


@pytest.fixture(scope="module")
def generator():
    path = TOOLS / "build_receipt_capstone.py"
    spec = importlib.util.spec_from_file_location("build_receipt_capstone", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cells():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return {c["id"]: "".join(c["source"]) for c in notebook["cells"]}


def markdown(cells):
    return "\n".join(text for cid, text in cells.items() if cid.startswith("md-"))


def test_rc_b1_colab_badge_points_at_the_default_branch(generator, cells):
    assert generator.BRANCH_FOR_BADGE == "main"
    badges = re.findall(r"colab\.research\.google\.com/github/([^)]+)\)", cells["md-00"])
    assert badges == [
        "kurtvalcorza/layoutlm-document-qa-pipeline/blob/main/tutorials/"
        "DIMER_Small_Business_Receipt_Intelligence_Capstone.ipynb"
    ]


def test_rc_m1_activity_changes_coverage_not_the_accuracy_target(generator, cells):
    text = markdown(cells)
    assert "may make a policy feasible" not in text
    assert "ACTIVITY_TARGET" not in "\n".join(cells.values())
    assert 0.05 <= generator.ACTIVITY_COVERAGE <= 1.0
    assert f"ACTIVITY_COVERAGE = {generator.ACTIVITY_COVERAGE:.2f} # @param" in cells["code-02"]
    assert "'--activity-coverage'" in cells["code-37"] and "coverage_" in cells["code-37"]
    assert "share of totals left unflagged" in cells["md-36"] and "unflagged_wrong" in cells["md-36"]


def test_rc_m1_canonical_review_targets_are_unchanged():
    import receipt_policy

    assert receipt_policy.DEFAULT_TARGETS == {"min_accuracy": 0.95, "min_coverage": 0.50, "min_accepted": 25}


@pytest.mark.parametrize(
    ("correct", "status", "recoverable", "expected"),
    [
        (True, "ok", True, "success"),
        (False, "failed", False, "failure_or_no_candidate"),
        (False, "no_candidate", True, "failure_or_no_candidate"),
        (False, "ambiguous", True, "numeric_ambiguity"),
        (False, "unsupported", False, "numeric_ambiguity"),
        (False, "parse_failed", False, "ocr_loss"),  # e.g. "800" read as "BOO" (review RC-m1)
        (False, "parse_failed", True, "extraction_error"),  # e.g. a label word chosen instead
        (False, "ok", True, "extraction_error"),
        (False, "ok", False, "ocr_loss"),
    ],
)
def test_rc_m1_failure_panel_categories(correct, status, recoverable, expected):
    import receipt_capstone

    assert receipt_capstone.panel_category(correct, status, recoverable) == expected
    assert expected in receipt_capstone.PANEL_CATEGORIES


def test_rc_m1_panel_categories_are_defined_for_the_learner(cells):
    for phrase in ("numeric ambiguity", "extraction error", "OCR loss", "checked in this order"):
        assert phrase in cells["md-29"], phrase


def test_rc_m2_controls_have_rerun_instructions(cells):
    assert "run that controls cell again" in cells["md-01"]
    assert "run the §1 controls cell again" in cells["md-36"]
    assert "run the §1 controls cell again" in cells["md-39"]
    assert "run the §1 controls cell again" in cells["code-40"]  # the "BYOD is off" message says how


def test_rc_m3_no_stale_learner_text(cells):
    text = markdown(cells)
    for stale in (
        "No elapsed time is promised before a qualified hosted run has measured it",
        "this is an open qualification item",
        "hosted qualification pending",
    ):
        assert stale not in text, stale
    assert "Colab with a Tesla T4 and **2 CPUs**" in cells["md-01"] and "Measured once each" in cells["md-01"]


def test_rc_m4_booleans_render_as_words(generator):
    tree = ast.parse(generator.HELPERS)
    fmt = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_fmt")
    namespace = {}
    exec(compile(ast.Module(body=[fmt], type_ignores=[]), "_fmt", "exec"), namespace)
    shown = {value: namespace["_fmt"](value) for value in (True, False, None, 0.5, 7, "x|y")}
    assert shown == {True: "true", False: "false", None: "undefined", 0.5: "0.5000", 7: "7", "x|y": "x/y"}


def test_rc_m5_evidence_records_are_current():
    ladder = (ROOT / "docs" / "receipt-release-evidence.md").read_text(encoding="utf-8")
    assert "Draft PR only" not in ladder and "ad47ef8" in ladder
    notes = (ROOT / "docs" / "receipt-capstone.md").read_text(encoding="utf-8")
    assert "Switch `BRANCH_FOR_BADGE` to `main` and regenerate in the merge commit" not in notes
