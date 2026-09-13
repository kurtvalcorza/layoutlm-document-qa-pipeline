"""Offline tests for the public validation, metric and evaluation stage helpers (DAT24 / EVAL21)."""

from __future__ import annotations

import pytest

from layoutlm_document_qa_pipeline import (
    ANLS_THRESHOLD,
    INPUT_SCHEMA,
    MAX_IMAGE_SIDE,
    MAX_QUESTION_CHARS,
    MAX_WORDS,
    MIN_IMAGE_SIDE,
    MODEL_ID,
    MODEL_REVISION,
    anls,
    evaluation_report,
    exact_match,
    normalize_answer,
    validate_inputs,
)

QUESTIONS = ["What is the invoice number?", "Who is the customer?"]
WORDS = ["Invoice", "number:", "NW-2026-0417"]
BOXES = [[70, 200, 140, 218], [146, 200, 220, 218], [300, 200, 420, 218]]
SIZE = (850, 1100)


def _result(answer: str, question: str = QUESTIONS[0], score: float = 0.9) -> dict:
    return {"answer": answer, "question": question, "score": score}


def test_validate_inputs_returns_manifest_with_schema_and_identity() -> None:
    manifest = validate_inputs(WORDS, BOXES, QUESTIONS, image_size=SIZE, names=["form.png"])
    assert manifest["verdict"] == "accepted"
    assert manifest["findings"] == []
    assert manifest["schema"] == INPUT_SCHEMA
    assert manifest["schema"]["image_side_px"] == [MIN_IMAGE_SIDE, MAX_IMAGE_SIDE]
    assert manifest["schema"]["question_chars"] == [1, MAX_QUESTION_CHARS]
    assert manifest["schema"]["words"] == [1, MAX_WORDS]
    assert manifest["inputs"] == [{"id": "form.png", "size": [850, 1100], "n_words": 3}]
    assert manifest["questions"] == QUESTIONS
    assert (manifest["model_id"], manifest["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_validate_inputs_default_id_and_whitespace() -> None:
    manifest = validate_inputs(WORDS, BOXES, ["  Who   is it? "], image_size=SIZE)
    assert [entry["id"] for entry in manifest["inputs"]] == ["page-0"]
    assert manifest["questions"] == ["Who is it?"]


def test_validate_inputs_rejects_like_answer() -> None:
    with pytest.raises(TypeError, match="non-empty sequence"):
        validate_inputs(WORDS, BOXES, "Who?", image_size=SIZE)
    with pytest.raises(TypeError, match="non-empty sequence"):
        validate_inputs(WORDS, BOXES, [], image_size=SIZE)
    with pytest.raises(ValueError, match="MAX_QUESTION_CHARS"):
        validate_inputs(WORDS, BOXES, ["x" * (MAX_QUESTION_CHARS + 1)], image_size=SIZE)
    with pytest.raises(ValueError, match="one pixel xyxy box per word"):
        validate_inputs(WORDS, BOXES[:1], QUESTIONS, image_size=SIZE)
    with pytest.raises(ValueError, match="not inside"):
        validate_inputs(WORDS, [[0, 0, 5, 5], [0, 0, 5, 5], [0, 0, 5, 2000]], QUESTIONS, image_size=SIZE)
    with pytest.raises(ValueError, match="exactly one entry"):
        validate_inputs(WORDS, BOXES, QUESTIONS, image_size=SIZE, names=["a", "b"])


def test_normalize_answer_and_exact_match() -> None:
    assert normalize_answer("  $1,099.20 ") == "1 099 20"
    assert exact_match("ACME, Corp", ["acme corp"]) is True
    assert exact_match("Acme Co", ["acme corp"]) is False


def test_anls_threshold_and_max_over_golds() -> None:
    assert anls("NW-2026-0417", ["NW-2026-0417"]) == 1.0
    assert anls("Acme Co", ["Acme Corp"]) == pytest.approx(7 / 9)
    assert anls("12", ["Acme Corp"]) == 0.0  # similarity below ANLS_THRESHOLD scores 0
    assert anls("1,099.20", ["$1,099.20", "1099.20"]) == 1.0
    assert 0 < ANLS_THRESHOLD < 1
    with pytest.raises(ValueError, match="golds"):
        anls("x", [])


def test_evaluation_report_not_measurable_without_golds() -> None:
    report = evaluation_report([_result("NW-2026-0417")], sample_kind="BYOD")
    assert report["verdict"] == "not-measurable"
    assert report["metrics"] == []
    assert report["n_questions"] == 1 and report["scores"] == [0.9]
    assert "ANLS" in report["needs"]
    assert (report["model_id"], report["model_revision"]) == (MODEL_ID, MODEL_REVISION)
    assert "not a calibrated probability" in report["score_semantics"]


def test_evaluation_report_sample_sanity_with_golds() -> None:
    results = [_result("NW-2026-0417"), _result("Blue Yonder", QUESTIONS[1], score=0.3)]
    report = evaluation_report(results, [["NW-2026-0417"], ["Blue Yonder Airlines"]])
    assert report["verdict"] == "sample-sanity"
    assert report["scores"] == [0.9, 0.3]
    assert [entry["score"] for entry in report["per_question"]] == [0.9, 0.3]
    by_id = {metric["id"]: metric for metric in report["metrics"]}
    assert by_id["exact_match"]["value"] == 0.5
    assert by_id["anls"]["threshold"] == ANLS_THRESHOLD
    assert by_id["anls"]["value"] == pytest.approx((1.0 + anls("Blue Yonder", ["Blue Yonder Airlines"])) / 2)
    assert [entry["exact_match"] for entry in report["per_question"]] == [True, False]
    assert report["per_question"][1]["golds"] == ["Blue Yonder Airlines"]


def test_evaluation_report_rejects_mismatched_or_empty_golds() -> None:
    with pytest.raises(ValueError, match="golds has"):
        evaluation_report([_result("a")], [["a"], ["b"]])
    with pytest.raises(ValueError, match="non-empty sequence"):
        evaluation_report([_result("a")], [[]])
    with pytest.raises(ValueError, match="results"):
        evaluation_report([], None)
