"""Corpus-level document-QA metrics and two non-neural baselines.

The per-question helpers (`normalize_answer`, `anls`, `exact_match`) live in `pipeline.py` and follow DocVQA's
conventions. This module averages them over a dataset and adds two baselines a fine-tuned reader must beat:
**last number** (the last word on the page that contains a digit — receipts end with their totals) and
**keyword lookup** (find the page word that best matches a content word of the question, e.g. `TOTAL` for
"What is the total amount?", and answer with the next word after it that contains a digit — a lookup with no
model that stands in for the "find the label, read the value" heuristic a person applies to a receipt).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from .pipeline import anls, exact_match, normalize_answer

_DIGIT_RE = re.compile(r"\d")
# Function words removed from a question before its remaining words are matched against the page.
QUESTION_STOP_WORDS = frozenset(
    [
        "what", "is", "the", "how", "much", "many", "was", "were", "paid", "given", "by", "of", "name",
        "first", "item", "items", "bought", "amount", "charge",
    ]
)  # fmt: skip
METRIC_DEFINITIONS = {
    "anls": (
        "mean over questions of the Average Normalised Levenshtein Similarity between the prediction and its "
        "best-matching accepted answer (lower-cased, punctuation removed, whitespace collapsed; a similarity "
        "below 0.5 scores 0); in 0..1"
    ),
    "exact_match": (
        "fraction of questions whose normalised prediction equals a normalised accepted answer; in 0..1"
    ),
}


def docqa_metrics(predictions: Sequence[str], golds: Sequence[Sequence[str]]) -> dict[str, Any]:
    """Mean ANLS and exact-match rate over parallel predictions and accepted-answer lists."""
    if len(predictions) != len(golds):
        raise ValueError(f"{len(predictions)} predictions but {len(golds)} gold lists")
    if not predictions:
        raise ValueError("no predictions to score")
    scores = [anls(p, g) for p, g in zip(predictions, golds, strict=True)]
    exact = [exact_match(p, g) for p, g in zip(predictions, golds, strict=True)]
    return {
        "n": len(predictions),
        "anls": sum(scores) / len(scores),
        "exact_match": sum(exact) / len(exact),
        "empty_rate": sum(1 for p in predictions if not p.strip()) / len(predictions),
        "definitions": dict(METRIC_DEFINITIONS),
    }


def _golds(records: Sequence[Mapping[str, Any]]) -> list[list[str]]:
    return [[str(a) for a in r["answers"]] for r in records]


def last_number_answer(words: Sequence[str]) -> str:
    """The last word on the page containing a digit; empty when there is none."""
    for word in reversed(list(words)):
        if _DIGIT_RE.search(word):
            return word
    return ""


def last_number_baseline(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Answer every question with the last numeric word of its page."""
    result = docqa_metrics([last_number_answer(r["words"]) for r in records], _golds(records))
    result["baseline"] = "last word on the page containing a digit"
    return result


def _overlap(a: str, b: str) -> int:
    """Length of the longest common prefix of two normalised words (`subtotal` vs `subtot` → 6)."""
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def keyword_lookup_answer(question: str, words: Sequence[str], *, min_overlap: int = 3) -> str:
    """Find the page word sharing the longest prefix (at least `min_overlap` characters, last such word on
    ties) with any content word of the question; answer with the first word after it that contains a digit,
    or the last numeric word of the page when no keyword matches."""
    content = [w for w in normalize_answer(question).split() if w not in QUESTION_STOP_WORDS]
    page = [normalize_answer(w) for w in words]
    best_index, best_overlap = None, min_overlap - 1
    for index, word in enumerate(page):
        overlap = max((_overlap(word, q) for q in content), default=0)
        if overlap >= best_overlap and overlap >= min_overlap:
            best_index, best_overlap = index, overlap
    if best_index is not None:
        for word in list(words)[best_index + 1 :]:
            if _DIGIT_RE.search(word):
                return word
    return last_number_answer(words)


def keyword_lookup_baseline(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """A label-then-value lookup with no model."""
    predictions = [keyword_lookup_answer(r["question"], r["words"]) for r in records]
    result = docqa_metrics(predictions, _golds(records))
    result["baseline"] = "keyword lookup (best-matching page word, then the next numeric word)"
    return result
