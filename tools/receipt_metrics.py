"""Evaluator-owned metric definitions (spec section 12). Predictors never import this module."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from receipt_common import FIELDS

METRIC_DEFINITIONS = {
    "total_em": "exactly matching canonical total / receipts with a present_usable total reference; OCR, "
    "model, parse and ambiguity failures count as incorrect",
    "field_em": "exact canonical match on that field's present_usable references",
    "field_macro_em": "unweighted mean of field EM over fields with at least one usable reference",
    "all_annotated_fields_match": "receipts with a usable total whose every present_usable field matches; "
    "says nothing about unannotated fields",
    "text_em_anls": "diagnostic answer-string agreement with the reference value words",
    "candidate_coverage": "fraction of frozen receipts producing any candidate string",
    "parse_coverage": "fraction of frozen receipts producing a supported normalized amount",
    "not_annotated": "predictions on not_annotated fields are unscored and unverified",
}
_PUNCT = re.compile(r"[^\w\s]")


def correct(prediction: Mapping[str, Any], reference: Mapping[str, Any]) -> bool | None:
    """None when the reference is not usable; otherwise exact canonical equality."""
    if reference["state"] != "present_usable":
        return None
    return (
        prediction.get("parse_status") == "ok"
        and prediction.get("normalized_amount") == reference["normalized_amount"]
    )


def _norm(text: str | None) -> str:
    return " ".join(_PUNCT.sub(" ", (text or "").lower()).split())


def _lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def anls(prediction: str | None, gold: str | None, threshold: float = 0.5) -> float:
    p, g = _norm(prediction), _norm(gold)
    longest = max(len(p), len(g))
    sim = 1.0 if longest == 0 else 1.0 - _lev(p, g) / longest
    return sim if sim >= threshold else 0.0


def rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> list[float] | None:
    if n == 0:
        return None
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0.0, centre - half), min(1.0, centre + half)]


def system_metrics(
    predictions: Mapping[str, Mapping[str, Any]],
    references: Mapping[str, Mapping[str, Any]],
    ids: Sequence[str],
) -> dict[str, Any]:
    """All primary and secondary metrics for one system on one frozen receipt set."""
    out: dict[str, Any] = {"frozen_receipts": len(ids), "fields": {}}
    for field in FIELDS:
        usable = [i for i in ids if references[i]["fields"][field]["state"] == "present_usable"]
        hits = [i for i in usable if correct(predictions[i]["fields"][field], references[i]["fields"][field])]
        states = Counter(predictions[i]["fields"][field]["parse_status"] for i in ids)
        usable_states = Counter(predictions[i]["fields"][field]["parse_status"] for i in usable)
        reasons = Counter(r for i in ids for r in predictions[i]["fields"][field].get("reasons", []))
        text_em = [
            (
                _norm(predictions[i]["fields"][field].get("raw_text"))
                == _norm(references[i]["fields"][field]["raw_text"])
            )
            for i in usable
        ]
        anls_values = [
            anls(predictions[i]["fields"][field].get("raw_text"), references[i]["fields"][field]["raw_text"])
            for i in usable
        ]
        unscored = sum(
            1
            for i in ids
            if references[i]["fields"][field]["state"] == "not_annotated"
            and predictions[i]["fields"][field]["parse_status"] == "ok"
        )
        out["fields"][field] = {
            "usable_references": len(usable),
            "correct": len(hits),
            "em": rate(len(hits), len(usable)),
            "reference_states": dict(
                sorted(Counter(references[i]["fields"][field]["state"] for i in ids).items())
            ),
            "prediction_states": dict(sorted(states.items())),
            "prediction_states_on_usable": dict(sorted(usable_states.items())),
            "failure_reasons": dict(sorted(reasons.items())),
            "candidate_coverage": rate(
                sum(1 for i in ids if predictions[i]["fields"][field].get("raw_text")), len(ids)
            ),
            "parse_coverage": rate(states.get("ok", 0), len(ids)),
            "text_em": rate(sum(text_em), len(usable)),
            "anls": None if not usable else sum(anls_values) / len(usable),
            "unscored_predictions_on_not_annotated": unscored,
        }
    total = out["fields"]["total_amount"]
    out["total_em"] = total["em"]
    out["total_correct"] = total["correct"]
    out["total_usable"] = total["usable_references"]
    out["total_em_interval_wilson"] = wilson(total["correct"], total["usable_references"])
    contributing = [f for f in FIELDS if out["fields"][f]["usable_references"] > 0]
    out["field_macro_em"] = (
        None if not contributing else sum(out["fields"][f]["em"] for f in contributing) / len(contributing)
    )
    out["field_macro_contributors"] = contributing
    with_total = [i for i in ids if references[i]["fields"]["total_amount"]["state"] == "present_usable"]
    all_match = [
        i
        for i in with_total
        if all(
            correct(predictions[i]["fields"][f], references[i]["fields"][f])
            for f in FIELDS
            if references[i]["fields"][f]["state"] == "present_usable"
        )
    ]
    out["all_annotated_fields_match"] = rate(len(all_match), len(with_total))
    out["all_annotated_fields_denominator"] = len(with_total)
    return out


def paired_bootstrap(
    a: Mapping[str, Mapping[str, Any]],
    b: Mapping[str, Mapping[str, Any]],
    references: Mapping[str, Mapping[str, Any]],
    ids: Sequence[str],
    *,
    field: str = "total_amount",
    resamples: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Receipt-level paired bootstrap of EM(a) - EM(b) on receipts with a usable reference."""
    import numpy as np

    usable = [i for i in ids if references[i]["fields"][field]["state"] == "present_usable"]
    if not usable:
        return {"n": 0, "difference": None, "interval": None, "resamples": resamples, "seed": seed}
    xa = np.array(
        [bool(correct(a[i]["fields"][field], references[i]["fields"][field])) for i in usable], float
    )
    xb = np.array(
        [bool(correct(b[i]["fields"][field], references[i]["fields"][field])) for i in usable], float
    )
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(usable), size=(resamples, len(usable)))
    diffs = xa[draws].mean(axis=1) - xb[draws].mean(axis=1)
    return {
        "n": len(usable),
        "difference": float(xa.mean() - xb.mean()),
        "interval": [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))],
        "resamples": resamples,
        "seed": seed,
        "unit": "receipt",
        "pairing": "same receipts, both systems",
        "limitations": "ignores merchant/template dependence, checkpoint overlap and annotation bias",
    }


def recoverability(document: Mapping[str, Any], reference: Mapping[str, Any], policy: str) -> str:
    """Evaluator-only: is the correct amount present in a spatially compatible OCR span?"""
    from receipt_training import align_field

    if reference["state"] != "present_usable":
        return "not_scoreable"
    if document.get("ocr_status") != "ok":
        return "not_recoverable"
    status = align_field(document, reference, policy)["status"]
    return {"aligned": "recoverable", "ambiguous_alignment": "recoverable_ambiguous"}.get(
        status, "not_recoverable"
    )
