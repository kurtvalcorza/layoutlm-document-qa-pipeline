"""Total-amount review routing and validation-only threshold selection (spec section 13).

States are ``needs_review`` and ``total_unflagged``; neither means human-verified. Scores are ranking
signals (LayoutLM span score; rule minimum OCR confidence), not calibrated probabilities.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from receipt_common import digest, finite_score
from receipt_metrics import correct, rate, wilson

DEFAULT_TARGETS = {"min_accuracy": 0.95, "min_coverage": 0.50, "min_accepted": 25}


def hard_failure(total: Mapping[str, Any]) -> str | None:
    if total["parse_status"] != "ok":
        return f"total_{total['parse_status']}"
    if not finite_score(total.get("span_score")):
        return "total_score_unavailable"
    return None


def route(total: Mapping[str, Any], threshold: float | None, policy_id: str) -> dict[str, Any]:
    reason = hard_failure(total)
    if reason:
        return {"state": "needs_review", "reasons": [reason], "policy_id": policy_id}
    if threshold is None:
        return {"state": "needs_review", "reasons": ["refer_all_policy"], "policy_id": policy_id}
    if float(total["span_score"]) >= threshold:
        return {"state": "total_unflagged", "reasons": [], "policy_id": policy_id}
    return {"state": "needs_review", "reasons": ["score_below_threshold"], "policy_id": policy_id}


def sweep(
    predictions: Mapping[str, Mapping[str, Any]],
    references: Mapping[str, Mapping[str, Any]],
    ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Coverage/accuracy at every observed eligible score, accept-all-eligible and refer-all."""
    scoreable = [i for i in ids if references[i]["fields"]["total_amount"]["state"] == "present_usable"]
    eligible = [
        (
            float(predictions[i]["fields"]["total_amount"]["span_score"]),
            bool(correct(predictions[i]["fields"]["total_amount"], references[i]["fields"]["total_amount"])),
        )
        for i in scoreable
        if hard_failure(predictions[i]["fields"]["total_amount"]) is None
    ]
    thresholds = sorted({s for s, _ in eligible})
    rows = []
    for kind, threshold in [("observed_score", t) for t in thresholds] + [("refer_all", None)]:
        accepted = [c for s, c in eligible if threshold is not None and s >= threshold]
        n = len(accepted)
        rows.append(
            {
                "kind": kind,
                "threshold": threshold,
                "accepted": n,
                "scoreable": len(scoreable),
                "coverage": rate(n, len(scoreable)),
                "accuracy": rate(sum(accepted), n),
                "errors": n - sum(accepted),
            }
        )
    if thresholds:
        rows[0] = {**rows[0], "kind": "accept_all_eligible"}
    return rows


def select(
    predictions: Mapping[str, Mapping[str, Any]],
    references: Mapping[str, Mapping[str, Any]],
    ids: Sequence[str],
    system_id: str,
    targets: Mapping[str, float] = DEFAULT_TARGETS,
) -> dict[str, Any]:
    """Maximise coverage subject to the declared accuracy/coverage/support targets; else refer all."""
    rows = sweep(predictions, references, ids)
    feasible = [
        r
        for r in rows
        if r["threshold"] is not None
        and r["accuracy"] is not None
        and r["accuracy"] >= targets["min_accuracy"]
        and r["coverage"] >= targets["min_coverage"]
        and r["accepted"] >= targets["min_accepted"]
    ]
    if feasible:
        best = sorted(feasible, key=lambda r: (-r["coverage"], r["errors"] / r["accepted"], -r["threshold"]))[
            0
        ]
    else:
        best = next(r for r in rows if r["kind"] == "refer_all")
    accepted_ids = sorted(
        i
        for i in ids
        if best["threshold"] is not None
        and hard_failure(predictions[i]["fields"]["total_amount"]) is None
        and float(predictions[i]["fields"]["total_amount"]["span_score"]) >= best["threshold"]
    )
    record = {
        "system_id": system_id,
        "targets": dict(targets),
        "policy_feasible": bool(feasible),
        "threshold": best["threshold"],
        "selected_state": best["kind"],
        "selection_role": "validation_policy",
        "selection_support": best["scoreable"],
        "accepted": best["accepted"],
        "empirical_accuracy": best["accuracy"],
        "empirical_accuracy_interval_wilson": wilson(best["accepted"] - best["errors"], best["accepted"]),
        "coverage": best["coverage"],
        "selected_ids": accepted_ids,
        "tie_rule": "max coverage, then lower empirical error, then higher threshold",
        "score_semantics": "uncalibrated ranking score",
    }
    record["policy_id"] = digest(
        {k: record[k] for k in ("system_id", "targets", "threshold", "selected_state")}
    )
    return {"policy": record, "sweep": rows}


def review_metrics(
    predictions: Mapping[str, Mapping[str, Any]],
    references: Mapping[str, Mapping[str, Any]],
    ids: Sequence[str],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply a frozen policy unchanged to held-out receipts."""
    routes = {
        i: route(predictions[i]["fields"]["total_amount"], policy["threshold"], policy["policy_id"])
        for i in ids
    }
    scoreable = [i for i in ids if references[i]["fields"]["total_amount"]["state"] == "present_usable"]
    unflagged = [i for i in ids if routes[i]["state"] == "total_unflagged"]
    unflagged_scored = [i for i in unflagged if i in set(scoreable)]
    wrong = [
        i
        for i in unflagged_scored
        if not correct(predictions[i]["fields"]["total_amount"], references[i]["fields"]["total_amount"])
    ]
    hard = [i for i in ids if hard_failure(predictions[i]["fields"]["total_amount"])]
    return {
        "system_id": policy["system_id"],
        "policy_id": policy["policy_id"],
        "threshold": policy["threshold"],
        "policy_feasible": policy["policy_feasible"],
        "frozen_receipts": len(ids),
        "usable_total_references": len(scoreable),
        "scored_coverage": rate(len(unflagged_scored), len(scoreable)),
        "selective_error": rate(len(wrong), len(unflagged_scored)),
        "selective_error_interval_wilson": wilson(len(wrong), len(unflagged_scored)),
        "unflagged_scored": len(unflagged_scored),
        "unflagged_wrong": len(wrong),
        "operational_routing_coverage": rate(len(unflagged), len(ids)),
        "unflagged_without_usable_reference": len(unflagged) - len(unflagged_scored),
        "review_count": len(ids) - len(unflagged),
        "hard_failures": len(hard),
        "routes": routes,
    }


def coverage_operating_point(predictions, references, ids, system_id: str, coverage: float) -> dict[str, Any]:
    """Change-one-thing: leave the top-scored share of totals unflagged; validation_policy data only.

    The one changed setting is the requested coverage. The highest threshold whose coverage reaches it is
    used, so the system leaves its best-scored totals unflagged and refers the rest. There is no accuracy
    target and no feasibility test: the activity shows what a coverage costs in unflagged errors. When the
    eligible totals cannot reach the request, every eligible total is accepted and ``reached_requested``
    is false. The canonical policy is never touched.
    """
    if math.isnan(coverage) or not (0.05 <= coverage <= 1.0):
        raise ValueError("The activity coverage must be between 0.05 and 1.0 (for example 0.30)")
    rows = sweep(predictions, references, ids)
    candidates = [r for r in rows if r["threshold"] is not None]
    reaching = [r for r in candidates if r["coverage"] is not None and r["coverage"] >= coverage]
    if reaching:
        best = max(reaching, key=lambda r: r["threshold"])
    elif candidates:
        best = min(candidates, key=lambda r: r["threshold"])  # accept every eligible total
    else:
        best = next(r for r in rows if r["kind"] == "refer_all")
    point = {
        "system_id": system_id,
        "requested_coverage": coverage,
        "reached_requested": bool(reaching),
        "threshold": best["threshold"],
        "selected_state": "coverage_operating_point" if best["threshold"] is not None else "refer_all",
        "selection_role": "validation_policy",
        "selection_support": best["scoreable"],
        "accepted": best["accepted"],
        "coverage": best["coverage"],
        "empirical_accuracy": best["accuracy"],
        "empirical_accuracy_interval_wilson": wilson(best["accepted"] - best["errors"], best["accepted"]),
        "unflagged_wrong": best["errors"],
        "score_semantics": "uncalibrated ranking score",
    }
    return {"point": point, "sweep": rows}
