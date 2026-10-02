"""Denominators, bootstrap, review-policy feasibility and the validation-only activity (spec 12-13)."""

import pytest
import receipt_metrics as m
import receipt_policy as p
from receipt_common import FIELDS


def pred(value=None, status="ok", score=0.9):
    return {
        "raw_text": value,
        "normalized_amount": value if status == "ok" else None,
        "parse_status": status,
        "token_ids": [],
        "span_score": score,
        "reasons": [],
    }


def rec(total, **others):
    fields = {f: pred(None, "no_candidate", None) for f in FIELDS}
    fields["total_amount"] = total
    fields.update(others)
    return {"fields": fields}


def ref(value=None, state="present_usable"):
    return {"state": state, "normalized_amount": value, "raw_text": value}


def refs_for(values):
    out = {}
    for i, v in enumerate(values):
        fields = {f: ref(state="not_annotated") for f in FIELDS}
        fields["total_amount"] = ref(v) if v is not None else ref(state="ambiguous_reference")
        out[f"r{i}"] = {"fields": fields}
    return out


def test_failures_stay_in_the_primary_denominator():
    refs = refs_for(["10", "20", "30", None])
    preds = {
        "r0": rec(pred("10")),
        "r1": rec(pred(None, "failed", None)),
        "r2": rec(pred(None, "ambiguous", None)),
        "r3": rec(pred("5")),
    }
    out = m.system_metrics(preds, refs, sorted(refs))
    assert out["total_usable"] == 3 and out["total_correct"] == 1 and out["total_em"] == pytest.approx(1 / 3)
    assert out["frozen_receipts"] == 4  # the unusable reference stays in operational counts


def test_not_annotated_predictions_are_unscored_not_false_positives():
    refs = refs_for(["10"])
    preds = {"r0": rec(pred("10"), tax_amount=pred("1"))}
    tax = m.system_metrics(preds, refs, ["r0"])["fields"]["tax_amount"]
    assert (
        tax["usable_references"] == 0
        and tax["em"] is None
        and tax["unscored_predictions_on_not_annotated"] == 1
    )


def test_paired_bootstrap_is_deterministic_and_paired():
    refs = refs_for([str(i) for i in range(20)])
    a = {k: rec(pred(v["fields"]["total_amount"]["normalized_amount"])) for k, v in refs.items()}
    b = {k: rec(pred("x")) if i % 2 else a[k] for i, k in enumerate(sorted(refs))}
    one = m.paired_bootstrap(a, b, refs, sorted(refs), resamples=500)
    assert one == m.paired_bootstrap(a, b, refs, sorted(refs), resamples=500)
    assert one["difference"] == pytest.approx(0.5) and one["interval"][0] > 0


def test_wilson_is_undefined_for_empty_sets():
    assert m.wilson(0, 0) is None and m.rate(0, 0) is None


def policy_set(n_correct, n_wrong, wrong_score=0.2):
    values = [str(i) for i in range(n_correct + n_wrong)]
    refs = refs_for(values)
    preds = {}
    for i, k in enumerate(sorted(refs, key=lambda x: int(x[1:]))):
        correct = i < n_correct
        preds[k] = rec(
            pred(values[i] if correct else "wrong-value", score=0.9 - i * 0.001 if correct else wrong_score)
        )
        if not correct:
            preds[k]["fields"]["total_amount"]["normalized_amount"] = "999999"
    return preds, refs


def test_policy_selects_max_coverage_meeting_targets():
    preds, refs = policy_set(40, 10)
    result = p.select(preds, refs, sorted(refs), "layoutlm_adapted")["policy"]
    assert result["policy_feasible"] and result["accepted"] == 40 and result["empirical_accuracy"] == 1.0
    assert result["coverage"] == pytest.approx(0.8)


def test_policy_refers_all_when_infeasible_and_keeps_targets():
    preds, refs = policy_set(10, 10, wrong_score=0.95)
    result = p.select(preds, refs, sorted(refs), "rules_baseline")["policy"]
    assert not result["policy_feasible"] and result["threshold"] is None and result["accepted"] == 0
    assert result["empirical_accuracy"] is None and result["targets"]["min_accuracy"] == 0.95


def test_minimum_accepted_support_is_enforced():
    preds, refs = policy_set(20, 0)
    assert not p.select(preds, refs, sorted(refs), "s")["policy"]["policy_feasible"]  # 20 < 25 accepted


def test_hard_failures_always_need_review():
    policy = {"threshold": 0.0, "policy_id": "x"}
    assert p.route(pred(None, "ambiguous", None), 0.0, "x")["state"] == "needs_review"
    assert p.route(pred("1", score=None), 0.0, "x")["reasons"] == ["total_score_unavailable"]
    assert p.route(pred("1", score=0.5), None, "x")["reasons"] == ["refer_all_policy"]
    assert p.route(pred("1", score=0.5), policy["threshold"], "x")["state"] == "total_unflagged"


def test_review_metrics_selective_error_undefined_when_nothing_unflagged():
    preds, refs = policy_set(5, 5)
    pol = {"system_id": "s", "policy_id": "x", "threshold": None, "policy_feasible": False}
    out = p.review_metrics(preds, refs, sorted(refs), pol)
    assert out["unflagged_scored"] == 0 and out["selective_error"] is None and out["review_count"] == 10


def test_coverage_activity_accepts_the_top_scored_share_only():
    preds, refs = policy_set(20, 30, wrong_score=0.1)
    point = p.coverage_operating_point(preds, refs, sorted(refs), "s", 0.30)["point"]
    assert point["reached_requested"] and point["accepted"] == 15 and point["coverage"] == pytest.approx(0.30)
    assert point["empirical_accuracy"] == 1.0 and point["unflagged_wrong"] == 0
    wider = p.coverage_operating_point(preds, refs, sorted(refs), "s", 0.60)["point"]
    assert wider["accepted"] == 50 and wider["unflagged_wrong"] == 30  # ties at 0.1: the step admits all
    for bad in (0.0, 0.01, 1.5, float("nan")):
        with pytest.raises(ValueError):
            p.coverage_operating_point(preds, refs, sorted(refs), "s", bad)


def test_coverage_activity_exposes_confidently_wrong_totals():
    preds, refs = policy_set(20, 30, wrong_score=0.95)  # every wrong total outranks every right one
    point = p.coverage_operating_point(preds, refs, sorted(refs), "s", 0.30)["point"]
    assert point["accepted"] == 30 and point["unflagged_wrong"] == 30 and point["empirical_accuracy"] == 0.0


def test_coverage_activity_caps_at_the_eligible_totals():
    preds, refs = policy_set(5, 0)
    for i in range(5, 20):
        refs[f"r{i}"] = refs_for(["x"])["r0"]
        preds[f"r{i}"] = rec(pred(None, "parse_failed", None))  # hard failures are never eligible
    point = p.coverage_operating_point(preds, refs, sorted(refs), "s", 0.50)["point"]
    assert not point["reached_requested"] and point["accepted"] == 5
    assert point["coverage"] == pytest.approx(0.25)


def test_coverage_activity_contrasts_with_an_infeasible_canonical_policy():
    """Review RC-M1: 49 usable totals and 20 correct cannot meet 95% on 25 accepted, at any ranking.

    The canonical policy therefore refers all, and the old accuracy-target activity could never differ from
    it. The coverage activity at the notebook's default still accepts totals and reports their errors.
    """
    import importlib.util

    from _receipt_fixtures import TOOLS

    spec = importlib.util.spec_from_file_location("gen", TOOLS / "build_receipt_capstone.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    preds, refs = policy_set(20, 29, wrong_score=0.5)
    canonical = p.select(preds, refs, sorted(refs), "s")["policy"]
    assert not canonical["policy_feasible"] and canonical["accepted"] == 0
    point = p.coverage_operating_point(preds, refs, sorted(refs), "s", gen.ACTIVITY_COVERAGE)["point"]
    assert point["accepted"] >= 0.05 * 49 and point["accepted"] != canonical["accepted"]
    assert canonical["targets"] == p.DEFAULT_TARGETS  # the canonical targets are unchanged
