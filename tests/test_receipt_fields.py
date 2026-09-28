"""Amount grammar, geometry helpers and the rule baseline (spec sections 8-9, 19.1)."""

import json

import pytest
import receipt_fields as rf

from _receipt_fixtures import document


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("75,000", "75000"),
        ("75.000", "75000"),
        ("75,000.00", "75000"),
        ("75.000,00", "75000"),
        ("1.500.000", "1500000"),
        ("12,50", "12.50"),
        ("12.50", "12.50"),
        ("0", "0"),
        ("075.000", "75000"),
        ("Rp 75.000", "75000"),
        ("Rp75.000", "75000"),
        ("1.234.567,89", "1234567.89"),
    ],
)
def test_cord_grammar_accepts_declared_forms(text, expected):
    out = rf.amount(text)
    assert out["parse_status"] == "ok" and out["normalized_amount"] == expected
    assert isinstance(out["normalized_amount"], str)


@pytest.mark.parametrize(
    ("text", "status"),
    [
        ("1,23,456", "ambiguous"),
        ("1.5", "ambiguous"),
        ("1.2345", "ambiguous"),
        ("1,234.567", "ambiguous"),
        ("10%", "unsupported"),
        ("-5.000", "unsupported"),
        ("(5.000)", "unsupported"),
        ("5.000-", "unsupported"),
        ("75.OOO", "parse_failed"),
        ("TOTAL 75.000", "ambiguous"),
        ("10.000 20.000", "ambiguous"),
        ("", "no_candidate"),
        (None, "no_candidate"),
        ("Rp", "parse_failed"),
    ],
)
def test_cord_grammar_never_guesses(text, status):
    out = rf.amount(text)
    assert out["parse_status"] == status and out["normalized_amount"] is None


def test_letter_o_is_never_repaired_to_zero():
    assert rf.amount("7O.000")["parse_status"] == "parse_failed"


def test_zero_is_a_value_distinct_from_missing():
    assert rf.amount("0")["normalized_amount"] == "0"
    assert rf.amount("")["normalized_amount"] is None


def test_currency_marker_is_recorded_not_inferred():
    assert rf.amount("Rp 10.000")["currency"] == "IDR"
    assert rf.amount("10.000")["currency"] == "unspecified"
    assert rf.amount("10.000", currency="PHP")["currency"] == "PHP"


def test_byod_policies_are_distinct_from_cord_grammar():
    assert rf.amount("1,234.50", "dot_decimal_comma_grouping")["normalized_amount"] == "1234.50"
    assert rf.amount("1.500", "dot_decimal_comma_grouping")["parse_status"] != "ok"
    assert rf.amount("1.234,50", "comma_decimal_dot_grouping")["normalized_amount"] == "1234.50"
    assert rf.amount("1,500", "comma_decimal_dot_grouping")["parse_status"] != "ok"
    with pytest.raises(rf.ContractError):
        rf.amount("1", "guess_the_format")


def test_decimal_strings_round_trip_through_json():
    value = rf.amount("75.000,50")
    assert json.loads(json.dumps(value))["normalized_amount"] == "75000.50"


def test_intersection_over_area_and_box_checks():
    assert rf.intersection_over_area([0, 0, 10, 10], [5, 0, 20, 10]) == pytest.approx(0.5)
    assert (
        rf.valid_box([0, 0, 1, 1], 10, 10)
        and not rf.valid_box([5, 5, 5, 9])
        and not rf.valid_box([0, 0, 20, 1], 10, 10)
    )


RULES = {
    "fields": {
        "total_amount": [{"phrase": ["total"], "support": 10, "purity": 1.0}],
        "subtotal_amount": [
            {"phrase": ["sub", "total"], "support": 5, "purity": 1.0},
            {"phrase": ["subtotal"], "support": 9, "purity": 1.0},
        ],
        "tax_amount": [{"phrase": ["tax"], "support": 8, "purity": 1.0}],
        "service_charge": [{"phrase": ["service"], "support": 4, "purity": 1.0}],
    }
}


def test_rules_extract_right_aligned_amounts_and_exclude_cash_change():
    doc = document(
        [
            ["SUB", "TOTAL", "70.000"],
            ["TAX", "7.000"],
            ["TOTAL", "77.000"],
            ["CASH", "100.000"],
            ["CHANGE", "23.000"],
        ]
    )
    out = rf.rule_extract(doc, RULES)
    assert out["total_amount"]["normalized_amount"] == "77000"
    assert out["subtotal_amount"]["normalized_amount"] == "70000"  # "sub total" is never read as "total"
    assert out["tax_amount"]["normalized_amount"] == "7000"
    assert out["service_charge"]["parse_status"] == "no_candidate"
    assert out["total_amount"]["span_score"] == pytest.approx(0.9)
    assert out["total_amount"]["key_token_ids"] == [5]


def test_rules_distinct_tied_values_are_ambiguous():
    out = rf.rule_extract(document([["TOTAL", "10.000"], ["TOTAL", "12.000"]]), RULES)
    assert (
        out["total_amount"]["parse_status"] == "ambiguous"
        and out["total_amount"]["normalized_amount"] is None
    )


def test_rules_next_line_fallback_and_missing_confidence():
    doc = document([["TOTAL"], ["50.000"]], confidence=None)
    out = rf.rule_extract(doc, RULES)
    assert out["total_amount"]["normalized_amount"] == "50000"
    assert out["total_amount"]["span_score"] is None  # ineligible for unflagged routing


def test_rules_rightmost_amount_on_key_line():
    out = rf.rule_extract(document([["TOTAL", "3", "30.000"]]), RULES)
    assert out["total_amount"]["normalized_amount"] == "30000"


def test_document_failures_are_structured():
    empty = {"receipt_id": "x", "image_size": [10, 10], "tokens": [], "ocr_status": "ok"}
    assert rf.check_document(empty) == "ocr_empty"
    long = document([["w"] * 5] * 401)
    assert rf.check_document(long) == "input_limit_exceeded"
    failed = rf.rule_extract(empty, RULES)
    assert all(v["parse_status"] == "failed" and v["reasons"] == ["ocr_empty"] for v in failed.values())


def test_keyword_derivation_uses_purity_and_support():
    counts = {
        "total": {"total.total_price": 20, "total.cashprice": 1},
        "cash": {"total.cashprice": 30},
        "sub total": {"sub_total.subtotal_price": 6},
        "tax": {"sub_total.tax_price": 2},
        "discount": {"sub_total.discount_price": 9},
    }
    rules = rf.derive_keywords(counts)
    phrases = {f: [" ".join(k["phrase"]) for k in v] for f, v in rules["fields"].items()}
    assert phrases["total_amount"] == ["total"] and phrases["subtotal_amount"] == ["sub total"]
    assert phrases["tax_amount"] == []  # support below 3
    assert "cash" not in sum(phrases.values(), []) and "discount" not in sum(phrases.values(), [])
