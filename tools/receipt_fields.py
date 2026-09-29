"""Conservative amount normalisation, OCR box helpers and the fixed rule baseline (System A).

This module is new in the capstone build (the earlier unpublished ``receipt_fields.py`` draft was not
available); it implements specification sections 8-9. Nothing here reads references: the rule
extractor receives OCR tokens, the frozen keyword dictionary and a number-format policy only.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

from receipt_common import FIELDS, ContractError, finite_score

POLICIES = ("cord_mixed_v1", "dot_decimal_comma_grouping", "comma_decimal_dot_grouping")
DEFAULT_POLICY = "cord_mixed_v1"
MAX_ANSWER_WORDS = 4
# Currency markers are removed only when separately recognised; they never change digits.
CURRENCY_MARKERS = {"rp": "IDR", "rp.": "IDR", "idr": "IDR", "php": "PHP", "\u20b1": "PHP", "p": None}
_MARKER_RE = re.compile(r"^(rp\.?|idr|php|\u20b1)\s*", re.IGNORECASE)
_SPACE_RE = re.compile(r"\s+")
PARSE_STATES = ("ok", "no_candidate", "ambiguous", "unsupported", "parse_failed", "failed")


def _clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    return _SPACE_RE.sub(" ", text).strip()


def _canonical(integer: str, fraction: str | None) -> str:
    value = integer.lstrip("0") or "0"
    if fraction is None or set(fraction) == {"0"}:
        return value
    return f"{value}.{fraction}"


def _grouped(body: str, group: str) -> str | None:
    """Integer digits if ``body`` uses ``group`` consistently every three digits, else None."""
    parts = body.split(group)
    if len(parts) < 2 or not 1 <= len(parts[0]) <= 3 or not all(len(p) == 3 for p in parts[1:]):
        return None
    if not all(p.isdigit() for p in parts):
        return None
    return "".join(parts)


def _parse_policy(body: str, policy: str) -> tuple[str, str | None] | str:
    """Return (integer, fraction) or a failure state for a marker-free candidate string."""
    if body.isdigit():
        return body, None
    if policy == "cord_mixed_v1":
        separators = (".", ",")
    elif policy == "dot_decimal_comma_grouping":
        separators = (",",)
    elif policy == "comma_decimal_dot_grouping":
        separators = (".",)
    else:
        raise ContractError(f"Unsupported number-format policy {policy!r}")
    decimals = {
        "cord_mixed_v1": (".", ","),
        "dot_decimal_comma_grouping": (".",),
        "comma_decimal_dot_grouping": (",",),
    }[policy]
    # Grouping only: 75,000 / 75.000 / 1.500.000.
    for group in separators:
        digits = _grouped(body, group)
        if digits is not None:
            return digits, None
    # A two-digit decimal suffix after the *other* separator, or after a lone separator.
    for decimal in decimals:
        head, sep, tail = body.rpartition(decimal)
        if not sep or len(tail) != 2 or not tail.isdigit():
            continue
        if head.isdigit():
            return head, tail
        for group in separators:
            if group == decimal:
                continue
            digits = _grouped(head, group)
            if digits is not None:
                return digits, tail
    return "ambiguous"


def amount(text: Any, policy: str = DEFAULT_POLICY, currency: str = "unspecified") -> dict[str, Any]:
    """Normalise one candidate string under a named grammar. Never repairs or guesses digits.

    Returns ``{parse_status, normalized_amount, currency, number_format_policy_id, raw_text}``;
    ``normalized_amount`` is a decimal string such as ``"75000"`` or ``"12.50"`` or ``None``.
    """
    if policy not in POLICIES:
        raise ContractError(f"Unsupported number-format policy {policy!r}")
    raw = "" if text is None else str(text)
    out = {
        "raw_text": raw,
        "normalized_amount": None,
        "currency": currency,
        "number_format_policy_id": policy,
    }
    body = _clean(raw)
    if not body:
        return {**out, "parse_status": "no_candidate"}
    match = _MARKER_RE.match(body)
    if match:
        marker = CURRENCY_MARKERS.get(match.group(1).lower().replace(" ", ""))
        if marker:
            out["currency"] = marker
        body = body[match.end() :]
    if not body:
        return {**out, "parse_status": "parse_failed", "reason": "currency_marker_only"}
    if "%" in body:
        return {**out, "parse_status": "unsupported", "reason": "percentage"}
    if body[0] in "-+(\u2212" or body[-1] in "-)" or "\u2212" in body:
        return {**out, "parse_status": "unsupported", "reason": "sign_or_refund"}
    if " " in body:
        return {**out, "parse_status": "ambiguous", "reason": "multiple_or_spaced_amounts"}
    if not re.fullmatch(r"[0-9.,]+", body):
        return {**out, "parse_status": "parse_failed", "reason": "non_numeric_characters"}
    parsed = _parse_policy(body, policy)
    if isinstance(parsed, str):
        return {**out, "parse_status": parsed, "reason": "separator_grammar"}
    integer, fraction = parsed
    return {**out, "parse_status": "ok", "normalized_amount": _canonical(integer, fraction)}


def is_amount_token(text: str, policy: str = DEFAULT_POLICY) -> bool:
    return amount(text, policy)["parse_status"] == "ok"


# ---- geometry -------------------------------------------------------------------------------------


def box_union(boxes: Sequence[Sequence[float]]) -> list[float]:
    if not boxes:
        raise ContractError("box_union needs at least one box")
    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]


def valid_box(box: Any, width: int | None = None, height: int | None = None) -> bool:
    if not isinstance(box, Sequence) or isinstance(box, str) or len(box) != 4:
        return False
    if not all(finite_score(v) for v in box):
        return False
    x0, y0, x1, y1 = (float(v) for v in box)
    if not (x0 < x1 and y0 < y1 and x0 >= 0 and y0 >= 0):
        return False
    return not (width is not None and (x1 > width or y1 > height))


def intersection_over_area(candidate: Sequence[float], region: Sequence[float]) -> float:
    """Intersection area divided by the candidate's own area."""
    ix = max(0.0, min(candidate[2], region[2]) - max(candidate[0], region[0]))
    iy = max(0.0, min(candidate[3], region[3]) - max(candidate[1], region[1]))
    area = (candidate[2] - candidate[0]) * (candidate[3] - candidate[1])
    return 0.0 if area <= 0 else ix * iy / area


# ---- documents ------------------------------------------------------------------------------------


def check_document(document: Mapping[str, Any], max_words: int = 2000) -> str | None:
    """Return a structured receipt-failure code, or None when the OCR document is usable."""
    tokens = document.get("tokens")
    if document.get("ocr_status", "ok") != "ok":
        return str(document.get("ocr_status"))
    if not isinstance(tokens, list):
        raise ContractError("document tokens must be a list")
    if not tokens:
        return "ocr_empty"
    if len(tokens) > max_words:
        return "input_limit_exceeded"
    width, height = document["image_size"]
    for index, token in enumerate(tokens):
        if token.get("id") != index or not str(token.get("text", "")).strip():
            raise ContractError("OCR tokens must have dense ids and nonempty text")
        if not valid_box(token.get("box"), width, height):
            raise ContractError(f"OCR token {index} has invalid geometry")
    return None


def field_result(
    tokens: Sequence[Mapping[str, Any]],
    token_ids: Sequence[int] | None,
    *,
    score: float | None,
    policy: str,
    currency: str = "unspecified",
    status: str | None = None,
    reasons: Sequence[str] = (),
) -> dict[str, Any]:
    """The common per-field output contract for every system."""
    if status is not None:
        return {
            "raw_text": None,
            "normalized_amount": None,
            "parse_status": status,
            "token_ids": [],
            "span_score": score,
            "currency": currency,
            "number_format_policy_id": policy,
            "reasons": list(reasons),
            "evidence_status": "machine_extracted_unverified",
        }
    ids = list(token_ids or [])
    raw = " ".join(str(tokens[i]["text"]) for i in ids)
    parsed = amount(raw, policy, currency)
    out = {
        "raw_text": raw,
        "normalized_amount": parsed["normalized_amount"],
        "parse_status": parsed["parse_status"],
        "token_ids": ids,
        "span_score": score,
        "currency": parsed["currency"],
        "number_format_policy_id": policy,
        "reasons": list(reasons) + ([parsed["reason"]] if parsed.get("reason") else []),
        "evidence_status": "machine_extracted_unverified",
    }
    return out


# ---- rule baseline (System A) ---------------------------------------------------------------------

_KEY_STRIP = re.compile(r"[^\w]+", re.UNICODE)


def key_text(text: str) -> str:
    """Case/Unicode normalisation for keyword matching: NFKC, casefold, punctuation removed."""
    return _KEY_STRIP.sub("", unicodedata.normalize("NFKC", str(text)).casefold())


def derive_keywords(
    train_key_phrases: Mapping[str, Mapping[str, int]], *, min_support: int = 3, min_purity: float = 0.8
) -> dict[str, Any]:
    """Freeze a field-keyword dictionary from TRAINING key-phrase counts only.

    ``train_key_phrases`` maps normalised phrase (space-separated key_text tokens) to
    ``{category: count}`` over every training annotation line category. A phrase becomes a keyword
    for a target field when its support is at least ``min_support`` and at least ``min_purity`` of its
    occurrences label that field's category, so ``cash``/``change``/``discount`` phrases cannot
    silently become total keywords.
    """
    from receipt_data import CATEGORY

    by_category = {v: k for k, v in CATEGORY.items()}
    fields: dict[str, list[dict[str, Any]]] = {f: [] for f in FIELDS}
    for phrase, counts in sorted(train_key_phrases.items()):
        total = sum(counts.values())
        category, best = max(sorted(counts.items()), key=lambda kv: kv[1])
        field = by_category.get(category)
        if field and total >= min_support and best / total >= min_purity and phrase.strip():
            fields[field].append({"phrase": phrase.split(" "), "support": best, "purity": best / total})
    for field in fields:
        # Longer phrases first so "sub total" is consumed before "total"; then support, then text.
        fields[field].sort(key=lambda k: (-len(k["phrase"]), -k["support"], k["phrase"]))
    return {
        "schema": "org.dimer.receipt-keyword-rules.v1",
        "min_support": min_support,
        "min_purity": min_purity,
        "source": "official training annotations (is_key words) only",
        "matching": "NFKC casefold, punctuation removed, whole OCR tokens, longest phrase first",
        "fields": fields,
    }


def _lines(tokens: Sequence[Mapping[str, Any]]) -> list[list[Mapping[str, Any]]]:
    lines: dict[Any, list[Mapping[str, Any]]] = {}
    for token in tokens:
        lines.setdefault(tuple(token["line"]), []).append(token)
    return list(lines.values())  # Tesseract reading order is preserved (dict insertion order)


def _phrase_hits(lines: list[list[Mapping[str, Any]]], rules: Mapping[str, Any]) -> list[dict[str, Any]]:
    """All non-overlapping key-phrase matches, longest phrase wins across every field."""
    catalogue = sorted(
        ((f, k) for f in FIELDS for k in rules["fields"][f]),
        key=lambda fk: (-len(fk[1]["phrase"]), -fk[1]["support"], fk[0]),
    )
    hits, used = [], set()
    for line_index, line in enumerate(lines):
        normal = [key_text(t["text"]) for t in line]
        for field, keyword in catalogue:
            phrase = keyword["phrase"]
            for start in range(len(line) - len(phrase) + 1):
                ids = [line[start + j]["id"] for j in range(len(phrase))]
                if normal[start : start + len(phrase)] == phrase and not used.intersection(ids):
                    used.update(ids)
                    hits.append(
                        {
                            "field": field,
                            "line": line_index,
                            "end": start + len(phrase) - 1,
                            "key_ids": ids,
                            "priority": catalogue.index((field, keyword)),
                        }
                    )
    return hits


def _amounts_right(line: list[Mapping[str, Any]], after: int, policy: str) -> list[list[int]]:
    """Parseable single-token (or currency-marker + number) spans right of index ``after``."""
    spans = []
    for i in range(after + 1, len(line)):
        text = str(line[i]["text"])
        if is_amount_token(text, policy):
            if i > after + 1 and _MARKER_RE.fullmatch(str(line[i - 1]["text"]).strip() + " "):
                spans.append([line[i - 1]["id"], line[i]["id"]])
            else:
                spans.append([line[i]["id"]])
    return spans


def rule_extract(
    document: Mapping[str, Any],
    rules: Mapping[str, Any],
    policy: str = DEFAULT_POLICY,
    currency: str = "unspecified",
    next_line_gap: float = 1.5,
) -> dict[str, Any]:
    """System A. Key phrase -> rightmost parseable amount on its line -> else the next OCR line.

    Ranking: key-phrase priority, then same line before next line, then the rightmost amount
    (receipts right-align prices). Distinct values at the best rank are ``ambiguous``.
    """
    failure = check_document(document)
    if failure:
        return {
            f: field_result(
                [], None, score=None, policy=policy, currency=currency, status="failed", reasons=[failure]
            )
            for f in FIELDS
        }
    tokens = document["tokens"]
    lines = _lines(tokens)
    hits = _phrase_hits(lines, rules)
    out = {}
    for field in FIELDS:
        candidates = []
        for hit in (h for h in hits if h["field"] == field):
            line = lines[hit["line"]]
            spans = _amounts_right(line, hit["end"], policy)
            level = 0
            if not spans and hit["line"] + 1 < len(lines):
                below = lines[hit["line"] + 1]
                key_box = box_union([tokens[i]["box"] for i in hit["key_ids"]])
                below_box = box_union([t["box"] for t in below])
                height = key_box[3] - key_box[1]
                if 0 <= below_box[1] - key_box[3] <= next_line_gap * height:
                    spans, level = _amounts_right(below, -1, policy), 1
            if spans:
                best = max(spans, key=lambda s: tokens[s[-1]]["box"][2])
                candidates.append({"rank": (hit["priority"], level), "span": best, "key_ids": hit["key_ids"]})
        if not candidates:
            out[field] = field_result(
                tokens,
                None,
                score=None,
                policy=policy,
                currency=currency,
                status="no_candidate",
                reasons=["no_keyword_amount"],
            )
            continue
        top = min(c["rank"] for c in candidates)
        tied = [c for c in candidates if c["rank"] == top]
        values = {
            amount(" ".join(tokens[i]["text"] for i in c["span"]), policy)["normalized_amount"] for c in tied
        }
        if len(values) > 1:
            out[field] = field_result(
                tokens,
                None,
                score=None,
                policy=policy,
                currency=currency,
                status="ambiguous",
                reasons=["tied_distinct_candidates"],
            )
            out[field]["candidate_token_ids"] = [c["span"] for c in tied]
            continue
        chosen = tied[0]
        confidences = [tokens[i].get("confidence") for i in chosen["key_ids"] + chosen["span"]]
        score = (
            min(float(c) for c in confidences) / 100.0
            if all(finite_score(c) and 0 <= float(c) <= 100 for c in confidences)
            else None
        )
        result = field_result(tokens, chosen["span"], score=score, policy=policy, currency=currency)
        result["key_token_ids"] = chosen["key_ids"]
        result["candidate_count"] = len(candidates)
        out[field] = result
    return out
