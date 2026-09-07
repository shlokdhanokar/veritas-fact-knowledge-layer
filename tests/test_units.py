"""Phase 3 tests: unit math must be provably correct.

This is the table that lets the system claim "₹8,142 Cr and ₹81,415.38 million
are the same fact" without hand-waving. No LLM involved, no API key needed.
"""

import pytest

from app.normalize.units import (
    normalize_quantity,
    parse_number,
    relative_gap,
    values_agree,
)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("8,142", 8142.0),
        ("81,415.38", 81415.38),
        ("₹ 74,540.82", 74540.82),
        ("(1,679.68)", -1679.68),  # accounting negative
        ("-8.87", -8.87),
        ("1,23,456.78", 123456.78),  # Indian grouping
        ("6.4 per cent", 6.4),
        ("33,000", 33000.0),
        ("", None),
        ("n/a", None),
        ("1.234.567", None),  # ambiguous, must refuse rather than guess
    ],
)
def test_parse_number(text, expected):
    assert parse_number(text) == expected


def test_crore_and_million_reconcile():
    """The Case 1 corroboration, proved arithmetically."""
    deck, _ = normalize_quantity("8,142", None, "Cr", "₹")
    report, _ = normalize_quantity("81,415.38", None, "million", "₹")
    assert deck == pytest.approx(8.142e10)
    assert report == pytest.approx(8.141538e10)
    assert values_agree(deck, report), "rounding must not read as disagreement"
    assert relative_gap(deck, report) < 0.0001


def test_standalone_and_consolidated_do_not_agree_numerically():
    """Case 2: the numbers really do differ — only context explains it.

    The reconciliation must come from the `basis` key, never from a sloppy
    tolerance that pretends these are the same number.
    """
    standalone, _ = normalize_quantity("74,540.82", None, "million", "₹")
    consolidated, _ = normalize_quantity("81,415.38", None, "million", "₹")
    assert not values_agree(standalone, consolidated)
    assert relative_gap(standalone, consolidated) > 0.08


def test_gdp_vintage_gap_is_small_but_real():
    """Case 3: 6.4% vs 6.5% must NOT be smoothed away by tolerance."""
    survey, unit_a = normalize_quantity("6.4", None, None, "percent")
    rbi, unit_b = normalize_quantity("6.5", None, None, "per cent")
    assert (unit_a, unit_b) == ("percent", "percent")
    assert not values_agree(survey, rbi)


@pytest.mark.parametrize(
    "raw,magnitude,unit,value,canon_unit",
    [
        ("8,142", "crore", "₹", 8.142e10, "INR"),
        ("500", "million", "$", 5.0e8, "USD"),
        ("2.5", "lakh", "Rs.", 2.5e5, "INR"),
        ("6.5", None, "per cent", 6.5, "percent"),
        ("50", None, "bps", 0.5, "percent"),  # 50 bps == 0.5%
        ("122", None, None, 122.0, "count"),  # gateways
        ("33,000", None, None, 33000.0, "count"),
    ],
)
def test_normalize_quantity_table(raw, magnitude, unit, value, canon_unit):
    got_value, got_unit = normalize_quantity(raw, None, magnitude, unit)
    assert got_value == pytest.approx(value)
    assert got_unit == canon_unit


def test_gateway_arithmetic_reconciliation():
    """Case 2's provable half: 82 (ex-Spoton) + 40 (Spoton) == 122 (total)."""
    ex_spoton, _ = normalize_quantity("82", None, None, None)
    spoton, _ = normalize_quantity("40", None, None, None)
    total, _ = normalize_quantity("122", None, None, None)
    assert values_agree(ex_spoton + spoton, total)


def test_tolerance_does_not_swallow_real_disagreement():
    a, _ = normalize_quantity("6.0", None, None, "percent")
    b, _ = normalize_quantity("7.8", None, None, "percent")
    assert not values_agree(a, b)


# --------------------------------------------------------------------------
# Unit phrases as they actually arrive from a model
# --------------------------------------------------------------------------

from app.normalize.units import infer_unit_from_text, parse_unit_phrase  # noqa: E402


@pytest.mark.parametrize(
    "phrase,unit,magnitude",
    [
        ("INR", "INR", None),
        ("Indian Rupees in million", "INR", "million"),   # magnitude hidden in the unit
        ("₹ Cr", "INR", "cr"),
        ("Rs.", "INR", None),
        ("million", None, "million"),                      # a magnitude, not a unit
        ("crore", None, "crore"),
        ("US$ million", "USD", "million"),
        ("per cent", "percent", None),
        ("%", "percent", None),
        ("bps", "bps", None),
        ("I", None, None),                                 # model noise, not a unit
        ("days", "days", None),
        ("", None, None),
        (None, None, None),
    ],
)
def test_parse_unit_phrase(phrase, unit, magnitude):
    assert parse_unit_phrase(phrase) == (unit, magnitude)


def test_embedded_magnitude_is_applied():
    """'Indian Rupees in million' must scale even with no magnitude field."""
    value, unit = normalize_quantity("81,415.38", None, None, "Indian Rupees in million")
    assert value == pytest.approx(8.141538e10)
    assert unit == "INR"


def test_currency_spellings_unify():
    """The bug this fixes: one currency fragmented across five spellings.

    Each of these must land on the same canonical value, or facts that should
    corroborate never get compared at all.
    """
    variants = [
        ("8,142", "crore", "INR"),
        ("8,142", "crore", "₹"),
        ("8,142", "crore", "Rs."),
        ("8,142", None, "₹ Cr"),
        ("81,415.38", None, "Indian Rupees in million"),
    ]
    results = {normalize_quantity(r, None, m, u) for r, m, u in variants}
    units = {u for _, u in results}
    assert units == {"INR"}
    values = sorted(v for v, _ in results)
    assert values_agree(values[0], values[-1])


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Revenue stood at ₹8,142 Cr for FY24", "INR"),
        ("Rs. 127 Cr of EBITDA", "INR"),
        ("growth of 6.5 per cent", "percent"),
        ("a spread of 50 bps", "bps"),
        ("US$ 500 million raised", "USD"),
        ("122 gateways across India", None),
    ],
)
def test_infer_unit_from_evidence_text(text, expected):
    assert infer_unit_from_text(text) == expected
