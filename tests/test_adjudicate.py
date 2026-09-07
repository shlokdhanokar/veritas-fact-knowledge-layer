"""Phase 4 tests: the four required cases, as executable specifications.

These use facts shaped exactly like the ones the extractor produces from the
starter documents, so they pin the reasoning rather than the plumbing. No API
key needed — the rule engine is deterministic, which is the point.
"""

import pytest

from app.normalize.units import normalize_quantity
from app.reasoning.adjudicate import (
    aggregation_signature,
    metric_delta,
    identity_relation,
    adjudicate,
    compare_context,
    periods_equivalent,
)
from app.schema import Evidence, Fact, Quantity


def make_fact(
    subject="Delhivery Limited",
    metric="Revenue from Operations",
    raw="81,415.38",
    magnitude="million",
    unit="INR",
    context=None,
    doc="doc-a",
    page=1,
    valid_from=None,
    valid_to=None,
    value_text=None,
    confidence=1.0,
) -> Fact:
    quantity = None
    if raw is not None:
        cv, cu = normalize_quantity(raw, None, magnitude, unit)
        quantity = Quantity(
            raw=raw, magnitude=magnitude, unit=unit, canonical_value=cv, canonical_unit=cu
        )
    f = Fact(
        doc_id=doc,
        subject=subject,
        metric=metric,
        quantity=quantity,
        value_text=value_text,
        context=context or {},
        valid_from=valid_from,
        valid_to=valid_to,
        confidence=confidence,
        evidence=Evidence(
            doc_id=doc, filename=f"{doc}.pdf", page=page, start=0, end=10, quote="quote"
        ),
    )
    f.fact_id = f.compute_id()
    return f


# --------------------------------------------------------------------------
# Case 1 — corroboration across documents, expressed differently
# --------------------------------------------------------------------------


def test_case1_crore_and_million_corroborate():
    deck = make_fact(raw="8,142", magnitude="crore", unit="INR", doc="q4-deck",
                     context={"period": "FY24", "basis": "consolidated"})
    report = make_fact(raw="81,415.38", magnitude="million", unit="INR", doc="annual-report",
                       context={"period": "FY24", "basis": "consolidated"})

    rel = adjudicate(deck, report)
    assert rel.verdict == "CORROBORATES"
    assert rel.method == "arithmetic"
    assert "resolve to" in rel.reasoning


def test_case1_survives_fiscal_year_spelling_differences():
    """FY24 and FY2023-24 are the same year; the system must see that."""
    a = make_fact(context={"period": "FY24"}, doc="a")
    b = make_fact(context={"period": "FY2023-24"}, doc="b")
    assert adjudicate(a, b).verdict == "CORROBORATES"


# --------------------------------------------------------------------------
# Case 2 — apparent contradiction resolved by scope
# --------------------------------------------------------------------------


def test_case2_standalone_vs_consolidated_is_reconciled_not_contradiction():
    """The showpiece. These numbers really differ; only `basis` explains it."""
    standalone = make_fact(raw="74,540.82", context={"period": "FY24", "basis": "standalone"})
    consolidated = make_fact(raw="81,415.38", context={"period": "FY24", "basis": "consolidated"})

    rel = adjudicate(standalone, consolidated)
    assert rel.verdict == "RECONCILED", "a naive system reports this as a contradiction"
    assert rel.differing_keys == ["basis"]
    assert "subsidiaries" in rel.reasoning


def test_case2_entity_scope_reconciles_gateway_counts():
    """82 gateways excluding a subsidiary vs 122 including it."""
    ex = make_fact(metric="gateways", raw="82", magnitude=None, unit=None,
                   context={"period": "as of December 31, 2021", "entity_scope": "excluding Spoton"})
    incl = make_fact(metric="gateways", raw="122", magnitude=None, unit=None,
                     context={"period": "as of December 31, 2021", "entity_scope": "including Spoton"})

    rel = adjudicate(ex, incl)
    assert rel.verdict == "RECONCILED"
    assert "entity_scope" in rel.differing_keys


# --------------------------------------------------------------------------
# Case 3 — apparent contradiction resolved by data vintage
# --------------------------------------------------------------------------


def test_case3_gdp_vintage_is_reconciled_and_names_the_later_figure():
    survey = make_fact(
        subject="India", metric="real GDP growth", raw="6.4", magnitude=None, unit="percent",
        doc="economic-survey",
        context={"period": "FY25", "vintage": "first advance estimate", "measure": "real"},
    )
    rbi = make_fact(
        subject="India", metric="real GDP growth", raw="6.5", magnitude=None, unit="percent",
        doc="rbi-annual-report",
        context={"period": "FY25", "vintage": "provisional estimate", "measure": "real"},
    )

    rel = adjudicate(survey, rbi)
    assert rel.verdict == "RECONCILED"
    assert "vintage" in rel.differing_keys
    assert "supersedes" in rel.reasoning
    # It must tell the reader which figure to prefer.
    assert "provisional" in rel.reasoning


# --------------------------------------------------------------------------
# Case 4 — temporal state, not contradiction
# --------------------------------------------------------------------------


def test_case4_director_resignation_is_temporal_not_contradictory():
    serving = make_fact(
        subject="Sandeep Kumar Barasia", metric="board role", raw=None,
        value_text="Executive Director", doc="prospectus-2022",
        valid_from="2019-01-01", valid_to="2024-07-01",
    )
    ceased = make_fact(
        subject="Sandeep Kumar Barasia", metric="board role", raw=None,
        value_text="ceased to be a Director", doc="annual-report-fy24",
        valid_from="2024-07-01",
    )

    rel = adjudicate(serving, ceased)
    assert rel.verdict == "RECONCILED"
    assert rel.differing_keys == ["valid_time"]
    assert "supersedes" in rel.reasoning


# --------------------------------------------------------------------------
# The engine must still be able to say "this is a real contradiction"
# --------------------------------------------------------------------------


def test_genuine_contradiction_is_reported():
    """If nothing explains the gap, the system must not invent an excuse."""
    a = make_fact(raw="81,415.38", context={"period": "FY24", "basis": "consolidated"}, doc="a")
    b = make_fact(raw="90,000.00", context={"period": "FY24", "basis": "consolidated"}, doc="b")

    rel = adjudicate(a, b)
    assert rel.verdict == "CONTRADICTS"
    assert "Nothing in either document accounts for it" in rel.reasoning


def test_absent_context_key_is_not_treated_as_a_difference():
    """Silence must not count as disagreement.

    If a missing qualifier counted as "different", every contradiction could be
    explained away by one document simply being less explicit.
    """
    stated = make_fact(raw="81,415.38", context={"period": "FY24", "basis": "consolidated"})
    silent = make_fact(raw="90,000.00", context={"period": "FY24"}, doc="b")

    assert compare_context(stated, silent) == []


def test_absent_context_key_is_not_treated_as_agreement_either():
    """...but silence must not count as confirmation of a contradiction.

    The assignment asks for "a genuine or likely contradiction" and "sensible
    handling of ambiguity". An unstated qualifier is exactly that ambiguity, so
    the verdict is downgraded and the missing key is named as the thing to check.
    """
    stated = make_fact(raw="81,415.38", context={"period": "FY24", "basis": "consolidated"})
    silent = make_fact(raw="90,000.00", context={"period": "FY24"}, doc="b")

    rel = adjudicate(stated, silent)
    assert rel.verdict == "LIKELY_CONTRADICTS"
    assert rel.unstated_keys == ["basis"]
    assert "does not state" in rel.reasoning
    assert rel.confidence < 1.0


def test_genuine_contradiction_requires_both_sides_to_state_their_qualifiers():
    """A confirmed contradiction: nothing is left unsaid that could explain it."""
    a = make_fact(raw="81,415.38", context={"period": "FY24", "basis": "consolidated"}, doc="a")
    b = make_fact(raw="90,000.00", context={"period": "FY24", "basis": "consolidated"}, doc="b")

    rel = adjudicate(a, b)
    assert rel.verdict == "CONTRADICTS"
    assert rel.unstated_keys == []
    assert rel.confidence > adjudicate(a, make_fact(
        raw="90,000.00", context={"period": "FY24"}, doc="c")).confidence


# --------------------------------------------------------------------------
# Period equivalence — the quiet workhorse
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "a,b,same",
    [
        ("FY24", "FY2024", True),
        ("FY24", "FY2023-24", True),
        ("FY2023-24", "2023-24", True),
        ("FY24", "FY23", False),
        ("Q4 FY24", "FY24", False),  # a quarter is not its year
        ("Q4 FY24", "Q4 FY2023-24", True),
        ("Q3 FY24", "Q4 FY24", False),
        ("as of March 31, 2024", "FY24", False),  # point-in-time vs aggregate
        ("FY2024/25", "FY25", True),
    ],
)
def test_periods_equivalent(a, b, same):
    assert periods_equivalent(a, b) is same


def test_quarter_and_annual_do_not_corroborate():
    """Q4 revenue equal to annual revenue would be a coincidence, not agreement."""
    q4 = make_fact(raw="2,076", magnitude="crore", context={"period": "Q4 FY24"})
    fy = make_fact(raw="2,076", magnitude="crore", context={"period": "FY24"}, doc="b")
    rel = adjudicate(q4, fy)
    assert rel.verdict == "RECONCILED"
    assert "period" in rel.differing_keys


# --------------------------------------------------------------------------
# Metric identity — the guard that stopped the system crying wolf
# --------------------------------------------------------------------------


def test_different_line_items_are_not_contradictions():
    """Found by running on real data: these share five of six words.

    Before this guard the engine reported 89 contradictions on one earnings
    deck, nearly all of them pairs like this. Afterwards: 0.
    """
    operating = make_fact(metric="Net cash from / (used in) operating activities",
                          raw="(30)", magnitude="crore", context={"period": "FY23"})
    investing = make_fact(metric="Net cash from / (used in) investing activities",
                          raw="(3,411)", magnitude="crore", context={"period": "FY23"}, doc="b")

    rel = adjudicate(operating, investing)
    assert rel.verdict == "UNRELATED"
    assert "investing" in rel.reasoning or "operating" in rel.reasoning


def test_subtotal_is_not_a_contradiction_of_its_component():
    """`(B)` versus `(A+B)` — a component and the sum that contains it."""
    component = make_fact(metric="Cash equivalents at the end of the year(3) (B)",
                          raw="5,213", magnitude="crore", context={"period": "FY23"})
    total = make_fact(metric="Cash & cash equivalents at the end of the year (A+B)",
                      raw="5,508", magnitude="crore", context={"period": "FY23"}, doc="b")

    rel = adjudicate(component, total)
    assert rel.verdict != "CONTRADICTS"
    assert "aggregation-level" in metric_delta(component, total)


def test_adjusted_measure_is_distinct_from_unadjusted():
    plain = make_fact(metric="EBITDA margin", raw="1.6", magnitude=None, unit="percent",
                      context={"period": "FY24"})
    adjusted = make_fact(metric="Adjusted EBITDA margin", raw="0.9", magnitude=None,
                         unit="percent", context={"period": "FY24"}, doc="b")
    assert adjudicate(plain, adjusted).verdict != "CONTRADICTS"


def test_synonyms_do_not_block_a_real_match():
    """Case 1 depends on this: 'revenue from operations' == 'revenue from services'."""
    ops = make_fact(metric="Revenue from Operations", raw="8,142", magnitude="crore",
                    context={"period": "FY24", "basis": "consolidated"})
    svc = make_fact(metric="Revenue from services", raw="81,415.38", magnitude="million",
                    context={"period": "FY24", "basis": "consolidated"}, doc="b")

    rel = adjudicate(ops, svc)
    assert rel.verdict == "CORROBORATES"
    assert "stronger evidence" in rel.reasoning


def test_aggregation_signature_reads_component_labels():
    assert aggregation_signature("Cash equivalents (B)") == (False, "B")
    assert aggregation_signature("Cash & cash equivalents (A+B)") == (True, "AB")
    assert aggregation_signature("Revenue from operations") == (False, "")


# --------------------------------------------------------------------------
# Identity is three-way, not two-way
#
# Found by testing on documents the system had never seen. Collapsing the
# middle case into either neighbour was measurably wrong in both directions.
# --------------------------------------------------------------------------


def test_identical_names_permit_a_confirmed_contradiction():
    a = make_fact(metric="formal employment rate", raw="11", magnitude=None,
                  unit="percent", context={"geography": "India"})
    b = make_fact(metric="formal employment rate", raw="11.2", magnitude=None,
                  unit="percent", context={"geography": "India"}, doc="b")
    assert identity_relation(a, b)[0] == "same"
    assert adjudicate(a, b).verdict == "CONTRADICTS"


def test_one_name_containing_another_is_only_a_suspicion():
    """A total and its component share every word of the shorter name.

    Asserting CONTRADICTS here produced dozens of false positives on the real
    corpus (EBITDA vs Service EBITDA, ESOPs ungranted vs Time-based ESOPs
    ungranted). It must be flagged, never asserted.
    """
    total = make_fact(metric="ESOPs ungranted", raw="40424975", magnitude=None,
                      unit=None, context={"period": "FY24"})
    part = make_fact(metric="Time-based ESOPs ungranted", raw="25364975",
                     magnitude=None, unit=None, context={"period": "FY24"}, doc="b")

    kind, _ = identity_relation(total, part)
    assert kind == "specialisation"
    rel = adjudicate(total, part)
    assert rel.verdict == "LIKELY_CONTRADICTS"
    assert rel.unstated_keys == ["metric_identity"]
    assert "component and its total" in rel.reasoning


def test_specialisation_still_surfaces_a_real_disagreement():
    """The recall side: this pair was silently dropped before the three-way split.

    Two documents counting the same fleet, worded differently, disagreeing.
    """
    a = make_fact(subject="Northwind plc", metric="number of turbines", raw="84",
                  magnitude=None, unit=None, context={"period": "as of December 31, 2024"})
    b = make_fact(subject="Northwind plc", metric="turbines operated", raw="91",
                  magnitude=None, unit=None,
                  context={"period": "as of December 31, 2024"}, doc="b")

    assert adjudicate(a, b).verdict == "LIKELY_CONTRADICTS"


def test_negation_makes_names_different_not_nested():
    """'current' and 'non-current' are opposites, not a general and a special case."""
    cur = make_fact(metric="Current lease liabilities", raw="2,001",
                    magnitude="million", context={"period": "FY24"})
    non = make_fact(metric="Non-current lease liabilities", raw="8,436",
                    magnitude="million", context={"period": "FY24"}, doc="b")

    assert identity_relation(cur, non)[0] == "different"
    assert adjudicate(cur, non).verdict == "UNRELATED"


def test_two_sided_difference_is_different():
    a = make_fact(subject="Interest at amortised cost to banks", metric="Finance costs",
                  raw="130.38", magnitude="million", context={"period": "FY24"})
    b = make_fact(subject="Interest at amortised cost to others", metric="Finance costs",
                  raw="1.49", magnitude="million", context={"period": "FY24"}, doc="b")
    assert identity_relation(a, b)[0] == "different"
    assert adjudicate(a, b).verdict == "UNRELATED"


def test_spelled_out_fiscal_years_unify():
    """An unseen document wrote 'financial year 2024' where its sibling wrote FY2024."""
    a = make_fact(metric="generation", raw="1,240", magnitude=None, unit="GWh",
                  context={"period": "FY2024"})
    b = make_fact(metric="generation", raw="1,240", magnitude=None, unit="GWh",
                  context={"period": "financial year 2024"}, doc="b")
    assert adjudicate(a, b).verdict == "CORROBORATES"


def test_open_ended_tenures_are_temporal_not_contradictory():
    """Both intervals open, on opposite sides: a beginning and an end of one tenure."""
    holds = make_fact(subject="Elena Voss", metric="role", raw=None,
                      value_text="Chief Operating Officer", valid_from="2019-06-01")
    left = make_fact(subject="Elena Voss", metric="Chief Operating Officer role",
                     raw=None, value_text="stepped down", valid_to="2025-03-01", doc="b")

    rel = adjudicate(holds, left)
    assert rel.verdict == "RECONCILED"
    assert rel.differing_keys == ["valid_time"]
