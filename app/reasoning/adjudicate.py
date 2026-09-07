"""Deciding what a pair of facts means to each other.

This is the centre of the system, and it is deliberately small.

    Do the context keys match?
      no  -> RECONCILED   (name the key that differs; the gap is explained)
      yes -> do the values agree?
               yes -> CORROBORATES
               no  -> CONTRADICTS

Everything else here is in service of making that question answerable: comparing
periods properly, spotting when two facts describe different points in time, and
checking whether a difference is arithmetically explained rather than merely
plausible.

Rules run first and settle the clear cases deterministically. The LLM is used
only where rules genuinely cannot decide — whether two differently-worded
metrics mean the same thing — and its verdict is recorded as `method="llm"` so a
reader can always tell machine judgement from arithmetic.
"""

from __future__ import annotations

import re

from app.normalize.units import relative_gap, values_agree
from app.reasoning.pairing import tokenize
from app.schema import Fact, Relation

# Wording that varies freely between documents without changing what is measured.
# Anything OUTSIDE this set that appears in one metric name but not the other is
# treated as meaningful — see `metric_delta`.
METRIC_SYNONYMS: tuple[frozenset[str], ...] = (
    frozenset({"revenue", "revenues", "income", "turnover"}),
    # NOTE: "operations" and "services" are deliberately NOT synonyms. Revenue
    # from services excludes traded goods, so the wording marks a real (if often
    # immaterial) distinction. Keeping them separate means such a pair is judged
    # on its values: equal values corroborate across phrasings, unequal values
    # are reported as different metrics rather than as a contradiction.
    frozenset({"operations", "operation", "operating"}),
    frozenset({"profit", "profits", "earnings"}),
    frozenset({"loss", "losses"}),
    frozenset({"employees", "employee", "headcount", "people"}),
    frozenset({"year", "yr", "annual", "annualised", "annualized"}),
    frozenset({"growth", "increase", "rise"}),
)


def _synonym_class(token: str) -> frozenset[str] | None:
    for group in METRIC_SYNONYMS:
        if token in group:
            return group
    return None


_AGGREGATION = re.compile(r"\+|\btotal\b|\bsum\b|\baggregate\b|\bcombined\b|\boverall\b", re.I)
_COMPONENT_LABEL = re.compile(r"\(\s*([A-Z](?:\s*\+\s*[A-Z])*)\s*\)")


def aggregation_signature(metric: str) -> tuple[bool, str]:
    """Whether a metric name denotes a total, and which components it sums.

    Financial tables label rows as components and subtotals: "Cash equivalents (B)"
    against "Cash & cash equivalents (A+B)". Those parenthesised labels are the
    only thing distinguishing a part from the whole, and a word tokenizer throws
    them away as single-character noise — which made the system report a real
    subtotal as a contradiction of its own component.

    Reading the label directly is both more accurate and fully general: any
    document using this convention benefits, and documents that don't are
    unaffected.
    """
    label = _COMPONENT_LABEL.search(metric or "")
    components = "".join(sorted(re.findall(r"[A-Z]", label.group(1)))) if label else ""
    return bool(_AGGREGATION.search(metric or "")), components


def metric_delta(a: Fact, b: Fact) -> set[str]:
    """Words that distinguish two metric names, ignoring pure synonyms.

    This exists because token overlap alone is dangerously permissive. "Net cash
    from operating activities" and "net cash from investing activities" share
    five of six words, yet they are different line items and comparing their
    values is meaningless. The residual word — "operating" vs "investing" — is
    exactly the signal that they are not the same metric.

    Returns the meaningful residual. Empty means "these name the same thing".
    """
    # A subtotal is not its own component, even when the words match exactly.
    agg_a, comp_a = aggregation_signature(a.metric)
    agg_b, comp_b = aggregation_signature(b.metric)
    if comp_a != comp_b or agg_a != agg_b:
        return {"aggregation-level"}

    ta, tb = tokenize(a.metric), tokenize(b.metric)
    residual = ta.symmetric_difference(tb)
    if not residual:
        return set()
    return _meaningful_residual(residual, ta, tb)


# Words that change *what is being measured* rather than merely how it is
# phrased. If one side carries one of these and the other does not, the two are
# different quantities even though one phrase contains the other: "EBITDA margin"
# and "Adjusted EBITDA margin" are not the same number.
MEASURE_QUALIFIERS = frozenset({
    "adjusted", "unadjusted", "underlying", "normalised", "normalized",
    "restated", "proforma", "forma", "excluding", "including", "ex",
    "net", "gross", "diluted", "basic", "organic", "comparable",
    "nominal", "real", "constant", "current", "average", "median",
    "annualised", "annualized", "estimated", "projected", "budgeted",
    "standalone", "consolidated", "reported", "recurring", "continuing",
    # Negation and contrast. These invert meaning outright, so a phrase that
    # merely *contains* another is not a more specific version of it:
    # "current lease liabilities" vs "NON-current lease liabilities" are
    # opposites, and "profit considered in consolidation" vs "NOT considered"
    # is its complement. Treating these as incidental produced the largest
    # class of false contradictions when the guard was first loosened.
    "non", "not", "un", "without", "less", "pre", "post", "deferred",
    "comprehensive", "share", "other", "total", "sub",
})


def _subset_relation(ta: set[str], tb: set[str]) -> bool:
    """Is one phrase simply a more specific version of the other?

    The distinction that matters for identity:

      "number of turbines" vs "turbines operated"  -> one adds a word: SAME thing
      "interest ... to banks" vs "... to others"   -> each has a word the other
                                                       lacks: DIFFERENT things

    A subset means one document was more specific about the same quantity. A
    two-sided difference means the documents are naming different quantities.
    Testing on an unseen document showed the earlier rule - any residual at all
    means different - was suppressing a genuine contradiction between two
    phrasings of the same count.
    """
    if not ta or not tb or not (ta <= tb or tb <= ta):
        return False
    # The extra words decide. Incidental phrasing ("number of", "operated") still
    # names the same quantity; a measure qualifier ("adjusted", "net") does not.
    extra = ta.symmetric_difference(tb)
    return not (extra & MEASURE_QUALIFIERS)


def _meaningful_residual(residual: set[str], ta: set[str], tb: set[str]) -> set[str]:
    meaningful = set()
    for token in residual:
        group = _synonym_class(token)
        # A token is explained away only if the other side carries a synonym of it.
        other = tb if token in ta else ta
        if group and (group & other):
            continue
        meaningful.add(token)
    return meaningful


def subject_delta(a: Fact, b: Fact) -> set[str]:
    """Words that distinguish two subjects.

    The same trap as metric names, one field over. A finance-costs table breaks
    out "Interest at amortised cost to banks" and "...to others"; the metric is
    identical and only the subject separates them, so comparing their values
    produces a confident and completely spurious contradiction.

    Subjects that reduce to nothing (a document saying "the Company" about
    itself) are not used as a discriminator — absence of a name is not evidence
    of a different entity.
    """
    ta, tb = tokenize(a.subject), tokenize(b.subject)
    if not ta or not tb:
        return set()
    return _meaningful_residual(ta.symmetric_difference(tb), ta, tb)


def identity_delta(a: Fact, b: Fact) -> set[str]:
    """Everything that says these two facts are not about the same thing."""
    return metric_delta(a, b) | subject_delta(a, b)


def identity_relation(a: Fact, b: Fact) -> tuple[str, set[str]]:
    """Classify how two facts' names relate. Returns (kind, residual words).

    Three outcomes, and the middle one is the point:

    ``same``
        The names agree modulo synonyms. A value disagreement here is a real
        contradiction, because there is nothing left to explain it.

    ``specialisation``
        One name contains the other plus extra words ("EBITDA" inside "Service
        EBITDA", "ESOPs ungranted" inside "Time-based ESOPs ungranted"). These
        are *usually* different quantities — a segment, a component, a variant —
        but sometimes just a terser phrasing of one thing ("number of turbines"
        vs "turbines operated"). We cannot tell from the words alone.

    ``different``
        Each name carries something the other lacks ("... to banks" vs "... to
        others"), or a qualifier inverts the meaning ("current" vs
        "non-current"). Different quantities; no comparison is meaningful.

    Collapsing ``specialisation`` into either neighbour was wrong in both
    directions when measured. Calling it ``same`` produced dozens of confident
    contradictions between a total and its own components. Calling it
    ``different`` silently suppressed a genuine 84-vs-91 disagreement found
    while testing on an unseen document. So it gets its own verdict: flagged for
    a human, never asserted.
    """
    residual = identity_delta(a, b)
    if not residual:
        return "same", set()

    ma, mb = tokenize(a.metric), tokenize(b.metric)
    sa, sb = tokenize(a.subject), tokenize(b.subject)
    metric_nested = bool(ma and mb and (ma <= mb or mb <= ma))
    subject_nested = not sa or not sb or (sa <= sb or sb <= sa)

    if metric_nested and subject_nested and not (residual & MEASURE_QUALIFIERS):
        return "specialisation", residual
    return "different", residual

# Context keys whose disagreement fully explains a difference in value. These
# are the qualifiers that change *what is being measured*, not merely how it is
# described.
EXPLANATORY_KEYS = ("basis", "entity_scope", "period", "vintage", "geography", "segment", "measure")

# Vintage ordering: later stages supersede earlier ones for the same period.
VINTAGE_RANK = {
    "projection": 0, "forecast": 0, "target": 0, "estimate": 1,
    "advance estimate": 1, "first advance estimate": 1, "second advance estimate": 2,
    "provisional": 3, "provisional estimate": 3, "revised": 4, "revised estimate": 4,
    "actual": 5, "final": 5,
}


def _norm(value: str | None) -> str:
    return (value or "").strip().lower()


def _vintage_rank(value: str | None) -> int | None:
    v = _norm(value)
    if not v:
        return None
    for name, rank in sorted(VINTAGE_RANK.items(), key=lambda kv: -len(kv[0])):
        if name in v:
            return rank
    return None


# Fiscal years as documents actually spell them. Beyond "FY24" and "FY2023-24"
# this covers "financial year 2024", "fiscal year 2024" and "year ended March 31,
# 2024". Found by testing on an unseen document that wrote "financial year 2024"
# where its sibling wrote "FY2024": without this the same period reads as two
# different ones, and a perfect corroboration is downgraded to a meaningless
# reconciliation that admits in its own reasoning that the values coincide.
_FY = re.compile(
    r"\b(?:fy|financial\s+year|fiscal\s+year|fiscal|year\s+end(?:ed|ing))\s*"
    r"(?:[a-z]+\s+\d{1,2},?\s*)?"  # optional "March 31," in "year ended March 31, 2024"
    r"(\d{2,4})(?:\s*[-/]\s*(\d{2,4}))?\b",
    re.I,
)
_QUARTER = re.compile(r"\bq([1-4])\b", re.I)
_YEAR_RANGE = re.compile(r"\b(20\d{2})\s*[-/]\s*(\d{2,4})\b")
_AS_OF = re.compile(r"\b(?:as\s+(?:of|at|on))\b", re.I)


def _fiscal_year(text: str) -> int | None:
    """Reduce many spellings of a fiscal year to its ending year.

    FY24, FY2024, FY2023-24, 2023-24 and FY2023/24 all denote the year ending
    2024. Without this, the same period looks like several different ones and
    every genuine comparison is missed.
    """
    t = _norm(text)
    m = _FY.search(t)
    if m:
        first, second = m.group(1), m.group(2)
        end = second or first
        end_i = int(end)
        if end_i < 100:  # two-digit year
            end_i += 2000
        # "FY2023-24" -> ends 2024; "FY24" -> ends 2024
        return end_i
    m = _YEAR_RANGE.search(t)
    if m:
        start, end = int(m.group(1)), m.group(2)
        end_i = int(end)
        if end_i < 100:
            end_i = start - (start % 100) + end_i
        return end_i
    return None


def periods_equivalent(a: str | None, b: str | None) -> bool:
    """Do two period strings denote the same span?

    Conservative: only returns True when we can positively identify the same
    fiscal year AND the same quarter (or neither being a quarter). Anything we
    cannot parse falls back to a plain string comparison, so an unknown format
    is treated as different rather than silently merged.
    """
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return False
    if na == nb:
        return True

    # A point-in-time statement ("as of March 31, 2024") is not the same kind of
    # claim as a period aggregate ("FY24"), even when the dates line up.
    if bool(_AS_OF.search(na)) != bool(_AS_OF.search(nb)):
        return False

    qa, qb = _QUARTER.search(na), _QUARTER.search(nb)
    if bool(qa) != bool(qb):
        return False  # a quarter is not its fiscal year
    if qa and qb and qa.group(1) != qb.group(1):
        return False

    fa, fb = _fiscal_year(na), _fiscal_year(nb)
    if fa is not None and fb is not None:
        return fa == fb
    return False


def asymmetric_context(a: Fact, b: Fact) -> list[str]:
    """Explanatory keys that one fact states and the other leaves silent.

    This is the uncertainty the system must not paper over in either direction.

    Treating silence as *agreement* would let us declare a hard contradiction
    between a figure explicitly labelled "standalone" and one whose basis the
    document never stated — even though the unlabelled figure may well be
    standalone too, or consolidated, and we simply cannot tell.

    Treating silence as *difference* would be worse: every genuine contradiction
    could then be waved away because one side happened to be less explicit, and
    the system could never report a disagreement at all.

    So silence is neither. It is recorded as a named hypothesis, downgrades the
    verdict from CONTRADICTS to LIKELY_CONTRADICTS, and is surfaced to the
    reader as the specific thing to go and check.
    """
    return [
        key
        for key in EXPLANATORY_KEYS
        if bool(a.context.get(key)) != bool(b.context.get(key))
    ]


def compare_context(a: Fact, b: Fact) -> list[str]:
    """Context keys that are present on both facts but disagree.

    A key absent from one side is NOT a disagreement — it is unknown. Treating
    absence as difference would let us "explain away" every real contradiction
    just because one document was less explicit, which would make the system
    useless. This asymmetry is the most important judgement call in the file.
    """
    differing = []
    for key in EXPLANATORY_KEYS:
        left, right = a.context.get(key), b.context.get(key)
        if not left or not right:
            continue
        if key == "period":
            if not periods_equivalent(left, right):
                differing.append(key)
        elif _norm(left) != _norm(right):
            differing.append(key)
    return differing


def _temporal_conflict(a: Fact, b: Fact) -> bool:
    """Do these facts describe different points on the same timeline?

    Two shapes count, and the second was missed until an unseen document
    exposed it:

    1. Non-overlapping intervals — one state ends before the other begins.

    2. Different endpoints for the same subject. A prospectus says a person
       holds an office "since June 2019" (open-ended); a later filing says they
       stepped down "with effect from March 1, 2025" (open on the other side).
       Neither interval is closed, so no ordering test fires, yet these are
       plainly a beginning and an end of one tenure rather than a disagreement.

    Only the endpoints are compared, so two facts asserting the identical
    interval are left alone for the value comparison to judge.
    """
    if (a.valid_to and b.valid_from and a.valid_to <= b.valid_from) or (
        b.valid_to and a.valid_from and b.valid_to <= a.valid_from
    ):
        return True

    both_dated = (a.valid_from or a.valid_to) and (b.valid_from or b.valid_to)
    return bool(both_dated and (a.valid_from, a.valid_to) != (b.valid_from, b.valid_to))


def _explain(key: str, a: Fact, b: Fact) -> str:
    left, right = a.context.get(key, "?"), b.context.get(key, "?")
    templates = {
        "basis": (
            f"Not a contradiction: these report different accounting bases "
            f"({left!r} vs {right!r}). A consolidated figure includes subsidiaries "
            f"that a standalone figure excludes, so the two are expected to differ."
        ),
        "entity_scope": (
            f"Not a contradiction: the figures cover different scopes "
            f"({left!r} vs {right!r}), so they count different things."
        ),
        "period": (
            f"Not a contradiction: these cover different periods ({left!r} vs {right!r})."
        ),
        "vintage": (
            f"Not a contradiction: these are different vintages of the same estimate "
            f"({left!r} vs {right!r}). Statistical agencies revise figures as more "
            f"data arrives, so the later vintage supersedes the earlier one."
        ),
        "geography": f"Different geographies ({left!r} vs {right!r}).",
        "segment": f"Different segments ({left!r} vs {right!r}).",
        "measure": (
            f"Not a contradiction: different measurement bases ({left!r} vs {right!r}), "
            f"for example real versus nominal."
        ),
    }
    return templates.get(key, f"Differing {key}: {left!r} vs {right!r}.")


def adjudicate(a: Fact, b: Fact, similarity: float = 1.0) -> Relation:
    """Apply the rule. Returns a Relation with its reasoning."""
    differing = compare_context(a, b)

    qa, qb = a.quantity, b.quantity
    have_values = bool(qa and qb and qa.is_normalized and qb.is_normalized)
    agree = (
        values_agree(qa.canonical_value, qb.canonical_value) if have_values else False
    )
    gap = relative_gap(qa.canonical_value, qb.canonical_value) if have_values else None

    base_conf = min(a.confidence, b.confidence) * similarity

    # --- Temporal supersession -------------------------------------------------
    # A state that ended before another began is a timeline, not a conflict.
    if _temporal_conflict(a, b):
        return Relation(
            left_id=a.fact_id, right_id=b.fact_id,
            verdict="RECONCILED", differing_keys=["valid_time"],
            reasoning=(
                "Not a contradiction: the two facts hold over different intervals "
                f"({a.valid_from or '?'}..{a.valid_to or 'open'} vs "
                f"{b.valid_from or '?'}..{b.valid_to or 'open'}). The later state "
                "supersedes the earlier one rather than conflicting with it."
            ),
            method="rule", confidence=round(base_conf, 3),
        )

    # --- Context explains the difference --------------------------------------
    if differing:
        note = " ".join(_explain(k, a, b) for k in differing)
        if have_values and agree:
            # Same number despite different qualifiers: weak evidence, worth flagging.
            note += (
                " Note: the values coincide anyway, so this pairing carries little "
                "information either way."
            )
        # A vintage difference where the later figure supersedes the earlier is
        # worth stating explicitly, since it is the resolution the reader wants.
        if "vintage" in differing:
            ra, rb = _vintage_rank(a.context.get("vintage")), _vintage_rank(b.context.get("vintage"))
            if ra is not None and rb is not None and ra != rb:
                later = a if ra > rb else b
                note += (
                    f" The more mature figure is {later.quantity.raw if later.quantity else later.value_text!r}"
                    f" ({later.context.get('vintage')}, from {later.evidence.filename})."
                )
        return Relation(
            left_id=a.fact_id, right_id=b.fact_id,
            verdict="RECONCILED", differing_keys=differing,
            reasoning=note, method="rule",
            confidence=round(base_conf * 0.95, 3),
        )

    # --- Are these even the same metric? --------------------------------------
    # Checked after context but before any contradiction claim. Two metrics whose
    # names differ meaningfully are different things, and different things are
    # allowed to hold different values. Claiming a contradiction here would be
    # the system's most common and most embarrassing error: on real extracted
    # facts this single check removed the large majority of false contradictions
    # (component-vs-total, operating-vs-investing, gross-vs-adjusted).
    kind, residual = identity_relation(a, b)
    if kind != "same":
        words = ", ".join(sorted(residual))
        if have_values and agree:
            return Relation(
                left_id=a.fact_id, right_id=b.fact_id,
                verdict="CORROBORATES", differing_keys=[],
                reasoning=(
                    f"Different wording ({a.subject!r}/{a.metric!r} vs "
                    f"{b.subject!r}/{b.metric!r}, differing on: {words}) but the values "
                    f"resolve to the same figure {qa.canonical_value:,.6g} "
                    f"{qa.canonical_unit} (difference {gap:.3%}). Agreement across "
                    f"independent phrasings is stronger evidence than agreement in "
                    f"identical wording."
                ),
                method="arithmetic", confidence=round(base_conf * 0.9, 3),
            )

        if kind == "specialisation" and have_values:
            # One name contains the other. Possibly a segment or component of the
            # other, possibly just terser wording for the same thing. Flag it;
            # never assert it.
            return Relation(
                left_id=a.fact_id, right_id=b.fact_id,
                verdict="LIKELY_CONTRADICTS", differing_keys=[],
                unstated_keys=["metric_identity"],
                reasoning=(
                    f"The values differ: {qa.canonical_value:,.6g} vs "
                    f"{qb.canonical_value:,.6g} {qa.canonical_unit} (a {gap:.1%} gap), "
                    f"and every qualifier stated on both sides agrees. But one name "
                    f"contains the other ({a.metric!r} vs {b.metric!r}, differing on: "
                    f"{words}), so these may be a component and its total, or a segment "
                    f"and the whole, rather than two readings of one quantity. Confirm "
                    f"they measure the same thing before treating this as an error."
                ),
                method="arithmetic", confidence=round(base_conf * 0.45, 3),
            )

        return Relation(
            left_id=a.fact_id, right_id=b.fact_id,
            verdict="UNRELATED", differing_keys=["metric"],
            reasoning=(
                f"These describe different things ({a.subject!r}/{a.metric!r} vs "
                f"{b.subject!r}/{b.metric!r}), differing on: {words}. Different "
                f"quantities may hold different values, so no contradiction is claimed."
            ),
            method="rule", confidence=round(base_conf * 0.5, 3),
        )

    # --- Context matches: the values must answer for themselves ---------------
    if not have_values:
        # Non-numeric facts under matching context: we cannot decide by rule.
        return Relation(
            left_id=a.fact_id, right_id=b.fact_id,
            verdict="UNRELATED", reasoning="No comparable numeric values.",
            method="rule", confidence=0.2,
        )

    if agree:
        return Relation(
            left_id=a.fact_id, right_id=b.fact_id,
            verdict="CORROBORATES", differing_keys=[],
            reasoning=(
                f"Context keys match and the values agree once normalized: "
                f"{qa.raw} {qa.magnitude or ''}{qa.unit or ''} and "
                f"{qb.raw} {qb.magnitude or ''}{qb.unit or ''} both resolve to "
                f"{qa.canonical_value:,.6g} {qa.canonical_unit} "
                f"(difference {gap:.3%}, within rounding tolerance)."
            ).replace("  ", " "),
            method="arithmetic", confidence=round(base_conf, 3),
        )

    # Values disagree. Before calling it a contradiction, ask whether one side
    # left a relevant qualifier unstated — in which case the gap may be explained
    # by something the document did not say, and we must not overclaim.
    unstated = asymmetric_context(a, b)
    if unstated:
        hypotheses = []
        for key in unstated:
            stated_side, silent_side = (a, b) if a.context.get(key) else (b, a)
            hypotheses.append(
                f"{silent_side.evidence.filename} (p{silent_side.evidence.page}) does not state "
                f"{key!r}, while the other reports {key}="
                f"{stated_side.context.get(key)!r}"
            )
        return Relation(
            left_id=a.fact_id, right_id=b.fact_id,
            verdict="LIKELY_CONTRADICTS", differing_keys=[], unstated_keys=unstated,
            reasoning=(
                f"The values differ: {qa.canonical_value:,.6g} vs "
                f"{qb.canonical_value:,.6g} {qa.canonical_unit} (a {gap:.1%} gap), and every "
                f"qualifier stated on both sides agrees. But this is not confirmed as a "
                f"genuine contradiction, because "
                + "; ".join(hypotheses)
                + f". If the unstated {'qualifiers differ' if len(unstated) > 1 else 'qualifier differs'}, "
                f"the gap would be explained. Check this before treating it as an error."
            ),
            method="arithmetic",
            # Each unresolved hypothesis makes the claim weaker.
            confidence=round(base_conf * (0.6 ** len(unstated)), 3),
        )

    return Relation(
        left_id=a.fact_id, right_id=b.fact_id,
        verdict="CONTRADICTS", differing_keys=[],
        reasoning=(
            f"Every context key that could explain a difference is stated on both sides "
            f"and agrees, yet the values differ: {qa.canonical_value:,.6g} vs "
            f"{qb.canonical_value:,.6g} {qa.canonical_unit} (a {gap:.1%} gap). "
            f"Nothing in either document accounts for it."
        ),
        method="arithmetic", confidence=round(base_conf, 3),
    )
