"""Finding fact pairs that are worth comparing at all.

Comparing every fact to every other fact is O(n^2): 10,000 facts would mean 50
million comparisons, most of them between things that share no subject matter.
So we block first — group facts by what they are *about* — and only compare
within blocks.

Blocking keys are derived from the facts themselves (their own words, reduced to
content tokens), never from a fixed taxonomy. A new document introducing an
unfamiliar metric forms its own blocks automatically, which is what lets the
knowledge layer grow without schema changes.
"""

from __future__ import annotations

import re
from collections import defaultdict
from itertools import combinations

from app.schema import Fact

# Words that carry no discriminating meaning for what a metric is about.
_STOP = {
    "the", "a", "an", "of", "for", "in", "on", "at", "to", "and", "or", "from",
    "by", "with", "as", "is", "was", "were", "be", "been", "total", "net",
    "gross", "our", "its", "their", "this", "that", "per", "value", "amount",
    "number", "no", "limited", "ltd", "company", "group", "inc",
}

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> set[str]:
    """Content tokens of a phrase, lowercased and de-noised."""
    return {t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def blocking_keys(fact: Fact) -> set[str]:
    """Cheap keys that two comparable facts are likely to share.

    We use individual metric tokens rather than the whole phrase, because the
    same quantity gets named differently across documents ("revenue from
    operations" vs "revenue from services"). Sharing any content token is enough
    to become a candidate; the real filtering happens in `similarity`.
    """
    return tokenize(fact.metric) | tokenize(fact.subject)


def similarity(a: Fact, b: Fact) -> float:
    """How likely two facts describe the same thing.

    Subject and metric are weighted separately so that a strong metric match
    with a vague subject (common when a document omits the company name because
    it is talking about itself) still scores well.
    """
    metric = jaccard(tokenize(a.metric), tokenize(b.metric))
    subject = jaccard(tokenize(a.subject), tokenize(b.subject))
    # An empty or generic subject should not veto an otherwise strong match.
    if not tokenize(a.subject) or not tokenize(b.subject):
        return metric
    return 0.7 * metric + 0.3 * subject


def comparable_units(a: Fact, b: Fact) -> bool:
    """Two quantities are only comparable if they resolve to the same unit.

    Rupees and percentages are never the same fact, however similar the wording.
    """
    qa, qb = a.quantity, b.quantity
    if qa is None or qb is None:
        return a.quantity is None and b.quantity is None  # both non-numeric
    if not (qa.is_normalized and qb.is_normalized):
        return False
    return qa.canonical_unit == qb.canonical_unit


def candidate_pairs(
    facts: list[Fact],
    *,
    threshold: float = 0.5,
    cross_document_only: bool = False,
    max_pairs: int | None = None,
) -> list[tuple[Fact, Fact, float]]:
    """Return (left, right, similarity) for pairs worth adjudicating.

    `cross_document_only` is available but off by default: some of the most
    interesting reconciliations are *within* one document, such as a network
    statistic quoted once including a subsidiary and once excluding it.
    """
    index: dict[str, list[int]] = defaultdict(list)
    for i, fact in enumerate(facts):
        for key in blocking_keys(fact):
            index[key].append(i)

    seen: set[tuple[int, int]] = set()
    out: list[tuple[Fact, Fact, float]] = []

    for bucket in index.values():
        # A token shared by a huge number of facts ("revenue") is not selective;
        # the pairs still get scored, but we avoid exploding on giant buckets.
        if len(bucket) > 400:
            continue
        for i, j in combinations(sorted(bucket), 2):
            if (i, j) in seen:
                continue
            seen.add((i, j))

            a, b = facts[i], facts[j]
            if cross_document_only and a.doc_id == b.doc_id:
                continue
            if a.fact_id == b.fact_id:
                continue
            if not comparable_units(a, b):
                continue

            score = similarity(a, b)
            if score >= threshold:
                out.append((a, b, score))

    out.sort(key=lambda t: t[2], reverse=True)
    return out[:max_pairs] if max_pairs else out
