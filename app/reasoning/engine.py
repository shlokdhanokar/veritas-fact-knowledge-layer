"""Running the reasoning pass over a set of facts.

Kept separate from `adjudicate` so the rule itself stays readable and testable
in isolation. This module is only orchestration: pair, judge, rank, and keep the
relations worth showing a human.
"""

from __future__ import annotations

from collections import Counter

from app.reasoning.adjudicate import adjudicate
from app.reasoning.pairing import candidate_pairs
from app.schema import Fact, Relation

# Verdicts worth surfacing. UNRELATED pairs are the majority and carry no signal.
INTERESTING = ("CONTRADICTS", "LIKELY_CONTRADICTS", "RECONCILED", "CORROBORATES")


def relate(
    facts: list[Fact],
    *,
    threshold: float = 0.5,
    min_confidence: float = 0.0,
    cross_document_only: bool = False,
    max_pairs: int | None = 20000,
) -> list[Relation]:
    """Adjudicate every candidate pair and return the interesting relations."""
    pairs = candidate_pairs(
        facts,
        threshold=threshold,
        cross_document_only=cross_document_only,
        max_pairs=max_pairs,
    )

    relations: list[Relation] = []
    for a, b, score in pairs:
        rel = adjudicate(a, b, similarity=score)
        if rel.verdict in INTERESTING and rel.confidence >= min_confidence:
            relations.append(rel)

    # Contradictions first: they are what a reader most needs to see, and they
    # are the claim the system is most accountable for.
    order = {"CONTRADICTS": 0, "LIKELY_CONTRADICTS": 1, "RECONCILED": 2, "CORROBORATES": 3}
    relations.sort(key=lambda r: (order[r.verdict], -r.confidence))
    return relations


def summarize(relations: list[Relation]) -> dict[str, int]:
    return dict(Counter(r.verdict for r in relations))


def index_facts(facts: list[Fact]) -> dict[str, Fact]:
    return {f.fact_id: f for f in facts}


def cross_document(relations: list[Relation], facts: dict[str, Fact]) -> list[Relation]:
    """Relations whose two sides come from different documents."""
    out = []
    for rel in relations:
        left, right = facts.get(rel.left_id), facts.get(rel.right_id)
        if left and right and left.doc_id != right.doc_id:
            out.append(rel)
    return out
