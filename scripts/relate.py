"""Run the reasoning pass over ingested facts and print what it found.

    python scripts/relate.py data/facts_q4.json data/facts_ar.json [--verdict CONTRADICTS]

Consumes no LLM quota: everything here is deterministic rule evaluation over
facts that were already extracted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.reasoning.engine import index_facts, relate, summarize  # noqa: E402
from app.schema import Fact  # noqa: E402


def load(paths: list[str]) -> list[Fact]:
    facts: list[Fact] = []
    for p in paths:
        data = json.loads(Path(p).read_text(encoding="utf-8"))
        facts.extend(Fact.model_validate(d) for d in data)
    # De-duplicate across files.
    return list({f.fact_id: f for f in facts}.values())


def show(rel, facts, width=100):
    left, right = facts[rel.left_id], facts[rel.right_id]
    print(f"\n[{rel.verdict}]  confidence={rel.confidence:.2f}  via={rel.method}")
    for side, f in (("A", left), ("B", right)):
        print(f"  {side}: {f.describe()[:width]}")
        print(f"     {f.evidence.filename} p{f.evidence.page}")
        print(f"     “{f.evidence.quote.strip()[:110]}”")
    if rel.differing_keys:
        print(f"  differing: {', '.join(rel.differing_keys)}")
    print(f"  reasoning: {rel.reasoning[:400]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("facts", nargs="+")
    ap.add_argument("--verdict", default=None, help="filter: CONTRADICTS|RECONCILED|CORROBORATES")
    ap.add_argument("--cross-doc", action="store_true", help="only cross-document relations")
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--min-confidence", type=float, default=0.0)
    args = ap.parse_args()

    facts = load(args.facts)
    index = index_facts(facts)
    print(f"loaded {len(facts)} facts from {len(args.facts)} file(s)")

    relations = relate(facts, min_confidence=args.min_confidence)
    print("verdicts:", summarize(relations))

    shown = relations
    if args.cross_doc:
        shown = [r for r in shown if index[r.left_id].doc_id != index[r.right_id].doc_id]
        print(f"cross-document relations: {len(shown)}")
    if args.verdict:
        shown = [r for r in shown if r.verdict == args.verdict.upper()]

    for rel in shown[: args.limit]:
        show(rel, index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
