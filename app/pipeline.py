"""End-to-end ingest: PDF in, facts and relations in the store.

Shared by the CLI and the API so that what a grader sees through the UI is
exactly what the scripts do — there is no second, divergent code path.

Incremental by construction. Adding a document only extracts that document's
windows (everything already read is a cache hit) and only adjudicates pairs that
involve its new facts. The existing knowledge layer is never rebuilt.
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.extraction.extractor import extract_document
from app.llm.provider import LLMProvider, get_provider
from app.parsing.pdf_parser import ParsedDoc, has_text_layer, parse_pdf
from app.parsing.segmenter import build_windows, select_windows
from app.reasoning.adjudicate import adjudicate
from app.reasoning.pairing import candidate_pairs
from app.reasoning.engine import INTERESTING
from app.schema import Fact
from app.store import Store, get_store

log = logging.getLogger(__name__)


class IngestError(RuntimeError):
    pass


def ingest_pdf(
    path: str | Path,
    *,
    store: Store | None = None,
    provider: LLMProvider | None = None,
    window_limit: int | None = None,
    max_workers: int = 3,
    relate_after: bool = True,
) -> dict:
    """Parse, extract, store, and relate one PDF. Returns a summary."""
    path = Path(path)
    store = store or get_store()

    if not has_text_layer(path):
        raise IngestError(
            f"{path.name} has no extractable text layer. It is probably a scan; "
            "this system does not perform OCR."
        )

    doc: ParsedDoc = parse_pdf(path)
    windows = select_windows(build_windows(doc), window_limit)
    if not windows:
        raise IngestError(f"{path.name} produced no fact-bearing passages.")

    facts, stats = extract_document(
        doc, windows, provider or get_provider(), max_workers=max_workers
    )

    store.upsert_document(
        doc_id=doc.doc_id,
        filename=doc.filename,
        page_count=doc.page_count,
        char_count=doc.char_count,
        title=doc.title,
        stats=stats.model_dump(),
    )
    store.save_facts(facts)

    summary = {
        "doc_id": doc.doc_id,
        "filename": doc.filename,
        "pages": doc.page_count,
        "windows": len(windows),
        "facts": len(facts),
        "extraction": stats.model_dump(),
        "grounding_rate": round(stats.grounding_rate, 3),
        "flagged_ambiguous": sum(1 for f in facts if f.notes),
    }

    if relate_after and facts:
        summary["relations"] = relate_new_facts(facts, store=store)

    return summary


def relate_new_facts(new_facts: list[Fact], *, store: Store | None = None) -> dict:
    """Adjudicate the new facts against everything already known.

    Only pairs touching a new fact are considered, so the cost of adding a
    document scales with that document rather than with the whole corpus.
    """
    store = store or get_store()
    existing = store.all_facts()

    new_ids = {f.fact_id for f in new_facts}
    pairs = candidate_pairs(existing)
    relevant = [(a, b, s) for a, b, s in pairs if a.fact_id in new_ids or b.fact_id in new_ids]

    relations = []
    for a, b, score in relevant:
        rel = adjudicate(a, b, similarity=score)
        if rel.verdict in INTERESTING:
            relations.append(rel)

    store.save_relations(relations)

    counts: dict[str, int] = {}
    for r in relations:
        counts[r.verdict] = counts.get(r.verdict, 0) + 1
    return {"pairs_considered": len(relevant), "relations_saved": len(relations), "verdicts": counts}


def rebuild_relations(store: Store | None = None) -> dict:
    """Re-adjudicate every pair. Used after a reasoning change, not during ingest."""
    store = store or get_store()
    # Full rebuild means the old verdicts are no longer authoritative.
    store.clear_relations()
    facts = store.all_facts()
    relations = []
    for a, b, score in candidate_pairs(facts):
        rel = adjudicate(a, b, similarity=score)
        if rel.verdict in INTERESTING:
            relations.append(rel)
    store.save_relations(relations)
    counts: dict[str, int] = {}
    for r in relations:
        counts[r.verdict] = counts.get(r.verdict, 0) + 1
    return {"facts": len(facts), "relations": len(relations), "verdicts": counts}
