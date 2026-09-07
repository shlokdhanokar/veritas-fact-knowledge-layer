"""Load facts from ingest JSON into the SQLite store, then relate them.

Bridges the CLI-era JSON files into the database the API serves. Real ingest
goes straight to the store; this exists so earlier runs are not wasted.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.parsing.pdf_parser import parse_pdf
from app.pipeline import rebuild_relations
from app.schema import Fact
from app.store import get_store

store = get_store()
seen = {}
for p in sys.argv[1:]:
    for d in json.load(open(p, encoding='utf-8')):
        f = Fact.model_validate(d)
        seen[f.fact_id] = f

by_doc = {}
for f in seen.values():
    by_doc.setdefault(f.doc_id, []).append(f)

for pdf in Path('data/starter-datasets').glob('*/*.pdf'):
    doc = parse_pdf(pdf)
    if doc.doc_id in by_doc:
        store.upsert_document(doc.doc_id, doc.filename, doc.page_count,
                              doc.char_count, doc.title,
                              {"proposed": 0, "kept": len(by_doc[doc.doc_id])})
        print(f"document {doc.filename}: {len(by_doc[doc.doc_id])} facts")

store.save_facts(list(seen.values()))
print(f"stored {len(seen)} facts; relating…")
print(rebuild_relations())
