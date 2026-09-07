"""Ingest PDFs into a JSON fact file. CLI harness for the pipeline.

    python scripts/ingest.py <pdf> [<pdf> ...] [--limit N] [--out facts.json]

Used during development and in the demo. The API server reuses the same
functions, so what you see here is what the endpoint does.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.extraction.extractor import extract_document  # noqa: E402
from app.llm.provider import get_provider  # noqa: E402
from app.parsing.pdf_parser import has_text_layer, parse_pdf  # noqa: E402
from app.parsing.segmenter import build_windows, select_windows  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdfs", nargs="+")
    ap.add_argument("--limit", type=int, default=None, help="max windows per document")
    ap.add_argument("--out", default="data/facts.json")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    provider = get_provider()
    print(f"provider: {provider.name}")

    all_facts = []
    for path in args.pdfs:
        path = Path(path)
        if not has_text_layer(path):
            print(f"!! {path.name}: no text layer (scanned?); skipping - we do not OCR")
            continue

        doc = parse_pdf(path)
        windows = select_windows(build_windows(doc), args.limit)
        print(f"\n== {path.name}  pages={doc.page_count} windows={len(windows)}")

        started = time.time()
        facts, stats = extract_document(doc, windows, provider, max_workers=args.workers)
        elapsed = time.time() - started

        print(f"   {stats.summary()}  in {elapsed:.1f}s")
        low = [f for f in facts if f.notes]
        print(f"   facts flagged for ambiguous binding: {len(low)}/{len(facts)}")
        all_facts.extend(facts)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps([f.model_dump() for f in all_facts], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nwrote {len(all_facts)} facts -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
