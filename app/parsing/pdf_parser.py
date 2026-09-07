"""Layout-aware PDF parsing.

Produces, for every document, a flat text stream per page *plus* a character-offset
index back to the on-page geometry. That index is what makes evidence grounding real:
given any character range in the extracted text we can name the page and draw the
exact boxes it came from.

Nothing here is document-specific. No filenames, no schemas, no keyword lists.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF


# A rectangle in PDF user-space: (x0, y0, x1, y1), origin top-left.
BBox = tuple[float, float, float, float]


@dataclass(slots=True)
class Span:
    """A run of characters that shares one bounding box on the page."""

    start: int  # inclusive, offset into Page.text
    end: int  # exclusive
    bbox: BBox


@dataclass(slots=True)
class Block:
    """A layout block — PyMuPDF's paragraph-ish grouping. Our unit of evidence."""

    index: int
    page_number: int  # 1-based
    start: int  # offset into Page.text
    end: int
    bbox: BBox
    text: str


@dataclass(slots=True)
class Page:
    number: int  # 1-based
    width: float
    height: float
    text: str
    spans: list[Span] = field(default_factory=list)
    blocks: list[Block] = field(default_factory=list)

    def boxes_for(self, start: int, end: int) -> list[BBox]:
        """Every bbox touched by the character range [start, end)."""
        return [s.bbox for s in self.spans if s.start < end and s.end > start]

    def quote(self, start: int, end: int) -> str:
        return self.text[start:end]


@dataclass(slots=True)
class ParsedDoc:
    doc_id: str
    path: str
    filename: str
    page_count: int
    pages: list[Page]
    title: str | None = None

    @property
    def blocks(self) -> list[Block]:
        return [b for p in self.pages for b in p.blocks]

    def page(self, number: int) -> Page:
        return self.pages[number - 1]

    @property
    def char_count(self) -> int:
        return sum(len(p.text) for p in self.pages)


def document_id(path: str | Path) -> str:
    """Content-addressed id: re-uploading the same bytes is the same document.

    This is what makes incremental ingest safe — identity comes from content,
    never from the filename.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _union(boxes: list[BBox]) -> BBox:
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _parse_page(page: fitz.Page, number: int) -> Page:
    """Rebuild page text while recording where every character sits.

    We walk PyMuPDF's block -> line -> span tree and append to a text buffer,
    recording (start, end, bbox) for each span as we go. Lines are joined with
    "\\n", blocks with "\\n\\n", so the reconstructed text reads naturally while
    offsets stay exact.
    """
    raw = page.get_text("dict")
    parts: list[str] = []
    cursor = 0
    spans: list[Span] = []
    blocks: list[Block] = []

    for block in raw.get("blocks", []):
        # type 0 == text. Images/drawings carry no characters to ground.
        if block.get("type") != 0:
            continue

        block_start = cursor
        block_boxes: list[BBox] = []
        first_line = True

        for line in block.get("lines", []):
            if not first_line:
                parts.append("\n")
                cursor += 1
            first_line = False

            for span in line.get("spans", []):
                text = span.get("text", "")
                if not text:
                    continue
                bbox = tuple(span["bbox"])  # type: ignore[assignment]
                spans.append(Span(start=cursor, end=cursor + len(text), bbox=bbox))
                block_boxes.append(bbox)
                parts.append(text)
                cursor += len(text)

        if cursor == block_start:  # block contributed nothing
            continue

        blocks.append(
            Block(
                index=len(blocks),
                page_number=number,
                start=block_start,
                end=cursor,
                bbox=_union(block_boxes),
                text="".join(parts)[block_start:cursor],
            )
        )
        parts.append("\n\n")
        cursor += 2

    text = "".join(parts)
    rect = page.rect
    return Page(
        number=number,
        width=rect.width,
        height=rect.height,
        text=text,
        spans=spans,
        blocks=blocks,
    )


def parse_pdf(path: str | Path) -> ParsedDoc:
    """Parse a PDF into pages, blocks, and a character->geometry index."""
    path = Path(path)
    doc = fitz.open(path)
    try:
        pages = [_parse_page(doc[i], i + 1) for i in range(doc.page_count)]
        meta_title = (doc.metadata or {}).get("title") or None
        return ParsedDoc(
            doc_id=document_id(path),
            path=str(path),
            filename=path.name,
            page_count=doc.page_count,
            pages=pages,
            title=meta_title.strip() if meta_title else None,
        )
    finally:
        doc.close()


def has_text_layer(path: str | Path, sample: int = 5) -> bool:
    """Cheap check for scanned PDFs, which would need OCR we do not attempt.

    Sampling a few pages keeps this O(1) on 100-page documents.
    """
    doc = fitz.open(path)
    try:
        if doc.page_count == 0:
            return False
        step = max(1, doc.page_count // sample)
        probed = [doc[i] for i in range(0, doc.page_count, step)][:sample]
        return any(len(p.get_text("text").strip()) > 50 for p in probed)
    finally:
        doc.close()
