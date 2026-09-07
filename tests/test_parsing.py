"""Phase 1 tests: is our evidence grounding actually trustworthy?

The whole system's credibility rests on being able to point at a claim and say
"this came from exactly here". These tests prove the char-offset index is exact.
"""

from pathlib import Path

import pytest

from app.parsing.pdf_parser import document_id, has_text_layer, parse_pdf

DATA = Path(__file__).resolve().parents[1] / "data" / "starter-datasets"
PDFS = sorted(DATA.glob("*/*.pdf"))


def test_dataset_present():
    assert PDFS, f"no starter PDFs found under {DATA}"


@pytest.mark.parametrize("pdf", PDFS, ids=lambda p: p.name)
def test_every_pdf_parses_and_has_text(pdf):
    assert has_text_layer(pdf), f"{pdf.name} looks scanned; we do not OCR"
    doc = parse_pdf(pdf)
    assert doc.page_count > 0
    assert doc.char_count > 1000
    assert doc.blocks, "no evidence units produced"


@pytest.mark.parametrize("pdf", PDFS, ids=lambda p: p.name)
def test_span_offsets_are_exact(pdf):
    """Every span must sit inside its page text, and spans must not overlap.

    If this fails, a highlighted quote would point at the wrong words.
    """
    doc = parse_pdf(pdf)
    for page in doc.pages:
        last_end = -1
        for span in page.spans:
            assert 0 <= span.start < span.end <= len(page.text)
            assert span.start >= last_end, "spans out of order or overlapping"
            last_end = span.end


@pytest.mark.parametrize("pdf", PDFS, ids=lambda p: p.name)
def test_block_text_matches_page_text(pdf):
    """A block's cached text must equal the page text at its own offsets."""
    doc = parse_pdf(pdf)
    for block in doc.blocks:
        page = doc.page(block.page_number)
        assert block.text == page.text[block.start : block.end]


@pytest.mark.parametrize("pdf", PDFS, ids=lambda p: p.name)
def test_quotes_resolve_to_boxes(pdf):
    """The round trip that matters: text range -> page geometry."""
    doc = parse_pdf(pdf)
    checked = 0
    for block in doc.blocks:
        if len(block.text.strip()) < 40:
            continue
        page = doc.page(block.page_number)
        boxes = page.boxes_for(block.start, block.end)
        assert boxes, "a real block produced no bounding boxes"
        # every box must lie within the page canvas (with a small tolerance
        # for glyphs that overhang their reported line box)
        for x0, y0, x1, y1 in boxes:
            assert -2 <= x0 <= page.width + 2
            assert -2 <= y0 <= page.height + 2
        checked += 1
        if checked >= 25:
            break
    assert checked, "no substantial blocks found to check"


def test_document_id_is_content_addressed():
    """Same bytes -> same id, so re-uploading a renamed file is not a new doc."""
    a = PDFS[0]
    assert document_id(a) == document_id(a)
    assert document_id(a) != document_id(PDFS[1])
    assert len(document_id(a)) == 16
