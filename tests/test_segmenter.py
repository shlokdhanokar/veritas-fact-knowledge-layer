"""Tests for windowing and — more importantly — the grounding gate.

`Window.locate` is the single mechanism that stops fabricated evidence from
entering the knowledge layer: a quote that cannot be found in the source is not
grounded, so the fact is dropped. These tests pin its behaviour, including
exactly how much leniency it allows.
"""

from pathlib import Path

import pytest

from app.parsing.pdf_parser import parse_pdf
from app.parsing.segmenter import Window, build_windows, fact_density, select_windows

DATA = Path(__file__).resolve().parents[1] / "data" / "starter-datasets"
PDFS = sorted(DATA.glob("*/*.pdf"))


def make_window(text: str, start: int = 100) -> Window:
    return Window(doc_id="d", page_number=3, start=start, end=start + len(text), text=text)


def test_locate_exact_match_returns_page_offsets():
    w = make_window("Revenue from operations stood at Rs 81,415.38 million.", start=500)
    got = w.locate("Rs 81,415.38 million")
    assert got is not None
    start, end = got
    assert w.text[start - 500 : end - 500] == "Rs 81,415.38 million"


def test_locate_tolerates_whitespace_differences():
    """Models re-flow PDF line breaks when quoting. That is formatting, not fiction."""
    w = make_window("Revenue from\noperations   stood at\n81,415.38")
    assert w.locate("Revenue from operations stood at 81,415.38") is not None


def test_locate_rejects_fabricated_quote():
    """The whole point: invented evidence must not resolve."""
    w = make_window("Revenue from operations stood at 81,415.38 million.")
    assert w.locate("Revenue from operations stood at 99,999.99 million.") is None
    assert w.locate("The company was founded in Mumbai.") is None


def test_locate_rejects_altered_digits():
    """A single changed digit is a different fact, not a formatting variation."""
    w = make_window("consolidated revenue of 81,415.38 million")
    assert w.locate("consolidated revenue of 81,415.38 million") is not None
    assert w.locate("consolidated revenue of 81,415.39 million") is None


def test_locate_empty_quote_is_not_grounded():
    assert make_window("some text").locate("") is None


def test_fact_density_prefers_quantified_prose():
    boilerplate = (
        "This report has been prepared in accordance with the applicable provisions "
        "and should be read together with the accompanying notes and disclosures."
    )
    factual = (
        "Revenue from operations for FY24 stood at Rs 81,415.38 million as against "
        "Rs 72,253.01 million for FY23, registering a growth of 12.68 per cent."
    )
    assert fact_density(factual) > fact_density(boilerplate)


def test_fact_density_ignores_tiny_fragments():
    assert fact_density("FY24") == 0.0  # a heading is not evidence


@pytest.mark.parametrize("pdf", PDFS, ids=lambda p: p.name)
def test_windows_are_consistent_with_pages(pdf):
    doc = parse_pdf(pdf)
    windows = build_windows(doc)
    assert windows
    for w in windows:
        page = doc.page(w.page_number)
        assert page.text[w.start : w.end] == w.text
        assert w.score > 0


def test_select_windows_takes_densest_then_restores_reading_order():
    ws = [
        Window("d", 5, 0, 10, "e", score=1.0),
        Window("d", 1, 0, 10, "a", score=9.0),
        Window("d", 3, 0, 10, "c", score=5.0),
    ]
    chosen = select_windows(ws, limit=2)
    assert [w.score for w in chosen] == [9.0, 5.0]  # dropped the weakest
    assert [w.page_number for w in chosen] == [1, 3]  # back in document order
