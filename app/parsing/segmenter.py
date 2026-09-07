"""Grouping parsed blocks into extraction windows, and deciding which are worth reading.

Two problems this solves.

**Cost.** A 100-page filing yields ~3,000 layout blocks. One LLM call per block is
both unaffordable on a free tier and wasteful, since most blocks are headings,
page furniture, or boilerplate. We group contiguous blocks on a page into windows
of a few thousand characters, which cuts calls by ~20x and gives the model enough
surrounding text to read a table row correctly.

**Focus.** Windows are then scored by a cheap deterministic heuristic for how
likely they are to contain assertable facts. No LLM, no keyword list tied to any
particular document — just signals that generalise: numbers, currency, units,
dates, and state-change verbs. Callers take the top-N.

This is the "precision over recall" trade-off stated plainly: we would rather read
the 60 densest windows carefully than all 200 badly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.parsing.pdf_parser import BBox, ParsedDoc

# Whole-page-sized by default. Measured trade-off: at ~3,000 chars the model saw
# table rows without their headers and grounded 75-81% of proposals; at page
# scale it sees the header block that says "Standalone | Consolidated" and
# grounded 100%, while using ~7x fewer requests. Bigger context is both cheaper
# and more accurate here, because the qualifiers a fact needs usually sit in
# nearby text rather than in the row itself.
DEFAULT_WINDOW_CHARS = 12000
MIN_WINDOW_CHARS = 120


@dataclass(slots=True)
class Window:
    """A contiguous run of blocks on one page, treated as one unit of reading."""

    doc_id: str
    page_number: int
    start: int  # char offset into the page text
    end: int
    text: str
    score: float = 0.0

    @property
    def key(self) -> str:
        return f"{self.doc_id}:{self.page_number}:{self.start}"

    def locate(self, quote: str) -> tuple[int, int] | None:
        """Find a quote inside this window, returning page-level char offsets.

        This is the grounding gate. A claim whose quote cannot be found verbatim
        in the window it came from is not grounded, and callers drop it — which
        is how hallucinated evidence gets filtered out mechanically rather than
        by trusting the model.
        """
        if not quote:
            return None
        idx = self.text.find(quote)
        if idx == -1:
            idx = _find_whitespace_insensitive(self.text, quote)
            if idx is None:
                return None
            start, end = idx
            return self.start + start, self.start + end
        return self.start + idx, self.start + idx + len(quote)


def _find_whitespace_insensitive(haystack: str, needle: str) -> tuple[int, int] | None:
    """Match ignoring whitespace differences.

    PDF text carries line breaks and doubled spaces that a model will silently
    normalise when quoting. That is a formatting difference, not a fabrication,
    so we still accept it — but we accept nothing looser than this.
    """
    pattern = re.compile(r"\s+".join(re.escape(tok) for tok in needle.split()))
    match = pattern.search(haystack)
    return (match.start(), match.end()) if match else None


# Signals that a passage asserts something checkable. Deliberately generic:
# none of these are tied to logistics, macroeconomics, or any source document.
_NUMBER = re.compile(r"\d")
_CURRENCY = re.compile(r"[₹$€£]|\b(?:rs\.?|inr|usd|eur)\b", re.I)
_MAGNITUDE = re.compile(r"\b(?:crore|lakh|million|billion|trillion|mn|bn|cr)\b", re.I)
_PERCENT = re.compile(r"%|\bper\s?cent\b|\bpercent\b|\bbps\b", re.I)
_PERIOD = re.compile(
    r"\b(?:FY\s?\d{2,4}|Q[1-4]\b|fiscal\s+\d{4}|20\d{2}-\d{2}|20\d{2}/\d{2}|as\s+(?:of|at))\b",
    re.I,
)
_DATE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\s+\d{1,2},?\s+\d{4}\b",
    re.I,
)
_STATE_VERB = re.compile(
    r"\b(?:resigned|appointed|ceased|retired|acquired|merged|approved|"
    r"increased|decreased|grew|declined|stood at|amounted to|reported|"
    r"projected|estimated|revised)\b",
    re.I,
)

# A claim sentence: a verb of assertion sitting next to a figure, e.g.
# "GDP growth for FY25 is estimated to be 6.4 per cent".
#
# This exists because the rest of the scorer rewards digit density, which
# systematically prefers statistical tables over prose — and the headline
# assertions a reader most wants are usually written as sentences. Measured on
# the starter set, the pages stating India's FY25 growth estimate ranked 58/89
# and 83/100 by density alone, so they fell outside any sane reading budget
# while far less meaningful appendix tables were read first.
_CLAIM = re.compile(
    r"(?:estimated|projected|expected|forecast|revised|reported|recorded|"
    r"stood|amounted|grew|rose|fell|declined|increased|decreased|moderated|"
    r"accelerated|is|was|at)\s+"
    r"(?:to\s+be\s+|to\s+|by\s+|at\s+)?"
    r"(?:around\s+|about\s+|nearly\s+|approximately\s+)?"
    r"[₹$€£]?\s?\d[\d,.]*\s?"
    r"(?:per\s?cent|percent|%|crore|lakh|million|billion|bn|mn|cr\b)",
    re.I,
)


def fact_density(text: str) -> float:
    """Cheap score for how fact-bearing a passage looks. Higher is better."""
    if len(text) < MIN_WINDOW_CHARS:
        return 0.0

    digits = len(_NUMBER.findall(text))
    digit_ratio = digits / max(len(text), 1)

    score = 0.0
    score += min(digit_ratio * 40, 4.0)  # numbers, saturating
    score += 2.0 * bool(_CURRENCY.search(text))
    score += 2.0 * bool(_MAGNITUDE.search(text))
    score += 1.5 * bool(_PERCENT.search(text))
    score += 2.0 * bool(_PERIOD.search(text))
    score += 1.5 * bool(_DATE.search(text))
    score += 1.5 * bool(_STATE_VERB.search(text))

    # Assertive prose is worth more than its digit count suggests. Weighted to
    # rival a dense table, and scaled by how many such claims appear, so a page
    # of headline findings can outrank a page of appendix figures.
    claims = len(_CLAIM.findall(text))
    score += min(claims * 1.6, 4.8)

    # Pages that are almost entirely digits are usually raw statistical tables
    # whose headers sit elsewhere; we cannot ground them reliably, so damp them.
    if digit_ratio > 0.35:
        score *= 0.5

    return round(score, 3)


def build_windows(doc: ParsedDoc, max_chars: int = DEFAULT_WINDOW_CHARS) -> list[Window]:
    """Group each page's blocks into scored windows."""
    windows: list[Window] = []

    for page in doc.pages:
        current: list = []
        for block in page.blocks:
            if current and (block.end - current[0].start) > max_chars:
                windows.append(_make_window(doc, page, current))
                current = []
            current.append(block)
        if current:
            windows.append(_make_window(doc, page, current))

    return [w for w in windows if w.score > 0]


def _make_window(doc: ParsedDoc, page, blocks: list) -> Window:
    start, end = blocks[0].start, blocks[-1].end
    text = page.text[start:end]
    return Window(
        doc_id=doc.doc_id,
        page_number=page.number,
        start=start,
        end=end,
        text=text,
        score=fact_density(text),
    )


def select_windows(windows: list[Window], limit: int | None = None) -> list[Window]:
    """Take the densest windows, then restore document order.

    Reading order matters: it keeps related facts adjacent in the extraction
    log, which makes manual review of the output far easier.
    """
    ranked = sorted(windows, key=lambda w: w.score, reverse=True)
    chosen = ranked[:limit] if limit else ranked
    return sorted(chosen, key=lambda w: (w.page_number, w.start))


def boxes_for_window_quote(doc: ParsedDoc, window: Window, start: int, end: int) -> list[BBox]:
    """Resolve page-level offsets back to geometry for highlighting."""
    return doc.page(window.page_number).boxes_for(start, end)
