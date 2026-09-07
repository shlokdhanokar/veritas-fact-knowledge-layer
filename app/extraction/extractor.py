"""Turning windows of text into grounded, normalized Facts.

The pipeline per window is: LLM proposes facts -> we verify each quote exists
verbatim in the source -> we resolve it to page geometry -> we normalize units
deterministically. A proposed fact that fails the quote check is dropped, not
repaired. That check is what makes "every fact is grounded in evidence" a
property of the system rather than a hope about the model.
"""

from __future__ import annotations

import concurrent.futures
import logging
import re
from typing import Literal

from pydantic import BaseModel, Field

from app.extraction.prompts import SYSTEM, build_user_prompt
from app.extraction.cache import ExtractionCache, get_cache
from app.llm.provider import LLMProvider, get_provider
from app.normalize.units import normalize_quantity, parse_number
from app.parsing.pdf_parser import ParsedDoc
from app.parsing.segmenter import Window
from app.schema import Evidence, Fact, Quantity

log = logging.getLogger(__name__)


class RawFact(BaseModel):
    """What the model returns. Flat on purpose - nested schemas degrade
    structured-output reliability, and flat fields are easier to validate."""

    subject: str
    metric: str
    fact_type: Literal["quantity", "state", "event", "attribute"] = "quantity"

    value_raw: str | None = None
    magnitude: str | None = None
    unit: str | None = None
    value_text: str | None = None

    period: str | None = None
    basis: str | None = None
    entity_scope: str | None = None
    vintage: str | None = None
    geography: str | None = None
    segment: str | None = None
    measure: str | None = None

    valid_from: str | None = None
    valid_to: str | None = None

    quote: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class RawFactList(BaseModel):
    facts: list[RawFact] = Field(default_factory=list)


class ExtractionStats(BaseModel):
    """Kept because the drop reasons are the honest story of extraction quality."""

    windows_read: int = 0
    proposed: int = 0
    kept: int = 0
    dropped_ungrounded: int = 0
    dropped_empty_value: int = 0
    failed_windows: int = 0
    cache_hits: int = 0

    @property
    def grounding_rate(self) -> float:
        return self.kept / self.proposed if self.proposed else 0.0

    def summary(self) -> str:
        return (
            f"windows={self.windows_read} proposed={self.proposed} kept={self.kept} "
            f"(grounded {self.grounding_rate:.0%}) "
            f"dropped_ungrounded={self.dropped_ungrounded} "
            f"dropped_empty={self.dropped_empty_value} failed_windows={self.failed_windows} "
            f"cache_hits={self.cache_hits}"
        )


def _context_from(raw: RawFact) -> dict[str, str]:
    """Collect stated qualifiers. Null and empty values are omitted entirely,
    so an absent qualifier never masquerades as a matching one."""
    candidates = {
        "period": raw.period,
        "basis": raw.basis,
        "entity_scope": raw.entity_scope,
        "vintage": raw.vintage,
        "geography": raw.geography,
        "segment": raw.segment,
        "measure": raw.measure,
    }
    out = {}
    for key, value in candidates.items():
        if value and str(value).strip().lower() not in {"", "null", "none", "n/a", "unknown"}:
            out[key] = str(value).strip()
    return out


_NUM_TOKEN = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _grounding_quality(raw: RawFact, quote: str) -> tuple[float, str | None]:
    """Detect the silent failure mode of table extraction.

    When a table row linearises to `Line haul expenses 724 616 696 708 2,517 2,684`
    the value is real but the *column binding* is a guess: which period does 708
    belong to? The model answers confidently either way and nothing in the output
    reveals the uncertainty.

    So we measure it. A quote carrying many candidate numbers is weak evidence for
    any single one of them, and a value that does not literally appear in its own
    quote is weaker still. Both cut confidence and leave a note, which keeps the
    ambiguity visible downstream instead of laundering it into a clean number.

    Returns (multiplier, note).
    """
    if not raw.value_raw:
        return 1.0, None

    value = str(raw.value_raw).strip()
    numbers = _NUM_TOKEN.findall(quote)

    if value not in quote:
        return 0.4, f"value {value!r} does not appear verbatim in its own quote"

    distinct = len(set(numbers))
    if distinct <= 2:
        return 1.0, None
    if value in numbers and numbers.count(value) > 1:
        return 0.5, f"value {value!r} appears {numbers.count(value)}x in quote; binding ambiguous"
    # Many candidates: likely a linearised table row with a two-level header.
    penalty = max(0.45, 1.0 - 0.12 * (distinct - 2))
    return penalty, (
        f"quote contains {distinct} numbers; column/period binding is inferred, not certain"
    )


def _to_fact(raw: RawFact, doc: ParsedDoc, window: Window) -> Fact | None:
    """Ground and normalize one proposed fact, or reject it."""
    located = window.locate(raw.quote)
    if located is None:
        return None  # quote not found in source: ungrounded, drop it
    start, end = located

    page = doc.page(window.page_number)
    quantity = None
    if raw.value_raw is not None and str(raw.value_raw).strip():
        value = parse_number(raw.value_raw)
        canonical_value, canonical_unit = normalize_quantity(
            raw.value_raw, value, raw.magnitude, raw.unit
        )
        quantity = Quantity(
            raw=str(raw.value_raw).strip(),
            value=value,
            magnitude=raw.magnitude,
            unit=raw.unit,
            canonical_value=canonical_value,
            canonical_unit=canonical_unit,
        )

    if quantity is None and not (raw.value_text and raw.value_text.strip()):
        return None  # a fact with no value asserts nothing

    quote_text = page.text[start:end]
    multiplier, note = _grounding_quality(raw, quote_text)

    fact = Fact(
        doc_id=doc.doc_id,
        subject=raw.subject.strip(),
        metric=raw.metric.strip(),
        fact_type=raw.fact_type,
        quantity=quantity,
        value_text=(raw.value_text or "").strip() or None,
        context=_context_from(raw),
        valid_from=raw.valid_from,
        valid_to=raw.valid_to,
        confidence=round(raw.confidence * multiplier, 3),
        notes=note,
        evidence=Evidence(
            doc_id=doc.doc_id,
            filename=doc.filename,
            page=window.page_number,
            start=start,
            end=end,
            # Store the source's own words, not the model's copy of them.
            quote=page.text[start:end],
            bboxes=page.boxes_for(start, end),
        ),
    )
    fact.subject_key = fact.subject.strip().lower()
    fact.metric_key = fact.metric.strip().lower()
    fact.fact_id = fact.compute_id()
    return fact


def extract_window(
    doc: ParsedDoc,
    window: Window,
    provider: LLMProvider,
    stats: ExtractionStats | None = None,
    cache: ExtractionCache | None = None,
) -> list[Fact]:
    """Extract facts from a single window, reading through a cache.

    Cache hits cost nothing and consume no quota, which is what makes re-runs
    and incremental ingest practical on a free tier.
    """
    stats = stats or ExtractionStats()
    cache = cache if cache is not None else get_cache()
    model = getattr(provider, "models", [getattr(provider, "model", provider.name)])[0]
    key = cache.make_key(window.text, str(model))

    payload = cache.get(key)
    if payload is not None:
        stats.cache_hits += 1
        result = RawFactList.model_validate(payload)
    else:
        try:
            result = provider.structured(
                SYSTEM,
                build_user_prompt(window.text, filename=doc.filename, page=window.page_number),
                RawFactList,
            )
        except Exception as exc:  # noqa: BLE001 - one bad window must not stop ingest
            log.warning("window %s failed: %s", window.key, exc)
            stats.failed_windows += 1
            return []
        cache.put(
            key,
            result.model_dump(),
            doc_id=doc.doc_id,
            page=window.page_number,
            model=str(model),
        )

    stats.windows_read += 1
    kept: list[Fact] = []
    for raw in result.facts:
        stats.proposed += 1
        fact = _to_fact(raw, doc, window)
        if fact is None:
            if window.locate(raw.quote) is None:
                stats.dropped_ungrounded += 1
            else:
                stats.dropped_empty_value += 1
            continue
        kept.append(fact)
        stats.kept += 1
    return kept


def extract_document(
    doc: ParsedDoc,
    windows: list[Window],
    provider: LLMProvider | None = None,
    *,
    max_workers: int = 3,
    progress: bool = False,
    cache: ExtractionCache | None = None,
) -> tuple[list[Fact], ExtractionStats]:
    """Extract across many windows concurrently.

    Modest concurrency: enough to make a 100-page document tolerable, low enough
    to stay inside free-tier rate limits. Failures are per-window and never abort
    the run, because a partial knowledge layer is still useful.
    """
    provider = provider or get_provider()
    cache = cache if cache is not None else get_cache()
    stats = ExtractionStats()
    facts: list[Fact] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(extract_window, doc, w, provider, stats, cache): w for w in windows}
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            facts.extend(future.result())
            if progress:
                print(
                    f"  [{done}/{len(windows)}] p{futures[future].page_number} "
                    f"kept={stats.kept}",
                    flush=True,
                )

    # De-duplicate: the same sentence can surface in overlapping windows.
    unique: dict[str, Fact] = {}
    for fact in facts:
        existing = unique.get(fact.fact_id)
        if existing is None or fact.confidence > existing.confidence:
            unique[fact.fact_id] = fact

    ordered = sorted(unique.values(), key=lambda f: (f.evidence.page, f.evidence.start))
    stats.kept = len(ordered)
    return ordered, stats
