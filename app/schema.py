"""The fact model.

The central design claim of this project: a value is meaningless without the
context that qualifies it. `₹74,540.82 million` and `₹81,415.38 million` are
both "Delhivery FY24 revenue" and both correct — they differ only in `basis`
(standalone vs consolidated). A schema of (subject, metric, value) cannot
represent that, so it must report a contradiction that does not exist.

So every Fact carries `context`: an open dict of qualifying keys. Two facts are
*comparable* only when their context keys agree. That single rule produces
corroboration, contradiction, and reconciliation from one mechanism.

`context` is deliberately an open dict rather than fixed fields, which lets the
schema grow as new kinds of documents introduce new qualifiers.
"""

from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel, Field

FactType = Literal["quantity", "state", "event", "attribute"]

# Context keys we understand well enough to reason about explicitly. Extractors
# may emit others; unknown keys still participate in comparability (they just
# get no special-cased reconciliation reasoning).
KNOWN_CONTEXT_KEYS = (
    "period",  # FY24, Q4 FY24, "as of 2021-12-31"
    "basis",  # standalone | consolidated
    "entity_scope",  # "excluding Spoton", "group", "domestic"
    "vintage",  # advance estimate | provisional | revised | actual | projection
    "geography",  # India, global
    "segment",  # Express Parcel, PTL, agriculture
    "measure",  # nominal | real | constant-prices | YoY
)


class Evidence(BaseModel):
    """Where a fact came from, precisely enough to highlight it."""

    doc_id: str
    filename: str
    page: int
    start: int = Field(description="char offset into the page text")
    end: int
    quote: str = Field(description="verbatim source text; never paraphrased")
    bboxes: list[tuple[float, float, float, float]] = Field(default_factory=list)


class Quantity(BaseModel):
    """A numeric value plus everything needed to compare it to another one."""

    raw: str = Field(description="as written in the document")
    value: float | None = None
    magnitude: str | None = Field(
        default=None, description="crore | lakh | million | billion | thousand"
    )
    unit: str | None = Field(default=None, description="INR | USD | percent | count | days")

    # Filled in by the deterministic normalizer, never by the LLM.
    canonical_value: float | None = None
    canonical_unit: str | None = None

    @property
    def is_normalized(self) -> bool:
        return self.canonical_value is not None and self.canonical_unit is not None


class Fact(BaseModel):
    fact_id: str = ""
    doc_id: str

    subject: str = Field(description="the entity the fact is about, as stated")
    metric: str = Field(description="what is being asserted, as stated")

    fact_type: FactType = "quantity"
    quantity: Quantity | None = None
    value_text: str | None = Field(
        default=None, description="the value for non-numeric facts, e.g. a role or status"
    )

    context: dict[str, str] = Field(
        default_factory=dict,
        description="qualifying keys; see KNOWN_CONTEXT_KEYS. Open by design.",
    )

    # Valid-time. A director who resigned is not a contradiction of a director
    # who was serving — the two facts hold over different intervals.
    valid_from: str | None = None
    valid_to: str | None = None

    evidence: Evidence
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    notes: str | None = None

    # Canonical forms, filled by the normalizer.
    subject_key: str = ""
    metric_key: str = ""

    def compute_id(self) -> str:
        """Stable id from content + location, so re-ingesting is idempotent."""
        basis = "|".join(
            [
                self.doc_id,
                str(self.evidence.page),
                str(self.evidence.start),
                self.subject_key or self.subject,
                self.metric_key or self.metric,
                str(self.quantity.raw if self.quantity else self.value_text),
            ]
        )
        return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]

    def context_signature(self) -> tuple[tuple[str, str], ...]:
        """The comparability fingerprint: sorted, normalized context keys."""
        return tuple(sorted((k, v.strip().lower()) for k, v in self.context.items() if v))

    def describe(self) -> str:
        if self.quantity and self.quantity.is_normalized:
            val = f"{self.quantity.canonical_value:,.4g} {self.quantity.canonical_unit}"
        elif self.quantity:
            val = self.quantity.raw
        else:
            val = self.value_text or "?"
        ctx = ", ".join(f"{k}={v}" for k, v in sorted(self.context.items()))
        return f"{self.subject} · {self.metric} = {val}" + (f"  [{ctx}]" if ctx else "")


Verdict = Literal["CORROBORATES", "CONTRADICTS", "RECONCILED", "UNRELATED"]


class Relation(BaseModel):
    """A judgement about two facts, with the reasoning that produced it."""

    left_id: str
    right_id: str
    verdict: Verdict
    # For RECONCILED: which context key differs and therefore explains the gap.
    differing_keys: list[str] = Field(default_factory=list)
    reasoning: str = ""
    # How the verdict was reached, so a reader can tell rules from model judgement.
    method: Literal["rule", "llm", "arithmetic"] = "rule"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
