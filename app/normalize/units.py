"""Deterministic unit normalization.

Explicit design decision: the LLM never does arithmetic. It reports what a
document *says* ("8,142", "Cr", "₹"); this module decides what that *means*.
Unit conversion is the one place where being subtly wrong is invisible and
fatal to every downstream comparison, so it is plain code with a test table.

Indian magnitudes matter here: 1 crore = 10^7, 1 lakh = 10^5. A system that
only knows million/billion silently mangles Indian filings.
"""

from __future__ import annotations

import re

# Multiplier to the base unit (a plain count of rupees/dollars/things).
MAGNITUDES: dict[str, float] = {
    "thousand": 1e3,
    "lakh": 1e5,
    "lakhs": 1e5,
    "lac": 1e5,
    "million": 1e6,
    "mn": 1e6,
    "mm": 1e6,
    "crore": 1e7,
    "crores": 1e7,
    "cr": 1e7,
    "billion": 1e9,
    "bn": 1e9,
    "trillion": 1e12,
    "tn": 1e12,
}

CURRENCIES: dict[str, str] = {
    "₹": "INR",
    "rs": "INR",
    "rs.": "INR",
    "inr": "INR",
    "rupees": "INR",
    "$": "USD",
    "us$": "USD",
    "usd": "USD",
    "€": "EUR",
    "eur": "EUR",
}

PERCENT_TOKENS = {"%", "percent", "per cent", "pct", "bps"}


def parse_number(text: str) -> float | None:
    """Parse a number as written in financial prose.

    Handles thousands separators, parenthesised negatives (accounting style),
    and stray currency/space noise. Returns None rather than guessing.
    """
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None

    negative = False
    # Accounting negatives: (1,679.68) means -1679.68
    if s.startswith("(") and s.endswith(")"):
        negative = True
        s = s[1:-1].strip()
    if s.startswith("-") or s.startswith("−"):  # hyphen or unicode minus
        negative = True
        s = s[1:].strip()

    # Keep digits, separators and the decimal point only.
    s = re.sub(r"[^\d.,]", "", s)
    if not s or not any(ch.isdigit() for ch in s):
        return None

    # Indian ("1,23,456.78") and Western ("123,456.78") grouping both reduce
    # to the same value once commas are dropped, since we never treat a comma
    # as a decimal separator in these documents.
    s = s.replace(",", "")
    if s.count(".") > 1:  # e.g. "1.234.567" — ambiguous, refuse
        return None

    try:
        value = float(s)
    except ValueError:
        return None
    return -value if negative else value


def canonical_magnitude(token: str | None) -> tuple[str | None, float]:
    """Map a magnitude word to (canonical name, multiplier)."""
    if not token:
        return None, 1.0
    key = token.strip().lower().rstrip(".")
    if key in MAGNITUDES:
        canonical = {
            "lakhs": "lakh",
            "lac": "lakh",
            "crores": "crore",
            "cr": "crore",
            "mn": "million",
            "mm": "million",
            "bn": "billion",
            "tn": "trillion",
        }.get(key, key)
        return canonical, MAGNITUDES[key]
    return None, 1.0


def canonical_unit(token: str | None) -> str | None:
    """Map a unit token to a canonical unit name."""
    if not token:
        return None
    key = token.strip().lower()
    if key in CURRENCIES:
        return CURRENCIES[key]
    if key in {"percent", "per cent", "%", "pct"}:
        return "percent"
    if key == "bps":
        return "bps"
    if key in {"inr", "usd", "eur"}:
        return key.upper()
    return key or None


def normalize_quantity(
    raw: str | None,
    value: float | None,
    magnitude: str | None,
    unit: str | None,
) -> tuple[float | None, str | None]:
    """Reduce a stated quantity to (canonical_value, canonical_unit).

    Currency amounts collapse to the base currency unit, so ₹8,142 crore and
    ₹81,415.38 million both become ~8.14e10 INR and can be compared directly.
    Percentages stay percentages; bps become percent so 50 bps == 0.5%.
    """
    if value is None:
        value = parse_number(raw or "")
    if value is None:
        return None, None

    _, multiplier = canonical_magnitude(magnitude)
    unit_c = canonical_unit(unit)

    if unit_c == "bps":
        return value / 100.0, "percent"
    if unit_c == "percent":
        # A percentage with a magnitude word would be nonsense; ignore it.
        return value, "percent"
    if unit_c in {"INR", "USD", "EUR"}:
        return value * multiplier, unit_c

    # Dimensionless counts (gateways, customers, employees) still scale.
    return value * multiplier, unit_c or "count"


def values_agree(a: float, b: float, *, rel_tol: float = 0.005, abs_tol: float = 1e-9) -> bool:
    """Do two canonical values agree once rounding is allowed for?

    Default tolerance is 0.5%, which absorbs the rounding in "₹8,142 Cr" vs
    "₹81,415.38 million" (a 0.006% gap) without waving through a real
    disagreement like 6.4% vs 7.8% growth.
    """
    if a == b:
        return True
    scale = max(abs(a), abs(b))
    if scale == 0:
        return abs(a - b) <= abs_tol
    return abs(a - b) / scale <= rel_tol


def relative_gap(a: float, b: float) -> float:
    """Relative difference, used to describe how far apart two facts are."""
    scale = max(abs(a), abs(b))
    return 0.0 if scale == 0 else abs(a - b) / scale
