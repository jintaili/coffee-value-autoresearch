"""Allowlisted review fields and removal of explicit target quotations."""

from __future__ import annotations

import re

ADAPTER_VERSION = "review-1"
FIELDS = ("bean", "location", "origin", "blind_assessment", "notes", "bottom_line")

# The original rating, component scores, est_price and price-derived columns are
# never read. This pattern handles quoted values embedded in otherwise useful prose.
TARGET_QUOTES = [
    re.compile(r"\((?:8\d|9\d|100)\)(?!\s*(?:hours?\b|days?\b|meters?\b|masl\b))", re.I),
    re.compile(r"\b(?:rating|rated|score|scored|points?)\s*(?::|of|is|at|=)?\s*(?:an?\s+)?\d{1,3}(?:\.\d+)?\s*(?:/\s*100|out of\s*100)?\b", re.I),
    re.compile(r"\b\d{1,3}(?:\.\d+)?\s*(?:/\s*100|out of\s*100|points?)\b", re.I),
    re.compile(r"\b\d{1,3}(?:\.\d+)?[-\s]points?\b", re.I),
    re.compile(r"(?<!\w)(?:US\$|C\$|NT\$|\$|£|€|¥)\s*\d[\d,.]*(?:\s*/\s*\d+(?:g|kg|oz|lb))?", re.I),
    re.compile(r"\b(?:USD|CAD|EUR|GBP|JPY|TWD|NTD|HKD|AUD)\s*\d[\d,.]*\b", re.I),
    re.compile(r"\b\d[\d,.]*\s*(?:USD|CAD|EUR|GBP|JPY|TWD|NTD|HKD|AUD)\b", re.I),
    re.compile(r"\b(?:price|cost|retail|MSRP)\s*(?::|is|of|at|=)?\s*\d[\d,.]*\b", re.I),
]


def scrub_target_quotes(value: str) -> str:
    for pattern in TARGET_QUOTES:
        value = pattern.sub("[target quotation removed]", value)
    return re.sub(r"\s+", " ", value).strip()


def build_review_state(row: dict[str, str]) -> dict[str, str]:
    return {field: scrub_target_quotes(str(row.get(field) or "")) for field in FIELDS}
