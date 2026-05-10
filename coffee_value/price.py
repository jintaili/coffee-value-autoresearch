"""Deterministic price parsing and normalization.

This module defines the shared price target contract for price-model
autoresearch. It is intentionally conservative: rows that are ambiguous,
non-weight-based, or unsupported are marked with explicit statuses instead of
being guessed into the training target.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime


OK = "ok"
MISSING = "missing"
UNSUPPORTED_CURRENCY = "unsupported_currency"
UNSUPPORTED_UNIT = "unsupported_unit"
UNPARSEABLE_AMOUNT = "unparseable_amount"
UNPARSEABLE_PACKAGE = "unparseable_package"
AMBIGUOUS = "ambiguous"


# Static approximate USD conversion rates. These are intentionally a hook, not
# a final historical-FX table. Price autoresearch should use this deterministic
# contract until a committed monthly FX table replaces it.
USD_PER_CURRENCY = {
    "USD": 1.0,
    "CAD": 0.75,
    "TWD": 0.031,
    "HKD": 0.128,
    "CNY": 0.14,
    "THB": 0.028,
    "KRW": 0.00074,
    "GBP": 1.27,
    "JPY": 0.0067,
    "AUD": 0.66,
    "EUR": 1.08,
    "MYR": 0.21,
    "AED": 0.272,
    "IDR": 0.000064,
    "CZK": 0.044,
}


# CPI adjustment hook. A full committed monthly CPI table should replace this
# before treating real-price modeling as final. With only the base entry, real
# and nominal prices are equal and reproducible.
CPI_BY_MONTH = {
    "2025-12": 1.0,
}
REAL_BASE_MONTH = "2025-12"


CURRENCY_PATTERNS = [
    ("TWD", r"\b(?:TWD|NTD|NT)\b|NT\s*\$|\$NT"),
    ("HKD", r"\b(?:HKD|HK)\b"),
    ("CAD", r"\bCAD\b|C\$"),
    ("CNY", r"\b(?:CNY|RMB)\b"),
    ("THB", r"\bTHB\b"),
    ("KRW", r"\bKRW\b"),
    ("GBP", r"\bGBP\b|£"),
    ("JPY", r"\bJPY\b|¥"),
    ("AUD", r"\bAUD\b"),
    ("EUR", r"\bEUR\b|€"),
    ("MYR", r"\bMYR\b|RM\s*\d"),
    ("AED", r"\bAED\b"),
    ("IDR", r"\bIDR\b"),
    ("CZK", r"\bCZK\b|Kč"),
    ("USD", r"\b(?:USD|US\$|US\s*\$)\b|^\s*\$"),
]

KNOWN_UNSUPPORTED_CURRENCY_RE = re.compile(r"\b(?:LAK|GTQ|PESOS?|PHP|CHF|SGD)\b", re.I)


UNIT_ALIASES = {
    "ounce": "oz",
    "ounces": "oz",
    "ouces": "oz",
    "oz": "oz",
    "oz.": "oz",
    "gram": "g",
    "grams": "g",
    "g": "g",
    "g.": "g",
    "kilogram": "kg",
    "kilograms": "kg",
    "kg": "kg",
    "pound": "lb",
    "pounds": "lb",
    "lb": "lb",
    "lbs": "lb",
}

GRAMS_PER_UNIT = {
    "g": 1.0,
    "kg": 1000.0,
    "oz": 28.349523125,
    "lb": 453.59237,
}

UNSUPPORTED_UNIT_RE = re.compile(
    r"\b(?:capsule|capsules|k[-\s]?cups?|pods?|servings?|sachets?|sticks?|bags?|packets?)\b",
    re.I,
)

AMOUNT_RE = re.compile(r"(?<!\d)(\d{1,6}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)(?!\d)")


@dataclass(frozen=True)
class ParsedPrice:
    raw: str
    status: str
    amount: float | None = None
    currency: str | None = None
    package_quantity: float | None = None
    package_unit: str | None = None
    package_grams: float | None = None
    usd_nominal: float | None = None
    usd_per_100g_nominal: float | None = None
    usd_per_100g_real: float | None = None
    real_base_month: str = REAL_BASE_MONTH

    def as_dict(self) -> dict[str, str]:
        def fmt(value: float | None) -> str:
            return "" if value is None else f"{value:.6f}"

        return {
            "price_raw": self.raw,
            "price_parse_status": self.status,
            "price_amount": fmt(self.amount),
            "price_currency": self.currency or "",
            "package_quantity": fmt(self.package_quantity),
            "package_unit": self.package_unit or "",
            "package_grams": fmt(self.package_grams),
            "price_usd_nominal": fmt(self.usd_nominal),
            "price_usd_per_100g_nominal": fmt(self.usd_per_100g_nominal),
            "price_usd_per_100g_real": fmt(self.usd_per_100g_real),
            "price_real_base_month": self.real_base_month,
        }


def normalize_space(value: str | None) -> str:
    return re.sub(r"\s+", " ", (value or "").replace("\ufeff", " ")).strip()


def review_month(review_date: str | None) -> str | None:
    raw = normalize_space(review_date)
    if not raw:
        return None
    for fmt in ("%Y-%m-%d", "%B %Y"):
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m")
        except ValueError:
            pass
    match = re.search(r"\b((?:19|20)\d{2})[-/](\d{1,2})\b", raw)
    if match:
        return f"{match.group(1)}-{int(match.group(2)):02d}"
    year = re.search(r"\b((?:19|20)\d{2})\b", raw)
    return f"{year.group(1)}-01" if year else None


def parse_currency(raw: str) -> str | None:
    if KNOWN_UNSUPPORTED_CURRENCY_RE.search(raw):
        return None
    for currency, pattern in CURRENCY_PATTERNS:
        if re.search(pattern, raw, re.I):
            return currency
    if re.search(r"\bE\s*\d", raw, re.I):
        return "EUR"
    return None


def parse_amount(raw: str) -> float | None:
    match = AMOUNT_RE.search(raw.replace(",", ""))
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", ""))
    except ValueError:
        return None


def split_package_part(raw: str) -> str | None:
    if "/" not in raw:
        match = re.search(r"\bfor\s+(.+)$", raw, re.I)
        return match.group(1).strip() if match else None
    return raw.split("/", 1)[1].strip()


def parse_package_grams(raw: str) -> tuple[float | None, float | None, str | None, str]:
    package = split_package_part(raw)
    if not package:
        return None, None, None, UNPARSEABLE_PACKAGE
    if UNSUPPORTED_UNIT_RE.search(package):
        return None, None, None, UNSUPPORTED_UNIT

    # Prefer parenthetical metric quantities when they are present in otherwise
    # mixed strings like "250 grams (8.8 oz.)".
    candidates = re.findall(
        r"(\d+(?:\.\d+)?)\s*(kilograms?|kg|grams?|g\.?|ounces?|ouces|oz\.?|pounds?|lbs?|lb)\b",
        package,
        flags=re.I,
    )
    if not candidates:
        bare_unit = re.search(r"^\s*(kilogram|kg|pound|lb)\s*$", package, re.I)
        if bare_unit:
            candidates = [("1", bare_unit.group(1))]
        else:
            return None, None, None, UNSUPPORTED_UNIT

    quantity_s, unit_raw = candidates[0]
    unit = UNIT_ALIASES.get(unit_raw.lower().rstrip("."))
    if unit is None:
        return None, None, None, UNSUPPORTED_UNIT
    quantity = float(quantity_s)
    grams = quantity * GRAMS_PER_UNIT[unit]
    if grams <= 0:
        return None, None, None, UNPARSEABLE_PACKAGE
    return grams, quantity, unit, OK


def cpi_multiplier(month: str | None, base_month: str = REAL_BASE_MONTH) -> float:
    if not month:
        return 1.0
    review_cpi = CPI_BY_MONTH.get(month)
    base_cpi = CPI_BY_MONTH.get(base_month)
    if not review_cpi or not base_cpi:
        return 1.0
    return base_cpi / review_cpi


def parse_est_price(raw_price: str | None, review_date: str | None = None) -> ParsedPrice:
    raw = normalize_space(raw_price)
    if not raw or raw.upper() == "NA":
        return ParsedPrice(raw=raw, status=MISSING)
    lowered = raw.lower()
    if any(token in lowered for token in ("not available", "see website", "varies", "contact")):
        return ParsedPrice(raw=raw, status=MISSING)

    currency = parse_currency(raw)
    if currency is None or currency not in USD_PER_CURRENCY:
        return ParsedPrice(raw=raw, status=UNSUPPORTED_CURRENCY)

    amount = parse_amount(raw)
    if amount is None:
        return ParsedPrice(raw=raw, status=UNPARSEABLE_AMOUNT, currency=currency)

    package_grams, package_quantity, package_unit, package_status = parse_package_grams(raw)
    if package_status != OK:
        return ParsedPrice(raw=raw, status=package_status, amount=amount, currency=currency)

    usd_nominal = amount * USD_PER_CURRENCY[currency]
    usd_per_100g_nominal = usd_nominal / package_grams * 100.0
    real = usd_per_100g_nominal * cpi_multiplier(review_month(review_date))

    return ParsedPrice(
        raw=raw,
        status=OK,
        amount=amount,
        currency=currency,
        package_quantity=package_quantity,
        package_unit=package_unit,
        package_grams=package_grams,
        usd_nominal=usd_nominal,
        usd_per_100g_nominal=usd_per_100g_nominal,
        usd_per_100g_real=real,
    )


def normalize_price_row(row: dict[str, str]) -> ParsedPrice:
    return parse_est_price(row.get("est_price"), row.get("review_date"))
