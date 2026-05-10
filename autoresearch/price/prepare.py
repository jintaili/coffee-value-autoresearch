"""Prepare canonical parsed-price rows for price autoresearch."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from coffee_value.features import extract_rows, feature_fieldnames
from coffee_value.price import OK, normalize_price_row


SOURCE = ROOT / "data" / "coffee.csv"
MODELING_PRICE = ROOT / "data" / "modeling_price.csv"
ARTIFACT_DIR = ROOT / "artifacts" / "price"
PARSE_REPORT = ARTIFACT_DIR / "parse_report.json"


PRICE_FIELDNAMES = [
    "price_raw",
    "price_parse_status",
    "price_amount",
    "price_currency",
    "package_quantity",
    "package_unit",
    "package_grams",
    "price_usd_nominal",
    "price_usd_per_100g_nominal",
    "price_usd_per_100g_real",
    "price_real_base_month",
]


def read_source_rows() -> list[dict[str, str]]:
    with SOURCE.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_modeling_price(rows: list[dict[str, str]]) -> None:
    MODELING_PRICE.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = feature_fieldnames() + PRICE_FIELDNAMES
    with MODELING_PRICE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    out = {}
    for q in [0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]:
        idx = round(q * (len(ordered) - 1))
        out[f"p{int(q * 100):02d}"] = ordered[idx]
    return out


def sample_rows(rows: list[dict[str, str]], status: str, n: int = 12) -> list[dict[str, str]]:
    out = []
    for row in rows:
        if row["price_parse_status"] != status:
            continue
        out.append(
            {
                "row_id": row["row_id"],
                "coffee_name": row["coffee_name"],
                "roaster": row["roaster"],
                "review_date": row["review_date"],
                "price_raw": row["price_raw"],
                "price_parse_status": row["price_parse_status"],
            }
        )
        if len(out) >= n:
            break
    return out


def build_report(rows: list[dict[str, str]]) -> dict[str, object]:
    statuses = Counter(row["price_parse_status"] for row in rows)
    currencies = Counter(row["price_currency"] for row in rows if row["price_parse_status"] == OK)
    units = Counter(row["package_unit"] for row in rows if row["price_parse_status"] == OK)
    prices = [
        float(row["price_usd_per_100g_real"])
        for row in rows
        if row["price_parse_status"] == OK and row["price_usd_per_100g_real"]
    ]
    status_examples = {status: sample_rows(rows, status) for status in sorted(statuses)}
    high_outliers = sorted(
        [
            {
                "row_id": row["row_id"],
                "coffee_name": row["coffee_name"],
                "roaster": row["roaster"],
                "review_date": row["review_date"],
                "price_raw": row["price_raw"],
                "price_usd_per_100g_real": float(row["price_usd_per_100g_real"]),
            }
            for row in rows
            if row["price_parse_status"] == OK
            and row["price_usd_per_100g_real"]
            and float(row["price_usd_per_100g_real"]) > 300
        ],
        key=lambda row: row["price_usd_per_100g_real"],
        reverse=True,
    )

    return {
        "source": str(SOURCE.relative_to(ROOT)),
        "modeling_price": str(MODELING_PRICE.relative_to(ROOT)),
        "rows": len(rows),
        "ok_rows": statuses.get(OK, 0),
        "status_counts": dict(statuses.most_common()),
        "currency_counts_ok": dict(currencies.most_common()),
        "unit_counts_ok": dict(units.most_common()),
        "price_usd_per_100g_real_quantiles": quantiles(prices),
        "status_examples": status_examples,
        "high_outliers_over_300_usd_per_100g": high_outliers,
    }


def main() -> None:
    source_rows = read_source_rows()
    feature_rows = extract_rows(SOURCE)
    out_rows = []
    for source, features in zip(source_rows, feature_rows):
        price = normalize_price_row(source)
        out_rows.append({**features.__dict__, **price.as_dict()})

    write_modeling_price(out_rows)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    report = build_report(out_rows)
    PARSE_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"wrote {MODELING_PRICE.relative_to(ROOT)} rows={len(out_rows)}")
    print(f"wrote {PARSE_REPORT.relative_to(ROOT)}")
    print(json.dumps({"ok_rows": report["ok_rows"], "status_counts": report["status_counts"]}, indent=2))


if __name__ == "__main__":
    main()
