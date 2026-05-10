"""Fetch deterministic FX and CPI reference snapshots for price normalization.

Outputs are committed small CSVs used by `coffee_value.price`. Training and
autoresearch should read those local files instead of making live network calls.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "coffee.csv"
REFERENCE_DIR = ROOT / "coffee_value" / "reference"
FX_OUT = REFERENCE_DIR / "fx_usd_monthly.csv"
CPI_OUT = REFERENCE_DIR / "cpi_us_monthly.csv"

FRED_CPI_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL"
FRANKFURTER_URL = "https://api.frankfurter.dev/v2/rates"


def review_month(raw: str) -> str | None:
    raw = (raw or "").strip()
    if not raw:
        return None
    for fmt in ("%Y-%m-%d", "%B %Y"):
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m")
        except ValueError:
            pass
    return None


def read_needed_months_and_currencies() -> tuple[list[str], list[str]]:
    months = set()
    currencies = set()
    if not SOURCE.exists():
        raise SystemExit(f"missing {SOURCE}; place coffee.csv before fetching references")
    sys.path.insert(0, str(ROOT))
    from coffee_value.price import parse_currency

    with SOURCE.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            month = review_month(row.get("review_date", ""))
            currency = parse_currency(row.get("est_price", ""))
            if month:
                months.add(month)
            if currency and currency != "USD":
                currencies.add(currency)
    return sorted(months), sorted(currencies)


def fetch_url(url: str) -> bytes:
    try:
        return subprocess.check_output(
            ["curl", "-L", "--fail", "--silent", "--show-error", url],
            timeout=60,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        request = urllib.request.Request(url, headers={"User-Agent": "coffee-value-autoresearch/0.1"})
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.read()


def fetch_cpi() -> list[dict[str, str]]:
    raw = fetch_url(FRED_CPI_URL).decode("utf-8")
    rows = []
    for row in csv.DictReader(raw.splitlines()):
        value = row.get("CPIAUCSL", "").strip()
        if not value or value == ".":
            continue
        month = row["observation_date"][:7]
        rows.append({"month": month, "cpi": value, "source": "FRED:CPIAUCSL"})
    return rows


def fetch_fx_for_currency(currency: str, start_month: str, end_month: str) -> list[dict[str, str]]:
    params = urllib.parse.urlencode(
        {
            "from": f"{start_month}-01",
            "to": f"{end_month}-28",
            "base": currency,
            "quotes": "USD",
            "group": "month",
        }
    )
    url = f"{FRANKFURTER_URL}?{params}"
    data = json.loads(fetch_url(url).decode("utf-8"))
    rows = []
    if not isinstance(data, list):
        raise RuntimeError(f"unexpected Frankfurter response for {currency}: {data!r}")
    for item in data:
        if item.get("quote") != "USD":
            continue
        rows.append(
            {
                "month": item["date"][:7],
                "currency": currency,
                "usd_per_currency": f"{float(item['rate']):.10f}",
                "source": "frankfurter.dev",
            }
        )
    return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    months, currencies = read_needed_months_and_currencies()
    if not months:
        raise SystemExit("no review months found")

    cpi_rows = fetch_cpi()
    write_csv(CPI_OUT, ["month", "cpi", "source"], cpi_rows)
    print(f"wrote {CPI_OUT.relative_to(ROOT)} rows={len(cpi_rows)}")

    fx_by_key = {}
    start_month, end_month = months[0], months[-1]
    for currency in currencies:
        try:
            rows = fetch_fx_for_currency(currency, start_month, end_month)
        except Exception as exc:
            print(f"warning: failed to fetch {currency}: {exc}", file=sys.stderr)
            continue
        for row in rows:
            fx_by_key[(row["month"], row["currency"])] = row
        print(f"fetched {currency} rows={len(rows)}")
        time.sleep(0.2)
    fx_rows = sorted(fx_by_key.values(), key=lambda row: (row["month"], row["currency"]))
    write_csv(FX_OUT, ["month", "currency", "usd_per_currency", "source"], fx_rows)
    print(f"wrote {FX_OUT.relative_to(ROOT)} rows={len(fx_rows)}")


if __name__ == "__main__":
    main()
