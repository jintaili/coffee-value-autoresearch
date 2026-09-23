"""Compare complete JEV and OpenAI extraction on the same freshly fetched pages."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT.parent / "coffee-value-app"
sys.path.insert(0, str(APP / "src"))

from coffee_value_app.analysis import build_page_context
from coffee_value_app.config import load_settings
from coffee_value_app.extractor import OpenAILLMExtractor
from coffee_value_app.fetcher import fetch_product_page
from coffee_value_app.jev_extractor import JevExtractor

URLS = (
    "https://hydrangea.coffee/products/salma-bermudez",
    "https://hydrangea.coffee/products/gesha-washed-crd-elida-estate-centro-loma-lot",
    "https://hydrangea.coffee/products/sub",
    "https://www.blackwhiteroasters.com/products/the-original-1",
    "https://www.blackwhiteroasters.com/products/r-the-new-school",
    "https://onyxcoffeelab.com/products/geometry",
    "https://onyxcoffeelab.com/products/roasters-choice",
    "https://onyxcoffeelab.com/products/kenya-kamunyaka-aa",
)


def load_key() -> None:
    if "TYPESAFE_API_KEY" not in os.environ:
        os.environ["TYPESAFE_API_KEY"] = subprocess.check_output(
            ["security", "find-generic-password", "-s", "typesafe-ai", "-w"], text=True
        ).strip()


def summary(rows: list[dict], provider: str) -> dict:
    good = [row for row in rows if row["provider"] == provider and row["status"] == "ok"]
    times = sorted(row["seconds"] for row in good)
    return {"successful_calls": len(good), "median_seconds": statistics.median(times) if times else None,
            "p95_seconds": times[max(0, int(.95 * len(times) + .999999) - 1)] if times else None,
            "coffee_pages_with_price": sum(row["page_type"] == "coffee_product" and row["listed_price"] is not None
                                           for row in good)}


async def run(rounds: int) -> dict:
    load_key()
    settings = load_settings()
    if not settings.openai_api_key:
        raise RuntimeError("The app .env needs OPENAI_API_KEY for the historical extractor comparison")
    extractors = {"openai": OpenAILLMExtractor(settings=settings), "jev": JevExtractor()}
    pages = []
    for url in URLS:
        page = await fetch_product_page(url)
        pages.append((url, page.final_url, page.text, build_page_context(page.text, page.final_url)))
    rng = random.Random(20260923)
    rows = []
    for round_no in range(1, rounds + 1):
        for url, final_url, html, context in pages:
            providers = list(extractors)
            rng.shuffle(providers)
            for provider in providers:
                start = time.perf_counter()
                try:
                    if provider == "jev":
                        page = (await extractors[provider].extract(url=final_url, page_text=context, html=html)).page
                    else:
                        page = await extractors[provider].extract(url=final_url, page_text=context)
                    row = {"round": round_no, "url": url, "provider": provider, "status": "ok",
                           "seconds": round(time.perf_counter() - start, 3), "page_type": page.page_type,
                           "listed_price": page.price.listed_price, "package_grams": page.price.package_grams,
                           "origin_country": page.coffee.origin_country, "quality": page.quality.extraction_quality}
                except Exception as exc:
                    row = {"round": round_no, "url": url, "provider": provider, "status": "error",
                           "seconds": round(time.perf_counter() - start, 3), "error": type(exc).__name__}
                rows.append(row)
                print(json.dumps(row), flush=True)
    return {"scope": "extraction only; fetch and prediction excluded; same HTML and page context per pair",
            "pages": len(pages), "rounds": rounds, "openai": summary(rows, "openai"),
            "jev": summary(rows, "jev"), "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()
    result = asyncio.run(run(args.rounds))
    path = ROOT / "artifacts" / "jev_serving_benchmark" / "complete_extraction.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"summary": {key: result[key] for key in ("scope", "pages", "rounds", "openai", "jev")},
                      "saved": str(path)}))


if __name__ == "__main__":
    main()
