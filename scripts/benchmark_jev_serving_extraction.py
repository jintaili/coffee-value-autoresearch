"""Paired, uncached extraction pilot on frozen app product-page contexts.

Run with the coffee-value-app virtualenv. This benchmark compares the current
full app extraction call with the shared A+B JEV semantic questions; it does not
claim equivalent response completeness or end-to-end appraisal latency.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT.parent / "coffee-value-app"
sys.path[:0] = [str(ROOT), str(APP / "src")]

from coffee_value.extraction.client import INPUT_USD_PER_TOKEN, evaluate
from coffee_value.extraction.questions import MODEL, QUESTIONS
from coffee_value_app.analysis import build_page_context
from coffee_value_app.config import load_settings
from coffee_value_app.extractor import OpenAILLMExtractor, trim_page_text
from coffee_value_app.fetcher import fetch_product_page
from coffee_value_app.schemas import to_model_input

OUT = ROOT / "artifacts" / "jev_serving_benchmark"
SNAPSHOTS = OUT / "snapshots.json"
EVENTS = OUT / "events.jsonl"
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


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


async def prepare(limit: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    snapshots = json.loads(SNAPSHOTS.read_text()) if SNAPSHOTS.exists() else []
    by_url = {item["url"]: item for item in snapshots}
    for url in URLS[:limit]:
        if url in by_url:
            continue
        try:
            fetched = await fetch_product_page(url)
            context = trim_page_text(build_page_context(fetched.text, fetched.final_url), 60_000)
            title = BeautifulSoup(fetched.text, "html.parser").find("h1")
            snapshot = {"url": url, "final_url": fetched.final_url,
                        "product_name": title.get_text(" ", strip=True) if title else "",
                        "page_context": context, "context_sha256": digest(context),
                        "context_chars": len(context),
                        "captured_utc": datetime.now(timezone.utc).isoformat()}
            by_url[url] = snapshot
            snapshots = [by_url[item] for item in URLS if item in by_url]
            SNAPSHOTS.write_text(json.dumps(snapshots, ensure_ascii=False, indent=2))
            print(json.dumps({"url": url, "status": "captured", "context_chars": len(context)}), flush=True)
        except Exception as exc:
            print(json.dumps({"url": url, "status": "fetch_failed", "error": f"{type(exc).__name__}: {exc}"}), flush=True)


def load_typesafe_key() -> None:
    if os.getenv("TYPESAFE_API_KEY"):
        return
    result = subprocess.run(["security", "find-generic-password", "-s", "typesafe-ai", "-w"],
                            capture_output=True, text=True, check=True)
    key = result.stdout.strip()
    if not key:
        raise RuntimeError("typesafe-ai Keychain entry is empty")
    os.environ["TYPESAFE_API_KEY"] = key


def jev_state(snapshot: dict) -> dict[str, str]:
    return {"bean": snapshot["product_name"], "location": "", "origin": "",
            "blind_assessment": "", "notes": snapshot["page_context"], "bottom_line": ""}


async def run(limit: int, rounds: int) -> None:
    if not SNAPSHOTS.exists():
        raise SystemExit("Run --prepare first")
    snapshots = json.loads(SNAPSHOTS.read_text())[:limit]
    if not snapshots:
        raise SystemExit("No product-page snapshots available")
    load_typesafe_key()
    settings = load_settings()
    if not settings.openai_api_key:
        raise SystemExit("OPENAI_API_KEY is not configured in the companion app")
    extractor = OpenAILLMExtractor(settings=settings)
    prior = {(item["url"], item["round"], item["provider"])
             for line in EVENTS.read_text().splitlines()
             if (item := json.loads(line)).get("status") == "complete"} if EVENTS.exists() else set()
    jobs = [(snapshot, round_no, provider)
            for round_no in range(1, rounds + 1)
            for snapshot in snapshots
            for provider in ("openai_app", "jev_ab")
            if (snapshot["url"], round_no, provider) not in prior]
    random.Random(20260922).shuffle(jobs)
    OUT.mkdir(parents=True, exist_ok=True)
    with EVENTS.open("a") as stream:
        for snapshot, round_no, provider in jobs:
            start = time.perf_counter()
            item = {"url": snapshot["url"], "round": round_no, "provider": provider,
                    "context_sha256": snapshot["context_sha256"],
                    "context_chars": snapshot["context_chars"], "cache_hit": False,
                    "started_utc": datetime.now(timezone.utc).isoformat()}
            try:
                if provider == "openai_app":
                    extraction = await extractor.extract(url=snapshot["final_url"],
                                                         page_text=snapshot["page_context"])
                    coffee = extraction.coffee
                    model_input = to_model_input(coffee, extraction.price)
                    item.update({"status": "complete", "model": extractor.model,
                                 "page_type": extraction.page_type,
                                 "semantic_fields": {
                                     "origin_country": model_input.origin_country,
                                     "process_method": model_input.process_method,
                                     "variety": model_input.variety,
                                     "is_blend": model_input.is_blend,
                                     "is_decaf": model_input.is_decaf,
                                     "producer_or_farm_present": model_input.producer_or_farm_present,
                                     "altitude_present": model_input.altitude_present},
                                 "has_display_notes": bool(coffee.display_tasting_notes),
                                 "has_price": extraction.price.listed_price is not None,
                                 "has_package_grams": extraction.price.package_grams is not None})
                else:
                    result, _, attempts = await asyncio.to_thread(
                        evaluate, jev_state(snapshot), questions=QUESTIONS, model=MODEL)
                    answers = result.get("answers")
                    if result.get("model") != MODEL or not isinstance(answers, dict) or set(answers) != set(QUESTIONS):
                        raise ValueError("JEV returned an incomplete or different question catalog")
                    usage = result.get("usage", {})
                    item.update({"status": "complete", "model": result["model"],
                                 "question_count": len(answers), "attempts": attempts,
                                 "input_tokens": usage.get("input_tokens"),
                                 "estimated_input_cost_usd":
                                     usage.get("input_tokens", 0) * INPUT_USD_PER_TOKEN,
                                 "semantic_fields": {key: answers[key]["choice"] for key in (
                                     "origin_country", "process_washed", "process_natural",
                                     "variety_gesha", "blend", "decaf", "producer_identified",
                                     "altitude_provided")}})
            except Exception as exc:
                item.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            item["latency_seconds"] = time.perf_counter() - start
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
            stream.flush()
            print(json.dumps({"url": snapshot["url"], "round": round_no,
                              "provider": provider, "status": item["status"],
                              "latency_seconds": round(item["latency_seconds"], 3),
                              **({"error": item["error"]} if item["status"] == "failed" else {})}), flush=True)


def summary(limit: int, rounds: int) -> None:
    snapshots = json.loads(SNAPSHOTS.read_text())[:limit]
    keys = {(snapshot["url"], round_no) for snapshot in snapshots for round_no in range(1, rounds + 1)}
    events = [json.loads(line) for line in EVENTS.read_text().splitlines()] if EVENTS.exists() else []
    latest = {(item["url"], item["round"], item["provider"]): item for item in events
              if (item["url"], item["round"]) in keys}
    result = {"snapshot_count": len(snapshots), "planned_pairs": len(keys), "contexts": [
        {key: snapshot[key] for key in ("url", "final_url", "product_name", "context_sha256",
                                              "context_chars", "captured_utc")} for snapshot in snapshots],
        "providers": {}, "paired": {},
        "scope": "current full app extraction call versus A+B JEV semantic model-feature call",
        "preflight_failures_before_client_fix": sum(item["status"] == "failed" for item in events)}
    for provider in ("openai_app", "jev_ab"):
        observations = [item for (url, round_no, name), item in latest.items() if name == provider]
        success = sorted(item["latency_seconds"] for item in observations if item["status"] == "complete")
        result["providers"][provider] = {
            "complete": len(success), "failed": sum(item["status"] == "failed" for item in observations),
            "p50_seconds": statistics.median(success) if success else None,
            "p95_seconds": success[math.ceil(.95 * len(success)) - 1] if success else None,
            "estimated_input_cost_usd": (sum(item.get("estimated_input_cost_usd", 0)
                                             for item in observations) if provider == "jev_ab" else None),
        }
    pairs = [(latest[(url, round_no, "openai_app")], latest[(url, round_no, "jev_ab")])
             for url, round_no in keys if (url, round_no, "openai_app") in latest and
             (url, round_no, "jev_ab") in latest and
             latest[(url, round_no, "openai_app")]["status"] == "complete" and
             latest[(url, round_no, "jev_ab")]["status"] == "complete"]
    if pairs:
        faster = sum(jev["latency_seconds"] < old["latency_seconds"] for old, jev in pairs)
        ratio = [jev["latency_seconds"] / old["latency_seconds"] for old, jev in pairs]
        result["paired"] = {"complete_pairs": len(pairs), "jev_faster_pairs": faster,
                            "median_per_pair_latency_ratio": statistics.median(ratio),
                            "median_per_pair_reduction_percent": 100 * (1 - statistics.median(ratio))}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result["providers"] | {"paired": result["paired"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "run", "summary"))
    parser.add_argument("--limit", type=int, default=len(URLS))
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.limit <= len(URLS) or not 1 <= args.rounds <= 3:
        raise SystemExit("limit must be 1–8 and rounds 1–3")
    if args.mode == "prepare":
        asyncio.run(prepare(args.limit))
    elif args.mode == "run":
        asyncio.run(run(args.limit, args.rounds))
    else:
        summary(args.limit, args.rounds)


if __name__ == "__main__":
    main()
