"""Resumable JEV extraction, with a development-only pilot and spend cap."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from coffee_value.extraction.cache import cache_key, digest, load_record, record_path, save_record
from coffee_value.extraction.catalog import BASE, Catalog, selected
from coffee_value.extraction.client import INPUT_USD_PER_TOKEN, evaluate
from coffee_value.extraction.encoding import validate
from coffee_value.extraction.state import ADAPTER_VERSION, build_review_state


def contract_args(catalog: Catalog) -> dict:
    return {"contract_version": catalog.version, "questions_hash": catalog.questions_hash,
            "model": catalog.model}


def validation_args(catalog: Catalog) -> dict:
    return {**contract_args(catalog), "questions": catalog.questions}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def eligible_ids(task_scope: str = "both") -> tuple[set[str], set[str]]:
    split = ROOT / "data" / "splits"
    tasks = ("rating", "price") if task_scope == "both" else (task_scope,)
    valid = {r["row_id"] for task in tasks for r in read_csv(split / f"{task}_validation.csv")}
    all_ids = valid | {r["row_id"] for task in tasks for r in read_csv(split / f"{task}_train.csv")}
    return all_ids, valid


def estimate_incremental_tokens(pending: list[dict[str, str]], catalog: Catalog) -> float | None:
    """Conservative payload-ratio forecast from already billed base records."""
    if catalog.name == "base" or not pending:
        return None
    sample = pending[::max(1, len(pending) // 300)]
    estimates = []
    for row in sample:
        state = build_review_state(row)
        base_record = load_record(record_path(ROOT / "artifacts" / "jev" / "cache", state))
        if not base_record or base_record.get("status") != "complete":
            continue
        actual_tokens = base_record.get("usage", {}).get("input_tokens")
        if not isinstance(actual_tokens, int) or actual_tokens <= 0:
            continue
        original_bytes = len(json.dumps({"state": state, "model": BASE.model, "questions": BASE.questions},
                                        separators=(",", ":")).encode())
        followup_bytes = len(json.dumps({"state": state, "model": catalog.model, "questions": catalog.questions},
                                        separators=(",", ":")).encode())
        estimates.append(actual_tokens * followup_bytes / original_bytes)
    return 1.25 * statistics.mean(estimates) if estimates else None


def extract_one(row: dict[str, str], cache_dir: Path, catalog: Catalog = BASE) -> dict:
    state = build_review_state(row)
    path = record_path(cache_dir, state, **contract_args(catalog))
    prior = load_record(path)
    if prior and prior.get("status") == "complete":
        validate(prior, **validation_args(catalog))
        return {"row_id": row["row_id"], "status": "complete", "cache_hit": True,
                "contract_version": catalog.version, "question_hash": catalog.questions_hash,
                "cache_key": cache_key(state, **contract_args(catalog)),
                "input_tokens": 0, "latency_seconds": 0, "cost_usd": 0}
    try:
        result, elapsed, attempts = evaluate(state, questions=catalog.questions, model=catalog.model)
        usage = result.get("usage", {})
        tokens = usage.get("input_tokens")
        if not isinstance(tokens, int) or tokens < 0:
            raise ValueError("provider omitted input token usage")
        record = {
            "status": "complete", "contract_version": catalog.version,
            "adapter_version": ADAPTER_VERSION, "question_hash": catalog.questions_hash,
            "requested_model": catalog.model, "resolved_model": result.get("model"),
            "state_hash": digest(state), "cache_key": cache_key(state, **contract_args(catalog)),
            "answers": result.get("answers"), "usage": usage,
            "latency_seconds": elapsed, "attempts": attempts,
        }
        validate(record, **validation_args(catalog))
        save_record(path, record)
        return {"row_id": row["row_id"], "status": "complete", "cache_hit": False,
                "contract_version": catalog.version, "question_hash": catalog.questions_hash,
                "cache_key": cache_key(state, **contract_args(catalog)), "input_tokens": tokens,
                "latency_seconds": elapsed, "cost_usd": tokens * INPUT_USD_PER_TOKEN}
    except Exception as exc:
        # Failed rows remain visible and retryable. Never encode failure as not_stated.
        return {"row_id": row["row_id"], "status": "failed", "cache_hit": False,
                "contract_version": catalog.version, "question_hash": catalog.questions_hash,
                "cache_key": cache_key(state, **contract_args(catalog)), "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("pilot", "full"))
    parser.add_argument("--pilot-size", type=int, default=40)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-cost-usd", type=float, default=5.0)
    parser.add_argument("--catalog", choices=("base", "followup", "auction"), default="base")
    parser.add_argument("--task", choices=("both", "rating", "price"), default="both")
    parser.add_argument("--pilot-ids-file", type=Path, help="newline-delimited development row IDs for a targeted pilot")
    parser.add_argument("--estimate-only", action="store_true", help="show pending rows and estimated cost without API calls")
    args = parser.parse_args()
    catalog = selected(args.catalog)
    output = ROOT / "artifacts" / ("jev" if catalog.name == "base" else f"jev_{catalog.name}")
    if not args.estimate_only and not os.getenv("TYPESAFE_API_KEY"):
        raise SystemExit("TYPESAFE_API_KEY is missing; no live requests were made")
    if args.workers < 1 or args.workers > 8:
        raise SystemExit("workers must be 1 to 8")
    all_ids, validation_ids = eligible_ids(args.task)
    rows = [{**r, "row_id": str(i)} for i, r in enumerate(read_csv(ROOT / "data" / "coffee.csv")) if str(i) in all_ids]
    if args.mode == "pilot":
        if args.pilot_ids_file:
            requested = [line.strip() for line in args.pilot_ids_file.read_text().splitlines() if line.strip()]
            if not requested or len(requested) != len(set(requested)):
                raise SystemExit("pilot IDs file must contain unique, nonempty row IDs")
            if not set(requested) <= (all_ids - validation_ids):
                raise SystemExit("targeted pilot contains an ineligible or validation row ID")
            by_id = {row["row_id"]: row for row in rows}
            rows = [by_id[rid] for rid in requested]
        else:
            rows = sorted((r for r in rows if r["row_id"] not in validation_ids),
                          key=lambda r: digest(r["row_id"]))[:args.pilot_size]
    elif args.pilot_ids_file:
        raise SystemExit("--pilot-ids-file is valid only in pilot mode")
    cache_dir = output / "cache"
    status_path = output / f"{args.mode}_status.jsonl"
    if not args.estimate_only:
        output.mkdir(parents=True, exist_ok=True)
    prior = {r["row_id"]: r for r in (json.loads(line) for line in status_path.read_text().splitlines())
             if r.get("contract_version") == catalog.version and r.get("question_hash") == catalog.questions_hash} if status_path.exists() else {}
    current_ids = {r["row_id"] for r in rows}
    prior = {rid: item for rid, item in prior.items() if rid in current_ids}
    pending = []
    for row in rows:
        state = build_review_state(row)
        record = load_record(record_path(cache_dir, state, **contract_args(catalog)))
        if record and record.get("status") == "complete":
            validate(record, **validation_args(catalog))
            prior[row["row_id"]] = {"row_id": row["row_id"], "status": "complete", "cache_hit": True,
                                     "contract_version": catalog.version, "question_hash": catalog.questions_hash,
                                     "cache_key": cache_key(state, **contract_args(catalog)),
                                     "input_tokens": 0, "latency_seconds": 0, "cost_usd": 0}
        else:
            pending.append(row)
    # Conservative preflight from observed pilot tokens; first run uses 10k/request.
    observed = [r["input_tokens"] for r in prior.values() if r.get("input_tokens", 0) > 0]
    for line in (output / "pilot_status.jsonl").read_text().splitlines() if (output / "pilot_status.jsonl").exists() else []:
        sample = json.loads(line)
        if (sample.get("contract_version") == catalog.version and
                sample.get("question_hash") == catalog.questions_hash and sample.get("input_tokens", 0) > 0):
            observed.append(sample["input_tokens"])
    incremental_estimate = estimate_incremental_tokens(pending, catalog) if not observed else None
    estimate = (sum(observed) / len(observed)) if observed else (incremental_estimate or 10000)
    estimate_basis = ("observed same-catalog calls" if observed else
                      "prior paid calls scaled by request bytes with 25% margin" if incremental_estimate else
                      "conservative 10000-token default")
    projected = len(pending) * estimate * INPUT_USD_PER_TOKEN
    if not args.estimate_only and projected > args.max_cost_usd:
        raise SystemExit(f"Projected uncached cost ${projected:.2f} exceeds cap ${args.max_cost_usd:.2f}; {len(pending)} rows pending")
    print(json.dumps({"mode": args.mode, "catalog": catalog.name, "task": args.task,
                      "eligible": len(rows), "pending": len(pending),
                      "questions": len(catalog.questions), "estimated_tokens_per_request": round(estimate),
                      "projected_cost_usd": round(projected, 4),
                      "estimate_basis": estimate_basis}), flush=True)
    if args.estimate_only:
        return
    spent = 0.0
    with ThreadPoolExecutor(max_workers=args.workers) as pool, status_path.open("a", encoding="utf-8") as log:
        # Keep only one small concurrent wave in flight so the spend cap has force.
        for start in range(0, len(pending), args.workers):
            if spent + args.workers * estimate * INPUT_USD_PER_TOKEN > args.max_cost_usd:
                print("spend cap reached; resume with a larger cap after review", flush=True)
                break
            futures = [pool.submit(extract_one, row, cache_dir, catalog) for row in pending[start:start + args.workers]]
            for future in as_completed(futures):
                item = future.result()
                spent += item.get("cost_usd", 0)
                log.write(json.dumps(item) + "\n")
                log.flush()
                prior[item["row_id"]] = item
            wave = [future.result() for future in futures]
            if any("HTTP 401" in item.get("error", "") or "HTTP 422" in item.get("error", "") for item in wave):
                print("authentication or request contract rejected; stopping", flush=True)
                break
            if wave and all(item["status"] == "failed" for item in wave):
                print("all requests in wave failed; stopping", flush=True)
                break
            seen = [r["input_tokens"] for r in prior.values() if r.get("input_tokens", 0) > 0]
            if seen:
                estimate = sum(seen) / len(seen)
            if start % 200 == 0:
                print(json.dumps({"done": len(prior), "failed": sum(r["status"] == "failed" for r in prior.values()),
                                  "spent_usd": round(spent, 4)}), flush=True)
    # Recompute current coverage from contract-keyed cache, not historical log lines.
    complete = sum(bool((record := load_record(record_path(cache_dir, build_review_state(row),
                                                            **contract_args(catalog)))) and record.get("status") == "complete") for row in rows)
    summary = {"mode": args.mode, "eligible": len(rows), "complete": complete,
               "failed_or_pending": len(rows) - complete, "successful_response_cost_estimate_this_run_usd": spent,
               "measured_tokens_per_request": estimate if observed or spent else None,
               "cache_dir": str(cache_dir), "contract_version": catalog.version, "model": catalog.model,
               "cost_note": "Estimate from returned input token usage; failed requests and retries may incur charges."}
    (output / f"{args.mode}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
