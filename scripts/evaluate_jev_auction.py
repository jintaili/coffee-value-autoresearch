"""Price-only comparison of an exact-lot auction signal and Panama × Gesha."""

from __future__ import annotations

import csv
import html
import json
import math
import pickle
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from autoresearch.price import train as price
from coffee_value.extraction.cache import load_record, record_path
from coffee_value.extraction.catalog import AUCTION
from coffee_value.extraction.encoding import encode, feature_names
from coffee_value.extraction.state import build_review_state
from scripts import evaluate_jev as prior

OUT = ROOT / "artifacts" / "jev_auction"
BASE_NAMES = feature_names()
AUCTION_NAMES = feature_names(questions=AUCTION.questions)
PANAMA = BASE_NAMES.index("jev:origin_country:Panama")
GESHA = BASE_NAMES.index("jev:variety_gesha:supported")


def auction_vectors(row_ids: set[str]) -> tuple[dict[str, np.ndarray], dict[str, dict], list[str]]:
    source = prior.rows(ROOT / "data" / "coffee.csv")
    vectors: dict[str, np.ndarray] = {}
    records: dict[str, dict] = {}
    missing = []
    for rid in sorted(row_ids, key=int):
        state = build_review_state(source[int(rid)])
        record = load_record(record_path(OUT / "cache", state, contract_version=AUCTION.version,
                                         questions_hash=AUCTION.questions_hash, model=AUCTION.model))
        if not record or record.get("status") != "complete":
            missing.append(rid)
            continue
        vectors[rid] = np.asarray(encode(record, questions=AUCTION.questions,
                                         contract_version=AUCTION.version,
                                         questions_hash=AUCTION.questions_hash,
                                         model=AUCTION.model), dtype=np.float64)
        records[rid] = record
    return vectors, records, missing


def pg_interaction(base: sparse.csr_matrix) -> sparse.csr_matrix:
    return base[:, PANAMA].multiply(base[:, GESHA]).tocsr()


def saved_bundle_matrix(bundle: dict, items: list[dict], base_vectors: dict[str, np.ndarray],
                        auction: dict[str, np.ndarray]) -> sparse.csr_matrix:
    base = prior.matrix_for(items, base_vectors)
    parts = [base]
    if bundle["use_auction"]:
        parts.append(prior.matrix_for(items, auction))
    if bundle["use_pg_interaction"]:
        parts.append(pg_interaction(base))
    text = bundle["text_encoder"].transform(prior.clean_text_rows(items))[:, bundle["text_column_offset"]:]
    parts.append(text)
    matrix = sparse.hstack(parts, format="csr")
    if matrix.shape[1] != len(bundle["feature_names"]):
        raise ValueError("auction bundle feature width mismatch")
    return matrix


def high_price_summary(truth: np.ndarray, predicted: np.ndarray) -> dict:
    threshold = float(np.quantile(truth, 0.9))
    mask = truth >= threshold
    return {"threshold_usd_per_100g": threshold, "rows": int(mask.sum()),
            "mean_true": float(truth[mask].mean()), "mean_pred": float(predicted[mask].mean()),
            "mean_bias": float(np.mean(predicted[mask] - truth[mask])),
            "rmsle": float(np.sqrt(np.mean((np.log1p(predicted[mask]) - np.log1p(truth[mask])) ** 2)))}


def read_saved_predictions(path: Path, *, baseline: bool = False) -> tuple[np.ndarray, np.ndarray, list[str]]:
    saved = prior.rows(path)
    predicted_field = "prediction_usd_per_100g_real" if baseline else "prediction"
    return (np.asarray([float(row["price_usd_per_100g_real"]) for row in saved]),
            np.asarray([float(row[predicted_field]) for row in saved]),
            [row["row_id"] for row in saved])


def write_predictions(run: str, rows: list[dict], predictions: list[float]) -> None:
    with (OUT / f"price_{run}_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["row_id", "price_usd_per_100g_real", "prediction"])
        writer.writeheader()
        for row, pred in zip(rows, predictions, strict=True):
            writer.writerow({"row_id": row["row_id"], "price_usd_per_100g_real": row["price_usd_per_100g_real"],
                             "prediction": pred})


def paired_intervals(validation: list[dict], predictions: dict[str, np.ndarray],
                     truth: np.ndarray, reps: int = 400) -> dict:
    """Exploratory paired bootstrap by roaster, not a fresh test set."""
    groups: dict[str, list[int]] = {}
    for i, row in enumerate(validation):
        groups.setdefault(row["roaster"], []).append(i)
    group_indices = list(groups.values())
    rng = np.random.default_rng(20260922)
    reference = ("incumbent", "jev_original")
    result = {}
    for candidate in ("jev_auction", "jev_pg", "jev_auction_pg"):
        result[candidate] = {}
        for base in reference:
            deltas = []
            for _ in range(reps):
                selected = rng.integers(0, len(group_indices), size=len(group_indices))
                index = np.concatenate([group_indices[j] for j in selected])
                candidate_error = price.rmsle_from_prices(truth[index], predictions[candidate][index])
                base_error = price.rmsle_from_prices(truth[index], predictions[base][index])
                deltas.append(candidate_error - base_error)
            result[candidate][f"rmsle_delta_vs_{base}_95pct_interval"] = np.quantile(deltas, [0.025, 0.975]).tolist()
    return {"resamples": reps, "unit": "roaster", "intervals": result}


def extraction_summary() -> dict:
    result = {}
    for mode in ("pilot", "full"):
        path = OUT / f"{mode}_status.jsonl"
        events = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        success = [item for item in events if item.get("status") == "complete" and not item.get("cache_hit")]
        times = sorted(item["latency_seconds"] for item in success)
        result[mode] = {"successful_calls": len(success),
                        "failed_events": sum(item.get("status") == "failed" for item in events),
                        "estimated_successful_response_cost_usd": sum(item.get("cost_usd", 0) for item in success),
                        "median_seconds": statistics.median(times) if times else None,
                        "p95_seconds": times[math.ceil(.95 * len(times)) - 1] if times else None}
    result["total_estimated_successful_response_cost_usd"] = sum(
        result[mode]["estimated_successful_response_cost_usd"] for mode in ("pilot", "full"))
    return result


def exact_auction_price_summary(items: list[dict], records: dict[str, dict]) -> dict:
    prices = [float(row["price_usd_per_100g_real"]) for row in items
              if records[row["row_id"]]["answers"]["lot_auction_status"]["choice"] == "exact_lot_auctioned"]
    return {"rows": len(prices), "mean_true_usd_per_100g": float(np.mean(prices)) if prices else None}


def render_html(report: dict) -> None:
    rows = []
    incumbent_rmsle = report.get("runs", {}).get("incumbent", {}).get("validation", {}).get("val_rmsle")
    for run, item in report.get("runs", {}).items():
        metrics = item["validation"]
        high = item["high_price_decile"]
        gain = incumbent_rmsle - metrics["val_rmsle"] if incumbent_rmsle is not None else 0.0
        relative_gain = 100 * gain / incumbent_rmsle if incumbent_rmsle else 0.0
        rows.append(f"<tr><td>{html.escape(run.replace('_', ' '))}</td><td>{metrics['val_rmsle']:.6f}</td>"
                    f"<td>{gain:+.6f}</td><td>{relative_gain:+.2f}%</td>"
                    f"<td>{metrics['val_mae']:.3f}</td><td>{metrics['val_p90_ae']:.3f}</td>"
                    f"<td>{high['mean_pred']:.2f}</td><td>{high['mean_bias']:+.2f}</td>"
                    f"<td>{item['validation_minus_train_rmsle']:+.6f}</td></tr>")
    table = "".join(rows) or "<tr><td colspan='9'>Waiting for complete auction extraction.</td></tr>"
    counts = report.get("choice_counts", {})
    counts_html = ", ".join(f"{html.escape(k)}: {v}" for k, v in sorted(counts.items())) or "pending"
    partitions = report.get("choice_counts_by_partition", {})
    exact_train = partitions.get("train", {}).get("exact_lot_auctioned", 0)
    exact_val = partitions.get("validation", {}).get("exact_lot_auctioned", 0)
    coefficient_lines = []
    for run, item in report.get("runs", {}).items():
        auction_coefficient = item.get("auction_coefficients", {}).get("jev:lot_auction_status:exact_lot_auctioned")
        pg_coefficient = item.get("pg_interaction_coefficient")
        if auction_coefficient is not None or pg_coefficient is not None:
            pieces = []
            if auction_coefficient is not None:
                pieces.append(f"exact-lot auction {auction_coefficient:+.4f}")
            if pg_coefficient is not None:
                pieces.append(f"Panama × Gesha {pg_coefficient:+.4f}")
            coefficient_lines.append(f"{html.escape(run)}: {', '.join(pieces)}")
    coefficient_html = "<br>".join(coefficient_lines) or "pending"
    paired = report.get("paired_intervals", {}).get("intervals", {})
    ci_lines = []
    for run, comparison in paired.items():
        interval = comparison.get("rmsle_delta_vs_jev_original_95pct_interval")
        if interval:
            ci_lines.append(f"{html.escape(run)} minus original JEV RMSLE: [{interval[0]:+.6f}, {interval[1]:+.6f}]")
    ci_html = "<br>".join(ci_lines) or "pending"
    cost = report["extraction"]["total_estimated_successful_response_cost_usd"]
    runs = report.get("runs", {})
    if runs:
        incumbent = runs["incumbent"]["validation"]["val_rmsle"]
        original = runs["jev_original"]["validation"]["val_rmsle"]
        auction = runs["jev_auction"]["validation"]["val_rmsle"]
        pg = runs["jev_pg"]["validation"]["val_rmsle"]
        combined = runs["jev_auction_pg"]["validation"]["val_rmsle"]
        decision = (f"All JEV hybrid point estimates improve on the incumbent. The best candidate, JEV plus "
                    f"Panama × Gesha, lowers RMSLE by {incumbent - pg:.6f} ({100 * (incumbent - pg) / incumbent:.2f}%). "
                    f"The original JEV hybrid improves it by {100 * (incumbent - original) / incumbent:.2f}%. "
                    f"The auction feature does not add to that gain: it changes RMSLE by {auction - original:+.6f} "
                    f"versus the original JEV hybrid, while the combined model changes it by {combined - original:+.6f}. "
                    "The best result remains short of the predeclared 3% target, and its exploratory paired interval "
                    "includes zero, so it is a provisional improvement rather than promotion evidence.")
    else:
        decision = "Waiting for the complete fixed-split comparison."
    auction_train = report["exact_auction_price_by_partition"].get("train", {})
    auction_val = report["exact_auction_price_by_partition"].get("validation", {})
    subgroup = (f"The validation split has only {auction_val.get('rows', 0)} exact-lot positives, "
                f"whose mean actual price is ${auction_val['mean_true_usd_per_100g']:.2f} per 100 g; "
                f"the {auction_train.get('rows', 0)} training positives average "
                f"${auction_train['mean_true_usd_per_100g']:.2f}.") if auction_train and auction_val else ""
    audit = report.get("manual_source_audit")
    audit_html = (f"<p>Manual review of all {audit['exact_positive_rows_reviewed']} exact-lot positives found "
                  "four without an explicit auction claim in the supplied text (row IDs 2798, 2864, "
                  f"4084, and 4836). Row 3497 is ambiguous. {html.escape(subgroup)} This small, shifted "
                  "subgroup limits what the auction comparison can establish.</p>") if audit else ""
    speed_path = ROOT / "artifacts" / "jev_serving_benchmark" / "summary.json"
    if speed_path.exists():
        speed = json.loads(speed_path.read_text())
        old_speed = speed["providers"]["openai_app"]
        jev_speed = speed["providers"]["jev_ab"]
        speed_html = (f"<p>In a local eight-page, 16-pair pilot, the current app extraction call took "
                      f"{old_speed['p50_seconds']:.2f} s at p50 and {old_speed['p95_seconds']:.2f} s at p95; "
                      f"the A+B JEV semantic call took {jev_speed['p50_seconds']:.2f} s and "
                      f"{jev_speed['p95_seconds']:.2f} s. JEV does not yet return the app's price, package, "
                      "and display fields, so this is a call-latency pilot, not a complete serving speedup. "
                      "See the <a href='jev-results-showcase.html'>combined results and method</a>.</p>")
    else:
        speed_html = ("<p>Not measured yet. The roughly 0.38-second median in the one-question training "
                      "backfill is not comparable with the current app’s full product-page extraction.</p>")
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Exact-lot auction JEV price comparison</title><style>body{{font:16px/1.5 system-ui;max-width:950px;margin:3rem auto;padding:0 1rem;color:#26342d}}table{{border-collapse:collapse;width:100%}}th,td{{padding:.65rem;border-bottom:1px solid #d4ddd6;text-align:left}}th{{background:#eef3ef}}.small{{color:#52665a}}@media(max-width:650px){{table{{display:block;overflow-x:auto}}}}</style></head><body>
<h1>Exact-lot auction JEV price comparison</h1><p>Status: {html.escape(report['status'])}. Auction coverage: {report['coverage']['complete']}/{report['coverage']['eligible']} price rows. The original A+B answers are reused.</p>
<p><strong>Decision:</strong> {html.escape(decision)}</p>
<table><thead><tr><th>Run</th><th>RMSLE ↓</th><th>Gain vs incumbent</th><th>Relative gain</th><th>MAE ($)</th><th>p90 AE ($)</th><th>Top-decile mean prediction ($)</th><th>Top-decile bias ($)</th><th>Val − train RMSLE</th></tr></thead><tbody>{table}</tbody></table>
<p>Price is real USD per 100 g. The top decile is selected by true validation price; negative bias means underprediction. The Panama × Gesha term is the product of two existing A+B JEV support probabilities. The auction term is a seven-outcome probability distribution for this exact reviewed lot.</p>
<p>Auction choices across price train and validation: {counts_html}. Exact-lot positives: {exact_train} training, {exact_val} validation. Estimated TypeSafe spend for this one-question extraction: ${cost:.4f} in successful-response charges; retries and failures may add charges.</p>
<p>Selected coefficients on the model's log-price scale:<br>{coefficient_html}</p>
<h2>Exploratory paired intervals</h2><p>{ci_html}. These roaster-cluster intervals use historical validation rows that were repeatedly used in earlier research, so they are not a fresh confirmatory test.</p>
<h2>Serving extraction speed</h2>{speed_html}
<p>The final serving report should compare identical saved page contexts with cache bypassed and equivalent successful outputs. Report extraction p50 and p95, end-to-end appraisal p50 and p95, paired per-page differences, completion rate, fallbacks, and request cost. The proposed gates—30% lower extraction p50, 20% lower extraction p95, and 20% lower end-to-end p50 with no p95 regression—are reasonable directional targets; measured before/after values should remain visible even when a gate is missed.</p>
<p>The targeted 61-row development pilot separated exact auction lots from generic auction-system history, blends containing an auction component, a different record auction, explicit non-auction lots, and no-auction controls. Row 3497, named “Special Auction,” remains ambiguous because the source does not directly say it was auctioned.</p>
{audit_html}
<p class="small">The baseline is the saved research price artifact; other runs refit the same fixed ElasticNet architecture and scrub explicit target quotations. Review prose may differ from roaster product pages. No serving model or inference path was changed.</p>
</body></html>"""
    (OUT / "summary.html").write_text(page, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train, val = prior.ordered_task_rows("price")
    all_ids = {row["row_id"] for row in train + val}
    base, base_missing = prior.load_jev_vectors(all_ids)
    auction, records, missing = auction_vectors(all_ids)
    if base_missing:
        raise ValueError(f"original JEV cache lacks {len(base_missing)} price rows")
    report = {"status": "pending_extraction" if missing else "complete",
              "auction_contract": AUCTION.version, "auction_question_hash": AUCTION.questions_hash,
              "base_contract": prior.CONTRACT_VERSION,
              "split_hashes": {partition: prior.split_hash(ROOT / "data" / "splits" / f"price_{partition}.csv")
                               for partition in ("train", "validation")},
              "coverage": {"eligible": len(all_ids), "complete": len(auction),
                           "missing": len(missing), "missing_sample": missing[:20]},
              "choice_counts": dict(Counter(record["answers"]["lot_auction_status"]["choice"]
                                            for record in records.values())),
              "choice_counts_by_partition": {
                  name: dict(Counter(records[row["row_id"]]["answers"]["lot_auction_status"]["choice"]
                                     for row in items if row["row_id"] in records))
                  for name, items in (("train", train), ("validation", val))},
              "exact_auction_price_by_partition": {
                  name: exact_auction_price_summary(items, records) if not missing else {}
                  for name, items in (("train", train), ("validation", val))},
              "extraction": extraction_summary(), "runs": {}}
    if missing:
        (OUT / "training_report.json").write_text(json.dumps(report, indent=2))
        render_html(report)
        print(json.dumps({"status": report["status"], "coverage": report["coverage"]}))
        return
    report["manual_source_audit"] = {"exact_positive_rows_reviewed": 45,
                                     "no_explicit_auction_claim_row_ids": ["2798", "2864", "4084", "4836"],
                                     "ambiguous_row_ids": ["3497"]}
    clean_train, clean_val = prior.clean_text_rows(train), prior.clean_text_rows(val)
    x_text_train, x_text_val, x_control_train, x_control_val, encoder, text_names = prior.price_text_matrices(clean_train, clean_val)
    base_train, base_val = prior.matrix_for(train, base), prior.matrix_for(val, base)
    auction_train, auction_val = prior.matrix_for(train, auction), prior.matrix_for(val, auction)
    pg_train, pg_val = pg_interaction(base_train), pg_interaction(base_val)
    candidates = {
        "scrubbed_control": (x_control_train, x_control_val, False, False),
        "jev_original": (sparse.hstack((base_train, x_text_train), format="csr"),
                         sparse.hstack((base_val, x_text_val), format="csr"), False, False),
        "jev_auction": (sparse.hstack((base_train, auction_train, x_text_train), format="csr"),
                        sparse.hstack((base_val, auction_val, x_text_val), format="csr"), True, False),
        "jev_pg": (sparse.hstack((base_train, pg_train, x_text_train), format="csr"),
                   sparse.hstack((base_val, pg_val, x_text_val), format="csr"), False, True),
        "jev_auction_pg": (sparse.hstack((base_train, auction_train, pg_train, x_text_train), format="csr"),
                           sparse.hstack((base_val, auction_val, pg_val, x_text_val), format="csr"), True, True),
    }
    truth = np.asarray([float(row["price_usd_per_100g_real"]) for row in val])
    incumbent = prior.baseline("price")
    baseline_truth, baseline_pred, baseline_ids = read_saved_predictions(
        ROOT / "artifacts" / "price" / "validation_predictions.csv", baseline=True)
    truth_by_id = dict(zip(baseline_ids, baseline_truth, strict=True))
    pred_by_id = dict(zip(baseline_ids, baseline_pred, strict=True))
    if not np.allclose(truth, [truth_by_id[row["row_id"]] for row in val]):
        raise ValueError("incumbent price predictions differ from this fixed validation split")
    incumbent_pred = np.asarray([pred_by_id[row["row_id"]] for row in val])
    report["runs"]["incumbent"] = {"train": incumbent["train"], "validation": incumbent["validation"],
                                    "high_price_decile": high_price_summary(truth, incumbent_pred),
                                    "validation_minus_train_rmsle": incumbent["validation"]["val_rmsle"] -
                                    incumbent["train"]["train_rmsle"],
                                    "provenance": incumbent["provenance"]}
    predictions: dict[str, np.ndarray] = {"incumbent": incumbent_pred}
    for run, (x_train, x_val, use_auction, use_pg) in candidates.items():
        fitted, pred, model = prior.predict_price(x_train, x_val, train, val)
        predicted = np.asarray(pred)
        predictions[run] = predicted
        report["runs"][run] = {**fitted, "feature_count": x_train.shape[1],
                                "high_price_decile": high_price_summary(truth, predicted),
                                "validation_minus_train_rmsle": fitted["validation"]["val_rmsle"] -
                                fitted["train"]["train_rmsle"]}
        if use_auction:
            start = len(BASE_NAMES)
            report["runs"][run]["auction_coefficients"] = dict(zip(
                AUCTION_NAMES, model.coef_[start:start + len(AUCTION_NAMES)].tolist(), strict=True))
        if use_pg:
            index = len(BASE_NAMES) + (len(AUCTION_NAMES) if use_auction else 0)
            report["runs"][run]["pg_interaction_coefficient"] = float(model.coef_[index])
        write_predictions(run, val, pred)
        if run in ("jev_auction", "jev_pg", "jev_auction_pg"):
            names = BASE_NAMES + (AUCTION_NAMES if use_auction else []) + (
                ["interaction:origin_country:Panama*variety_gesha:supported"] if use_pg else []) + text_names
            if len(names) != x_train.shape[1]:
                raise ValueError(f"{run} feature names do not match model matrix")
            bundle = {"model": model, "run": run, "text_encoder": encoder,
                      "text_column_offset": len(encoder.structured_vocab),
                      "base_contract": prior.CONTRACT_VERSION,
                      "auction_contract": AUCTION.version, "auction_question_hash": AUCTION.questions_hash,
                      "use_auction": use_auction, "use_pg_interaction": use_pg, "feature_names": names}
            path = OUT / f"price_{run}_model.pkl"
            with path.open("wb") as stream:
                pickle.dump(bundle, stream)
            with path.open("rb") as stream:
                restored = pickle.load(stream)
            check_x = saved_bundle_matrix(restored, val, base, auction)
            if (x_val != check_x).nnz and not np.allclose(x_val.toarray(), check_x.toarray(), rtol=1e-8, atol=1e-8):
                raise ValueError(f"{run} saved transforms changed features")
            check_pred = price.inverse_target(restored["model"].predict(check_x))
            if not np.allclose(check_pred, predicted, rtol=1e-8, atol=1e-8):
                raise ValueError(f"{run} saved bundle changed predictions")
    old = json.loads((ROOT / "artifacts" / "jev" / "training_report.json").read_text())["tasks"]["price"]
    for run, old_run in (("scrubbed_control", "scrubbed_text_control"), ("jev_original", "hybrid")):
        actual = report["runs"][run]["validation"]["val_rmsle"]
        expected = old[old_run]["validation"]["val_rmsle"]
        if abs(actual - expected) > 1e-8:
            raise ValueError(f"{run} no longer reproduces the prior comparison")
    report["paired_intervals"] = paired_intervals(val, predictions, truth)
    incumbent_rmsle = report["runs"]["incumbent"]["validation"]["val_rmsle"]
    original_rmsle = report["runs"]["jev_original"]["validation"]["val_rmsle"]
    report["rmsle_comparisons"] = {
        run: {"absolute_gain_vs_incumbent": incumbent_rmsle - item["validation"]["val_rmsle"],
              "relative_gain_vs_incumbent_percent":
                  100 * (incumbent_rmsle - item["validation"]["val_rmsle"]) / incumbent_rmsle,
              "absolute_gain_vs_original_jev": original_rmsle - item["validation"]["val_rmsle"]}
        for run, item in report["runs"].items()
    }
    (OUT / "training_report.json").write_text(json.dumps(report, indent=2))
    render_html(report)
    print(json.dumps({"status": report["status"], "coverage": report["coverage"],
                      "report": str(OUT / "training_report.json")}))


if __name__ == "__main__":
    main()
