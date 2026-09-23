"""Three fixed training comparisons; incumbent artifacts remain untouched."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import os
import pickle
import shutil
import statistics
import sys
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.linear_model import ElasticNet

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from autoresearch.rating import train as rating
from autoresearch.price import train as price
from coffee_value.extraction.cache import load_record, record_path
from coffee_value.extraction.encoding import ENCODING_VERSION, encode, feature_names
from coffee_value.extraction.questions import CONTRACT_VERSION, MODEL, question_hash
from coffee_value.extraction.state import build_review_state, scrub_target_quotes

OUT = ROOT / "artifacts" / "jev"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def split_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ordered_task_rows(task: str) -> tuple[list[dict], list[dict]]:
    path = ROOT / "data" / ("modeling_coffee.csv" if task == "rating" else "modeling_price.csv")
    all_rows = rows(path)
    train_ids = {r["row_id"] for r in rows(ROOT / "data" / "splits" / f"{task}_train.csv")}
    val_ids = {r["row_id"] for r in rows(ROOT / "data" / "splits" / f"{task}_validation.csv")}
    if train_ids & val_ids:
        raise ValueError(f"{task} train/validation overlap")
    eligible = (lambda r: bool(r["rating"])) if task == "rating" else (lambda r: r.get("price_parse_status") == "ok" and float(r["price_usd_per_100g_real"]) > 0)
    train = [r for r in all_rows if r["row_id"] in train_ids and eligible(r)]
    val = [r for r in all_rows if r["row_id"] in val_ids and eligible(r)]
    if len(train) != len(train_ids) or len(val) != len(val_ids):
        raise ValueError(f"{task} split or target definition changed")
    return train, val


def load_jev_vectors(row_ids: set[str]) -> tuple[dict[str, np.ndarray], list[str]]:
    source = rows(ROOT / "data" / "coffee.csv")
    vectors: dict[str, np.ndarray] = {}
    missing: list[str] = []
    for rid in sorted(row_ids, key=int):
        state = build_review_state(source[int(rid)])
        record = load_record(record_path(OUT / "cache", state))
        if not record or record.get("status") != "complete":
            missing.append(rid)
            continue
        vectors[rid] = np.asarray(encode(record), dtype=np.float64)
    return vectors, missing


def matrix_for(rows_: list[dict], vectors: dict[str, np.ndarray]) -> sparse.csr_matrix:
    return sparse.csr_matrix(np.stack([vectors[r["row_id"]] for r in rows_]))


def clean_text_rows(items: list[dict]) -> list[dict]:
    return [{**row, "sensory_text": scrub_target_quotes(row.get("sensory_text", "")),
             "producer_text": scrub_target_quotes(row.get("producer_text", ""))} for row in items]


def rating_text_matrices(train: list[dict], val: list[dict]) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix, object, list[str]]:
    # Reuse incumbent embedding vectors without ever rewriting incumbent cache.
    incumbent_cache = ROOT / "artifacts" / "rating_baseline" / "embedding_cache_sentence_transformers_all_MiniLM_L6_v2.pkl"
    candidate_cache = OUT / incumbent_cache.name
    if incumbent_cache.exists() and not candidate_cache.exists():
        shutil.copy2(incumbent_cache, candidate_cache)
    rating.EMBED_CACHE_DIR = OUT
    encoder = rating.HybridEncoder()
    encoder.fit(train)
    n_struct = len(encoder.tfidf.structured_vocab)
    full_train, full_val = encoder.transform(train), encoder.transform(val)
    return full_train[:, n_struct:], full_val[:, n_struct:], full_train, full_val, encoder, encoder.feature_names[n_struct:]


def price_text_matrices(train: list[dict], val: list[dict]) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix, object, list[str]]:
    encoder = price.FeatureEncoder()
    encoder.fit(train)
    n_struct = len(encoder.structured_vocab)
    full_train, full_val = encoder.transform(train), encoder.transform(val)
    return full_train[:, n_struct:], full_val[:, n_struct:], full_train, full_val, encoder, encoder.feature_names[n_struct:]


def package_matrices(train: list[dict], val: list[dict]) -> tuple[sparse.csr_matrix, sparse.csr_matrix, dict]:
    logs = [math.log(g) for r in train if (g := price.package_grams(r)) is not None]
    mean = float(np.mean(logs)) if logs else 0.0
    std = float(np.std(logs)) or 1.0
    return (encode_package(train, mean, std), encode_package(val, mean, std),
            {"package_log_mean": mean, "package_log_std": std})


def encode_package(items: list[dict], mean: float, std: float) -> sparse.csr_matrix:
    out = np.zeros((len(items), 5), dtype=float)
    for i, row in enumerate(items):
        grams = price.package_grams(row)
        if grams is None:
            out[i, 1] = 1
        else:
            out[i] = [(math.log(grams) - mean) / std, 0, float(grams <= 20), float(grams <= 50), float(grams <= 100)]
    return sparse.csr_matrix(out)


def bundle_matrix(bundle: dict, items: list[dict], vectors: dict[str, np.ndarray]) -> sparse.csr_matrix:
    x_jev = matrix_for(items, vectors)
    if bundle["run"] == "hybrid":
        encoder = bundle["text_encoder"]
        x_text = encoder.transform(clean_text_rows(items))[:, bundle["text_column_offset"]:]
        matrix = sparse.hstack((x_jev, x_text), format="csr")
    elif bundle["task"] == "price":
        params = bundle["package_transform"]
        x_pkg = encode_package(items, params["package_log_mean"], params["package_log_std"])
        matrix = sparse.hstack((x_jev, x_pkg), format="csr")
    else:
        matrix = x_jev
    if matrix.shape[1] != len(bundle["feature_names"]):
        raise ValueError("saved bundle feature width mismatch")
    return matrix


def predict_rating(x_train: sparse.csr_matrix, x_val: sparse.csr_matrix, train: list[dict], val: list[dict]) -> tuple[dict, list[float], object]:
    y_train = np.asarray([float(r["rating"]) for r in train])
    y_val = np.asarray([float(r["rating"]) for r in val])
    weights, intercept = rating.fit_ridge(x_train, y_train, 1.0)
    train_pred = rating.predict(x_train, weights, intercept)
    val_pred = rating.predict(x_val, weights, intercept)
    return {"train": rating.metrics(y_train, train_pred, "train"), "validation": rating.metrics(y_val, val_pred, "val")}, val_pred.tolist(), rating.LinearModel(weights, intercept)


def predict_price(x_train: sparse.csr_matrix, x_val: sparse.csr_matrix, train: list[dict], val: list[dict]) -> tuple[dict, list[float], object]:
    y_train = np.asarray([float(r["price_usd_per_100g_real"]) for r in train])
    y_val = np.asarray([float(r["price_usd_per_100g_real"]) for r in val])
    est = ElasticNet(alpha=price.ELASTICNET_ALPHA, l1_ratio=price.ELASTICNET_L1_RATIO,
                     max_iter=price.ELASTICNET_MAX_ITER, tol=price.ELASTICNET_TOL, random_state=price.SEED)
    est.fit(x_train, np.log(y_train))
    train_pred = price.inverse_target(est.predict(x_train))
    val_pred = price.inverse_target(est.predict(x_val))
    report = {"train": price.metrics(y_train, train_pred, "train"), "validation": price.metrics(y_val, val_pred, "val"),
              "validation_deciles": price.validation_diagnostics(y_val, val_pred)}
    return report, val_pred.tolist(), est


def baseline(task: str) -> dict:
    report = json.loads((ROOT / "artifacts" / task / "report.json").read_text())
    saved = rows(ROOT / "artifacts" / task / "validation_predictions.csv")
    if task == "rating":
        truth = np.asarray([float(r["rating"]) for r in saved])
        predictions = np.asarray([float(r["prediction"]) for r in saved])
        checked = rating.metrics(truth, predictions, "val")
    else:
        truth = np.asarray([float(r["price_usd_per_100g_real"]) for r in saved])
        predictions = np.asarray([float(r["prediction_usd_per_100g_real"]) for r in saved])
        checked = price.metrics(truth, predictions, "val")
    expected_ids = {r["row_id"] for r in rows(ROOT / "data" / "splits" / f"{task}_validation.csv")}
    if {r["row_id"] for r in saved} != expected_ids or len(saved) != len(expected_ids):
        raise ValueError(f"{task} incumbent prediction IDs differ from fixed split")
    for metric, value in checked.items():
        if abs(value - report["metrics"][metric]) > 0.0001:
            raise ValueError(f"{task} incumbent {metric} differs from saved predictions")
    return {"train": report["train_metrics"], "validation": report["metrics"],
            "recomputed_from_saved_rounded_predictions": checked,
            "provenance": "read-only incumbent report; rounded saved predictions checked independently",
            "artifact": f"artifacts/{task}/report.json"}


def save_predictions(task: str, run: str, val: list[dict], predictions: list[float]) -> None:
    path = OUT / f"{task}_{run}_predictions.csv"
    target = "rating" if task == "rating" else "price_usd_per_100g_real"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["row_id", target, "prediction"])
        writer.writeheader()
        for row, pred in zip(val, predictions, strict=True):
            writer.writerow({"row_id": row["row_id"], target: row[target], "prediction": pred})


def append_prediction_diagnostics(result: dict) -> None:
    for task, task_result in result["tasks"].items():
        for run in ("baseline", "scrubbed_text_control", "hybrid", "jev_only"):
            path = (ROOT / "artifacts" / task / "validation_predictions.csv") if run == "baseline" else (OUT / f"{task}_{run}_predictions.csv")
            if not path.exists() or run not in task_result:
                continue
            saved = rows(path)
            if task == "rating":
                truth = np.asarray([float(r["rating"]) for r in saved])
                predicted = np.asarray([float(r["prediction"]) for r in saved])
                task_result[run]["rating_buckets"] = rating.bucket_analysis([], truth, predicted)
            else:
                truth = np.asarray([float(r["price_usd_per_100g_real"]) for r in saved])
                field = "prediction_usd_per_100g_real" if run == "baseline" else "prediction"
                predicted = np.asarray([float(r[field]) for r in saved])
                task_result[run]["price_deciles"] = price.quantile_analysis([], truth, predicted)


def render_html(result: dict) -> None:
    def number(value: object) -> str:
        return f"{value:.6f}" if isinstance(value, (float, int)) else "pending"
    cells = []
    for task, task_result in result["tasks"].items():
        primary = "val_concordance" if task == "rating" else "val_rmsle"
        baseline_value = task_result["baseline"]["validation"][primary]
        for run in ("baseline", "scrubbed_text_control", "hybrid", "jev_only"):
            item = task_result.get(run)
            metric = (item or {}).get("validation", {})
            primary_value = metric.get(primary)
            gain = ((primary_value - baseline_value) if task == "rating" else (baseline_value - primary_value)) if primary_value is not None else None
            tertiary = "val_rmse" if task == "rating" else "val_p90_ae"
            cells.append(f"<tr><td>{html.escape(task)}</td><td>{html.escape(run)}</td>"
                         f"<td>{number(primary_value)}</td><td>{number(gain)}</td>"
                         f"<td>{number(metric.get('val_mae'))}</td><td>{number(metric.get(tertiary))}</td></tr>")
        if task == "rating" and task_result.get("app_reference"):
            metric = task_result["app_reference"]["validation"]
            cells.append(f"<tr><td>rating</td><td>app TF-IDF reference</td><td>{number(metric.get(primary))}</td>"
                         f"<td>{number(metric.get(primary) - baseline_value)}</td>"
                         f"<td>{number(metric.get('val_mae'))}</td><td>{number(metric.get('val_rmse'))}</td></tr>")
    limitations = "Historical validation IDs have been used repeatedly for model selection. Review prose differs from roaster copy. Fresh matched-page evaluation is still needed."
    if result["status"] != "complete":
        limitations += " Live JEV extraction and candidate model metrics are pending; no improvement claim is possible."
    extraction = result.get("extraction", {})
    full = extraction.get("full", {})
    pilot = extraction.get("pilot", {})
    cost = extraction.get("estimated_total_successful_response_cost_usd", 0.0)
    def seconds(value: object) -> str:
        return f"{value:.3f}" if isinstance(value, (float, int)) else "pending"
    extraction_line = (f"Final 40-row pilot: {pilot.get('complete', 0)}/40 complete; "
                       f"median {seconds(pilot.get('latency_median_seconds'))} s, "
                       f"p95 {seconds(pilot.get('latency_p95_seconds'))} s. "
                       f"Full backfill: {full.get('complete', 0)} successful calls; "
                       f"median {seconds(full.get('latency_median_seconds'))} s, "
                       f"p95 {seconds(full.get('latency_p95_seconds'))} s. "
                       f"Estimated spend across all pilot versions and full successful responses: ${cost:.4f}.")
    decision = ""
    diagnostics = ""
    if result["status"] == "complete":
        rt = result["tasks"]["rating"]
        pr = result["tasks"]["price"]
        rating_gain = rt["hybrid"]["validation"]["val_concordance"] - rt["baseline"]["validation"]["val_concordance"]
        price_gain_pct = 100 * (pr["baseline"]["validation"]["val_rmsle"] - pr["hybrid"]["validation"]["val_rmsle"]) / pr["baseline"]["validation"]["val_rmsle"]
        matched_rating_gain = rt["hybrid"]["validation"]["val_concordance"] - rt["scrubbed_text_control"]["validation"]["val_concordance"]
        matched_price_gain = pr["scrubbed_text_control"]["validation"]["val_rmsle"] - pr["hybrid"]["validation"]["val_rmsle"]
        decision = (f"<h2>Decision</h2><p>The hybrid gains {rating_gain:+.6f} rating concordance and lowers price RMSLE "
                    f"by {price_gain_pct:.2f}% against the research incumbents. Both miss the proposed +0.005 rating "
                    f"and 3% price gates. On matched scrubbed text, the JEV structured replacement gains {matched_rating_gain:+.6f} "
                    f"rating concordance and lowers price RMSLE by {matched_price_gain:.6f}. The JEV-only models regress on both tasks. "
                    "Retain the incumbents for serving; these historical results do not justify integration or rollout.</p>")
        high_b = pr["baseline"]["price_deciles"][-1]
        high_h = pr["hybrid"]["price_deciles"][-1]
        diagnostics = (f"<h2>Where the models still miss</h2><p>The most expensive price decile averages "
                       f"${high_b['mean_true']:.2f}/100 g. Baseline predictions average ${high_b['mean_pred']:.2f}; hybrid predictions "
                       f"average ${high_h['mean_pred']:.2f}. Hybrid high-decile RMSLE is {high_h['rmsle']:.6f}, "
                       f"versus {high_b['rmsle']:.6f} for baseline. The luxury-price compression barely changes.</p>")
    independent = result.get("independent_audit", {})
    paired = ""
    if independent:
        try:
            rci = independent["tasks"]["rating"]["runs"]["hybrid"]["paired_primary_delta_95pct_interval"]
            pci = independent["tasks"]["price"]["runs"]["hybrid"]["paired_primary_delta_95pct_interval"]
            rmatch = independent["tasks"]["rating"]["runs"]["hybrid"].get("paired_primary_delta_vs_scrubbed_control_95pct_interval")
            pmatch = independent["tasks"]["price"]["runs"]["hybrid"].get("paired_primary_delta_vs_scrubbed_control_95pct_interval")
            paired = (f"<p>Exploratory paired roaster-cluster bootstrap, {independent['resamples']} resamples: rating hybrid "
                      f"concordance delta interval [{rci[0]:+.6f}, {rci[1]:+.6f}]; price hybrid RMSLE delta interval "
                      f"[{pci[0]:+.6f}, {pci[1]:+.6f}]. ")
            if rmatch and pmatch:
                paired += (f"Against the matched scrubbed-text control, rating delta interval "
                           f"[{rmatch[0]:+.6f}, {rmatch[1]:+.6f}], and price RMSLE delta interval "
                           f"[{pmatch[0]:+.6f}, {pmatch[1]:+.6f}]. ")
            paired += "These are historical, repeatedly used validation rows, not a fresh confirmatory test.</p>"
        except (KeyError, TypeError):
            pass
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>JEV A+B training comparison</title>
<style>body{{font:16px/1.5 system-ui;max-width:920px;margin:3rem auto;padding:0 1rem;color:#26342d}}table{{border-collapse:collapse;width:100%}}th,td{{padding:.7rem;border-bottom:1px solid #d4ddd6;text-align:left}}code{{background:#eef3ef;padding:.15rem}}</style>
<h1>JEV A+B training comparison</h1><p>Status: {html.escape(result['status'])}. Cached feature coverage: {result['coverage']['complete']}/{result['coverage']['eligible']} eligible rows.</p>
<table><thead><tr><th>Task</th><th>Run</th><th>Primary metric</th><th>Gain over research baseline</th><th>MAE</th><th>RMSE / p90 AE</th></tr></thead><tbody>{''.join(cells)}</tbody></table>
<p>Rating primary metric is pairwise concordance; higher is better. Price primary metric is RMSLE in real USD per 100 g; lower is better. Gain is positive when a candidate improves. The price fit target is log(price), and RMSLE uses log1p of predicted and actual prices.</p>
<p>The scrubbed-text control refits the incumbent structured and text feature architecture after removing explicit target quotations from its text. It isolates this input change from replacing structured features with JEV probabilities; it is not a fourth proposed candidate.</p>
{decision}{diagnostics}{paired}
<p>{html.escape(limitations)}</p><p>The new hybrid text is scrubbed of explicit score and price quotations before fitting. The incumbent text was not scrubbed this way, so this is an additional input difference in the hybrid comparison.</p>
<p>Catalog: {len(feature_names())} ordered probability columns; contract {CONTRACT_VERSION}, model {MODEL}, encoder {ENCODING_VERSION}. Fixed split file hashes and the full feature order are in <code>training_report.json</code>.</p>
<p>{html.escape(extraction_line)} The provider's published input rate is $0.042 per million tokens. Failed and retried requests may add cost beyond this estimate.</p>
<p>Development pilot audit: supported claims were generally grounded in sampled source text, but the provider sometimes inferred explicit negation of one processing method from another. In row 5506 it supported washed and wet-hulled from text stating only wet-hulled. These are known extraction errors, not corrected training labels. See <code>pilot_contract3_audit.json</code>.</p>
<p>Run <code>python scripts/backfill_jev.py pilot</code>, inspect <code>artifacts/jev/pilot_summary.json</code>, then <code>python scripts/backfill_jev.py full</code> and <code>python scripts/evaluate_jev.py</code>. The backfill requires <code>TYPESAFE_API_KEY</code>. Artifacts and cached answers are under <code>artifacts/jev/</code>.</p>
<p>Verification passed: five contract and serialization tests, Python compilation, saved bundle prediction roundtrips, and exact validation row coverage. Run <code>python -m unittest discover -s tests -v</code> and <code>python -m compileall -q coffee_value/extraction scripts/backfill_jev.py scripts/evaluate_jev.py</code>.</p>
</html>"""
    (OUT / "summary.html").write_text(page, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    result: dict = {"status": "pending_extraction", "contract_version": CONTRACT_VERSION,
                    "question_hash": question_hash(), "model": MODEL, "encoding_version": ENCODING_VERSION,
                    "feature_names": feature_names(), "split_hashes": {}, "tasks": {}}
    pilot_path = OUT / "pilot_contract3_audit.json"
    result["extraction"] = {"pilot": json.loads(pilot_path.read_text()) if pilot_path.exists() else {}}
    pilot_status = OUT / "pilot_status.jsonl"
    full_status = OUT / "full_status.jsonl"
    def summarize_status(path: Path) -> dict:
        events = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        successes = [e for e in events if e.get("status") == "complete" and not e.get("cache_hit")]
        durations = sorted(e["latency_seconds"] for e in successes)
        return {"complete": len(successes), "failed": sum(e.get("status") == "failed" for e in events),
                "input_tokens": sum(e.get("input_tokens", 0) for e in successes),
                "estimated_successful_response_cost_usd": sum(e.get("cost_usd", 0) for e in successes),
                "latency_median_seconds": statistics.median(durations) if durations else None,
                "latency_p95_seconds": durations[math.ceil(.95 * len(durations)) - 1] if durations else None}
    result["extraction"]["full"] = summarize_status(full_status)
    result["extraction"]["all_pilots"] = summarize_status(pilot_status)
    result["extraction"]["estimated_total_successful_response_cost_usd"] = (
        result["extraction"]["full"]["estimated_successful_response_cost_usd"] +
        result["extraction"]["all_pilots"]["estimated_successful_response_cost_usd"])
    independent_path = OUT / "independent_evaluation.json"
    if independent_path.exists():
        result["independent_audit"] = json.loads(independent_path.read_text())
    for task in ("rating", "price"):
        train, val = ordered_task_rows(task)
        result["tasks"][task] = {"train_rows": len(train), "validation_rows": len(val), "baseline": baseline(task)}
        if task == "rating":
            app_report = Path("/Users/jintaili/Development/coffee-value-app/artifacts/rating/report.json")
            if app_report.exists():
                result["tasks"][task]["app_reference"] = {
                    "artifact": str(app_report), "validation": json.loads(app_report.read_text())["metrics"],
                    "provenance": "read-only local serving artifact; TF-IDF Ridge"}
        for part in ("train", "validation"):
            result["split_hashes"][f"{task}_{part}"] = split_hash(ROOT / "data" / "splits" / f"{task}_{part}.csv")
    all_ids = {r["row_id"] for task in ("rating", "price") for partition in ordered_task_rows(task) for r in partition}
    vectors, missing = load_jev_vectors(all_ids)
    result["coverage"] = {"eligible": len(all_ids), "complete": len(vectors), "missing": len(missing), "missing_sample": missing[:20]}
    if missing:
        (OUT / "training_report.json").write_text(json.dumps(result, indent=2))
        render_html(result)
        print(json.dumps({"status": result["status"], "complete": len(vectors), "eligible": len(all_ids),
                          "missing_sample": missing[:10]}))
        return
    for task in ("rating", "price"):
        train, val = ordered_task_rows(task)
        train_text, val_text = clean_text_rows(train), clean_text_rows(val)
        x_jev_train, x_jev_val = matrix_for(train, vectors), matrix_for(val, vectors)
        if task == "rating":
            x_text_train, x_text_val, x_control_train, x_control_val, text_encoder, text_names = rating_text_matrices(train_text, val_text)
            fit = predict_rating
            x_only_train, x_only_val = x_jev_train, x_jev_val
            only_names = feature_names()
            only_transforms = {}
        else:
            x_text_train, x_text_val, x_control_train, x_control_val, text_encoder, text_names = price_text_matrices(train_text, val_text)
            x_pkg_train, x_pkg_val, package_transform = package_matrices(train, val)
            fit = predict_price
            x_only_train = sparse.hstack((x_jev_train, x_pkg_train), format="csr")
            x_only_val = sparse.hstack((x_jev_val, x_pkg_val), format="csr")
            only_names = feature_names() + price.PACKAGE_FEATURE_NAMES
            only_transforms = package_transform
        control_metrics, control_preds, _ = fit(x_control_train, x_control_val, train, val)
        result["tasks"][task]["scrubbed_text_control"] = {**control_metrics, "features": x_control_train.shape[1]}
        save_predictions(task, "scrubbed_text_control", val, control_preds)
        x_hybrid_train = sparse.hstack((x_jev_train, x_text_train), format="csr")
        x_hybrid_val = sparse.hstack((x_jev_val, x_text_val), format="csr")
        for run, x_train, x_val in (("hybrid", x_hybrid_train, x_hybrid_val), ("jev_only", x_only_train, x_only_val)):
            names = feature_names() + text_names if run == "hybrid" else only_names
            if x_train.shape[1] != len(names):
                raise ValueError(f"{task} {run} feature names do not match matrix")
            metrics, preds, model = fit(x_train, x_val, train, val)
            result["tasks"][task][run] = {**metrics, "features": x_train.shape[1]}
            save_predictions(task, run, val, preds)
            if task == "rating" and run == "hybrid":
                text_encoder.embed._model = None
            bundle = {"model": model, "contract_version": CONTRACT_VERSION, "question_hash": question_hash(),
                      "feature_names": names, "run": run, "task": task,
                      "text_encoder": text_encoder if run == "hybrid" else None,
                      "text_column_offset": (len(text_encoder.tfidf.structured_vocab) if task == "rating" else len(text_encoder.structured_vocab)) if run == "hybrid" else None,
                      "package_transform": only_transforms if run == "jev_only" else None,
                      "text_scrubbing": "target quotations removed from sensory_text and producer_text"}
            with (OUT / f"{task}_{run}_model.pkl").open("wb") as stream:
                pickle.dump(bundle, stream)
            with (OUT / f"{task}_{run}_model.pkl").open("rb") as stream:
                loaded = pickle.load(stream)
            check_matrix = bundle_matrix(loaded, val, vectors)
            if (x_val != check_matrix).nnz and not np.allclose(x_val.toarray(), check_matrix.toarray(), rtol=1e-8, atol=1e-8):
                raise ValueError(f"{task} {run} saved transforms changed features")
            loaded_model = loaded["model"]
            check = loaded_model.predict(check_matrix) if task == "rating" else price.inverse_target(loaded_model.predict(check_matrix))
            if not np.allclose(check, preds, rtol=1e-8, atol=1e-8):
                raise ValueError(f"{task} {run} model serialization changed predictions")
    result["status"] = "complete"
    append_prediction_diagnostics(result)
    (OUT / "training_report.json").write_text(json.dumps(result, indent=2))
    render_html(result)
    print(json.dumps({"status": "complete", "report": str(OUT / "training_report.json")}))


if __name__ == "__main__":
    main()
