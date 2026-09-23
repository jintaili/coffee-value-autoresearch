"""Evaluate a compact JEV follow-up catalog and two fixed price interactions."""

from __future__ import annotations

import csv
import html
import json
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
from coffee_value.extraction.catalog import FOLLOWUP
from coffee_value.extraction.encoding import encode, feature_names
from coffee_value.extraction.state import build_review_state
from scripts import evaluate_jev as prior

OUT = ROOT / "artifacts" / "jev_followup"
INTERACTIONS = (
    ("origin_country", "Panama", "variety_gesha", "supported"),
    ("lot_auction", "supported", "documented_scarcity", "supported"),
)


def followup_vectors(row_ids: set[str]) -> tuple[dict[str, np.ndarray], list[str], dict[str, dict]]:
    source = prior.rows(ROOT / "data" / "coffee.csv")
    vectors: dict[str, np.ndarray] = {}
    records: dict[str, dict] = {}
    missing = []
    for rid in sorted(row_ids, key=int):
        state = build_review_state(source[int(rid)])
        record = load_record(record_path(OUT / "cache", state, contract_version=FOLLOWUP.version,
                                         questions_hash=FOLLOWUP.questions_hash, model=FOLLOWUP.model))
        if not record or record.get("status") != "complete":
            missing.append(rid)
            continue
        vectors[rid] = np.asarray(encode(record, questions=FOLLOWUP.questions,
                                         contract_version=FOLLOWUP.version,
                                         questions_hash=FOLLOWUP.questions_hash,
                                         model=FOLLOWUP.model), dtype=np.float64)
        records[rid] = record
    return vectors, missing, records


def interaction_matrix(base: sparse.csr_matrix, extension: sparse.csr_matrix) -> sparse.csr_matrix:
    """A fixed pairwise layer over probabilities; no validation-selected terms."""
    names = feature_names() + feature_names(questions=FOLLOWUP.questions)
    all_features = sparse.hstack((base, extension), format="csr")
    index = {name: i for i, name in enumerate(names)}
    columns = []
    for left_q, left_option, right_q, right_option in INTERACTIONS:
        left = all_features[:, index[f"jev:{left_q}:{left_option}"]]
        right = all_features[:, index[f"jev:{right_q}:{right_option}"]]
        columns.append(left.multiply(right))
    return sparse.hstack(columns, format="csr")


def bundle_matrix(bundle: dict, items: list[dict], base_vectors: dict[str, np.ndarray],
                  extension_vectors: dict[str, np.ndarray]) -> sparse.csr_matrix:
    x_base = prior.matrix_for(items, base_vectors)
    x_extra = prior.matrix_for(items, extension_vectors)
    text = bundle["text_encoder"].transform(prior.clean_text_rows(items))[:, bundle["text_column_offset"]:]
    parts = [x_base, x_extra]
    if bundle["run"] == "jev_interactions":
        parts.append(interaction_matrix(x_base, x_extra))
    parts.append(text)
    matrix = sparse.hstack(parts, format="csr")
    if matrix.shape[1] != len(bundle["feature_names"]):
        raise ValueError("saved follow-up bundle feature width mismatch")
    return matrix


def high_price_decile(truth: list[float], predictions: list[float]) -> dict:
    observed = np.asarray(truth)
    forecast = np.asarray(predictions)
    threshold = float(np.quantile(observed, 0.9))
    mask = observed >= threshold
    return {
        "threshold_usd_per_100g": threshold,
        "rows": int(mask.sum()),
        "mean_true": float(observed[mask].mean()),
        "mean_pred": float(forecast[mask].mean()),
        "mean_bias": float((forecast[mask] - observed[mask]).mean()),
        "rmsle": float(np.sqrt(np.mean((np.log1p(forecast[mask]) - np.log1p(observed[mask])) ** 2))),
    }


def save_predictions(task: str, run: str, val: list[dict], predictions: list[float]) -> None:
    target = "rating" if task == "rating" else "price_usd_per_100g_real"
    with (OUT / f"{task}_{run}_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["row_id", target, "prediction"])
        writer.writeheader()
        for row, prediction in zip(val, predictions, strict=True):
            writer.writerow({"row_id": row["row_id"], target: row[target], "prediction": prediction})


def feature_audit(records: dict[str, dict]) -> dict:
    source = prior.rows(ROOT / "data" / "coffee.csv")
    by_question: dict[str, dict] = {}
    for qid in FOLLOWUP.questions:
        choices = Counter(record["answers"][qid]["choice"] for record in records.values())
        examples = []
        for choice in choices:
            if choice in ("not_stated", "explicitly_negated", "conflicting"):
                continue
            candidates = [rid for rid, record in records.items() if record["answers"][qid]["choice"] == choice]
            candidates.sort(key=lambda rid: (-records[rid]["answers"][qid]["probabilities"][choice], int(rid)))
            for rid in candidates[:3]:
                state = build_review_state(source[int(rid)])
                examples.append({"row_id": rid, "choice": choice,
                                 "probability": records[rid]["answers"][qid]["probabilities"][choice],
                                 "state": state, "audit": "unreviewed"})
        by_question[qid] = {"counts": dict(sorted(choices.items())), "examples": examples}
    return by_question


def extraction_summary() -> dict:
    result = {}
    for mode in ("pilot", "full"):
        path = OUT / f"{mode}_status.jsonl"
        events = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        success = [event for event in events if event.get("status") == "complete" and not event.get("cache_hit")]
        times = sorted(event["latency_seconds"] for event in success)
        result[mode] = {"successful_calls": len(success),
                        "failed_calls": sum(event.get("status") == "failed" for event in events),
                        "estimated_successful_response_cost_usd": sum(event.get("cost_usd", 0) for event in success),
                        "median_seconds": statistics.median(times) if times else None,
                        "p95_seconds": times[int(np.ceil(.95 * len(times))) - 1] if times else None}
    result["estimated_total_successful_response_cost_usd"] = sum(
        result[mode]["estimated_successful_response_cost_usd"] for mode in ("pilot", "full"))
    return result


def fit_task(task: str, base_vectors: dict[str, np.ndarray], extension_vectors: dict[str, np.ndarray]) -> dict:
    train, val = prior.ordered_task_rows(task)
    clean_train, clean_val = prior.clean_text_rows(train), prior.clean_text_rows(val)
    x_base_train, x_base_val = prior.matrix_for(train, base_vectors), prior.matrix_for(val, base_vectors)
    x_extra_train, x_extra_val = prior.matrix_for(train, extension_vectors), prior.matrix_for(val, extension_vectors)
    if task == "rating":
        x_text_train, x_text_val, x_control_train, x_control_val, encoder, text_names = prior.rating_text_matrices(clean_train, clean_val)
        fit = prior.predict_rating
    else:
        x_text_train, x_text_val, x_control_train, x_control_val, encoder, text_names = prior.price_text_matrices(clean_train, clean_val)
        fit = prior.predict_price
    candidates = {
        "scrubbed_text_control": (x_control_train, x_control_val),
        "jev_hybrid_original": (sparse.hstack((x_base_train, x_text_train), format="csr"),
                                sparse.hstack((x_base_val, x_text_val), format="csr")),
        "jev_hybrid_extended": (sparse.hstack((x_base_train, x_extra_train, x_text_train), format="csr"),
                                sparse.hstack((x_base_val, x_extra_val, x_text_val), format="csr")),
    }
    if task == "price":
        interactions_train = interaction_matrix(x_base_train, x_extra_train)
        interactions_val = interaction_matrix(x_base_val, x_extra_val)
        candidates["jev_interactions"] = (
            sparse.hstack((x_base_train, x_extra_train, interactions_train, x_text_train), format="csr"),
            sparse.hstack((x_base_val, x_extra_val, interactions_val, x_text_val), format="csr"),
        )
    incumbent = prior.baseline(task)
    incumbent_item = {"train": incumbent["train"], "validation": incumbent["validation"],
                      "provenance": incumbent["provenance"]}
    if task == "price":
        saved = prior.rows(ROOT / "artifacts" / "price" / "validation_predictions.csv")
        incumbent_item["high_price_decile"] = high_price_decile(
            [float(row["price_usd_per_100g_real"]) for row in saved],
            [float(row["prediction_usd_per_100g_real"]) for row in saved])
        incumbent_item["validation_minus_train_rmsle"] = (
            incumbent["validation"]["val_rmsle"] - incumbent["train"]["train_rmsle"])
    result = {"train_rows": len(train), "validation_rows": len(val), "runs": {"incumbent": incumbent_item}}
    for run, (x_train, x_val) in candidates.items():
        metrics, predictions, model = fit(x_train, x_val, train, val)
        item = {**metrics, "feature_count": x_train.shape[1]}
        if task == "price":
            truth = [float(row["price_usd_per_100g_real"]) for row in val]
            item["high_price_decile"] = high_price_decile(truth, predictions)
            item["validation_minus_train_rmsle"] = (metrics["validation"]["val_rmsle"] -
                                                    metrics["train"]["train_rmsle"])
            if run == "jev_interactions":
                offset = x_base_train.shape[1] + x_extra_train.shape[1]
                item["interaction_names"] = [f"{a}:{b} × {c}:{d}" for a, b, c, d in INTERACTIONS]
                item["interaction_train_nonzero"] = [int(interactions_train[:, i].count_nonzero()) for i in range(len(INTERACTIONS))]
                item["interaction_validation_nonzero"] = [int(interactions_val[:, i].count_nonzero()) for i in range(len(INTERACTIONS))]
                item["interaction_coefficients"] = model.coef_[offset:offset + len(INTERACTIONS)].tolist()
        result["runs"][run] = item
        save_predictions(task, run, val, predictions)
        if run in ("jev_hybrid_extended", "jev_interactions"):
            if task == "rating":
                encoder.embed._model = None
            names = feature_names() + feature_names(questions=FOLLOWUP.questions)
            if run == "jev_interactions":
                names += [f"interaction:{a}:{b}*{c}:{d}" for a, b, c, d in INTERACTIONS]
            names += text_names
            if len(names) != x_train.shape[1]:
                raise ValueError(f"{task} {run} feature width mismatch")
            offset = (len(encoder.tfidf.structured_vocab) if task == "rating" else
                      len(encoder.structured_vocab))
            bundle = {"task": task, "run": run, "model": model, "text_encoder": encoder,
                      "text_column_offset": offset,
                      "feature_names": names, "base_contract": prior.CONTRACT_VERSION,
                      "followup_contract": FOLLOWUP.version, "followup_question_hash": FOLLOWUP.questions_hash}
            path = OUT / f"{task}_{run}_model.pkl"
            with path.open("wb") as stream:
                pickle.dump(bundle, stream)
            with path.open("rb") as stream:
                restored = pickle.load(stream)
            restored_x = bundle_matrix(restored, val, base_vectors, extension_vectors)
            if (x_val != restored_x).nnz and not np.allclose(x_val.toarray(), restored_x.toarray(),
                                                             rtol=1e-8, atol=1e-8):
                raise ValueError(f"{task} {run} saved transforms changed features")
            check = (restored["model"].predict(restored_x) if task == "rating" else
                     price.inverse_target(restored["model"].predict(restored_x)))
            if not np.allclose(check, predictions, rtol=1e-8, atol=1e-8):
                raise ValueError(f"{task} {run} saved model changed predictions")
    return result


def render_html(report: dict) -> None:
    cells = []
    for task, task_report in report.get("tasks", {}).items():
        primary = "val_concordance" if task == "rating" else "val_rmsle"
        train_primary = "train_concordance" if task == "rating" else "train_rmsle"
        for run, metrics in task_report["runs"].items():
            validation = metrics["validation"]
            gap = metrics.get("validation_minus_train_rmsle")
            gap_cell = f"{gap:+.6f}" if gap is not None else "—"
            bias = metrics.get("high_price_decile", {}).get("mean_bias")
            bias_cell = f"{bias:+.3f}" if bias is not None else "—"
            cells.append(f"<tr><td>{html.escape(task)}</td><td>{html.escape(run.replace('_', ' '))}</td>"
                         f"<td>{metrics['train'][train_primary]:.6f}</td><td>{validation[primary]:.6f}</td>"
                         f"<td>{validation['val_mae']:.6f}</td><td>{gap_cell}</td><td>{bias_cell}</td></tr>")
    rows_html = "".join(cells) or "<tr><td colspan='7'>Awaiting approved extraction and model fitting.</td></tr>"
    audit_rows = []
    for qid, item in report["audit"].items():
        choices = ", ".join(f"{name}: {count}" for name, count in item["counts"].items()) or "No answers yet"
        examples = []
        for example in item["examples"][:4]:
            state = json.dumps(example["state"], ensure_ascii=False, indent=2)
            examples.append(f"<details><summary>Row {html.escape(example['row_id'])}: {html.escape(example['choice'])} "
                            f"({example['probability']:.3f})</summary><pre>{html.escape(state)}</pre></details>")
        audit_rows.append(f"<tr><td>{html.escape(qid)}</td><td>{html.escape(choices)}</td><td>{''.join(examples) or '—'}</td></tr>")
    audit_html = "".join(audit_rows)
    price_runs = report.get("tasks", {}).get("price", {}).get("runs", {})
    interaction = price_runs.get("jev_interactions", {})
    interaction_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{train_count}</td><td>{val_count}</td><td>{coef:+.6f}</td></tr>"
        for name, train_count, val_count, coef in zip(
            interaction.get("interaction_names", []), interaction.get("interaction_train_nonzero", []),
            interaction.get("interaction_validation_nonzero", []), interaction.get("interaction_coefficients", []), strict=True))
    extraction = report["extraction"]
    cost = extraction["estimated_total_successful_response_cost_usd"]
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>JEV finish and lot-status follow-up</title><style>body{{font:16px/1.5 system-ui;max-width:900px;margin:3rem auto;padding:0 1rem;color:#26342d}}table{{border-collapse:collapse;width:100%}}th,td{{padding:.65rem;border-bottom:1px solid #d4ddd6;text-align:left}}code{{background:#eef3ef;padding:.1rem .25rem}}</style></head><body>
<h1>JEV finish and lot-status follow-up</h1><p>Status: {report['status']}. New feature coverage: {report['coverage']['complete']}/{report['coverage']['eligible']} rows. Base A+B answers are reused.</p>
<table><thead><tr><th>Task</th><th>Run</th><th>Train primary</th><th>Validation primary</th><th>Validation MAE</th><th>Price generalization gap</th><th>High-price mean bias ($/100 g)</th></tr></thead><tbody>{rows_html}</tbody></table>
<p>Rating primary metric is concordance (higher is better); price primary metric is RMSLE (lower is better). The price interaction run adds only Panama × Gesha and lot auction × documented scarcity to the extended hybrid. All runs use the same fixed historical train/validation IDs and scrubbed text.</p>
<p>Estimated charges from successful responses in this follow-up: ${cost:.4f}. The reported price gap is validation RMSLE minus training RMSLE; a large positive gap is evidence of overfit. The expensive-lot mean bias is predicted minus actual price for the top true-price decile, so a negative value means underprediction.</p>
<h2>Interaction support</h2><table><thead><tr><th>Pair</th><th>Nonzero train rows</th><th>Nonzero validation rows</th><th>Coefficient</th></tr></thead><tbody>{interaction_rows or '<tr><td colspan="4">Pending extraction.</td></tr>'}</tbody></table>
<h2>Question coverage and source examples</h2><p>These are model outputs and source excerpts for manual review, not certified labels. Open a row to inspect its exact score- and price-scrubbed state.</p><table><thead><tr><th>Question</th><th>Choice counts</th><th>Examples</th></tr></thead><tbody>{audit_html}</tbody></table>
<p>The new questions require explicit finish or lot-specific evidence. Auction history of a roaster, awards for other lots, generic rarity, review scores, and retail prices do not qualify. The eight-question catalog is versioned independently so the prior 38-question cache remains reusable.</p>
<p>The validation set has been reused in prior research. Results are exploratory. Review prose may differ from live roaster pages; no serving-speed claim follows from this training experiment.</p>
</body></html>"""
    (OUT / "summary.html").write_text(page, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = {task: prior.ordered_task_rows(task) for task in ("rating", "price")}
    all_ids = {row["row_id"] for train, val in tasks.values() for row in train + val}
    base_vectors, base_missing = prior.load_jev_vectors(all_ids)
    extension_vectors, missing, records = followup_vectors(all_ids)
    if base_missing:
        raise ValueError(f"prior A+B cache incomplete: {len(base_missing)} rows")
    report = {"status": "pending_extraction" if missing else "complete",
              "base_contract": prior.CONTRACT_VERSION,
              "followup_contract": FOLLOWUP.version,
              "followup_question_hash": FOLLOWUP.questions_hash,
              "followup_model": FOLLOWUP.model,
              "followup_features": feature_names(questions=FOLLOWUP.questions),
              "coverage": {"eligible": len(all_ids), "complete": len(extension_vectors),
                           "missing": len(missing), "missing_sample": missing[:20]},
              "extraction": extraction_summary(), "audit": feature_audit(records), "tasks": {}}
    if not missing:
        for task in tasks:
            report["tasks"][task] = fit_task(task, base_vectors, extension_vectors)
    (OUT / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    render_html(report)
    print(json.dumps({"status": report["status"], "coverage": report["coverage"],
                      "report": str(OUT / "training_report.json")}))


if __name__ == "__main__":
    main()
