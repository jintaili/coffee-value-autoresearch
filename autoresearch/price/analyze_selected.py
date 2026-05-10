"""Comprehensive analysis for the selected price model setup.

The accepted setup is represented in the results ledger as commit 6507aee. This script analyzes
the currently generated artifacts for that setup.
"""

from __future__ import annotations

import csv
import json
import pickle
import subprocess
import sys
import types
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SETUP_COMMIT = "6507aee"
LEDGER_COMMIT = "6507aee"
ARTIFACT_DIR = ROOT / "artifacts" / "price"
MODEL_PATH = ARTIFACT_DIR / "model.pkl"
REPORT_PATH = ARTIFACT_DIR / "report.json"
ANALYSIS_JSON = ARTIFACT_DIR / "6507aee_analysis.json"
ANALYSIS_MD = ARTIFACT_DIR / "6507aee_analysis.md"
MODELING = ROOT / "data" / "modeling_price.csv"
TRAIN_SPLIT = ROOT / "data" / "splits" / "price_train.csv"
VALIDATION_SPLIT = ROOT / "data" / "splits" / "price_validation.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_ids(path: Path) -> set[str]:
    return {row["row_id"] for row in read_csv(path)}


def load_setup_module():
    source = subprocess.check_output(
        ["git", "show", f"{SETUP_COMMIT}:autoresearch/price/train.py"],
        cwd=ROOT,
        text=True,
    )
    module = types.ModuleType("price_train_6507aee_setup")
    module.__file__ = str((ROOT / "autoresearch" / "price" / "train.py").resolve())
    sys.modules[module.__name__] = module
    exec(compile(source, f"{SETUP_COMMIT}:autoresearch/price/train.py", "exec"), module.__dict__)
    return module


def load_model(setup_module):
    # The pickle references classes from the dynamically loaded module.
    sys.modules[setup_module.__name__] = setup_module
    sys.modules["__main__"].FeatureEncoder = setup_module.FeatureEncoder
    with MODEL_PATH.open("rb") as f:
        return pickle.load(f)


def valid_price_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out = []
    for row in rows:
        if row.get("price_parse_status") != "ok":
            continue
        try:
            price = float(row["price_usd_per_100g_real"])
        except (KeyError, ValueError):
            continue
        if price > 0:
            out.append(row)
    return out


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred = np.maximum(0.0, y_pred)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


def abs_error_stats(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    ae = np.abs(err)
    return {
        "count": int(len(y_true)),
        "mean_true": float(np.mean(y_true)),
        "mean_pred": float(np.mean(y_pred)),
        "mean_error": float(np.mean(err)),
        "mae": float(np.mean(ae)),
        "median_ae": float(np.median(ae)),
        "p90_ae": float(np.quantile(ae, 0.9)),
        "rmsle": rmsle(y_true, y_pred),
    }


def slice_stats(rows, y_true, y_pred, field: str, min_count: int = 20, top_n: int = 25):
    groups: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row.get(field) or "unknown"].append(i)
    out = []
    for value, idx in groups.items():
        if len(idx) < min_count:
            continue
        arr = np.array(idx)
        item = {"field": field, "value": value, **abs_error_stats(y_true[arr], y_pred[arr])}
        out.append(item)
    out.sort(key=lambda x: (-x["rmsle"], -x["count"]))
    return out[:top_n]


def decile_stats(y_true: np.ndarray, y_pred: np.ndarray) -> list[dict[str, float]]:
    order = np.argsort(y_true)
    out = []
    for decile, idx in enumerate(np.array_split(order, 10)):
        stats = abs_error_stats(y_true[idx], y_pred[idx])
        stats["decile"] = decile
        stats["min_true"] = float(np.min(y_true[idx]))
        stats["max_true"] = float(np.max(y_true[idx]))
        out.append(stats)
    return out


def worst_cases(rows, y_true, y_pred, n: int = 30):
    items = []
    for row, true, pred in zip(rows, y_true, y_pred):
        items.append(
            {
                "row_id": row["row_id"],
                "coffee_name": row["coffee_name"],
                "roaster": row["roaster"],
                "review_date": row["review_date"],
                "price_raw": row["price_raw"],
                "true": float(true),
                "pred": float(pred),
                "error": float(pred - true),
                "abs_error": float(abs(pred - true)),
                "log_error": float(np.log1p(max(pred, 0.0)) - np.log1p(true)),
            }
        )
    by_abs = sorted(items, key=lambda x: x["abs_error"], reverse=True)[:n]
    over = sorted(items, key=lambda x: x["error"], reverse=True)[:n]
    under = sorted(items, key=lambda x: x["error"])[:n]
    by_log_abs = sorted(items, key=lambda x: abs(x["log_error"]), reverse=True)[:n]
    return {"largest_abs_error": by_abs, "largest_overpredictions": over, "largest_underpredictions": under, "largest_log_error": by_log_abs}


def coefficient_analysis(model_bundle, top_n: int = 80):
    encoder = model_bundle["encoder"]
    model = model_bundle.get("model")
    if model is not None:
        coefs = np.asarray(model.weights if hasattr(model, "weights") else model.coef_)
    else:
        coefs = np.asarray(model_bundle["weights"])
    names = encoder.feature_names
    pairs = [(names[i], float(coefs[i])) for i in range(min(len(names), len(coefs)))]
    nonzero = [(name, coef) for name, coef in pairs if abs(coef) > 1e-12]

    def kind(name: str) -> str:
        if name.startswith("tfidf:"):
            return "tfidf"
        return name.split("=", 1)[0]

    by_kind = Counter(kind(name) for name, _ in nonzero)
    positive = sorted(nonzero, key=lambda x: x[1], reverse=True)[:top_n]
    negative = sorted(nonzero, key=lambda x: x[1])[:top_n]
    strongest = sorted(nonzero, key=lambda x: abs(x[1]), reverse=True)[:top_n]
    return {
        "n_features": len(names),
        "n_nonzero": len(nonzero),
        "nonzero_by_kind": dict(by_kind.most_common()),
        "top_positive": [{"feature": name, "coef": coef} for name, coef in positive],
        "top_negative": [{"feature": name, "coef": coef} for name, coef in negative],
        "top_abs": [{"feature": name, "coef": coef} for name, coef in strongest],
    }


def markdown_table(rows: list[dict], columns: list[str], max_rows: int = 20) -> str:
    shown = rows[:max_rows]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in shown:
        vals = []
        for col in columns:
            val = row.get(col, "")
            if isinstance(val, float):
                val = f"{val:.4f}"
            vals.append(str(val).replace("|", "/"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_markdown(analysis: dict) -> None:
    lines = [
        "# Price Model Analysis: 6507aee",
        "",
        f"Ledger commit: `{LEDGER_COMMIT}`",
        f"Setup commit: `{SETUP_COMMIT}`",
        "",
        "## Metrics",
        "",
        markdown_table([analysis["metrics"]], ["train_rmsle", "val_rmsle", "overfit_gap", "val_spearman", "val_mae", "val_median_ae", "val_p90_ae"]),
        "",
        "## Validation Deciles",
        "",
        markdown_table(analysis["decile_stats"], ["decile", "count", "min_true", "max_true", "mean_true", "mean_pred", "mean_error", "mae", "median_ae", "rmsle"], 10),
        "",
        "## Worst Absolute Errors",
        "",
        markdown_table(analysis["worst_cases"]["largest_abs_error"], ["coffee_name", "roaster", "price_raw", "true", "pred", "error", "abs_error"], 15),
        "",
        "## Largest Log Errors",
        "",
        markdown_table(analysis["worst_cases"]["largest_log_error"], ["coffee_name", "roaster", "price_raw", "true", "pred", "log_error"], 15),
        "",
        "## High-RMSLE Slices",
        "",
        "### Origin Country",
        markdown_table(analysis["slices"]["origin_country"], ["value", "count", "mean_true", "mean_pred", "mean_error", "mae", "rmsle"], 15),
        "",
        "### Roaster Country",
        markdown_table(analysis["slices"]["roaster_country"], ["value", "count", "mean_true", "mean_pred", "mean_error", "mae", "rmsle"], 15),
        "",
        "## Feature Importance",
        "",
        f"Nonzero coefficients: {analysis['feature_importance']['n_nonzero']} / {analysis['feature_importance']['n_features']}",
        "",
        "### Top Positive Coefficients",
        markdown_table(analysis["feature_importance"]["top_positive"], ["feature", "coef"], 25),
        "",
        "### Top Negative Coefficients",
        markdown_table(analysis["feature_importance"]["top_negative"], ["feature", "coef"], 25),
    ]
    ANALYSIS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    setup_module = load_setup_module()
    model_bundle = load_model(setup_module)
    rows = valid_price_rows(read_csv(MODELING))
    train_ids = read_ids(TRAIN_SPLIT)
    validation_ids = read_ids(VALIDATION_SPLIT)
    train_rows = [row for row in rows if row["row_id"] in train_ids]
    validation_rows = [row for row in rows if row["row_id"] in validation_ids]

    x_train = model_bundle["encoder"].transform(train_rows)
    x_validation = model_bundle["encoder"].transform(validation_rows)
    y_train = np.array([float(row["price_usd_per_100g_real"]) for row in train_rows])
    y_validation = np.array([float(row["price_usd_per_100g_real"]) for row in validation_rows])
    model = model_bundle.get("model")
    if model is not None:
        train_pred_raw = model.predict(x_train)
        validation_pred_raw = model.predict(x_validation)
    else:
        train_pred_raw = np.asarray(x_train @ model_bundle["weights"] + model_bundle["intercept"]).reshape(-1)
        validation_pred_raw = np.asarray(x_validation @ model_bundle["weights"] + model_bundle["intercept"]).reshape(-1)
    train_pred = setup_module.inverse_target(train_pred_raw)
    validation_pred = setup_module.inverse_target(validation_pred_raw)

    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    metrics = {
        **{f"train_{k}": v for k, v in abs_error_stats(y_train, train_pred).items() if k != "count"},
        **{f"val_{k}": v for k, v in abs_error_stats(y_validation, validation_pred).items() if k != "count"},
        "overfit_gap": (rmsle(y_validation, validation_pred) - rmsle(y_train, train_pred)) / rmsle(y_train, train_pred),
        "val_spearman": report["metrics"]["val_spearman"],
        "val_p90_ae": report["metrics"]["val_p90_ae"],
    }
    analysis = {
        "ledger_commit": LEDGER_COMMIT,
        "setup_commit": SETUP_COMMIT,
        "metrics": metrics,
        "config": report["config"],
        "decile_stats": decile_stats(y_validation, validation_pred),
        "slices": {
            field: slice_stats(validation_rows, y_validation, validation_pred, field)
            for field in ["origin_country", "roaster_country", "process_method", "variety", "is_espresso", "is_blend"]
        },
        "worst_cases": worst_cases(validation_rows, y_validation, validation_pred),
        "feature_importance": coefficient_analysis(model_bundle),
    }
    ANALYSIS_JSON.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    write_markdown(analysis)
    print(f"wrote {ANALYSIS_JSON.relative_to(ROOT)}")
    print(f"wrote {ANALYSIS_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
