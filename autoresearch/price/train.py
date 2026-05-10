"""Baseline price model for price autoresearch.

Only this file should be edited during ordinary price autoresearch.
"""

from __future__ import annotations

import csv
import json
import math
import pickle
import random
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


ROOT = Path(__file__).resolve().parents[2]
MODELING = ROOT / "data" / "modeling_price.csv"
ARTIFACT_DIR = ROOT / "artifacts" / "price"
REPORT = ARTIFACT_DIR / "report.json"
PREDICTIONS = ARTIFACT_DIR / "validation_predictions.csv"
MODEL = ARTIFACT_DIR / "model.pkl"
RESULTS = ROOT / "autoresearch" / "price" / "results.tsv"
SPLIT_DIR = ROOT / "data" / "splits"
TRAIN_SPLIT = SPLIT_DIR / "price_train.csv"
VALIDATION_SPLIT = SPLIT_DIR / "price_validation.csv"


RUN_DESCRIPTION = "baseline: exp15-aligned deterministic features + unigram/bigram tfidf + ridge on log1p real USD/100g"

SEED = 20260509
VALIDATION_FRAC = 0.15

STRUCTURED_FIELDS = [
    "origin_country",
    "process_method",
    "variety",
    "is_blend",
    "is_espresso",
    "is_decaf",
    "producer_or_farm_present",
    "altitude_present",
    "roaster_country",
]
TEXT_FIELDS = ["sensory_text", "producer_text"]
MAX_FEATURES = 6000
MIN_DF = 5
MAX_DF = 0.85
NGRAM_MAX = 2
RIDGE_ALPHA = 10.0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_id_split(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["row_id"])
        writer.writeheader()
        for row_id in ids:
            writer.writerow({"row_id": row_id})


def read_ids(path: Path) -> set[str]:
    return {row["row_id"] for row in read_csv(path)}


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


def price_bucket(price: float, quantile_edges: list[float]) -> str:
    for i, edge in enumerate(quantile_edges):
        if price <= edge:
            return str(i)
    return str(len(quantile_edges))


def ensure_splits(rows: list[dict[str, str]]) -> None:
    if TRAIN_SPLIT.exists() and VALIDATION_SPLIT.exists():
        return
    prices = sorted(math.log1p(float(row["price_usd_per_100g_real"])) for row in rows)
    edges = [prices[round(q * (len(prices) - 1))] for q in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]
    by_bucket: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        bucket = price_bucket(math.log1p(float(row["price_usd_per_100g_real"])), edges)
        by_bucket[bucket].append(row["row_id"])

    rng = random.Random(SEED)
    train_ids: list[str] = []
    validation_ids: list[str] = []
    for bucket in sorted(by_bucket):
        ids = by_bucket[bucket][:]
        rng.shuffle(ids)
        n_validation = max(1, round(len(ids) * VALIDATION_FRAC))
        validation_ids.extend(ids[:n_validation])
        train_ids.extend(ids[n_validation:])
    write_id_split(TRAIN_SPLIT, sorted(train_ids, key=int))
    write_id_split(VALIDATION_SPLIT, sorted(validation_ids, key=int))


def tokenize(text: str) -> list[str]:
    text = (text or "").lower().replace("’", "'")
    return re.findall(r"[a-z0-9][a-z0-9'\-]*", text)


def ngrams(text: str, n_max: int = NGRAM_MAX) -> list[str]:
    toks = tokenize(text)
    grams = list(toks)
    for n in range(2, n_max + 1):
        for i in range(len(toks) - n + 1):
            grams.append(" ".join(toks[i : i + n]))
    return grams


class FeatureEncoder:
    def __init__(self, max_features: int = MAX_FEATURES, min_df: int = MIN_DF, max_df: float = MAX_DF):
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.structured_vocab: dict[tuple[str, str], int] = {}
        self.text_vocab: dict[str, int] = {}
        self.idf: np.ndarray | None = None
        self.feature_names: list[str] = []

    def fit(self, rows: list[dict[str, str]]) -> None:
        vocab: dict[tuple[str, str], int] = {}
        for field in STRUCTURED_FIELDS:
            values = sorted({row.get(field) or "unknown" for row in rows})
            for value in values:
                vocab[(field, value)] = len(vocab)
        self.structured_vocab = vocab

        df = Counter()
        for row in rows:
            terms = set(ngrams(" ".join(row.get(field, "") for field in TEXT_FIELDS)))
            df.update(terms)
        n_docs = len(rows)
        max_doc_count = max(1, math.floor(self.max_df * n_docs))
        terms = [
            (term, count)
            for term, count in df.items()
            if count >= self.min_df and count <= max_doc_count
        ]
        terms.sort(key=lambda x: (-x[1], x[0]))
        terms = terms[: self.max_features]
        self.text_vocab = {term: i for i, (term, _) in enumerate(terms)}
        self.idf = np.array([math.log((1 + n_docs) / (1 + count)) + 1 for _, count in terms])

        self.feature_names = []
        for (field, value), _ in sorted(self.structured_vocab.items(), key=lambda x: x[1]):
            self.feature_names.append(f"{field}={value}")
        for term, _ in sorted(self.text_vocab.items(), key=lambda x: x[1]):
            self.feature_names.append(f"tfidf:{term}")

    def transform(self, rows: list[dict[str, str]]) -> sparse.csr_matrix:
        assert self.idf is not None
        n_struct = len(self.structured_vocab)
        indptr = [0]
        indices = []
        data = []
        for row in rows:
            for field in STRUCTURED_FIELDS:
                idx = self.structured_vocab.get((field, row.get(field) or "unknown"))
                if idx is not None:
                    indices.append(idx)
                    data.append(1.0)

            counts = Counter(ngrams(" ".join(row.get(field, "") for field in TEXT_FIELDS)))
            text_items = []
            norm = 0.0
            for term, count in counts.items():
                idx = self.text_vocab.get(term)
                if idx is None:
                    continue
                value = (1.0 + math.log(count)) * float(self.idf[idx])
                text_items.append((n_struct + idx, value))
                norm += value * value
            norm = math.sqrt(norm) or 1.0
            for idx, value in text_items:
                indices.append(idx)
                data.append(value / norm)
            indptr.append(len(indices))
        return sparse.csr_matrix((data, indices, indptr), shape=(len(rows), len(self.structured_vocab) + len(self.text_vocab)))


def fit_ridge(x_train: sparse.csr_matrix, y_train: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    intercept = float(y_train.mean())
    centered = y_train - intercept
    xtx = x_train.T @ x_train
    reg = sparse.eye(xtx.shape[0], format="csr") * alpha
    weights = spsolve((xtx + reg).tocsc(), x_train.T @ centered)
    return np.asarray(weights), intercept


def predict(x: sparse.csr_matrix, weights: np.ndarray, intercept: float) -> np.ndarray:
    return np.asarray(x @ weights + intercept).reshape(-1)


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0
        i = j
    return ranks


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    rt = rankdata(y_true)
    rp = rankdata(y_pred)
    if np.std(rt) == 0 or np.std(rp) == 0:
        return float("nan")
    return float(np.corrcoef(rt, rp)[0, 1])


def inverse_target(y_log: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, np.expm1(y_log))


def rmsle_from_prices(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred = np.maximum(0.0, y_pred)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


def metrics(y_true_price: np.ndarray, y_pred_price: np.ndarray, prefix: str) -> dict[str, float]:
    err = y_pred_price - y_true_price
    return {
        f"{prefix}_rmsle": rmsle_from_prices(y_true_price, y_pred_price),
        f"{prefix}_spearman": spearman(y_true_price, y_pred_price),
        f"{prefix}_mae": float(np.mean(np.abs(err))),
        f"{prefix}_median_ae": float(np.median(np.abs(err))),
    }


def quantile_analysis(rows: list[dict[str, str]], y_true: np.ndarray, y_pred: np.ndarray) -> list[dict[str, object]]:
    order = np.argsort(y_true)
    chunks = np.array_split(order, 10)
    out = []
    for i, idx in enumerate(chunks):
        if len(idx) == 0:
            continue
        true = y_true[idx]
        pred = y_pred[idx]
        err = pred - true
        out.append(
            {
                "quantile": i,
                "count": int(len(idx)),
                "min_true": float(np.min(true)),
                "max_true": float(np.max(true)),
                "mean_true": float(np.mean(true)),
                "mean_pred": float(np.mean(pred)),
                "mean_error": float(np.mean(err)),
                "median_abs_error": float(np.median(np.abs(err))),
                "rmsle": rmsle_from_prices(true, pred),
            }
        )
    return out


def write_predictions(rows: list[dict[str, str]], y_true: np.ndarray, y_pred: np.ndarray) -> None:
    PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "row_id",
        "coffee_name",
        "roaster",
        "review_date",
        "price_raw",
        "price_usd_per_100g_real",
        "prediction_usd_per_100g_real",
        "error_usd_per_100g_real",
    ]
    with PREDICTIONS.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row, true, pred in zip(rows, y_true, y_pred):
            writer.writerow(
                {
                    "row_id": row["row_id"],
                    "coffee_name": row["coffee_name"],
                    "roaster": row["roaster"],
                    "review_date": row["review_date"],
                    "price_raw": row["price_raw"],
                    "price_usd_per_100g_real": f"{true:.6f}",
                    "prediction_usd_per_100g_real": f"{pred:.6f}",
                    "error_usd_per_100g_real": f"{pred - true:.6f}",
                }
            )


def short_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short=7", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def append_results(metric_values: dict[str, float], status: str = "keep") -> None:
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["commit", "val_rmsle", "overfit_gap", "status", "description"]
    write_header = not RESULTS.exists() or RESULTS.stat().st_size == 0
    with RESULTS.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "commit": short_commit(),
                "val_rmsle": f"{metric_values['val_rmsle']:.6f}",
                "overfit_gap": f"{metric_values['overfit_gap']:.6f}",
                "status": status,
                "description": RUN_DESCRIPTION,
            }
        )


def main() -> None:
    if not MODELING.exists():
        raise SystemExit("Run python3 autoresearch/price/prepare.py first.")

    rows = valid_price_rows(read_csv(MODELING))
    ensure_splits(rows)
    train_ids = read_ids(TRAIN_SPLIT)
    validation_ids = read_ids(VALIDATION_SPLIT)
    train_rows = [row for row in rows if row["row_id"] in train_ids]
    validation_rows = [row for row in rows if row["row_id"] in validation_ids]

    y_train_price = np.array([float(row["price_usd_per_100g_real"]) for row in train_rows])
    y_validation_price = np.array([float(row["price_usd_per_100g_real"]) for row in validation_rows])
    y_train = np.log1p(y_train_price)

    encoder = FeatureEncoder()
    encoder.fit(train_rows)
    x_train = encoder.transform(train_rows)
    x_validation = encoder.transform(validation_rows)
    weights, intercept = fit_ridge(x_train, y_train, RIDGE_ALPHA)

    y_train_pred_price = inverse_target(predict(x_train, weights, intercept))
    y_validation_pred_price = inverse_target(predict(x_validation, weights, intercept))

    train_metrics = metrics(y_train_price, y_train_pred_price, prefix="train")
    val_metrics = metrics(y_validation_price, y_validation_pred_price, prefix="val")
    metric_values = {**train_metrics, **val_metrics}
    metric_values["overfit_gap"] = metric_values["val_rmsle"] - metric_values["train_rmsle"]

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    write_predictions(validation_rows, y_validation_price, y_validation_pred_price)
    report = {
        "config": {
            "target": "log1p(price_usd_per_100g_real)",
            "model": "ridge",
            "alpha": RIDGE_ALPHA,
            "max_features": MAX_FEATURES,
            "min_df": MIN_DF,
            "max_df": MAX_DF,
            "ngram_max": NGRAM_MAX,
            "structured_fields": STRUCTURED_FIELDS,
            "text_fields": TEXT_FIELDS,
            "train_rows": len(train_rows),
            "validation_rows": len(validation_rows),
        },
        "metrics": val_metrics,
        "train_metrics": train_metrics,
        "overfitting": {"overfit_gap": metric_values["overfit_gap"]},
        "quantile_analysis": quantile_analysis(validation_rows, y_validation_price, y_validation_pred_price),
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with MODEL.open("wb") as f:
        pickle.dump({"encoder": encoder, "weights": weights, "intercept": intercept, "config": report["config"]}, f)
    append_results(metric_values)

    for key in ["train_rmsle", "val_rmsle", "overfit_gap", "val_spearman", "val_mae", "val_median_ae"]:
        print(f"{key}: {metric_values[key]:.6f}")
    print(f"report: {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
