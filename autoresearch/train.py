"""Baseline rating model for the fixed autoresearch validation split."""

from __future__ import annotations

import csv
import json
import math
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


ROOT = Path(__file__).resolve().parents[1]
MODELING = ROOT / "data" / "modeling_coffee.csv"
TRAIN_SPLIT = ROOT / "data" / "splits" / "rating_train.csv"
VALIDATION_SPLIT = ROOT / "data" / "splits" / "rating_validation.csv"
ARTIFACT_DIR = ROOT / "artifacts" / "rating_baseline"
PREDICTIONS = ARTIFACT_DIR / "validation_predictions.csv"
REPORT = ARTIFACT_DIR / "report.json"
MODEL = ARTIFACT_DIR / "model.pkl"
RESULTS = ROOT / "autoresearch" / "results.tsv"


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
MAX_FEATURES = 2000
MIN_DF = 5
MAX_DF = 0.85
RIDGE_ALPHA = 10.0
RUN_NAME = "exp01_alpha10"
RUN_DESCRIPTION = "ridge alpha 25 -> 10; same features as baseline (decompress tail predictions)"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_ids(path: Path) -> set[str]:
    return {row["row_id"] for row in read_csv(path)}


def tokenize(text: str) -> list[str]:
    text = (text or "").lower().replace("’", "'")
    return re.findall(r"[a-z0-9][a-z0-9'\-]*", text)


def rating_bucket(rating: float) -> str:
    if rating <= 89:
        return "<=89"
    if rating >= 96:
        return ">=96"
    return str(int(rating))


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
        structured = {}
        for field in STRUCTURED_FIELDS:
            values = sorted({(row.get(field) or "unknown") for row in rows})
            for value in values:
                structured[(field, value)] = len(structured)
        self.structured_vocab = structured

        df = Counter()
        for row in rows:
            terms = set(tokenize(" ".join(row.get(field, "") for field in TEXT_FIELDS)))
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
        self.idf = np.array([math.log((1 + n_docs) / (1 + count)) + 1 for _, count in terms], dtype=float)

        self.feature_names = []
        for (field, value), _ in sorted(self.structured_vocab.items(), key=lambda x: x[1]):
            self.feature_names.append(f"{field}={value}")
        for term, _ in sorted(self.text_vocab.items(), key=lambda x: x[1]):
            self.feature_names.append(f"tfidf:{term}")

    def transform(self, rows: list[dict[str, str]]) -> sparse.csr_matrix:
        assert self.idf is not None
        n_struct = len(self.structured_vocab)
        n_text = len(self.text_vocab)
        indptr = [0]
        indices = []
        data = []
        for row in rows:
            for field in STRUCTURED_FIELDS:
                key = (field, row.get(field) or "unknown")
                idx = self.structured_vocab.get(key)
                if idx is not None:
                    indices.append(idx)
                    data.append(1.0)

            counts = Counter(tokenize(" ".join(row.get(field, "") for field in TEXT_FIELDS)))
            norm = 0.0
            text_items = []
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
        return sparse.csr_matrix((data, indices, indptr), shape=(len(rows), n_struct + n_text))


def fit_ridge(x_train: sparse.csr_matrix, y_train: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    y_mean = float(y_train.mean())
    centered = y_train - y_mean
    xtx = x_train.T @ x_train
    reg = sparse.eye(xtx.shape[0], format="csr") * alpha
    weights = spsolve((xtx + reg).tocsc(), x_train.T @ centered)
    return np.asarray(weights), y_mean


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
        avg_rank = (i + j - 1) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    rt = rankdata(y_true)
    rp = rankdata(y_pred)
    if np.std(rt) == 0 or np.std(rp) == 0:
        return float("nan")
    return float(np.corrcoef(rt, rp)[0, 1])


def pairwise_concordance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    correct = 0
    total = 0
    n = len(y_true)
    for i in range(n):
        true_diff = y_true[i] - y_true[i + 1 :]
        pred_diff = y_pred[i] - y_pred[i + 1 :]
        mask = true_diff != 0
        if not np.any(mask):
            continue
        total += int(mask.sum())
        correct += int((np.sign(true_diff[mask]) == np.sign(pred_diff[mask])).sum())
    return correct / total if total else float("nan")


def metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "val") -> dict[str, float]:
    err = y_pred - y_true
    return {
        f"{prefix}_concordance": pairwise_concordance(y_true, y_pred),
        f"{prefix}_spearman": spearman(y_true, y_pred),
        f"{prefix}_mae": float(np.mean(np.abs(err))),
        f"{prefix}_rmse": float(np.sqrt(np.mean(err * err))),
        f"{prefix}_within_1": float(np.mean(np.abs(err) <= 1.0)),
        f"{prefix}_within_2": float(np.mean(np.abs(err) <= 2.0)),
    }


def bucket_analysis(rows: list[dict[str, str]], y_true: np.ndarray, y_pred: np.ndarray) -> list[dict[str, object]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, y in enumerate(y_true):
        groups[rating_bucket(float(y))].append(i)
    out = []
    order = ["<=89", "90", "91", "92", "93", "94", "95", ">=96"]
    for bucket in order:
        idx = groups.get(bucket, [])
        if not idx:
            continue
        t = y_true[idx]
        p = y_pred[idx]
        err = p - t
        out.append(
            {
                "bucket": bucket,
                "count": len(idx),
                "mean_true": float(np.mean(t)),
                "mean_pred": float(np.mean(p)),
                "mean_error": float(np.mean(err)),
                "median_error": float(np.median(err)),
                "mae": float(np.mean(np.abs(err))),
            }
        )
    return out


def coverage(rows: list[dict[str, str]]) -> dict[str, object]:
    out = {}
    for field in STRUCTURED_FIELDS + TEXT_FIELDS:
        values = [row.get(field, "") for row in rows]
        non_unknown = [v for v in values if v and v != "unknown"]
        out[field] = {
            "non_unknown_rows": len(non_unknown),
            "coverage": len(non_unknown) / len(rows),
            "unique": len(set(values)),
        }
    return out


def write_predictions(rows: list[dict[str, str]], y_true: np.ndarray, y_pred: np.ndarray) -> None:
    PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["row_id", "coffee_name", "roaster", "rating", "prediction", "error"] + STRUCTURED_FIELDS
    with PREDICTIONS.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row, true, pred in zip(rows, y_true, y_pred):
            out = {field: row.get(field, "") for field in fieldnames}
            out["prediction"] = f"{pred:.4f}"
            out["error"] = f"{pred - true:.4f}"
            writer.writerow(out)


def write_results(metric_values: dict[str, float]) -> None:
    fieldnames = [
        "run",
        "description",
        "train_concordance",
        "val_concordance",
        "concordance_gap",
        "val_spearman",
        "val_mae",
        "val_rmse",
        "val_within_1",
        "val_within_2",
    ]
    row = {
        "run": RUN_NAME,
        "description": RUN_DESCRIPTION,
        **{k: f"{metric_values[k]:.6f}" for k in fieldnames if k in metric_values},
    }
    write_header = not RESULTS.exists() or RESULTS.stat().st_size == 0
    with RESULTS.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def top_coefficients(encoder: FeatureEncoder, weights: np.ndarray, n: int = 40) -> dict[str, list[dict[str, object]]]:
    pairs = list(zip(encoder.feature_names, weights))
    pairs.sort(key=lambda x: x[1])
    negative = [{"feature": f, "coef": float(c)} for f, c in pairs[:n]]
    positive = [{"feature": f, "coef": float(c)} for f, c in pairs[-n:][::-1]]
    return {"positive": positive, "negative": negative}


def worst_errors(rows: list[dict[str, str]], y_true: np.ndarray, y_pred: np.ndarray, n: int = 25) -> dict[str, list[dict[str, object]]]:
    items = []
    for row, true, pred in zip(rows, y_true, y_pred):
        items.append(
            {
                "row_id": row["row_id"],
                "coffee_name": row["coffee_name"],
                "roaster": row["roaster"],
                "rating": float(true),
                "prediction": float(pred),
                "error": float(pred - true),
            }
        )
    items.sort(key=lambda x: x["error"])
    return {"false_low": items[:n], "false_high": items[-n:][::-1]}


def main() -> None:
    if not MODELING.exists() or not TRAIN_SPLIT.exists() or not VALIDATION_SPLIT.exists():
        raise SystemExit("Run python3 autoresearch/prepare.py first.")

    rows = read_csv(MODELING)
    train_ids = read_ids(TRAIN_SPLIT)
    validation_ids = read_ids(VALIDATION_SPLIT)
    train_rows = [row for row in rows if row["row_id"] in train_ids and row["rating"]]
    validation_rows = [row for row in rows if row["row_id"] in validation_ids and row["rating"]]
    y_train = np.array([float(row["rating"]) for row in train_rows], dtype=float)
    y_validation = np.array([float(row["rating"]) for row in validation_rows], dtype=float)

    encoder = FeatureEncoder()
    encoder.fit(train_rows)
    x_train = encoder.transform(train_rows)
    x_validation = encoder.transform(validation_rows)
    weights, intercept = fit_ridge(x_train, y_train, RIDGE_ALPHA)
    y_train_pred = predict(x_train, weights, intercept)
    y_pred = predict(x_validation, weights, intercept)

    train_metric_values = metrics(y_train, y_train_pred, prefix="train")
    validation_metric_values = metrics(y_validation, y_pred, prefix="val")
    metric_values = {**train_metric_values, **validation_metric_values}
    metric_values["concordance_gap"] = metric_values["train_concordance"] - metric_values["val_concordance"]
    bucket_rows = bucket_analysis(validation_rows, y_validation, y_pred)
    coef_rows = top_coefficients(encoder, weights)
    error_rows = worst_errors(validation_rows, y_validation, y_pred)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    write_predictions(validation_rows, y_validation, y_pred)
    report = {
        "config": {
            "model": "ridge",
            "alpha": RIDGE_ALPHA,
            "max_features": MAX_FEATURES,
            "min_df": MIN_DF,
            "max_df": MAX_DF,
            "structured_fields": STRUCTURED_FIELDS,
            "text_fields": TEXT_FIELDS,
            "train_rows": len(train_rows),
            "validation_rows": len(validation_rows),
        },
        "metrics": validation_metric_values,
        "train_metrics": train_metric_values,
        "overfitting": {
            "concordance_gap": metric_values["concordance_gap"],
        },
        "feature_coverage_train": coverage(train_rows),
        "feature_coverage_validation": coverage(validation_rows),
        "bucket_analysis": bucket_rows,
        "top_coefficients": coef_rows,
        "worst_errors": error_rows,
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with MODEL.open("wb") as f:
        pickle.dump({"encoder": encoder, "weights": weights, "intercept": intercept, "config": report["config"]}, f)
    write_results(metric_values)

    print(json.dumps({"metrics": metric_values, "report": str(REPORT.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
