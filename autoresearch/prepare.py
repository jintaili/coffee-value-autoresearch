"""Prepare canonical features and the fixed rating validation split."""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

from features import extract_rows, write_features


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "coffee.csv"
MODELING = ROOT / "data" / "modeling_coffee.csv"
SPLIT_DIR = ROOT / "data" / "splits"
TRAIN_SPLIT = SPLIT_DIR / "rating_train.csv"
VALIDATION_SPLIT = SPLIT_DIR / "rating_validation.csv"
SEED = 20260509
VALIDATION_FRAC = 0.15


def rating_bucket(rating: float) -> str:
    if rating <= 89:
        return "<=89"
    if rating >= 96:
        return ">=96"
    return str(int(rating))


def valid_rating(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def write_split(path: Path, row_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["row_id"])
        writer.writeheader()
        for row_id in row_ids:
            writer.writerow({"row_id": row_id})


def main() -> None:
    rows = extract_rows(SOURCE)
    write_features(rows, MODELING)

    rng = random.Random(SEED)
    by_bucket: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if valid_rating(row.rating):
            by_bucket[rating_bucket(float(row.rating))].append(row.row_id)

    train_ids: list[str] = []
    validation_ids: list[str] = []
    for bucket in sorted(by_bucket):
        ids = by_bucket[bucket][:]
        rng.shuffle(ids)
        n_validation = max(1, round(len(ids) * VALIDATION_FRAC))
        validation_ids.extend(ids[:n_validation])
        train_ids.extend(ids[n_validation:])

    train_ids = sorted(train_ids, key=lambda x: int(x))
    validation_ids = sorted(validation_ids, key=lambda x: int(x))
    write_split(TRAIN_SPLIT, train_ids)
    write_split(VALIDATION_SPLIT, validation_ids)

    print(f"wrote {MODELING.relative_to(ROOT)} rows={len(rows)}")
    print(f"wrote {TRAIN_SPLIT.relative_to(ROOT)} rows={len(train_ids)}")
    print(f"wrote {VALIDATION_SPLIT.relative_to(ROOT)} rows={len(validation_ids)}")
    print(f"seed={SEED} validation_frac={VALIDATION_FRAC}")


if __name__ == "__main__":
    main()
