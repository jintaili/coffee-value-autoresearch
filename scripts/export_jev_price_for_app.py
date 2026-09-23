"""Export the selected JEV price model without training-module pickle classes."""

from __future__ import annotations

import argparse
import hashlib
import pickle
from pathlib import Path

import numpy as np

from coffee_value.extraction.encoding import ENCODING_VERSION, feature_names
from coffee_value.extraction.questions import CONTRACT_VERSION, MODEL, question_hash


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts" / "jev_auction" / "price_jev_pg_model.pkl"


def export(source: Path, destination: Path) -> dict:
    with source.open("rb") as stream:
        bundle = pickle.load(stream)
    if bundle["run"] != "jev_pg" or bundle["use_auction"] or not bundle["use_pg_interaction"]:
        raise ValueError("source is not the selected JEV + Panama × Gesha price model")

    encoder = bundle["text_encoder"]
    base_names = feature_names()
    names = bundle["feature_names"]
    if names[:len(base_names)] != base_names:
        raise ValueError("JEV feature order has changed")
    if names[len(base_names)] != "interaction:origin_country:Panama*variety_gesha:supported":
        raise ValueError("Panama × Gesha feature is missing")
    if len(names) != len(bundle["model"].coef_):
        raise ValueError("model coefficient width does not match feature names")
    if bundle["base_contract"] != CONTRACT_VERSION:
        raise ValueError("JEV extraction contract has changed")

    portable = {
        "format": "coffee-jev-price-v1",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "contract_version": CONTRACT_VERSION,
        "encoding_version": ENCODING_VERSION,
        "jev_model": MODEL,
        "question_hash": question_hash(),
        "feature_names": names,
        "text_column_offset": bundle["text_column_offset"],
        "structured_vocab": encoder.structured_vocab,
        "text_vocab": encoder.text_vocab,
        "idf": np.asarray(encoder.idf, dtype=np.float64),
        "package_log_mean": float(encoder.package_log_mean),
        "package_log_std": float(encoder.package_log_std),
        "coefficients": np.asarray(bundle["model"].coef_, dtype=np.float64),
        "intercept": float(bundle["model"].intercept_),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as stream:
        pickle.dump(portable, stream, protocol=pickle.HIGHEST_PROTOCOL)
    with destination.open("rb") as stream:
        restored = pickle.load(stream)
    if restored["source_sha256"] != portable["source_sha256"] or not np.array_equal(
        restored["coefficients"], portable["coefficients"]
    ):
        raise ValueError("portable artifact failed round-trip validation")
    return {"source": str(source), "destination": str(destination), "bytes": destination.stat().st_size,
            "features": len(names), "question_hash": portable["question_hash"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(export(args.source, args.destination))


if __name__ == "__main__":
    main()
