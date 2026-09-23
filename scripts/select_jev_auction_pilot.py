"""Pick a source-diverse, price-development-only auction audit sample."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from coffee_value.extraction.cache import digest
from coffee_value.extraction.state import build_review_state
from scripts.backfill_jev import read_csv

OUT = ROOT / "artifacts" / "jev_auction"
STRICT = re.compile(r"\bauction lot\b|\b(?:this|the) coffee\b.{0,90}\bauction\b|\b(?:bought|purchased|sold)\b.{0,80}\bat auction\b", re.I)
GENERIC = re.compile(r"\bauction system\b", re.I)
ANY = re.compile(r"\bauction\b", re.I)


def main() -> None:
    source = read_csv(ROOT / "data" / "coffee.csv")
    train_ids = {row["row_id"] for row in read_csv(ROOT / "data" / "splits" / "price_train.csv")}
    groups: dict[str, list[str]] = {name: [] for name in ("strict", "generic", "other_auction", "no_auction")}
    for rid in sorted(train_ids, key=int):
        text = " ".join(build_review_state(source[int(rid)]).values())
        name = ("strict" if STRICT.search(text) else "generic" if GENERIC.search(text)
                else "other_auction" if ANY.search(text) else "no_auction")
        groups[name].append(rid)
    limits = {"strict": 31, "generic": 15, "other_auction": 5, "no_auction": 10}
    chosen = {name: sorted(ids, key=digest)[:limits[name]] for name, ids in groups.items()}
    selected = [rid for name in limits for rid in chosen[name]]
    if len(selected) != len(set(selected)):
        raise ValueError("auction pilot selected duplicate rows")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "targeted_pilot_ids.txt").write_text("\n".join(selected) + "\n")
    (OUT / "pilot_selection.json").write_text(json.dumps({
        "source": "price_train only; fixed hash selection within source-context strata",
        "available": {name: len(ids) for name, ids in groups.items()},
        "selected": chosen,
        "selected_total": len(selected),
    }, indent=2))
    print(json.dumps({"selected_total": len(selected),
                      "selected_per_group": {name: len(ids) for name, ids in chosen.items()},
                      "ids_file": str(OUT / "targeted_pilot_ids.txt")}))


if __name__ == "__main__":
    main()
