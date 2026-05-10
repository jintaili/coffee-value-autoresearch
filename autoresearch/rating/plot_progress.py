"""Plot rating autoresearch progress from results.tsv."""

from __future__ import annotations

import csv
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "autoresearch" / "rating" / "results.tsv"
OUT = ROOT / "autoresearch" / "rating" / "progress.png"


def read_results() -> list[dict[str, str]]:
    with RESULTS.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return [row for row in rows if row.get("val_concordance")]


def short_label(row: dict[str, str]) -> str:
    run = row.get("run", "").strip()
    desc = row.get("description", "").strip()
    desc = desc.replace("[reset, ", "[").replace("[reset,", "[")
    if run == "main":
        return "baseline"
    if not desc:
        return run
    return f"{run}: {textwrap.shorten(desc, width=32, placeholder='...')}"


def main() -> None:
    rows = read_results()
    xs = list(range(len(rows)))
    vals = [float(row["val_concordance"]) for row in rows]

    kept_idx: list[int] = []
    running_best: list[float] = []
    best = float("-inf")
    for i, val in enumerate(vals):
        if val > best:
            best = val
            kept_idx.append(i)
        running_best.append(best)

    fig, ax = plt.subplots(figsize=(12, 5.6), dpi=160)
    ax.scatter(xs, vals, s=12, c="#cfcfcf", alpha=0.55, label="Discarded", zorder=1)
    ax.scatter(
        [xs[i] for i in kept_idx],
        [vals[i] for i in kept_idx],
        s=26,
        c="#18b66a",
        edgecolor="#0f7f49",
        linewidth=0.7,
        label="Kept",
        zorder=3,
    )
    ax.step(xs, running_best, where="post", color="#18b66a", linewidth=1.2, label="Running best", zorder=2)

    y_span = max(vals) - min(vals)
    y_offset = max(y_span * 0.045, 0.0007)
    for i in kept_idx:
        label = short_label(rows[i])
        ax.annotate(
            label,
            xy=(xs[i], vals[i]),
            xytext=(xs[i] + 0.4, vals[i] + y_offset),
            fontsize=7,
            color="#16894f",
            rotation=27,
            ha="left",
            va="bottom",
        )

    ax.set_title(f"Rating Autoresearch Progress: {len(rows)} Experiments, {len(kept_idx)} Kept Improvements")
    ax.set_xlabel("Experiment #")
    ax.set_ylabel("Validation Concordance (higher is better)")
    ax.grid(True, color="#e8e8e8", linewidth=0.8, alpha=0.7)
    ax.legend(loc="upper left", fontsize=8, frameon=True)

    pad = max(y_span * 0.12, 0.002)
    ax.set_ylim(min(vals) - pad, max(vals) + pad * 2.2)
    ax.set_xlim(-1, len(rows))
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
