# Coffee Value Autoresearch

This repo develops coffee rating and price predictors from production-compatible coffee features. The project uses an agentic autoresearch workflow: an agent makes one focused experiment, commits the experimental code before running it, evaluates against a fixed validation split, keeps the commit if it improves, and records the run in a ledger.

The rating autoresearch run has concluded and is preserved as a research trace:

- [Rating program](autoresearch/rating/program.md)
- [Rating results ledger](autoresearch/rating/results.tsv)
- [Rating summary](autoresearch/rating/notes.md)

![Rating autoresearch progress](autoresearch/rating/progress.png)

The reusable extraction contract lives in [coffee_value/features.py](coffee_value/features.py). That deterministic extractor is shared by rating and future price work so the two regression models use the same canonical feature representation.

## Layout

```text
coffee_value/
  features.py              # locked shared deterministic extractor

autoresearch/
  rating/
    program.md             # agent instructions and validation contract
    prepare.py             # creates canonical rows and fixed split
    train.py               # final rating research script
    results.tsv            # experiment ledger
    notes.md               # human-readable run summary
```

Local datasets and artifacts are intentionally ignored:

```text
data/
artifacts/
```

## Run Rating Baseline

Place `coffee.csv` at `data/coffee.csv`, then run:

```bash
python3 autoresearch/rating/prepare.py
python3 autoresearch/rating/train.py
```

The scripts write generated files under `data/`, `data/splits/`, and `artifacts/rating/`.
