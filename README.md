# Coffee Value Autoresearch

This repo develops coffee rating and price predictors from production-compatible coffee features. The project uses an agentic autoresearch workflow: an agent makes one focused experiment, commits the experimental code before running it, evaluates against a fixed validation split, keeps the commit if it improves, and records the run in a ledger.

The selected rating and price models are summarized in [MODEL_SELECTION.md](MODEL_SELECTION.md).

The rating and price autoresearch runs have concluded and are preserved as research traces:

- [Rating program](autoresearch/rating/program.md)
- [Rating results ledger](autoresearch/rating/results.tsv)
- [Rating summary](autoresearch/rating/notes.md)
- [Price program](autoresearch/price/program.md)
- [Price results ledger](autoresearch/price/results.tsv)
- [Selected price analysis](artifacts/price/6507aee_analysis.md)

![Rating autoresearch progress](autoresearch/rating/progress.png)

The reusable extraction contract lives in [coffee_value/features.py](coffee_value/features.py). That deterministic extractor is shared by rating and price work so the two regression models use the same canonical feature representation.

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
  price/
    program.md             # agent instructions and validation contract
    prepare.py             # creates canonical rows and fixed split
    train.py               # final selected price research script
    results.tsv            # experiment ledger
    analyze_selected.py    # selected model diagnostic report generator
```

Local datasets and artifacts are intentionally ignored:

```text
data/
artifacts/
```

## Run Selected Models

Place `coffee.csv` at `data/coffee.csv`, then run:

```bash
python3 autoresearch/rating/prepare.py
python3 autoresearch/rating/train.py
python3 autoresearch/price/prepare.py
python3 autoresearch/price/train.py
python3 autoresearch/price/analyze_selected.py
```

The scripts write generated files under `data/`, `data/splits/`, `artifacts/rating/`, and `artifacts/price/`.
