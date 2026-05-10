# Coffee Price Autoresearch Program

You are improving a coffee price predictor. The shared feature extractor and price normalization are locked; focus on model selection and output engineering.

## Goal

Predict comparable retail coffee price:

```text
price_usd_per_100g_real
```

Do not change price parsing during ordinary model autoresearch.

The goal is simple:

```text
get the lowest val_rmsle
```

Lower is better.

## Data Contract

Before experiments, run:

```bash
python3 autoresearch/price/prepare.py
```

This writes:

- `data/modeling_price.csv`
- `artifacts/price/parse_report.json`

Use only rows where:

```text
price_parse_status == ok
```

The first run should always establish the baseline by running `train.py` as-is.

## Editable Surface

Allowed for model experiments:

- `autoresearch/price/train.py`

Locked during ordinary price autoresearch:

- `coffee_value/features.py`
- `coffee_value/price.py`
- `autoresearch/price/prepare.py`

## Feature Contract

Align the price model with the accepted rating configuration (`ship_exp15`) unless an experiment explicitly tests a change.

Use production-compatible shared features:

- `origin_country`
- `process_method`
- `variety`
- `is_blend`
- `is_espresso`
- `is_decaf`
- `producer_or_farm_present`
- `altitude_present`
- `roaster_country`
- `sensory_text`
- `producer_text`

Do not use `rating` as an input unless explicitly requested. The rating and price heads should be independent estimates from the same webpage-compatible feature contract.

## Results TSV

Log every experiment to `autoresearch/price/results.tsv`, tab-separated:

```text
commit	val_rmsle	overfit_gap	status	description
```

- `commit`: short git commit hash for the experimental code.
- `val_rmsle`: primary validation metric; use `0.000000` for crashes.
- `overfit_gap`: `val_rmsle - train_rmsle`; use `0.000000` for crashes.
- `status`: `keep`, `discard`, or `crash`.
- `description`: short free text description of what changed.

## Experiment Loop

Run on a dedicated branch. Loop:

1. Look at git state and the current best result in `autoresearch/price/results.tsv`.
2. Edit `autoresearch/price/train.py` with one experimental idea.
3. Commit the experimental code before running it.
4. Run:

   ```bash
   python3 autoresearch/price/train.py > run.log 2>&1
   ```

5. Read the summary:

   ```bash
   grep "^val_rmsle:\\|^train_rmsle:\\|^overfit_gap:" run.log
   ```

6. Append the result to `autoresearch/price/results.tsv`.
7. If `val_rmsle` improved, keep the commit.
8. If the result is equal/worse, crashed, or badly overfit, reset back to where the experiment started.
9. Continue until manually stopped.

Do not pause to ask whether to continue once the loop begins.

## Candidate Experiments

Focus on:

- target transform: raw, log, log1p, and others
- output calibration
- model family
- regularization
- robust treatment of high-price outliers

Avoid:

- changing price parsing
- changing the validation split to improve score
- using rating as an input unless explicitly requested
- LLM extraction

## Metrics

Primary:

- `val_rmsle`

Secondary diagnostics:

- `val_spearman`
- MAE in USD per 100g after inverse transform
- median absolute error in USD per 100g
- bias by price quantile
- `train_rmsle`
- `overfit_gap = val_rmsle - train_rmsle`

Guardrails:

- Prefer `overfit_gap <= 0.05`.
- Treat `overfit_gap > 0.10` as severe overfitting unless the validation improvement is large and diagnostics are sane.
- Flag candidates that improve RMSLE by collapsing predictions or badly worsening low/high price quantile bias.
