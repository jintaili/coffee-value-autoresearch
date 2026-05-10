# Coffee Price Autoresearch Program

You are improving a coffee price predictor. The shared feature extractor and price normalization are locked; focus on model selection and output engineering.

## Goal

Predict comparable retail coffee price:

```text
price_usd_per_100g_real
```

The initial price parser uses deterministic nominal FX and a CPI hook. Do not change price parsing during ordinary model autoresearch.

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

## Editable Surface

Allowed for model experiments:

- `autoresearch/price/train.py`
- `autoresearch/price/program.md`

Locked during ordinary price autoresearch:

- `coffee_value/features.py`
- `coffee_value/price.py`
- `autoresearch/price/prepare.py`

## Candidate Experiments

Focus on:

- target transform: raw, log, log1p
- output calibration and clipping
- model family
- text representation
- regularization
- robust treatment of high-price outliers

Avoid:

- changing price parsing
- changing the validation split to improve score
- using rating as an input unless explicitly requested
- LLM extraction

## Metrics

Prioritize price ranking and usable dollar error:

- validation Spearman
- MAE in USD per 100g after inverse transform
- median absolute error in USD per 100g
- RMSLE
- bias by price quantile
- train/validation ranking gap
