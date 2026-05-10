# Model Selection

This repo now has selected rating and price model configurations. The experiment
ledgers remain the source of truth for the autoresearch trace; this file records
the shipping choices and the main caveats.

## Rating Model

Selected result: `ship_exp15` in `autoresearch/rating/results.tsv`.

Configuration:

- TF-IDF text features with 6000 max features and bigrams.
- MiniLM sentence embeddings in a hybrid feature matrix.
- Ridge regression with `alpha=1`.
- No roaster identity feature.

Validation summary:

- `val_rmse`: 0.939847
- `train_rmse`: 0.891044
- `overfit_gap`: 0.048803
- `val_spearman`: 0.870150
- `val_mae`: 1.049560
- `val_p90_ae`: 1.557227

This was selected over later mpnet and roaster-aware variants because those
variants either regressed validation error or increased overfit risk without a
strong enough validation gain.

## Price Model

Selected result: `6507aee` in `autoresearch/price/results.tsv`.

Configuration:

- Target: `log(price_usd_per_100g_real)`.
- Model: ElasticNet with `alpha=0.0001`, `l1_ratio=0.1`.
- Structured production-compatible features from `coffee_value/features.py`.
- TF-IDF over sensory and producer text, 24000 max features, bigrams.
- Package-size features added in `autoresearch/price/train.py`:
  - standardized `log(package_grams)`
  - missing package flag
  - small package flags for `<=20g`, `<=50g`, and `<=100g`

Validation summary:

- `val_rmsle`: 0.259166
- `train_rmsle`: 0.173046
- `overfit_gap`: 0.497668
- `val_spearman`: 0.787833
- `val_mae`: 3.875031
- `val_median_ae`: 0.968472
- `val_p90_ae`: 5.098572
- `val_low_decile_rmsle`: 0.374266
- `val_high_decile_rmsle`: 0.453428
- `val_max_abs_quantile_mean_error`: 21.953748

The package-size features materially improved rare luxury coffee handling. The
prior best model had `val_rmsle=0.301006`, high-decile RMSLE `0.590634`, and
worst quantile mean error `29.931073`.

Known limitation: the most expensive decile is still compressed downward. In
the selected price analysis, the top validation decile has mean true price
`56.21`, mean prediction `34.26`, and mean error `-21.95`. The remaining worst
cases are extreme luxury lots such as Panama Geisha, Kona, Ninety Plus, and
civet/luwak coffees.

Post-hoc top-decile calibration and high-price sample weighting were tested
after the package-size change. They improved some high-decile diagnostics but
were discarded because they either regressed the primary validation metric or
looked too marginal relative to overfit and p90/median error risk.

## Reproduction

Rating:

```bash
python3 autoresearch/rating/prepare.py
python3 autoresearch/rating/train.py
```

Price:

```bash
python3 autoresearch/price/prepare.py
python3 autoresearch/price/train.py
python3 autoresearch/price/analyze_selected.py
```

Generated datasets and model artifacts are local outputs under `data/` and
`artifacts/`. The selected price analysis report is tracked separately for
review convenience.
