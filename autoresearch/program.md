# Coffee Rating Autoresearch Program

You are improving a coffee rating predictor. Work like an autoresearch agent: make one focused change, run the fixed validation, read the diagnostics, record the result, and repeat.

## Goal

Maximize fixed-validation pairwise rating concordance for CoffeeReview-style ratings using features that can later be extracted from roaster product pages.

Primary score:

```text
val_concordance
```

Higher is better. Equal true-rating pairs are ignored.

## Data Contract

Before the first experiment, run:

```bash
python3 autoresearch/prepare.py
```

This writes:

- `data/modeling_coffee.csv`
- `data/splits/rating_train.csv`
- `data/splits/rating_validation.csv`

The validation split is fixed:

- 85% train / 15% validation
- stratified random by rating bucket
- seed `20260509`

Do not change the validation split while running experiments. If extraction logic changes, rerun `prepare.py`; it must regenerate both train and validation features using the same shared extractor.

## Main Loop

Repeat this loop:

1. Inspect the previous report in `artifacts/rating_baseline/report.json` and prior runs in `autoresearch/results.tsv`.
2. Choose one hypothesis to test.
3. Edit only the files needed for that hypothesis.
4. Run:

   ```bash
   python3 autoresearch/prepare.py
   python3 autoresearch/train.py
   ```

5. Compare the new `val_concordance` to the previous best.
6. Inspect calibration diagnostics, especially rating-bucket mean error.
7. Keep the change if it improves `val_concordance` without creating severe bucket bias.
8. If the result is worse or suspicious, revert or revise the change before continuing.
9. Leave `autoresearch/results.tsv` as the running experiment ledger.

Prefer small, attributable changes over broad rewrites.

## Editable Surface

Allowed to edit:

- `autoresearch/features.py`
- `autoresearch/prepare.py`
- `autoresearch/train.py`
- `autoresearch/program.md`

Be careful editing `prepare.py`: the split seed and split policy are part of the validation contract. Model-selection comparisons are only meaningful when validation row IDs stay fixed.

## Features In Scope

Tier 1 deterministic features:

- `origin_country`
- `origin_region`
- `process_method`
- `variety`
- `is_blend`
- `is_espresso`
- `is_decaf`
- `producer_or_farm_present`
- `altitude_present`
- `roaster_country`

Tier 2 text fields:

- `sensory_text`
- `producer_text`

Hold off on sensory buckets and LLM extraction until deterministic extraction and the baseline loop are stable.

## Candidate Experiments

Good first experiments:

- improve country alias extraction
- improve roaster-country normalization
- improve process-method extraction
- improve variety extraction
- test excluding high-cardinality `origin_region`
- test using `origin_region` only when frequent enough
- test text source choices: `sensory_text` only vs `sensory_text + producer_text`
- tune TF-IDF `max_features`, `min_df`, `max_df`, and unigram/bigram settings
- tune Ridge `alpha`
- add concordance-by-rating-gap diagnostics

Avoid for now:

- reviewer component score inputs
- price inputs
- LLM extraction
- neural fine-tuning
- changing the validation split to improve score

## Required Run Outputs

Each call to `python3 autoresearch/train.py` should produce:

- `artifacts/rating_baseline/report.json`
- `artifacts/rating_baseline/validation_predictions.csv`
- `artifacts/rating_baseline/model.pkl`
- appended row in `autoresearch/results.tsv`

The report must include:

- `val_concordance`
- Spearman correlation
- MAE
- RMSE
- within-1 and within-2 accuracy
- feature coverage
- rating-bucket bias table
- top positive and negative coefficients when available
- worst false-high predictions
- worst false-low predictions

## Acceptance Rule

Select models by `val_concordance`.

Tie-breakers:

1. Lower absolute bucket bias, especially in `<=89`, `95`, and `>=96`.
2. Lower MAE.
3. Simpler feature processing.

Flag any candidate that raises concordance by compressing all predictions toward the center. A useful ranking model still needs plausible displayed ratings.
