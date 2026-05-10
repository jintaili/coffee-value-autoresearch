# Coffee Rating Autoresearch Program

You are improving a coffee rating predictor. Work like the following: make one focused change, run the fixed validation, read the diagnostics, record the result, and repeat.

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
python3 autoresearch/rating/prepare.py
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

## Run Branch

Create one new branch for the whole autonomous run before starting the loop, for example:

```bash
git checkout -b run/rating-autoresearch-YYYYMMDD
```

Stay on that branch for all experiments in the run. Do not create a new branch for each individual experiment. Each experiment should be committed before it is run; keep the commit if it improves, or reset it away if it does not.

## Main Loop

Repeat this loop:

1. Inspect the previous report in `artifacts/rating/report.json` and prior runs in `autoresearch/rating/results.tsv`.
2. Choose one hypothesis to test.
3. Edit only the files needed for that hypothesis.
4. Commit the experimental code before running it.
5. Run:

   ```bash
   python3 autoresearch/rating/prepare.py
   python3 autoresearch/rating/train.py
   ```

6. Compare the new `val_concordance` to the previous best.
7. Inspect calibration diagnostics, especially rating-bucket mean error.
8. Leave `autoresearch/rating/results.tsv` as the running experiment ledger.
9. If the change improves `val_concordance` without creating severe bucket bias or overfitting, keep the commit and continue from it.
10. If the result is worse or suspicious, reset the branch back to where it started before that experiment.
11. Push the run branch periodically so the remote history reflects the kept experiment sequence.

To determine overfitting, use these thresholds for concordance_gap = train_concordance - val_concordance:

- <= 0.030: acceptable
- 0.030 to 0.050: suspicious; keep only with clear validation improvement and sane diagnostics
- \> 0.050: reject/reset unless explicitly justified

## Stopping Criteria

Stop the autonomous run when 9 consecutive experiments fail to produce a new best `val_concordance`.

## Editable Surface

Allowed to edit:

- `autoresearch/rating/prepare.py`
- `autoresearch/rating/train.py`

Do not edit the locked shared extractor during ordinary rating-model autoresearch:

- `coffee_value/features.py`

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

## Candidate Experiments

The following are example experiments. But you should not limit yourself to only these:

- improve process-method extraction
- improve variety extraction
- test ways to incorporate information from high-cardinality `origin_region` without overfitting
- test text source choices: `sensory_text` only vs `sensory_text + producer_text`
- tune TF-IDF `max_features`, `min_df`, `max_df`, and unigram/bigram settings
- test compressed text representations such as TF-IDF followed by SVD before Ridge or gradient boosting
- test sentence embeddings for `sensory_text` and `producer_text`
- test word-vector mean embeddings such as GloVe/FastText as a compact baseline if local vectors are available
- test different regression models
- other (hyper)parameter tuning

Avoid for now:

- price inputs
- LLM extraction
- neural fine-tuning
- changing the validation split to improve score

For any embedding experiment, preserve offline/online parity: the exact embedding model, preprocessing, text fields, vector dimensions, and artifact version must be recorded and reusable at inference time.

## Required Run Outputs

Each call to `python3 autoresearch/rating/train.py` should produce:

- `artifacts/rating/report.json`
- `artifacts/rating/validation_predictions.csv`
- `artifacts/rating/model.pkl`
- appended row in `autoresearch/rating/results.tsv`

The report must include:

- `val_concordance`
- Spearman correlation
- MAE
- RMSE
- within-1 and within-2 accuracy
- feature coverage
- rating-bucket bias table
- top impact features when available
- worst false-high predictions
- worst false-low predictions

## Acceptance Rule

Select models by `val_concordance`.

Tie-breakers:

1. Lower absolute bucket bias, especially in `<=89`, `95`, and `>=96`.
2. Lower MAE.
3. Simpler feature processing.
4. Lower overfitting.

Flag any candidate that raises concordance by compressing all predictions toward the center, exploiting unstable high-cardinality features, or otherwise showing signs of overfitting. A useful ranking model still needs plausible displayed ratings and robust generalization.
