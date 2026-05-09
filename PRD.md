# Coffee Rating Predictor PRD

## Purpose

Build the first training pipeline for a coffee rating predictor using only `data/coffee.csv`. The model should estimate a CoffeeReview-style rating from features that can later be extracted from arbitrary roaster product webpages.

The broader product is a coffee rating and price web app. A user provides a coffee product page, an LLM extracts model-compatible coffee features, and the app predicts likely rating and price to help judge value. This PRD covers the first milestone: deterministic feature extraction and a rating baseline.

## Goals

- Train a rating model on the larger CoffeeReview dataset only.
- Use a shared canonical feature schema that can later support a separate price model.
- Keep feature extraction deterministic for the first milestone.
- Favor operationalizable features: fields that a production LLM can extract from roaster pages.
- Prioritize ranking/concordance metrics over pure point prediction.
- Produce fixed validation artifacts so autoresearch runs are comparable.

## Non-Goals

- Do not train the price predictor yet.
- Do not use an LLM to extract training features yet.
- Do not use reviewer component scores as model inputs.
- Do not build the web app yet.
- Do not optimize on changing validation splits across runs.

## Dataset

Primary source:

- `data/coffee.csv`

Known shape from inspection:

- 8,918 rows
- 20 columns
- review dates from 1997-02-01 through 2026-03-01
- rating available for 8,911 rows
- price available for 7,007 rows, reserved for later price work

The older files, `data/coffee_analysis.csv` and `data/simplified_coffee.csv`, should not be used as additional training rows because they substantially overlap with `data/coffee.csv`.

## Excluded Inputs

The rating model must not use these columns as inputs:

- `aroma`
- `body`
- `flavor`
- `aftertaste`
- `with_milk`
- `acid_structure`
- `acidity`
- `rating`
- `est_price`

Reason: reviewer sub-scores leak the rating process, and price should not be used to predict rating in the first production-compatible model.

## Canonical Feature Schema

The deterministic extractor should produce one row per coffee with these fields.

### Identifiers

- `row_id`: stable source row index
- `coffee_name`: from `bean`
- `roaster`: raw roaster name, retained for analysis and optional capped feature experiments
- `review_date`: parsed date
- `review_year`: integer year

### Tier 1 Model Features

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

### Tier 2 Text Fields

- `sensory_text`
- `producer_text`

Definitions:

- `sensory_text`: tasting notes, cup profile, flavor, aroma, body, acidity, sweetness, and finish language.
- `producer_text`: origin, farm, producer, process, variety, altitude, certification, lot, and roaster background details.

For `data/coffee.csv`:

- `sensory_text` should initially come from `blind_assessment`.
- `producer_text` should initially come from `notes` plus `bottom_line`.

### Targets

- `rating`: numeric target for this milestone
- `price_100g_usd`: reserved for later, not required in the rating pipeline

## Deterministic Extraction Rules

The first extractor should use simple, reviewable rules rather than LLM calls.

### `origin_country`

Parse from `origin` using a controlled country list and common coffee-origin aliases. If multiple countries are present, use a normalized multi-country label or `blend_multi_origin`.

Examples:

- `Nyeri growing region, south-central Kenya` -> `Kenya`
- `Guji Zone, southern Ethiopia` -> `Ethiopia`
- `Panama; Ethiopia` -> `blend_multi_origin`

### `origin_region`

Extract a coarse region/farm string from `origin` after removing the country when possible. This should remain conservative. Unknown is acceptable.

### `roaster_country`

Parse from `location`. For US state names, map to `United States`. Normalize common Taiwan city names to `Taiwan`. Keep unknown rather than guessing aggressively.

### `process_method`

Extract from `origin`, `notes`, `bottom_line`, and `blind_assessment` using keyword rules.

Allowed values should include:

- `washed`
- `natural`
- `honey`
- `anaerobic`
- `carbonic_maceration`
- `wet_hulled`
- `lactic`
- `unknown`

Support multi-label extraction if several methods appear.

### `variety`

Extract known coffee varieties from text.

Initial vocabulary:

- `gesha`
- `geisha`
- `bourbon`
- `typica`
- `caturra`
- `catuai`
- `sl28`
- `sl34`
- `pacamara`
- `maragogipe`
- `mokka`
- `mokkita`
- `pink_bourbon`
- `ruiru`
- `castillo`
- `java`

Normalize spelling variants where reasonable.

### Boolean Flags

- `is_blend`: true if blend language appears or multiple origins are clearly listed.
- `is_espresso`: true if espresso appears in coffee name or assessment.
- `is_decaf`: true if decaf/decaffeinated appears.
- `producer_or_farm_present`: true if text contains farm, estate, producer, cooperative, mill, station, finca, hacienda, or named producer-like cues.
- `altitude_present`: true if text contains altitude/elevation or meter/feet patterns.

## Text Feature Processing Options

The autoresearch loop should compare text representations, but the validation set must stay fixed.

Initial options:

### TF-IDF

Use saved vectorizers to maintain offline/online parity. The fitted vocabulary and preprocessing must be serialized with the model artifact.

Recommended grid:

- input: `sensory_text`
- input: `sensory_text + producer_text`
- `max_features`: 500, 1000, 2000
- `min_df`: 5, 10
- `max_df`: 0.85
- `ngram_range`: `(1, 1)` first; `(1, 2)` only as a later ablation
- `sublinear_tf`: true

### Sentence Embeddings

Use the same embedding model in training and production. Embed `sensory_text` and `producer_text` separately, then concatenate with structured features.

This is likely the most production-aligned NLP representation if roaster-page language differs from CoffeeReview critic prose.

### GloVe / Word-Vector Means

Optional low-priority baseline. It is compact, but likely weaker for specialty coffee phrases and short text.

## Model Options

Start with Ridge regression.

Candidate models:

- `Ridge`
- `ElasticNet`
- gradient boosting on structured features plus embeddings or SVD-compressed text features

Do not start with neural fine-tuning.

## Validation Protocol

Validation must be fixed before model iteration.

Create persistent split files under a reproducible path such as:

- `data/splits/rating_train.csv`
- `data/splits/rating_validation.csv`

Recommended fixed split:

- train: 85% of valid rating rows
- validation: 15% of valid rating rows
- stratification: rating bucket
- seed: fixed and recorded, initially `20260509`

Rationale:

- The first phase is feature/model research, not final production evaluation.
- A fixed stratified random validation set gives a stable, representative score while preserving rare high/low ratings.
- A temporal split is useful later, but early experiments should not conflate feature quality with historical drift in reviews, roasters, and language.
- Keeping the split files fixed prevents autoresearch from chasing split noise.

Use rating buckets rather than exact rating values for stratification:

- `<=89`
- `90`
- `91`
- `92`
- `93`
- `94`
- `95`
- `>=96`

There is no required test split in the first milestone. Later, after the feature stack stabilizes, add a separate final evaluation such as recent-year holdout or unseen-roaster holdout.

The agent may report temporal or roaster-group diagnostics as secondary analysis, but selection should be based on the fixed validation set.

## Primary Metrics

Ranking/concordance should be prioritized for rating.

Primary validation metrics:

- pairwise concordance accuracy, sampled deterministically if necessary

This is the single model-selection metric for autoresearch. Higher is better.

Definition: for validation pairs with different true ratings, count the pair as correct when the predicted rating difference has the same sign as the true rating difference. Ignore equal-rating pairs.

Secondary metrics:

- Spearman rank correlation
- Kendall tau
- MAE
- RMSE
- within-1-point accuracy
- within-2-point accuracy

Calibration and bias diagnostics:

- mean error by true rating quantile
- median error by true rating quantile
- MAE by true rating quantile
- predicted rating distribution by true rating quantile
- residual plot data by rating bucket

The model should be roughly unbiased across rating quantiles. A high-ranking model that systematically underpredicts high-rated coffees or overpredicts low-rated coffees should be flagged.

If concordance improves but quantile bias becomes severe, prefer the simpler or less biased candidate unless the improvement is large and explainable.

## Required Outputs

Each autoresearch run should produce:

- feature coverage table
- missingness table
- validation metrics
- quantile error analysis
- top positive and negative coefficients for Ridge where available
- worst false-high predictions
- worst false-low predictions
- prediction CSV for the fixed validation set
- model artifact with serialized preprocessing

## First Milestone

Build:

- deterministic extractor script
- canonical modeling CSV
- fixed rating split files
- Ridge rating baseline
- validation report

Suggested files:

- `autoresearch/program.md`
- `autoresearch/prepare.py`
- `autoresearch/train.py`
- `autoresearch/results.tsv`
- `data/modeling_coffee.csv`
- `data/splits/rating_train.csv`
- `data/splits/rating_validation.csv`

## Open Decisions

- Whether `roaster` should be excluded entirely from rating or tested as capped `top_k`.
- Which sentence embedding model to use if embedding experiments are added.
- Whether `origin_region` should be categorical, text, or omitted until extraction quality is clearer.
