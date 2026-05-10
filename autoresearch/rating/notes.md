# Rating Autoresearch Summary

## Objective

Predict CoffeeReview-style ratings from features that can later be extracted from arbitrary roaster product pages.

## Validation

- Fixed 85/15 train/validation split
- Stratified by rating bucket
- Seed: `20260509`
- Primary metric: `val_concordance`
- Overfitting signal: `train_concordance - val_concordance`

## Feature Contract

The rating model consumes canonical deterministic features produced by `coffee_value/features.py`.

Tier 1 fields:

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

## Current Result

See `autoresearch/rating/results.tsv` for the current accepted row.

The accepted configuration is a production-oriented hybrid using deterministic Tier 1 fields, TF-IDF text features, MiniLM sentence embeddings, and Ridge regression.

## Notable Research Lessons

- High-cardinality `origin_region` needs careful handling because it can add many sparse parameters.
- Text carries strong signal, but it must preserve offline/online parity for production use.
- Concordance is the selection metric, but bucket bias and concordance gap are required guardrails.
