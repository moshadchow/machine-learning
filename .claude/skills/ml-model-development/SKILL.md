---
name: ml-model-development
description: Scikit-learn model development best practices for this project — feature/target separation, train/test splits, evaluation metrics, and guardrails.
---

# ML Model Development

Project conventions for building and evaluating ML models (scikit-learn `LinearRegression` predicting broker order counts from turnover).

## When to Use

- Before training or modifying any model
- When changing features, targets, or evaluation methodology
- When adding guardrails for insufficient data

## Core Rules

**Feature/target convention:**
- Feature: `totalValue` (turnover in BDT)
- Targets: `total` (Submitted Orders), `totalTrade` (Executed Trades)
- Column names: `Date`, `BrokerId`, `totalValue`, `total`, `totalTrade`
- `Date` and `BrokerId` are metadata — never model features

**Estimator:** `sklearn.linear_model.LinearRegression` — do not change without a documented reason.

**Train/test split:** Use a **temporal split** — train on earlier dates, test on the most recent dates. Daily observations from the same broker are correlated; a random split leaks future days into training.

**Minimum data guards:**
- `MIN_OBS_TO_TRAIN = 2` — below this, refuse to train
- `MIN_OBS_FOR_METRICS = 10` — below this, train but withhold metrics (R²/MAE undefined on <2 test rows)
- `MIN_DATES_FOR_METRICS = 2` — below this, raise `InsufficientDataError`

**Never:**
- Fabricate samples or treat aggregate totals as individual transactions
- Replace missing/zero-filled values with observations
- Train on incomplete or invalid datasets
- Write API responses, tokens, or the dataset to disk

## Verification

- Confirm metrics printed for both train and test sets (or an explicit insufficient-data notice)
- Confirm `predictions_vs_actual` table includes `Date`/`BrokerId`
- Confirm `predict_orders(model, turnover)` returns `{total: int, totalTrade: int}`

## References

- `model.py` — training/evaluation/prediction logic
- `specs/02-oms-api-integration.md` — methodology changes vs the notebook
