---
name: temporal-split-evaluation
description: Temporal split methodology — train on earlier dates, test on most recent dates — to prevent leakage from correlated daily observations.
---

# Temporal Split Evaluation

The temporal split used by `model.py` to evaluate the broker order prediction model.

## Why Temporal, Not Random

Daily observations from the same broker are correlated. A random `train_test_split` leaks future days into training, producing misleadingly optimistic metrics. The temporal split holds out the most recent dates for evaluation.

## Method

Given a DataFrame with column `Date`:
1. Collect sorted distinct dates
2. Test set = most recent `ceil(TEST_SIZE × n_dates)` distinct dates (at least one)
3. Train set = all earlier dates
4. Split **by date** (not by row) so no same-day data crosses the boundary

## Requirements

- At least `MIN_DATES_FOR_METRICS = 2` distinct dates
- At least `MIN_OBS_FOR_METRICS = 10` total observations for meaningful metrics
- Below these, withhold metrics with an explicit notice (do not train metrics-less models silently)

## Outputs

`evaluate_model()` returns:
- `r2_per_target`, `mae_per_target`, `r2_avg`, `mae_avg`
- `predictions_vs_actual` table with `Date`, `BrokerId`, feature, actuals, predictions
- `train_rows`, `test_rows`, `train_dates`, `test_dates`

## References

- `model.py::temporal_split` — the split implementation
- `model.py::evaluate_model` — evaluation wrapper
- `AGENTS.md` — split rationale and small-data policy
