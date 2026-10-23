---
description: Create a spec file and feature branch for the next machine-learning step
argument-hint: "Step number and feature name e.g. 02 Loan Approval Evaluation"
allowed-tools: Read, Write, Glob, Bash(git:*)
---

You are a senior machine-learning developer spinning up a
new experiment/step for this project.

User input: $ARGUMENTS

## Step 1 — Parse the arguments
From $ARGUMENTS extract:

1. `step_number` — zero-padded to 2 digits: 2 → 02, 11 → 11

2. `feature_title` — human readable title in Title Case
   - Example: "Loan Approval Evaluation" or "Order Prediction Cleanup"

3. `feature_slug` — git and file safe slug
   - Lowercase, kebab-case
   - Only a-z, 0-9 and -
   - Maximum 40 characters
   - Example: loan-approval-evaluation, order-prediction-cleanup

If you cannot infer these from $ARGUMENTS, ask the user
to clarify before proceeding.

## Step 2 — Research the codebase
Read these before writing the spec:
- All `.ipynb` notebooks in the project root — current models,
  features, targets, and metrics in use
  - `logistic_regression_eg.ipynb` — logistic regression on
    `dataset/loan_prediction_dataset_v2.csv`
  - `order_prediction_dataset_v2.ipynb` — multi-output linear
    regression on the broker summary report
- `loan_prediction_dataset.py` — existing script (may be empty)
- The header row (first line) of every CSV in `dataset/` —
  confirm exact column names and formats before writing the spec
- All files in `specs/` — avoid duplicating existing plans

Check `specs/` to confirm the requested step is not already
planned. If it is, warn the user and stop.

If a section of the requested step is ambiguous (e.g. which
dataset, which target column, which metric), ask the user
before writing the spec.

## Step 3 — Write the spec
Generate a spec document with this exact structure:

---
# Spec: <feature_title>

## Overview
One paragraph describing what this experiment does and why
it exists at this stage of the project.

## Depends on
Which previous steps/notebooks this step requires.

## Data
Which dataset file in `dataset/` is used, the feature columns,
the target column, and any cleaning needed (e.g. removing
commas from numeric strings, handling missing values).
Always verify column names against the actual CSV header.

## Approach
The model type and preprocessing strategy, e.g.:
- train/test split ratio
- scaling/encoding steps
- estimator to use (LogisticRegression, LinearRegression, etc.)

## Evaluation
Which metrics will be reported and what counts as success.
- Classification: accuracy, and optionally precision/recall/F1
- Regression: MAE and R² (via `mean_absolute_error`, `r2_score`)

## Files to change
Every existing file that will be modified.

## Files to create
Every new notebook or script that will be created.

## New dependencies
Any new pip packages installed into `mlvenv`. If none: state
"No new dependencies".

## Rules for implementation
Specific constraints to follow. Always include:
- Use pandas for data handling and scikit-learn for
  models/metrics only
- Always set `random_state=42` on `train_test_split` and on
  any stochastic estimator, so results are reproducible
- Fit scalers/encoders on training data only — never leak
  test data into preprocessing (prefer `sklearn.pipeline.Pipeline`)
- Datasets live in `dataset/` — never move or edit raw CSVs;
  clean them in code instead
- Never commit `mlvenv/` or large data files
- Install new packages into the `mlvenv` virtual environment
- Explore interactively in `.ipynb` notebooks; move reusable
  training code into `.py` scripts
- Notebooks must run cleanly top-to-bottom
  (Kernel → Restart & Run All) before being committed
- Every code section gets a short markdown header or comment
  explaining the step
- Print evaluation metrics (and a predictions-vs-actual table
  for regression), not just raw predictions
- Python: 4-space indent, `snake_case` names, no unused imports
- No new dependencies unless explicitly approved

## Definition of done
A specific testable checklist, e.g. "notebook runs without
errors", "test accuracy ≥ X", "metrics printed for both
train and test sets".
---

## Step 4 — Save the spec
Save to: `specs/<step_number>-<feature_slug>.md`

## Step 5 — Set up the branch
Check whether this is a git repository:
`git rev-parse --is-inside-work-tree`

- If it IS a repo: create and switch to the feature branch
  `feature/<step_number>-<feature_slug>`
- If it is NOT a repo: warn the user that no git repository
  exists yet and ask whether to run `git init` first.
  Do not initialize a repository on your own.

## Step 6 — Report to the user
Print a short summary in this exact format:
```
Branch:    <branch_name or "(no git repo yet)">
Spec file: specs/<step_number>-<feature_slug>.md
Title:     <feature_title>
```

Then tell the user:
"Review the spec at `specs/<step_number>-<feature_slug>.md`
then enter Plan Mode with Shift+Tab twice to begin implementation."

Do not print the full spec in chat unless explicitly asked.
