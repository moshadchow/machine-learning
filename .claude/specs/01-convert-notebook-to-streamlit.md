# Spec: Convert an Existing Machine Learning Notebook into a Streamlit Application

## Overview
This step packages the multi-output linear regression from `order_prediction_dataset_v2.ipynb` into an interactive Streamlit web app. The app lets a user enter a broker's turnover (BDT) and instantly see predicted submitted and executed order counts, alongside the model's evaluation metrics. This is the project's first deployment step: the notebook proved the modeling approach, and this step makes it usable by a non-technical audience without touching the raw data or the existing notebook.

## Depends on
- `order_prediction_dataset_v2.ipynb` — source of the modeling approach (multi-output `LinearRegression` on `Turnover (BDT)` → `Submitted Order`, `Executed Order`). The notebook is read-only for this step and is not modified.

## Data
- File: `dataset/Broker_Summary_Report_2026-01-01_to_2026-02-26.csv` (9 rows)
- Verified header columns: `SL`, `Broker Name`, `Total TWS`, `Active TWS`, `Submitted Order`, `Executed Order`, `Turnover (BDT)`
- Feature column: `Turnover (BDT)` (numeric, no cleaning needed)
- Target columns: `Submitted Order`, `Executed Order` — stored as strings with thousands separators (e.g. `"121,551"`), must be cleaned by removing commas and casting to `float` in code
- Raw CSV is never moved or edited; all cleaning happens in Python

## Approach
- Reuse the notebook's model exactly: `LinearRegression` handling both targets natively (multi-output), no scaling or encoding required since there is a single numeric feature
- Extract reusable code into a `.py` module (`model.py`) with small functions: load-and-clean data, train, evaluate
- Evaluation split stays identical to the notebook: `train_test_split(test_size=0.2, random_state=42)`
- The predictor served by the app is trained on **all rows** (standard practice after evaluation; 9 rows is too few to withhold data from the deployed model)
- Model is trained once at app startup and cached with `@st.cache_resource` — no model artifact files on disk, so nothing extra to gitignore
- Streamlit UI: a number input for `Turnover (BDT)` with sensible bounds from the data, a Predict action, predicted values displayed as formatted integers, plus an expander showing the evaluation metrics and a preview of the source data
- Note: with 9 rows the 80/20 split leaves only ~2 test rows, so test R² is unstable (currently ≈ -0.19). Metrics are displayed for transparency but are not a quality gate for this step

## Evaluation
- Metrics reported in the app (computed on the 80/20 test split): R² via `r2_score` and MAE via `mean_absolute_error`, per target and averaged
- Success for this step is functional, not metric-based (the tiny dataset makes test metrics statistically meaningless):
  - App launches and renders without errors
  - Predictions for turnover values 6,000,000,000 / 15,000,000,000 / 30,000,000,000 match the notebook's sample predictions
  - Predictions are non-negative and increase monotonically with turnover (a property of this univariate linear model)

## Files to change
- None. `order_prediction_dataset_v2.ipynb`, the dataset, and `.gitignore` are left untouched.

## Files to create
- `model.py` — data loading, comma cleaning, training, and evaluation functions (importable, no Streamlit code)
- `app.py` — Streamlit UI that imports `model.py`, caches the trained model, and renders inputs/predictions/metrics
- `specs/01-convert-notebook-to-streamlit.md` — this spec

## New dependencies
- `streamlit` — installed into `mlvenv` (`mlvenv\Scripts\pip install streamlit`). Everything else (`pandas`, `scikit-learn`, `numpy`) is already present.

## Rules for implementation
- Use pandas for data handling and scikit-learn for models/metrics only
- Always set `random_state=42` on `train_test_split` and on any stochastic estimator, so results are reproducible
- Fit scalers/encoders on training data only — never leak test data into preprocessing (prefer `sklearn.pipeline.Pipeline`)
- Datasets live in `dataset/` — never move or edit raw CSVs; clean them in code instead
- Never commit `mlvenv/` or large data files
- Install new packages into the `mlvenv` virtual environment
- Explore interactively in `.ipynb` notebooks; move reusable training code into `.py` scripts
- Notebooks must run cleanly top-to-bottom (Kernel → Restart & Run All) before being committed
- Every code section gets a short markdown header or comment explaining the step
- Print evaluation metrics (and a predictions-vs-actual table for regression), not just raw predictions
- Python: 4-space indent, `snake_case` names, no unused imports
- No new dependencies unless explicitly approved (`streamlit` is the only approved addition for this step)

## Definition of done
- [ ] `mlvenv\Scripts\pip install streamlit` completed inside `mlvenv`
- [ ] `model.py` reproduces the notebook's cleaning/split/model logic and runs standalone
- [ ] `mlvenv\Scripts\streamlit run app.py` launches the app without errors
- [ ] Entering a turnover value produces predicted Submitted and Executed orders displayed as formatted numbers
- [ ] App displays R² and MAE metrics for both targets
- [ ] Sample predictions (6B / 15B / 30B turnover) match the notebook's `--- Sample Predictions ---` output
- [ ] `order_prediction_dataset_v2.ipynb` still runs cleanly top-to-bottom and is unchanged
- [ ] Nothing from `mlvenv/` or `dataset/` is staged for commit
