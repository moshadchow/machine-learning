"""Model module for broker order prediction.

Multi-output LinearRegression predicting total orders and total trades
from turnover (totalValue), trained on day-wise data fetched live from
the OMS API (see oms_api.py; authentication via oms_auth.py). Each row is
one broker-day observation (Date, BrokerId, totalValue, total, totalTrade).

Evaluation uses a TEMPORAL split — train on earlier dates, test on the
most recent ~20% of dates — because daily observations from the same
broker are correlated and a random split would leak future days into
training (documented methodology change vs the notebook's random split;
see spec 02 §3.4). The estimator itself is unchanged: multi-output
LinearRegression.

Training data eligibility (spec 02 §9): only observations with
totalValue > 0 are included in the training dataset. Observations with
totalValue <= 0 are excluded (no zero-filling or fabrication). The filter
is applied before train/test splitting and model training.

No Streamlit code lives here — the app imports from this module, and it
can also be run standalone for verification (requires OMS_* env vars or
a local .env file):

    mlvenv\\Scripts\\python model.py
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# --- Constants ---
# Column names come from the OMS API field mapping (see oms_api.py).
DATE_COL = "Date"  # metadata only — never a model feature
BROKER_COL = "BrokerId"  # metadata only — never a model feature
FEATURE = "totalValue"  # turnover in BDT
TARGETS = ["total", "totalTrade"]
TEST_SIZE = 0.2  # most recent share of DATES held out for evaluation
RANDOM_STATE = 42  # for any stochastic estimator (the split itself is deterministic)

# Minimum data requirements for the modeling/evaluation approach:
# - Below MIN_OBS_TO_TRAIN, LinearRegression cannot be fit meaningfully.
# - The 80/20 TEMPORAL split needs at least 2 test rows for R² to be
#   defined and at least 2 distinct dates to split on. Below that,
#   metrics would be undefined or misleading, so they are withheld
#   (predictions still work).
MIN_OBS_TO_TRAIN = 2
MIN_OBS_FOR_METRICS = 10
MIN_DATES_FOR_METRICS = 2


class InsufficientDataError(RuntimeError):
    """Raised when the dataset is too small for the requested operation."""


def filter_eligible_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Filter training dataset to observations with totalValue > 0.

    Observations with totalValue <= 0 are excluded (no zero-filling or
    fabrication). Missing/malformed values are excluded if not numeric.
    Returns (eligible_df, stats) where stats includes counts for reporting.
    """
    if df is None or len(df) == 0:
        stats = {
            "total": 0,
            "eligible": 0,
            "excluded": 0,
            "excluded_le_zero": 0,
            "excluded_invalid": 0,
        }
        return df.copy() if df is not None else pd.DataFrame(), stats

    work = df.copy()
    total = len(work)

    # Coerce totalValue to numeric where possible
    if FEATURE in work.columns:
        work[FEATURE] = pd.to_numeric(work[FEATURE], errors="coerce")
    else:
        work[FEATURE] = pd.NA

    # Eligible only if totalValue > 0 and numeric
    eligible_mask = work[FEATURE].notna() & (work[FEATURE] > 0)
    eligible_df = work[eligible_mask].copy()
    eligible_count = len(eligible_df)

    excluded = total - eligible_count
    # Count how many excluded are <= 0 (and numeric) vs invalid
    le_zero_mask = work[FEATURE].notna() & (work[FEATURE] <= 0)
    excluded_le_zero = int(le_zero_mask.sum())
    excluded_invalid = int(excluded - excluded_le_zero)
    if excluded_invalid < 0:
        excluded_invalid = 0

    stats = {
        "total": int(total),
        "eligible": int(eligible_count),
        "excluded": int(excluded),
        "excluded_le_zero": int(excluded_le_zero),
        "excluded_invalid": int(excluded_invalid),
    }
    return eligible_df, stats


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by date: earlier dates -> train, most recent dates -> test.

    The test set is the most recent ceil(TEST_SIZE * n_dates) distinct
    dates (at least one). Splitting by DATE (not by row) keeps every
    broker's observations from a given day on the same side, so no same-day
    information leaks between train and test. Requires at least two
    distinct dates.
    """
    dates = sorted(df[DATE_COL].dropna().unique())
    if len(dates) < MIN_DATES_FOR_METRICS:
        raise InsufficientDataError(
            f"Temporal evaluation needs at least {MIN_DATES_FOR_METRICS} "
            f"distinct dates, but the data covers only {len(dates)} "
            f"({dates[0] if dates else 'none'}). Widen OMS_FROM_DATE..OMS_TO_DATE."
        )
    n_test_dates = max(1, int(len(dates) * TEST_SIZE + 0.999))  # ceil
    cutoff_date = dates[-n_test_dates]
    train_df = df[df[DATE_COL] < cutoff_date]
    test_df = df[df[DATE_COL] >= cutoff_date]
    return train_df, test_df


def evaluate_model(df: pd.DataFrame) -> dict:
    """Train on the earlier dates and evaluate on the most recent dates.

    Temporal split (see temporal_split) — no random_state needed because
    the split is deterministic. Applies training data eligibility filter
    (totalValue > 0) before splitting/training. Raises InsufficientDataError
    when the eligible dataset has fewer than MIN_OBS_FOR_METRICS rows or
    fewer than two distinct dates.

    Returns per-target and average R²/MAE, a predictions-vs-actual table,
    and the split-trained model itself. Also includes eligibility stats.
    """
    eligible_df, eligibility = filter_eligible_dataset(df)

    if eligibility["eligible"] == 0:
        raise InsufficientDataError(
            "No records satisfy totalValue > 0. All observations have "
            "totalValue <= 0 or invalid values; exclude none were fabricated."
        )

    if len(eligible_df) < MIN_OBS_FOR_METRICS:
        raise InsufficientDataError(
            f"Cannot compute evaluation metrics with {len(eligible_df)} eligible "
            f"observations (of {eligibility['total']} total; "
            f"{eligibility['excluded_le_zero']} excluded with totalValue <= 0): "
            f"the {int(TEST_SIZE * 100)}% temporal test split needs at least "
            f"2 rows, which requires at least {MIN_OBS_FOR_METRICS} total "
            f"eligible observations. Check the fetch report for failed "
            "broker-date pairs or eligibility exclusions."
        )

    train_df, test_df = temporal_split(eligible_df)

    X_train, y_train = train_df[[FEATURE]], train_df[TARGETS]
    X_test, y_test = test_df[[FEATURE]], test_df[TARGETS]
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    predictions_vs_actual = pd.DataFrame(
        {
            DATE_COL: test_df[DATE_COL].values,
            BROKER_COL: test_df[BROKER_COL].values,
            FEATURE: X_test[FEATURE].values,
            f"Actual {TARGETS[0]}": y_test[TARGETS[0]].values,
            f"Pred {TARGETS[0]}": y_pred[:, 0].round(2),
            f"Actual {TARGETS[1]}": y_test[TARGETS[1]].values,
            f"Pred {TARGETS[1]}": y_pred[:, 1].round(2),
        }
    )

    return {
        "r2_per_target": dict(
            zip(TARGETS, r2_score(y_test, y_pred, multioutput="raw_values"))
        ),
        "mae_per_target": dict(
            zip(TARGETS, mean_absolute_error(y_test, y_pred, multioutput="raw_values"))
        ),
        "r2_avg": float(r2_score(y_test, y_pred)),
        "mae_avg": float(mean_absolute_error(y_test, y_pred)),
        "predictions_vs_actual": predictions_vs_actual,
        "split_model": model,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_dates": f"{train_df[DATE_COL].min()} .. {train_df[DATE_COL].max()}",
        "test_dates": f"{test_df[DATE_COL].min()} .. {test_df[DATE_COL].max()}",
        "eligibility": eligibility,
        "eligible_df": eligible_df,
    }


def train_full_model(df: pd.DataFrame) -> LinearRegression:
    """Fit the serving model on all eligible rows (dataset is too small to withhold data).

    Applies training data eligibility filter (totalValue > 0) before training.
    Raises InsufficientDataError if no eligible observations remain or fewer
    than MIN_OBS_TO_TRAIN eligible observations.
    """
    eligible_df, eligibility = filter_eligible_dataset(df)

    if eligibility["eligible"] == 0:
        raise InsufficientDataError(
            "No records satisfy totalValue > 0. All observations have "
            "totalValue <= 0 or invalid values; exclude none were fabricated."
        )

    if len(eligible_df) < MIN_OBS_TO_TRAIN:
        raise InsufficientDataError(
            f"Cannot train the model with {len(eligible_df)} eligible observations "
            f"(of {eligibility['total']} total; {eligibility['excluded_le_zero']} "
            f"excluded with totalValue <= 0): at least {MIN_OBS_TO_TRAIN} are "
            "required. Check the fetch report above for failed broker IDs or "
            "eligibility exclusions."
        )
    model = LinearRegression()
    model.fit(eligible_df[[FEATURE]], eligible_df[TARGETS])
    return model


def predict_orders(model: LinearRegression, turnover: float) -> dict[str, int]:
    """Predict total orders and total trades for a single turnover value."""
    X_new = pd.DataFrame({FEATURE: [turnover]})
    predictions = model.predict(X_new)
    return {
        TARGETS[0]: int(round(predictions[0, 0])),
        TARGETS[1]: int(round(predictions[0, 1])),
    }


# --- CLI verification: authenticate, fetch from the API, train, and report ---
if __name__ == "__main__":
    import oms_api
    from oms_auth import OmsAuthError, OmsAuthenticator, load_auth_config

    try:
        config = oms_api.load_config()
        auth_config = load_auth_config()
    except (oms_api.OmsConfigError, OmsAuthError) as exc:
        raise SystemExit(str(exc))

    authenticator = OmsAuthenticator(auth_config)
    try:
        authenticator.get_access_token()
    except OmsAuthError as exc:
        raise SystemExit(f"Authentication failed: {exc}")

    df, report = oms_api.build_dataset(config, authenticator)

    print("--- Fetch Report ---")
    print(report.summary())
    oms_api.print_failure_summary(report)

    # Training data eligibility (totalValue > 0)
    eligible_df, eligibility = filter_eligible_dataset(df)
    print(
        f"\n--- Training Data Eligibility ---\n"
        f"Total observations: {eligibility['total']}\n"
        f"Eligible (totalValue > 0): {eligibility['eligible']}\n"
        f"Excluded: {eligibility['excluded']} "
        f"(totalValue <= 0: {eligibility['excluded_le_zero']}, "
        f"invalid/missing: {eligibility['excluded_invalid']})"
    )
    if eligibility["eligible"] == 0:
        raise SystemExit(
            "No records satisfy totalValue > 0. Training stopped "
            "(no zero-filling or fabrication)."
        )

    print(f"\n--- Dataset ({len(eligible_df)} eligible rows, {report.date_count} days "
          f"x up to {report.broker_count} brokers) ---")
    if len(eligible_df) > 20:
        print(eligible_df.head(10).to_string(index=False))
        print(f"... ({len(eligible_df) - 20} more rows) ...")
        print(eligible_df.tail(10).to_string(index=False))
    else:
        print(eligible_df.to_string(index=False))

    # Evaluation metrics (withheld when the dataset is too small).
    print("\n--- Model Evaluation (temporal split) ---")
    try:
        metrics = evaluate_model(df)
        print(
            f"Train: {metrics['train_rows']} rows ({metrics['train_dates']}) | "
            f"Test: {metrics['test_rows']} rows ({metrics['test_dates']})"
        )
        print("Test set predictions vs actual (most recent dates):")
        print(metrics["predictions_vs_actual"].to_string(index=False))
        print()
        for target in TARGETS:
            print(
                f"{target}: R2={metrics['r2_per_target'][target]:.4f}  "
                f"MAE={metrics['mae_per_target'][target]:.2f}"
            )
        print(f"Average: R2={metrics['r2_avg']:.4f}  MAE={metrics['mae_avg']:.4f}")
    except InsufficientDataError as exc:
        print(f"Metrics withheld: {exc}")

    # Serving model (trained on all rows).
    print("\n--- Serving Model Sample Predictions (trained on eligible rows) ---")
    try:
        serving_model = train_full_model(df)
    except InsufficientDataError as exc:
        raise SystemExit(str(exc))

    # Sample turnovers drawn from the fetched data range (not hardcoded).
    sample_values = eligible_df[FEATURE]
    sample_turnover = pd.DataFrame(
        {
            FEATURE: [
                float(sample_values.min()),
                float(sample_values.median()),
                float(sample_values.max()),
            ]
        }
    )
    serving_preds = serving_model.predict(sample_turnover)
    print(
        pd.DataFrame(
            {
                FEATURE: sample_turnover[FEATURE],
                f"Predicted {TARGETS[0]}": serving_preds[:, 0].round().astype(int),
                f"Predicted {TARGETS[1]}": serving_preds[:, 1].round().astype(int),
            }
        ).to_string(index=False)
    )
