"""Streamlit app for broker order prediction.

Authenticates against the OMS API (login/MFA/refresh via oms_auth.py),
fetches per-broker aggregates (oms_api.py), trains a multi-output
LinearRegression on eligible in-memory data (totalValue > 0) once at
startup (cached), and renders an interactive UI: enter a broker's
turnover in BDT to predict total orders and total trades, with evaluation
metrics in an expander.

Requires OMS_* environment variables or a local .env file (see
oms_auth.py and .env.example). Run with:

    mlvenv\\Scripts\\streamlit run app.py

Note: the model bundle (including the in-memory token) is cached with
@st.cache_resource, so changes to environment variables or an expired
MFA code require an app restart to take effect.
"""

import streamlit as st

import model
import oms_api
from oms_auth import OmsAuthError, OmsAuthenticator, load_auth_config


# --- Page configuration ---
st.set_page_config(
    page_title="Broker Order Predictor",
    page_icon="📈",
    layout="centered",
)


# --- Load data and train models (cached; runs once per app startup) ---
@st.cache_resource
def load_app_bundle():
    """Authenticate, fetch API data, evaluate, and train the serving model.

    Applies training data eligibility (totalValue > 0). Returns
    (eligible_df, fetch_report, metrics_or_none, serving_model, eligibility_stats).
    Raises OmsConfigError/OmsAuthError/InsufficientDataError on failure.
    """
    data_config = oms_api.load_config()
    auth_config = load_auth_config()
    authenticator = OmsAuthenticator(auth_config)

    # Authenticate once; one token serves every configured broker ID.
    authenticator.get_access_token()

    df, report = oms_api.build_dataset(data_config, authenticator)
    if df.empty:
        raise model.InsufficientDataError(
            "No broker data could be fetched. See the fetch report above for "
            "per-broker failure reasons."
        )

    eligible_df, eligibility = model.filter_eligible_dataset(df)
    if eligibility["eligible"] == 0:
        raise model.InsufficientDataError(
            "No records satisfy totalValue > 0. All observations have "
            "totalValue <= 0 or invalid values; none were fabricated."
        )

    # Evaluation metrics are withheld when the dataset is too small.
    metrics = None
    try:
        metrics = model.evaluate_model(df)
    except model.InsufficientDataError:
        pass

    serving_model = model.train_full_model(df)
    return eligible_df, report, metrics, serving_model, eligibility

# --- App header ---
st.title("Broker Order Predictor")
st.caption(
    "Enter a broker's turnover in BDT to predict how many total orders and "
    "total trades they will generate. The model is a multi-output linear "
    "regression trained on live data fetched from the OMS API."
)

# --- Load or surface errors ---
try:
    eligible_df, fetch_report, metrics, serving_model, eligibility = load_app_bundle()
except oms_api.OmsConfigError as exc:
    st.error(
        "Configuration error — fix the environment variables (or .env file) "
        "below and restart the app."
    )
    st.code(str(exc), language=None)
    st.stop()
except OmsAuthError as exc:
    # Authentication failure: distinct from data errors; no secrets shown.
    st.error(
        "Authentication failed — the model was not trained. Correct the "
        "credentials/MFA code below and restart the app."
    )
    st.code(str(exc), language=None)
    st.stop()
except model.InsufficientDataError as exc:
    st.error(str(exc))
    st.stop()

# --- Fetch report ---
with st.expander(
    f"API fetch report — {fetch_report.summary()}",
    expanded=bool(fetch_report.failed),
):
    st.caption(
        f"Coverage: {fetch_report.broker_count} brokers × "
        f"{fetch_report.date_count} days = {fetch_report.expected_pairs} "
        "expected broker-date pairs (one observation per valid pair)."
    )
    st.info(
        f"**Training data eligibility:** {eligibility['eligible']} / "
        f"{eligibility['total']} observations eligible (totalValue > 0). "
        f"Excluded {eligibility['excluded']} "
        f"({eligibility['excluded_le_zero']} with totalValue <= 0, "
        f"{eligibility['excluded_invalid']} invalid/missing)."
    )
    if fetch_report.failed:
        by_broker = fetch_report.failed_by_broker()
        st.warning(
            f"{len(fetch_report.failed)} broker-date pair(s) failed and were "
            "excluded from the training dataset. Failures aggregated by broker:"
        )
        for broker_id, failures in by_broker.items():
            reasons = sorted({reason for _, reason in failures})
            sample_dates = ", ".join(day for day, _ in failures[:3])
            more = "" if len(failures) <= 3 else ", ..."
            st.write(
                f"- **{broker_id}** — {len(failures)} date(s); "
                f"{reasons[0]} (e.g. {sample_dates}{more})"
            )
    else:
        st.write("All broker-date pairs fetched successfully.")

# --- Prediction input ---
min_turnover = float(eligible_df[model.FEATURE].min())
max_turnover = float(eligible_df[model.FEATURE].max())

# Clamp the default into the valid range so Streamlit doesn't reject it
# when the API data range differs from the old hardcoded default.
default_turnover = min(max(10_000_000_000.0, min_turnover), max_turnover)

turnover = st.number_input(
    "Turnover (BDT)",
    min_value=min_turnover,
    max_value=max_turnover,
    value=default_turnover,
    step=100_000_000.0,
    format="%.0f",
    help=f"Range from the fetched data: {min_turnover:,.0f} – {max_turnover:,.0f} BDT",
)
st.caption(f"Selected turnover: {turnover:,.0f} BDT")

if st.button("Predict orders", type="primary"):
    predictions = model.predict_orders(serving_model, turnover)
    st.subheader("Predicted orders")
    orders_col, trades_col = st.columns(2)
    orders_col.metric("Total Orders", f"{predictions[model.TARGETS[0]]:,}")
    trades_col.metric("Total Trades", f"{predictions[model.TARGETS[1]]:,}")

# --- Evaluation metrics and data preview ---
with st.expander("Model evaluation & data preview"):
    if metrics is None:
        st.info(
            f"**Evaluation metrics withheld.** Only {len(eligible_df)} eligible "
            f"observation(s) (of {eligibility['total']} fetched) were used, but at "
            f"least {model.MIN_OBS_FOR_METRICS} are needed for a meaningful "
            "temporal train/test evaluation. Predictions above are still served "
            "from a model trained on all eligible rows."
        )
    else:
        st.markdown(
            f"**Evaluation metrics** (temporal split: train on earlier dates, "
            f"test on the most recent {int(model.TEST_SIZE * 100)}% of dates — "
            "daily observations from the same broker are correlated, so a "
            "random split would leak future days into training). "
            f"Train: {metrics['train_rows']} rows ({metrics['train_dates']}); "
            f"Test: {metrics['test_rows']} rows ({metrics['test_dates']}). "
            f"Eligible: {eligibility['eligible']} / {eligibility['total']} "
            f"(excluded {eligibility['excluded']} with totalValue <= 0/invalid). "
            "Shown for transparency, not as a quality gate."
        )
        for target in model.TARGETS:
            st.write(
                f"- **{target}** — R²: {metrics['r2_per_target'][target]:.4f}, "
                f"MAE: {metrics['mae_per_target'][target]:,.2f}"
            )
        st.write(
            f"- **Average** — R²: {metrics['r2_avg']:.4f}, "
            f"MAE: {metrics['mae_avg']:,.2f}"
        )

        st.markdown("**Predictions vs actual (most recent test dates)**")
        st.dataframe(metrics["predictions_vs_actual"], hide_index=True)

    st.markdown(
        f"**Source data preview** (in-memory, {len(eligible_df)} eligible rows "
        f"from API; {eligibility['excluded']} excluded with totalValue <= 0)"
    )
    st.dataframe(eligible_df, hide_index=True)

    st.caption(
        "The predictor above is trained on all eligible rows; the metrics (when "
        "shown) use the temporal split-trained model — earlier dates for "
        "training, the most recent dates for evaluation. Observations with "
        "totalValue <= 0 are excluded (no zero-filling)."
    )
