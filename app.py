"""Streamlit app for broker order prediction.

Imports the reusable modeling code from model.py, trains the serving model
once at startup (cached), and renders an interactive UI: enter a broker's
turnover in BDT to predict submitted and executed order counts, with the
model's evaluation metrics available in an expander.

Run with:

    mlvenv\\Scripts\\streamlit run app.py
"""

import streamlit as st

import model

# --- Page configuration ---
st.set_page_config(
    page_title="Broker Order Predictor",
    page_icon="📈",
    layout="centered",
)


# --- Load data and train models (cached; runs once per app startup) ---
@st.cache_resource
def load_app_bundle():
    """Load/clean data, evaluate the split model, train the serving model."""
    df = model.load_and_clean_data()
    metrics = model.evaluate_model(df)
    serving_model = model.train_full_model(df)
    return df, metrics, serving_model


df, metrics, serving_model = load_app_bundle()

# --- App header ---
st.title("Broker Order Predictor")
st.caption(
    "Enter a broker's turnover in BDT to predict how many orders they will "
    "submit and execute. The model is a multi-output linear regression "
    "trained on the broker summary dataset (9 rows)."
)

# --- Prediction input ---
min_turnover = float(df[model.FEATURE].min())
max_turnover = float(df[model.FEATURE].max())

turnover = st.number_input(
    "Turnover (BDT)",
    min_value=min_turnover,
    max_value=max_turnover,
    value=10_000_000_000.0,
    step=100_000_000.0,
    format="%.0f",
    help=f"Range from the source data: {min_turnover:,.0f} – {max_turnover:,.0f} BDT",
)
st.caption(f"Selected turnover: {turnover:,.0f} BDT")

if st.button("Predict orders", type="primary"):
    predictions = model.predict_orders(serving_model, turnover)
    st.subheader("Predicted orders")
    submitted_col, executed_col = st.columns(2)
    submitted_col.metric("Submitted Order", f"{predictions[model.TARGETS[0]]:,}")
    executed_col.metric("Executed Order", f"{predictions[model.TARGETS[1]]:,}")

# --- Evaluation metrics and data preview ---
with st.expander("Model evaluation & data preview"):
    st.markdown(
        "**Evaluation metrics** (80/20 train/test split, `random_state=42`, "
        "mirrors the notebook). With only 9 rows the test split has ~2 rows, "
        "so R² is unstable — shown for transparency, not as a quality gate."
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

    st.markdown("**Predictions vs actual (test split)**")
    st.dataframe(metrics["predictions_vs_actual"], hide_index=True)

    st.markdown("**Source data preview** (cleaned)")
    st.dataframe(df, hide_index=True)

    st.caption(
        "The predictor above is trained on all 9 rows; the metrics use the "
        "80/20 split-trained model to match the notebook."
    )
