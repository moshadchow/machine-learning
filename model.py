"""Model module for broker order prediction.

Reproduces the modeling logic from order_prediction_dataset_v2.ipynb:
multi-output LinearRegression predicting Submitted Order and Executed Order
from Turnover (BDT). No Streamlit code lives here — the app imports from
this module, and it can also be run standalone for verification:

    mlvenv\\Scripts\\python model.py
"""

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# --- Constants (match the notebook exactly) ---
FEATURE = "Turnover (BDT)"
TARGETS = ["Submitted Order", "Executed Order"]
DATA_PATH = (
    Path(__file__).resolve().parent
    / "dataset"
    / "Broker_Summary_Report_2026-01-01_to_2026-02-26.csv"
)
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_and_clean_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the broker summary CSV and clean comma-formatted numeric columns."""
    df = pd.read_csv(path)
    for col in TARGETS:
        df[col] = df[col].astype(str).str.replace(",", "").astype(float)
    return df


def evaluate_model(df: pd.DataFrame) -> dict:
    """Train on the 80% split and evaluate on the 20% test split.

    Returns per-target and average R²/MAE, a predictions-vs-actual table,
    and the split-trained model itself (used for notebook-fidelity checks).
    """
    X = df[[FEATURE]]
    y = df[TARGETS]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    predictions_vs_actual = pd.DataFrame(
        {
            FEATURE: X_test[FEATURE].values,
            "Actual Submitted": y_test[TARGETS[0]].values,
            "Pred Submitted": y_pred[:, 0].round(2),
            "Actual Executed": y_test[TARGETS[1]].values,
            "Pred Executed": y_pred[:, 1].round(2),
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
    }


def train_full_model(df: pd.DataFrame) -> LinearRegression:
    """Fit the serving model on all rows (dataset is too small to withhold data)."""
    model = LinearRegression()
    model.fit(df[[FEATURE]], df[TARGETS])
    return model


def predict_orders(model: LinearRegression, turnover: float) -> dict[str, int]:
    """Predict submitted/executed orders for a single turnover value."""
    X_new = pd.DataFrame({FEATURE: [turnover]})
    predictions = model.predict(X_new)
    return {
        TARGETS[0]: int(round(predictions[0, 0])),
        TARGETS[1]: int(round(predictions[0, 1])),
    }


# --- CLI verification: reproduce the notebook's stored outputs ---
if __name__ == "__main__":
    df = load_and_clean_data()
    metrics = evaluate_model(df)

    print("--- Test Set Predictions vs Actual ---")
    print(metrics["predictions_vs_actual"])

    print("\n--- Model Evaluation Metrics (per target) ---")
    for target in TARGETS:
        print(
            f"{target}: R2={metrics['r2_per_target'][target]:.4f}  "
            f"MAE={metrics['mae_per_target'][target]:.2f}"
        )
    print(f"Average: R2={metrics['r2_avg']:.4f}  MAE={metrics['mae_avg']:.2f}")

    # Notebook fidelity check: the notebook printed uniform-average R2/MAE
    # and sample predictions from the split-trained model.
    print("\n--- Notebook Fidelity Check (split-trained model) ---")
    sample_turnover = pd.DataFrame(
        {FEATURE: [6_000_000_000, 15_000_000_000, 30_000_000_000]}
    )
    split_preds = metrics["split_model"].predict(sample_turnover)
    print(
        pd.DataFrame(
            {
                FEATURE: sample_turnover[FEATURE],
                "Predicted Submitted Order": split_preds[:, 0].round().astype(int),
                "Predicted Executed Order": split_preds[:, 1].round().astype(int),
            }
        )
    )

    # The app serves a model trained on all rows; record its predictions too.
    print("\n--- Serving Model Sample Predictions (trained on all rows) ---")
    serving_model = train_full_model(df)
    serving_preds = serving_model.predict(sample_turnover)
    print(
        pd.DataFrame(
            {
                FEATURE: sample_turnover[FEATURE],
                "Predicted Submitted Order": serving_preds[:, 0].round().astype(int),
                "Predicted Executed Order": serving_preds[:, 1].round().astype(int),
            }
        )
    )
