---
name: streamlit-ml-app
description: Streamlit UI patterns for ML apps — cached bundles, error handling, and sensitive-data handling.
---

# Streamlit ML App

Patterns for the Streamlit frontend that serves broker order predictions.

## When to Use

- When modifying `app.py` or adding UI components
- When changing how errors are surfaced to users
- When adjusting caching or data loading

## Core Rules

**Cached bundle:** Use `@st.cache_resource` for `load_app_bundle()` — authenticate once, fetch once, train once per app startup. Return `(df, fetch_report, metrics_or_none, serving_model)`. A data refresh requires an app restart.

**Error handling:**
- `OmsConfigError` → `st.error` + `st.stop()` with remediation hint
- `OmsAuthError` → separate catch block with corrective-action message (never show credential values)
- `InsufficientDataError` → `st.error` + `st.stop()`

**Never expose:**
- Credentials, tokens, MFA codes, authorization headers
- NID numbers (if applicable)
- Raw API responses in error messages

**Prediction input:** Turnover in BDT with `min_value`/`max_value` clamped to the fetched data range; default clamped into valid range.

**Display labels:** "Total Orders" for `total`, "Total Trades" for `totalTrade` — match the model's target names.

## Verification

- App launches headless and serves HTTP 200
- Error states render correctly (no credential leakage)
- Predictions match `model.predict_orders` output
- Metrics panel describes the temporal split (train/test rows + date ranges)
