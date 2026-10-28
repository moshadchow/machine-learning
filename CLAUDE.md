# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Setup

```bash
# Create and activate venv (Windows PowerShell)
python -m venv mlvenv
mlvenv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the app

```bash
# Launch Streamlit UI
mlvenv\Scripts\streamlit run app.py
```

### Run model CLI (auth → fetch → evaluate → predict)

```bash
mlvenv\Scripts\python model.py
```

### Run API client CLI

```bash
mlvenv\Scripts\python oms_api.py
```

### Lint / format

```bash
# Ruff check
ruff check .

# Black format
black .
```

### Tests

No test suite exists in this repo. Verification is manual via the CLIs above. To add tests, place them in the repo root and run with `pytest`.

## Architecture

This is a broker order-prediction Streamlit application with three layered modules:

1. **`oms_auth.py`** — Centralized authentication for the OMS API. Handles login (POST /api/login) with MFA, in-memory token state (access/refresh tokens + UTC expiry), proactive refresh within a safety margin, and bounded retry on 401. **Never logs or persists tokens.** Raises `OmsAuthError` on failure. Auth failures abort dataset construction entirely.

2. **`oms_api.py`** — HTTP client for the OMS broker-summary endpoints. Fetches both `orders-execution` and `orders` per broker per calendar day (FromDate = ToDate = loop date). Builds an in-memory pandas DataFrame with columns: `Date`, `BrokerId`, `totalValue`, `total`, `totalTrade`. Authentication is injected via a shared `OmsAuthenticator` instance. Partial per-broker failures are skip-and-reported; auth failures abort the entire build.

3. **`model.py`** — Machine-learning logic. Uses `LinearRegression` with feature=`totalValue` and targets=`[total, totalTrade]`. Employs a **temporal split** (train on earlier dates, test on most recent ~20% of dates) to avoid leakage from correlated daily observations. Includes guards: `MIN_OBS_TO_TRAIN=2`, `MIN_OBS_FOR_METRICS=10`, `MIN_DATES_FOR_METRICS=2`. Evaluation uses `r2_score` and `mean_absolute_error`.

**Flow**: Authenticator → API client → dataset (in-memory DataFrame) → temporal split → model train/evaluate → predictions → Streamlit UI.

## Code style

- Python 3.11+, PEP8, snake_case
- Type hints for all public functions
- FastAPI endpoints (if any) declare request/response models
- Business logic stays out of routers; use service helpers
- Centralized exception handling
- Structured JSON logging with correlation IDs
- Keep filenames lowercase for new modules

## What to avoid

- Never store plaintext passwords, OTPs, tokens, or biometrics
- Never bypass consent before collecting NID/photo/fingerprint
- Never expose NID numbers in logs
- Never trust frontend validation alone
- Never hard delete audit records
- Never hardcode secrets; use `.env` (gitignored)
- Tokens and secrets never appear in logs or error messages
- Do not persist tokens to disk, include them in training data, or expose them in Streamlit