# AGENTS.md

## Project
Broker order prediction: a multi-output `LinearRegression` predicts total orders and total trades from turnover (BDT), trained on data fetched live from the OMS API. Layers, kept in sync manually:

- `order_prediction_dataset_v2.ipynb` — original modeling approach (reference only; no longer the data source, keep it runnable top-to-bottom)
- `oms_auth.py` — authentication: login (with MFA), in-memory token state, expiry checks, refresh; raises `OmsAuthError`
- `oms_api.py` — HTTP data client: env-var config, fetches both OMS endpoints per broker, builds the in-memory dataset (no model/Streamlit code)
- `model.py` — training/evaluation/prediction logic; also a CLI verification script
- `app.py` — Streamlit UI importing `model.py`, `oms_api.py`, `oms_auth.py`

Environment: Windows, Python 3.11, virtualenv at `mlvenv/` (gitignored). There is **no** test, lint, typecheck, or CI tooling — verification means running `model.py`/`oms_api.py` and launching the app.

## Commands (run from repo root; venv paths are Windows-style)
- Launch the app: `mlvenv\Scripts\streamlit run app.py`
- Verify config + auth + API connectivity: `mlvenv\Scripts\python oms_api.py` — prints config (secrets redacted), login outcome, per-broker fetch report, and the dataset
- Verify the model end-to-end (closest thing to a test): `mlvenv\Scripts\python model.py` — authenticates, fetches, prints the fetch report, metrics (or an insufficient-data notice), and sample predictions
- Install packages into the venv: `mlvenv\Scripts\pip install <pkg>` — never install globally; `requirements.txt` is a full `pip freeze` (UTF-16 encoded), not a curated list; `requests` is already present

All run/verify commands above require `OMS_*` config (env vars or a local `.env`, see below) and network access to the API.

## Configuration
Set via environment variables, or a local `.env` file (gitignored; copy `.env.example`). Real env vars override `.env` values. Required: `OMS_API_BASE_URL`, `OMS_LOGIN_ID`, `OMS_PASSWORD`, `OMS_BROKER_IDS` (comma-separated), `OMS_FROM_DATE`, `OMS_TO_DATE`. Optional: `OMS_DEVICE_ID` (default `Android-01`), `OMS_STOCK_EXCHANGE` (default `1`), `OMS_MFA_KEY`/`OMS_MFA_CODE` (only for MFA-enabled accounts — when unset, empty values are sent and the API decides; if it reports MFA required, set them and restart).

Never hardcode or commit credentials/tokens; `.env` is gitignored, only `.env.example` (placeholders) is committed. The token/password/MFA code are never logged, raised in errors, or shown in the UI — errors name variables, never values.

## Workflow
- Work is spec-first: run `/create-spec <NN> <Title>` (defined in `.opencode/commands/create-spec.md`) to write `specs/NN-slug.md` and create branch `feature/NN-slug`; implement on that branch, merge via PR
- Steps are numbered and cumulative — check `specs/` before starting to avoid duplicating a planned step

## Data source (OMS API) and modeling rules
- Auth: `oms_auth.py` logs in via `POST /api/login` (loginId/password/deviceId/mfaKey/mfaCode/appType), keeps tokens in memory only, refreshes via `POST /api/login/refresh-token`. **HTTP 200 + top-level `success:true` do NOT mean auth succeeded** — the nested `data.success`/`isAuthenticated`/`isAuthorised` + non-empty tokens are the real gate (confirmed live: failed logins return HTTP 200 with `data.success:false`).
- **Day-wise iteration**: for every broker × every calendar date in the inclusive `OMS_FROM_DATE..OMS_TO_DATE` range, both endpoints are called with `FromDate = ToDate = that day` → one observation per valid broker-day (`Date`, `BrokerId`, `totalValue`, `total`, `totalTrade`). `Date`/`BrokerId` are metadata for traceability, never model features. A long range means thousands of requests (~299 days × 4 brokers ≈ 1,196 pairs ≈ 2,400 requests): fetched with a bounded pool (`FETCH_MAX_WORKERS=4`, ~5 min for that range). Keep ranges intentional; a `200` with numeric zeros is a valid observation (never zero-fill failed pairs).
- Data comes live from two OMS endpoints per broker-day (see `specs/02-oms-api-integration.md`): `totalValue` (feature, turnover BDT) from `/orders-execution`, `total` (target) from `/orders`, `totalTrade` (target) from `/orders-execution`. The old CSV in `dataset/` is reference-only; do not reintroduce it as a production source.
- One token authenticates all broker-date pairs. A 401 on a protected endpoint triggers one refresh + one retry; a second failure raises `OmsAuthError` (bounded, no loop). Auth failures abort dataset construction entirely — never train on an unauthenticated/partial-auth session.
- Partial per-broker-date DATA failures are skipped-and-reported (aggregated per broker in the UI/CLI); zero successes is a hard error
- **Evaluation uses a temporal split** (train on earlier dates, test on the most recent ~20% of dates) because daily observations from the same broker are correlated and a random split would leak future days into training — a documented methodology change from the notebook's random split (spec 02 §3.4). The estimator is unchanged.
- Keep API data and the trained model in memory — never write API responses or the dataset (or tokens) to disk
- Fit preprocessing on training data only (prefer `sklearn.pipeline.Pipeline`)
- No new dependencies unless explicitly approved (`requests` is already available; do not add `python-dotenv` — the `.env` loader is built into `oms_auth.py`)
- Notebooks must run cleanly top-to-bottom (Kernel → Restart & Run All) before commit
- The Streamlit bundle (including the in-memory token) is cached with `@st.cache_resource`, so changing `OMS_*` env vars or an expired MFA code requires an app restart

## Gotchas
- **Refresh can fail mid-run when the access token expires**: the test API appears to require a still-valid access token to refresh, so a long day-wise fetch may hit a 401 whose refresh is rejected (`Invalid token request!`). This is recovered automatically (one bounded re-login fallback in `oms_auth._refresh_locked`, plus `refresh_if_still` so only the first of 4 concurrent 401s refreshes). If BOTH refresh and re-login fail, the build aborts — update `OMS_MFA_CODE`/credentials and restart.
- The MFA code (`OMS_MFA_CODE`) may be one-time/rotating: when login is rejected ("Invalid Code!"/"Wrong Login ID or Password!"), update the code in `.env`/env and restart the app. Do not add automatic MFA retry.
- With few brokers, the 80/20 split needs ≥10 observations for a meaningful evaluation (≥2 test rows for R² to be defined). Below 10, the app still serves predictions from a model trained on all rows but **withholds metrics** with an explicit notice; below 2 it refuses to train. Do not "fix" this by loosening the guard — a negative/undefined R² on tiny data is expected, not a bug.
- `__pycache__/*.pyc` is tracked by git (accidentally); it may show as modified — don't commit changes to it
- The notebook's modeling methodology is preserved, but its column names (`Turnover (BDT)`, `Submitted Order`, `Executed Order`) differ from the API-driven ones in `model.py` (`totalValue`, `total`, `totalTrade`) — that naming change is intentional and documented in spec 02.

