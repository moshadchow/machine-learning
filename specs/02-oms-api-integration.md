# Task: Replace CSV Data Source with OMS API Integration

## 1. Identity Context

Act as a senior Python backend engineer and machine-learning engineer with expertise in REST API integration, pandas, scikit-learn, Python environment configuration, and Streamlit applications.

Implement the changes using production-quality practices, clear separation of concerns, secure credential handling, and appropriate error handling.

## 2. World Context

The existing project is a Streamlit application that predicts submitted and executed order counts using a multi-output `LinearRegression` model.

The project includes:

- `order_prediction_dataset_v2.ipynb`: Reference implementation of the existing modeling approach.

- `model.py`: Model training, preprocessing, and prediction logic.

- `app.py`: Streamlit user interface.

- `AGENTS.md`: Project conventions and implementation constraints.

The current dataset is loaded from an external CSV file. Replace this data source with two OMS API endpoints without unnecessarily changing the existing application architecture or prediction interface.

## 3. Task Context

## 3.1 Authentication and Token Lifecycle

Implement a centralized authentication module for the OMS API. The module must support login, MFA, access-token management, token refresh, and authenticated requests to both broker-summary endpoints.

### A. Environment Configuration

Read all authentication and API settings from environment variables.

```dotenv

OMS_API_BASE_URL=https://test-oms-api.ecosoftbd.com

OMS_LOGIN_ID=

OMS_PASSWORD=

OMS_DEVICE_ID=Android-01

OMS_MFA_KEY=

OMS_MFA_CODE=

OMS_BROKER_IDS=

OMS_FROM_DATE=

OMS_TO_DATE=

OMS_STOCK_EXCHANGE=1

```

Requirements:

- Store credentials in `.env` locally and exclude `.env` from version control.

- Include placeholder values only in `.env.example`.

- Never hardcode credentials or tokens in source code.

- Validate required configuration before making API requests.

- Support multiple comma-separated broker IDs.

- Use the same authenticated session for all configured broker IDs.

### B. Login API

**Endpoint:** `POST /api/login`

**Headers:**

- `accept: text/plain`

- `Content-Type: application/json-patch+json`

**Request body:**

```json

{

  "loginId": "${OMS_LOGIN_ID}",

  "password": "${OMS_PASSWORD}",

  "deviceId": "${OMS_DEVICE_ID}",

  "mfaKey": "${OMS_MFA_KEY}",

  "mfaCode": "${OMS_MFA_CODE}",

  "appType": 1

}

```

The MFA code is generated dynamically by an authenticator. Obtain the current code through the configured environment variable or an explicitly configured, secure TOTP integration.

Do not assume that a static MFA code remains valid indefinitely.

### C. Login Response Validation

The login API returns the authentication information inside the `data` object.

Extract the following fields:

- `data.userId`

- `data.accessToken`

- `data.refreshToken`

- `data.accessTokenExpiryDateTimeUtc`

- `data.success`

- `data.isAuthenticated`

- `data.isAuthorised`

- `data.isMfaRequired`

- `data.passwordChangeRequired`

- `data.errorMessage`

Accept authentication only when:

1. The HTTP request succeeds.

2. The response is valid JSON.

3. Top-level `success` is `true`.

4. `data.success` is `true`.

5. `data.isAuthenticated` is `true`.

6. `data.isAuthorised` is `true`.

7. `data.accessToken` is present and non-empty.

8. `data.refreshToken` is present and non-empty.

If any required condition fails, stop authentication and report a sanitized error. Do not call protected endpoints or train the model using an unauthenticated session.

Handle `isMfaRequired` and `passwordChangeRequired` explicitly. If a password change is required or the authentication flow cannot be completed, stop and report the appropriate action.

### D. In-Memory Token Management

Maintain a centralized in-memory token state containing:

- Access token

- Refresh token

- Access-token expiry time in UTC

- Authentication status

Parse `data.accessTokenExpiryDateTimeUtc` as a UTC timestamp. Check token expiry before making protected requests.

Refresh the token before expiry when appropriate. Use a configurable safety margin to avoid requests with nearly expired tokens.

Do not persist tokens to disk, include them in the training dataset, expose them in Streamlit, or write them to application logs.

### E. Refresh-Token API

**Endpoint:** `POST /api/login/refresh-token`

**Headers:**

- `accept: text/plain`

- `Content-Type: application/json-patch+json`

**Request body:**

```json

{

  "accessToken": "<current_access_token>",

  "refreshToken": "<current_refresh_token>",

  "deviceId": "<configured_device_id>"

}

```

Validate the refresh response using the same required success and authentication fields as the login response.

Extract the updated tokens from:

- `data.accessToken`

- `data.refreshToken`

- `data.accessTokenExpiryDateTimeUtc`

Replace the in-memory token state only after validating the returned credentials.

If refresh fails, stop protected requests and report an authentication error. Do not retry refresh indefinitely or silently fall back to the old CSV dataset.

### F. Protected Broker-Summary Endpoints

Use the current access token for both endpoints:

**1. Order execution summary**

`GET /api/broker-summary/orders-execution`

**2. Order summary**

`GET /api/broker-summary/orders`

Send the following headers with each request:

```text

accept: text/plain

Authorization: Bearer <current_access_token>

X-BrokerId: <configured_broker_id>

```

Send the configured query parameters:

- `FromDate`

- `ToDate`

- `StockExchange`

The same access token may be reused for every configured broker ID.

For each broker, call both endpoints with the same date range and stock exchange. Pair the responses using the broker ID and query configuration.

### G. Expiry, Refresh, and Retry Behavior

Implement the following sequence:

1. Check whether a valid access token is available.

2. Authenticate if no valid token state exists.

3. Refresh the token before expiry when appropriate.

4. Execute the protected API request.

5. If the API explicitly indicates an expired or invalid access token, refresh the token and retry the affected request once.

6. If refresh or the retry fails, stop the affected operation and report a sanitized error.

Do not refresh tokens for unrelated errors such as invalid parameters, missing permissions, or server-side failures unless the API explicitly identifies an authentication problem.

Prevent infinite retry loops and concurrent refresh operations that could invalidate rotated refresh tokens.

### H. API Client Architecture

Keep authentication separate from data retrieval and model training.

Use a dedicated authentication component and a reusable OMS API client. Share the token state across requests and brokers.

Reuse existing dependencies where possible. Do not add new dependencies without approval.

Configure request timeouts and bounded retries. Never disable TLS certificate verification.

### I. Security and Error Handling

- Never log passwords, MFA codes, access tokens, refresh tokens, or authorization headers.

- Do not expose sensitive API responses or credentials in Streamlit error messages.

- Distinguish authentication failures from data retrieval failures.

- Do not silently skip failed brokers or fabricate missing observations.

- Do not train on incomplete or invalid datasets.

- Stop dataset construction when authentication is invalid.

- Keep tokens in memory and clear the token state when authentication is invalidated.

- Handle missing configuration, invalid MFA codes, expired tokens, rejected refresh tokens, and required password changes explicitly.

### J. Definition of Done — Authentication

1. Environment configuration is validated.

2. Login uses the documented request schema.

3. MFA is supported through the configured authenticator workflow.

4. Login responses are validated using the documented fields.

5. Access tokens, refresh tokens, and expiry timestamps are extracted correctly.

6. Refresh requests use the documented request schema.

7. Successful refresh replaces the in-memory token state.

8. One access token can authenticate requests for all configured broker IDs.

9. Expiry handling and retry behavior are bounded.

10. Secrets are excluded from logs, source control, and the user interface.

11. Authentication failures prevent dataset construction and model training.

12. The implementation is verified using a successful API response or representative test fixtures, including invalid MFA, rejected refresh tokens, and expired access tokens.

### 3.2 API configuration and day-wise date range

Configure the following environment variables:
- OMS_API_BASE_URL: https://test-oms-api.ecosoftbd.com
- OMS_FROM_DATE: Inclusive start date in YYYY-MM-DD format.
- OMS_TO_DATE: Inclusive end date in YYYY-MM-DD format.
- OMS_BROKER_IDS: Multiple broker IDs stored as a comma-separated list.
- OMS_STOCK_EXCHANGE: Stock exchange identifier; default to 1 only when no value is configured and this default is approved.
Authentication settings are defined in Section 3.1. Do not configure or use a static OMS_AUTH_TOKEN; obtain and refresh tokens through the login lifecycle in oms_auth.py.

Validate the date format and ensure OMS_FROM_DATE is not later than OMS_TO_DATE. Generate every calendar date from the start date through the end date, inclusive. For each iteration, set both FromDate and ToDate to the current loop date so each API response represents one day only.

Never hardcode credentials or commit secrets.

### 3.3 API endpoints

Call both endpoints for every configured broker ID and every date generated by Section 3.2. Use the same current date, broker ID, stock exchange, and authenticated access token for the two requests that form one observation.

Endpoint A: Order execution summary

GET /api/broker-summary/orders-execution

Query parameters:
- FromDate: Current loop date in YYYY-MM-DD format.
- ToDate: The same current loop date in YYYY-MM-DD format.
- StockExchange: Value from OMS_STOCK_EXCHANGE.
Headers:
- accept: text/plain
- X-BrokerId: <broker_id>
- Authorization: Bearer <current_access_token>
Required response fields:
- data.totalValue
- data.totalTrade
Endpoint B: Order summary

GET /api/broker-summary/orders

Use the same query parameters and headers as Endpoint A for the same broker and loop date.

Required response field:
- data.total
Use the authentication, response-validation, expiry, and refresh rules defined in Section 3.1. Reuse the same authenticated session across all dates and broker IDs; do not log in separately for each request.

### 3.4 Day-wise dataset construction

Build the dataset in memory with pandas. Do not write API responses or generated training observations to disk.

For each configured broker ID, and for each calendar date in the inclusive configured date range:
1. Set FromDate and ToDate to the current loop date.
2. Call Endpoint A and retrieve data.totalValue and data.totalTrade.
3. Call Endpoint B and retrieve data.total.
4. Validate the HTTP status, JSON structure, API success indicators when present, and required numeric fields for both responses.
5. Pair the responses only when both are valid and belong to the same broker, date, and stock exchange.
6. Append exactly one observation with Date, BrokerId, totalValue, total, and totalTrade.
7. If either request fails or required data is invalid, do not create an observation for that broker-date pair. Record a sanitized failure reason and continue only when the failure is recoverable.
The model columns are:
- Feature: totalValue
- Target 1: total
- Target 2: totalTrade
The totalTrade target comes from Endpoint A; total comes from Endpoint B. Keep Date and BrokerId as metadata for coverage reporting and traceability, not as model features unless the model specification is explicitly changed.

Example of the expected dataset shape (values are illustrative only):
Date	BrokerId	totalValue	total	totalTrade
2026-01-01	Broker A	1000000	120	90
2026-01-02	Broker A	1500000	160	110
2026-01-03	Broker A	2000000	210	150
2026-01-01	Broker B	800000	95	70


Do not combine responses from different brokers or dates. Do not treat a multi-day aggregate as a daily observation, fabricate samples, or replace missing values with zero.

Report the number of expected broker-date pairs, successful observations, and failed pairs. Do not silently present a partial dataset as complete. Apply the existing skip-and-report policy to recoverable per-request or per-broker data failures; authentication failures must abort dataset construction as specified in Section 3.1.

Before training, verify that the dataset is non-empty, contains valid numeric values, and meets the model's minimum sample requirements. If the dataset is insufficient, report the issue rather than training or displaying misleading metrics. Daily observations from the same broker may be correlated, so evaluation must avoid temporal leakage; preserve the existing evaluation methodology unless a change is necessary and documented.

### 3.5 Machine-learning requirements

Use the following model specification:

- Feature: `totalValue`

- Targets: `total`, `totalTrade`

- Estimator: `sklearn.linear_model.LinearRegression`

- Prediction input: Turnover in BDT, using the same units as `totalValue`.

Preserve the existing model interface where practical. Keep the training dataset and trained model in memory.

Validate that the dataset contains enough valid observations for the existing training and evaluation approach. Clearly report insufficient data rather than producing misleading evaluation metrics.

Do not change the existing notebook's modeling methodology unless necessary and documented.

### 3.6 Streamlit integration

Update the application so that:

1. The user can enter turnover.

2. The application fetches the required API data and trains or reuses the model according to the existing caching design.

3. The model predicts `total` and `totalTrade`.

4. The predicted values are displayed with clear labels.

5. API, authentication, configuration, and insufficient-data errors are presented clearly.

6. Sensitive credentials never appear in the interface or logs.

### 3.7 API client design

Keep HTTP communication separate from model training.

Use the existing dependencies where possible. If `requests` is already installed, reuse it.

Implement:

- Request timeouts.

- HTTP error handling.

- JSON parsing and API success validation.

- Environment-variable validation.

- Handling for invalid broker IDs and incomplete responses.

- Safe handling of partial failures across multiple brokers.

- Clear diagnostics without exposing authorization headers or tokens.

Do not silently discard failed brokers. Report how many succeeded and failed, and make the policy for partial datasets explicit.

## 4. Example Context

Example Endpoint A response:

`data.totalValue = 540061527.1`

`data.totalTrade = 1130`

Example Endpoint B response:

`data.total = 623`

The resulting training observation is:

| totalValue | total | totalTrade |

|---:|---:|---:|

| 540061527.1 | 623 | 1130 |

This is an illustration of field mapping, not a complete training dataset.

Generate observations by pairing both endpoint responses for each broker. Do not combine responses from different brokers or different date ranges.

## 5. Constraint Context

- Follow `AGENTS.md` and inspect the existing code before making changes.

- Preserve the Streamlit UI and existing model behavior where possible.

- Use environment variables for dates, broker IDs, credentials, MFA configuration, and API configuration; obtain tokens through the Section 3.1 authentication lifecycle.
- Query both broker-summary endpoints once per broker per calendar day, with FromDate and ToDate set to that same day.
- Keep API data and model training data in memory.

- Do not modify or depend on the old CSV as the new production data source.

- Do not add dependencies without approval.

- Do not modify the reference notebook unnecessarily.

- Preserve existing preprocessing and train/test conventions unless a documented change is required.

- Do not invent historical samples or treat aggregate totals as individual transactions.

- Do not claim that the model is reliable merely because training and prediction execute successfully.

## 6. Definition of Done

1. Both endpoints are called with the correct parameters and headers.

2. Multiple broker IDs are supported.

3. Environment configuration is validated.

4. API responses are correctly paired and converted into a pandas dataset.

5. The feature and both targets are mapped correctly.

6. The model trains and predicts using in-memory API data.

7. Streamlit displays both predictions.

8. Errors and partial API failures are handled explicitly.

9. No API credentials or training dataset are written to disk.

10. Existing project verification commands are run where possible.

11. Changes, tests, limitations, and any remaining blockers are documented.

Before implementation, inspect the existing project and resolve any ambiguity that would materially affect correctness. In particular, assess whether the number of configured brokers provides enough independent observations for meaningful model training and evaluation.

---

## 7. Implementation Outcome

Implemented on branch `feature/api-endpoint`.

### Files changed

- **`oms_api.py` (new)** — HTTP client. `OmsConfig` dataclass + `load_config()` validates all `OMS_*` env vars up front and reports every problem in one aggregated error (token value never included). `fetch_broker()` calls both endpoints per broker with `FromDate`/`ToDate`/`StockExchange` params and `accept`/`X-BrokerId`/`Authorization` headers, `timeout=(5, 30)`. Failure classes: timeout, connection error, 401/403 (auth), 404 (invalid broker/base URL), 5xx, non-JSON body, missing/non-numeric required fields. `build_dataset()` pairs both responses per broker (never across brokers), skips-and-reports failed brokers, and returns `(DataFrame, FetchReport)` in memory. CLI: `mlvenv\Scripts\python oms_api.py`.

- **`model.py` (modified)** — `FEATURE="totalValue"`, `TARGETS=["total","totalTrade"]`; CSV loader removed. Added `MIN_OBS_TO_TRAIN=2` / `MIN_OBS_FOR_METRICS=10` guards (`InsufficientDataError`). `evaluate_model`/`train_full_model`/`predict_orders` signatures unchanged. CLI repurposed to fetch from the API and report metrics-or-notice plus sample predictions.

- **`app.py` (modified)** — cached bundle validates config → `st.error` + `st.stop()` (variable names only); fetch report shown in an expander (N/M succeeded + per-broker reasons); insufficient-data notice replaces metrics when <10 rows; labels "Total Orders"/"Total Trades"; number-input default clamped into the fetched [min, max] range.

- **`AGENTS.md` (updated)** — API as data source, `OMS_*` env vars required to run/verify, new column names, updated commands and gotchas.

- Unchanged: `order_prediction_dataset_v2.ipynb`, `dataset/*.csv`, `requirements.txt` (reuses existing `requests`), `.gitignore`.

### Decisions resolved before implementation

- **Column names** adopted from the API field mapping (`totalValue`/`total`/`totalTrade`) rather than remapping onto the old notebook labels, because the semantics differ (`totalTrade` ≠ "Executed Order"). The notebook's methodology is unchanged; its column names differ from the API-driven ones by design.

- **Partial-failure policy**: skip-and-report — successes are kept, failures are listed per broker in the UI and CLI; zero successes is a hard error.

- **Small-data policy**: predictions require ≥2 observations (else refuse to train); evaluation metrics are withheld below 10 observations, because the 80/20 split needs ≥2 test rows for R² to be defined.

- **Env loading**: pure environment variables (no `.env` support, no `python-dotenv`), per project rule against unapproved dependencies.

### Verification performed

- All three modules compile (`py_compile`).

- Config validation: missing vars, bad URL scheme, invalid date formats, reversed date range, and empty/duplicate broker ID lists all produce one aggregated, token-free error.

- Live API negative test (invalid token): HTTP 401 correctly classified as an auth failure; token value never printed.

- Connection-error path: unreachable host correctly classified and reported per broker.

- Model guards: 1 observation → refuses to train; 9 observations → trains but withholds metrics; 10 observations → metrics computed; `predict_orders` interface verified.

- Streamlit app launches headless and serves HTTP 200 with the config-error state (no credentials set).

### Limitations / blockers (first increment — superseded by Section 8)

- **Live success path not yet verified end-to-end**: the positive fetch → train → predict flow needs a valid `OMS_AUTH_TOKEN`, real broker IDs, and a date range. The user indicated they would provide credentials; run `mlvenv\Scripts\python model.py` and `mlvenv\Scripts\streamlit run app.py` with real `OMS_*` values to complete DoD items 1–7 against the live API.

- The API's full response schema is unknown beyond the documented `data.*` fields; API-level success is detected via common boolean flags (`success`/`isSuccess`/`status`) when present, plus required-field validation. If the API uses a different success indicator, extend `_request_json` in `oms_api.py`.

- No automated test suite exists in this repo (unchanged); verification is manual via the CLIs above.

---

## 8. Implementation Outcome — Authentication & Token Lifecycle (§3.1)

Implemented on branch `feature/api-endpoint`. This increment replaces the static `OMS_AUTH_TOKEN` with the login/MFA/refresh flow.

### Files changed

- **`oms_auth.py` (new)** — dedicated auth component. `OmsAuthConfig` + `load_auth_config()` validate login vars (aggregated, name-only errors). `OmsAuthenticator` handles: `login()` (POST `/api/login`, exact body/headers), `refresh()` (POST `/api/login/refresh-token` with actual current tokens), in-memory `TokenState` (access/refresh tokens + UTC expiry), proactive refresh within a 60s safety margin, and thread-safe single-flight refresh. `_validate_auth_payload()` enforces all 8 conditions from §3.1.C (top `success` → `data` object → `data.success` → `passwordChangeRequired` branch → non-empty `accessToken` (with `isMfaRequired` branch) → non-empty `refreshToken` → `isAuthenticated` → `isAuthorised`). Includes `parse_utc_expiry()` (ISO-8601 Z/offset, epoch s/ms, .NET `/Date()/`) and a built-in `.env` loader (`load_env_file()`; real env vars win, no new dependency). Raises `OmsAuthError`.

- **`oms_api.py` (modified)** — `OmsConfig`/`load_config()` no longer read `OMS_AUTH_TOKEN`; `fetch_broker`/`build_dataset`/`_request_json` take an `OmsAuthenticator`. Protected requests build `Authorization: Bearer <current token>` per request; HTTP 401/403 triggers exactly one `refresh()` + one retry (second failure → `OmsAuthError`, state invalidated). No refresh for 4xx-parameter/5xx errors. `build_dataset` lets `OmsAuthError` propagate to abort the whole build (never trains on an unauthenticated session) while per-broker `OmsApiError` failures remain skip-and-report.

- **`app.py` (modified)** — authenticates once in the cached bundle (one token serves all brokers); catches `OmsAuthError` separately from config/data errors with a corrective-action message (never credential values).

- **`model.py` (modified)** — CLI authenticates before fetching; docstring notes auth via `oms_auth.py`.

- **`.env.example` (new)** — placeholder-only template; **`.gitignore` (modified)** — added `.env`.

- **`AGENTS.md` (updated)** — auth layer, `.env`/env-var config, the "HTTP 200 ≠ authenticated" gate, MFA-restart gotcha.

### Decisions resolved

- **Auth source**: login flow only; static `OMS_AUTH_TOKEN` removed (§3.1.A's env block omits it; the §3.2 mention is stale spec text).

- **`.env` support**: built-in ~20-line loader (no `python-dotenv`), honoring the no-unapproved-dependencies rule; `.env` gitignored, `.env.example` committed.

- **MFA**: read from env; a stale/rotating code surfaces the API's own `errorMessage` with "update OMS_MFA_CODE and restart" guidance — no automatic retry.

### Verification performed

- All four modules compile (`py_compile`).

- **Auth validation matrix (mocked, 16 cases)**: success path; top-`success:false`; missing `data`; `data.success:false` (live-shaped "Wrong Login ID or Password!"); `passwordChangeRequired`; `isMfaRequired` with empty token ("Invalid Code!"); empty access/refresh tokens; `isAuthenticated:false`; `isAuthorised:false`; HTTP 400; non-JSON; refresh rotation (AT1→AT2, RT1→RT2); rejected refresh (state cleared, no loop); expiry parsing (Z, offset, epoch-ms, .NET, garbage→None); proactive refresh on near-expiry.

- **API retry matrix (mocked, 5 cases)**: 401→refresh→success; 401→refresh→401 (bounded `OmsAuthError`); 404 does **not** trigger refresh; auth failure aborts `build_dataset` (bounded GET attempts); partial per-broker data failures still skip-and-report.

- **`.env` loader**: loads values, does not override real env vars, skips comments/malformed lines; config-error messages name variables but never leak secret values.

- **Live negative test** (bad login creds against `test-oms-api.ecosoftbd.com`): `oms_api.py` prints config (secrets redacted) → "Authentication failed: login rejected by the OMS API: Wrong Login ID or Password!" → exit 1; no protected endpoint touched.

- **`.env` end-to-end**: with cleared shell vars and a temp `.env`, the CLI loaded config, attempted login, and rejected bad credentials correctly.

- **Streamlit**: launches headless, serves HTTP 200 with the auth/config-error state (no valid creds), no stderr errors.

### Limitations / blockers

- **Live success path VERIFIED end-to-end** (after this increment): real login succeeded against `test-oms-api.ecosoftbd.com` with MFA unset (account has no MFA); `oms_api.py` fetched 3/4 brokers (one timed out — skip-and-report worked); `model.py` withheld metrics at 3 obs (correct) and trained/predicted on all rows; Streamlit served HTTP 200 with real data. The successful login confirms the spec §3.1.C field names (`data.accessToken`/`data.refreshToken`/`data.isAuthenticated`/`data.isAuthorised` etc.) — validation would have failed otherwise.

- Refresh-response field set assumed identical to login (per spec §3.1.E); the refresh path is verified via mocked rotation/rejection tests, not a live token expiry.

- MFA code may be one-time/rotating for MFA-enabled accounts: rejected logins require updating `OMS_MFA_CODE` and restarting the app (cached bundle) — an operational limitation, not a silent retry. This account has no MFA, so the code is left as a 2-char placeholder.

- **MFA is optional**: `OMS_MFA_KEY`/`OMS_MFA_CODE` are not required at config-validation time (adjusted after implementation — this account has no MFA enabled). When unset, empty strings are sent and the API is the source of truth; if it reports `isMfaRequired`, the guidance branch surfaces "set OMS_MFA_KEY and the current OMS_MFA_CODE and restart".

- One configured broker (`68da39b44902f277ee457c8a`) timed out against `/orders-execution` during verification — possibly a slow/absent broker; the skip-and-report path handled it (3/4 rows). Re-run or remove the ID if it persistently times out.

- With only 3-4 brokers, evaluation metrics are correctly withheld (<10 obs). Add more `OMS_BROKER_IDS` to enable the 80/20 evaluation.

- No automated test suite exists in this repo (unchanged); verification is manual via the CLIs and the mocked matrices above.

---

## 9. Implementation Outcome — Day-wise Dataset Construction (§3.2–§3.4)

Implemented on branch `feature/api-endpoint`. This increment replaces per-broker whole-range fetching with per-broker-per-day fetching (`FromDate = ToDate = loop date`) and switches evaluation to a temporal split. The auth layer (§3.1) is unchanged.

### Training Data Eligibility Rule

Only include daily observations where `totalValue > 0` in the machine-learning training dataset.

For each broker and calendar date:

1. Fetch both OMS API responses for the same broker and date.
2. Validate the API responses and required fields.
3. Extract `totalValue` and `totalTrade` from the `orders-execution` endpoint and `total` from the `orders` endpoint.
4. Validate that the required values are numeric and not missing.
5. Include the observation in the training dataset only if `totalValue > 0`.
6. Exclude observations where `totalValue` is zero or negative.
7. Record the number of excluded observations for reporting.

**Dataset eligibility condition:**

```python
if totalValue > 0:
    include_observation = True
else:
    include_observation = False
```

The resulting dataset must contain only eligible observations with these columns:

- `Date` — metadata
- `BrokerId` — metadata
- `totalValue` — model feature
- `total` — target 1
- `totalTrade` — target 2

The model must use:

```python
X = dataset[["totalValue"]]
y = dataset[["total", "totalTrade"]]
```

Apply the filter before train/test splitting and model training. Do not replace excluded observations with zero or fabricate substitute records.

If no eligible observations remain, stop training and report that no records satisfy `totalValue > 0`. If too few eligible observations remain, report insufficient data instead of training an unreliable model.

### Definition of Done — Training Data Eligibility

1. Observations with `totalValue > 0` are eligible for training.
2. Observations with `totalValue <= 0` are excluded.
3. Missing, malformed, or invalid required fields are handled separately from the zero-value filter.
4. The application reports eligible and excluded observation counts.
5. Train/test splitting and model training use only eligible observations.
6. The serving model is trained on all eligible observations after evaluation, following the existing project conventions.

### Files changed

- **`oms_api.py` (modified)** — `iter_dates()` yields every calendar date inclusive. `fetch_broker_day(broker_id, day, cfg, auth)` replaces `fetch_broker`; `_request_json()` takes the loop date and sends `FromDate = ToDate = day`. `build_dataset(cfg, auth, max_workers=FETCH_MAX_WORKERS)` authenticates once in the main thread, then fetches all broker×day pairs via a `ThreadPoolExecutor` (4 workers); per-pair `OmsApiError` → recorded in `FetchReport.failed[(broker_id, date)]` and skipped; `OmsAuthError` → cancels pending futures and aborts the whole build. Final DataFrame sorted by `Date`, `BrokerId`. Columns: `Date`, `BrokerId`, `totalValue`, `total`, `totalTrade` (`Date`/`BrokerId` are metadata, never model features). `FetchReport` reworked: `expected_pairs`, `succeeded_count`, `failed` keyed by (broker, date), plus `failed_by_broker()` aggregation and `print_failure_summary()` so long ranges stay readable. A `200` with numeric zeros is a valid observation — failed pairs are never zero-filled.
- **`model.py` (modified)** — new `temporal_split(df)`: the most recent `ceil(TEST_SIZE × n_dates)` distinct dates form the test set, all earlier dates the train set; split by DATE (not row) so no same-day data crosses the boundary. `evaluate_model()` uses it and returns extra metadata (`train_rows`/`test_rows`/`train_dates`/`test_dates`); `predictions_vs_actual` now includes `Date`/`BrokerId`. New guard: `MIN_DATES_FOR_METRICS=2` (single-date data → `InsufficientDataError`). `train_full_model`/`predict_orders` unchanged. CLI prints the aggregated failure summary and caps dataset printing at head/tail 10 rows.
- **`app.py` (modified)** — fetch-report expander shows "N/M broker-date pairs" with per-broker aggregated failure details; metrics panel describes the temporal split with train/test row counts and date ranges.
- **`AGENTS.md` (updated)** — day-wise iteration rules, request-count/runtime warning, temporal-split rationale.

### Decisions resolved (user-approved)

- **Bounded concurrency**: `ThreadPoolExecutor(max_workers=4)` — ~1,196 pairs (299 days × 4 brokers) for the current `.env` range complete in minutes instead of ~20+ sequential; the authenticator's lock already prevents concurrent-refresh races.
- **Temporal split**: replaces the notebook's random `train_test_split` as a documented methodology change — daily observations from the same broker are correlated, so a random split would leak future days into training (spec §3.4 mandate). The estimator (`LinearRegression`) and feature/target mapping are unchanged.
- **Live verification strategy**: small probe first (2 days × 2 brokers), then the full configured range.

### Verification performed

- All four modules compile (`py_compile`).
- **Mocked matrix (9 groups)**: inclusive date iteration (multi-day, single-day, cross-month); `FromDate == ToDate` on every request; per-pair value pairing (no cross broker/date mixing); deterministic Date/BrokerId sort; partial-failure counts + `failed_by_broker()` aggregation; auth failure aborts the build with bounded GET attempts; numeric zeros accepted as valid observations; temporal split ordering (train max date < test min date; most recent 20% of dates held out); single-date data blocked; `predictions_vs_actual` carries Date/BrokerId.
- **Live small probe** (2 days × 2 brokers against `test-oms-api.ecosoftbd.com`): login succeeded, `FromDate == ToDate` accepted, 4/4 pairs fetched; low-activity days returned genuine zeros (valid observations, not failures).
- **Live full-range run** (299 days × 4 brokers ≈ 1,196 pairs, 4 workers) — see the run result below.
- Streamlit launch re-checked after UI changes.

### Limitations / blockers

- **Live mid-run token expiry observed and handled**: the first full-range run aborted partway with the API rejecting a refresh (`OmsAuthError: token refresh rejected by the OMS API: Invalid token request!`). Root cause: the test API appears to require a STILL-VALID access token to refresh, so a refresh issued after server-side expiry fails. Fix (spec-compliant, bounded — not an indefinite retry): `_refresh_locked()` now falls back to a full re-login once when the refresh is rejected (spec §3.1.G "Authenticate if no valid token state exists"); combined error surfaces if re-login also fails. A concurrency guard `refresh_if_still(stale_token)` ensures that when the 4 workers 401 simultaneously, only the first triggers refresh/re-login and the rest reuse the rotated token (no refresh/login storm). `EXPIRY_SAFETY_MARGIN` raised 60s→300s so proactive refresh fires well before expiry during long concurrent runs. Verified with a 5-case mocked matrix (refresh-rejected→re-login-succeeds; both-fail→combined error; stale-token reuse without redundant refresh; current-token triggers refresh; end-to-end build_dataset recovery) and re-run live.

- A full-range run issues ~2,400 HTTP requests against the test API; very long ranges (multi-year) would need batching/rate-limit awareness if the API starts throttling.
- With `@st.cache_resource`, the day-wise fetch runs once per app startup — a data refresh requires an app restart.
- Broker-days with genuinely no activity return zeros from the API and enter the dataset as real observations; they are indistinguishable (by design) from active low-volume days.
- The known-slow broker (`68da39b44902f277ee457c8a`) may add failed pairs on days when it times out; the skip-and-report path handles this per pair.