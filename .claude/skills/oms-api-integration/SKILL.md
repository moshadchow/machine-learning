---
name: oms-api-integration
description: OMS API client patterns — authentication, day-wise fetching, dataset construction, and error handling.
---

# OMS API Integration

Patterns for `oms_api.py` — the HTTP client that fetches broker data from the OMS API and builds the in-memory training dataset.

## Authentication

- `OmsAuthenticator` (in `oms_auth.py`) handles login/MFA/refresh via `POST /api/login` and `POST /api/login/refresh-token`
- One token serves all configured broker IDs; no separate login per request
- Tokens live in memory only — never logged, persisted, or exposed in UI
- Auth failures abort dataset construction entirely (never train unauthenticated)
- HTTP 200 ≠ authenticated — validate `data.success`, `data.isAuthenticated`, `data.isAuthorised`, and non-empty tokens

## Day-wise Fetching

For each broker × each calendar date in `OMS_FROM_DATE..OMS_TO_DATE` (inclusive):
- Set `FromDate = ToDate = loop date` (one day only)
- Call both endpoints: `orders-execution` (`totalValue`, `totalTrade`) and `orders` (`total`)
- Pair responses per broker/date — never across brokers or dates
- Append one observation: `Date`, `BrokerId`, `totalValue`, `total`, `totalTrade`
- Failed pairs → skip-and-report (never zero-fill); zero successes → hard error
- Use `ThreadPoolExecutor(max_workers=FETCH_MAX_WORKERS)` for bounded concurrency

## Error Handling

- Timeout, connection error, 401/403, 404, 5xx, non-JSON body, missing/non-numeric fields → classified per-broker
- 401/403 triggers exactly one `refresh()` + one retry; second failure → `OmsAuthError`
- 404 does **not** trigger refresh
- `build_dataset` aborts on `OmsAuthError`; per-broker `OmsApiError` remains skip-and-report

## Configuration

All settings via `OMS_*` env vars or `.env` (gitignored). Required: `OMS_API_BASE_URL`, `OMS_LOGIN_ID`, `OMS_PASSWORD`, `OMS_BROKER_IDS`, `OMS_FROM_DATE`, `OMS_TO_DATE`. Optional: `OMS_DEVICE_ID`, `OMS_STOCK_EXCHANGE`, `OMS_MFA_KEY`, `OMS_MFA_CODE`. Never hardcode credentials.

## References

- `oms_api.py` — HTTP client and dataset builder
- `oms_auth.py` — authentication component
- `specs/02-oms-api-integration.md` — full integration spec (§3.1–§3.9)
