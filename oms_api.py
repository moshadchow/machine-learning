"""OMS API client for broker order data.

Fetches per-broker, per-day aggregates from two OMS endpoints and builds
the in-memory training dataset. Authentication is handled by oms_auth.py
(login/MFA/refresh); this module only retrieves data. No Streamlit or
model code lives here.

Day-wise iteration (spec 02 §3.2–§3.4): for every configured broker ID and
every calendar date in the inclusive OMS_FROM_DATE..OMS_TO_DATE range,
both endpoints are called with FromDate == ToDate == that day, so each
valid pair of responses becomes exactly one daily observation (Date,
BrokerId, totalValue, total, totalTrade). Pairs are fetched with a small
bounded thread pool (FETCH_MAX_WORKERS) — a long date range means
thousands of requests, so keep the range intentional.

  Endpoint A  GET /api/broker-summary/orders-execution  (FromDate=ToDate=day)
              -> data.totalValue (feature), data.totalTrade (target 2)
  Endpoint B  GET /api/broker-summary/orders            (FromDate=ToDate=day)
              -> data.total (target 1)

Configuration is read from environment variables (never hardcoded; a local
.gitignored .env is loaded when present):

  OMS_API_BASE_URL    e.g. https://test-oms-api.ecosoftbd.com
  OMS_FROM_DATE       YYYY-MM-DD (inclusive start)
  OMS_TO_DATE         YYYY-MM-DD (inclusive end)
  OMS_BROKER_IDS      comma-separated broker IDs
  OMS_STOCK_EXCHANGE  optional, defaults to "1"

Login credentials (OMS_LOGIN_ID/OMS_PASSWORD/OMS_MFA_*) are validated by
oms_auth.load_auth_config().

Run standalone to verify configuration and connectivity (requires login):

    mlvenv\\Scripts\\python oms_api.py
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Iterator

import pandas as pd
import requests

from oms_auth import OmsAuthError, OmsAuthenticator, load_auth_config

# --- Endpoint paths (relative to the configured base URL) ---
ORDERS_EXECUTION_PATH = "/api/broker-summary/orders-execution"
ORDERS_PATH = "/api/broker-summary/orders"

# --- Request behavior ---
REQUEST_TIMEOUT = (5, 30)  # (connect, read) seconds
FETCH_MAX_WORKERS = 4  # bounded concurrency for the broker x day matrix

# Column names for the constructed dataset (match the spec's field mapping).
# Date and BrokerId are metadata for traceability/coverage, not model features.
DATE_COL = "Date"
BROKER_COL = "BrokerId"
FEATURE_COL = "totalValue"
TARGET_ORDER_COL = "total"  # from Endpoint B (orders)
TARGET_TRADE_COL = "totalTrade"  # from Endpoint A (orders-execution)


class OmsConfigError(ValueError):
    """Raised when required environment configuration is missing or invalid."""


class OmsApiError(RuntimeError):
    """Raised when an OMS API call fails (HTTP, or bad payload).

    Authentication problems raise OmsAuthError (from oms_auth) instead, so
    callers can distinguish auth failures from data-retrieval failures.
    """


@dataclass(frozen=True)
class OmsConfig:
    """Validated OMS API configuration (data retrieval only)."""

    base_url: str
    from_date: str
    to_date: str
    broker_ids: tuple[str, ...]
    stock_exchange: str


@dataclass
class FetchReport:
    """Outcome of fetching the broker x day matrix.

    Failed pairs keep their sanitized reason keyed by (broker_id, date);
    the aggregated per-broker view keeps the UI/CLI readable when one
    broker fails on every day of a long range.
    """

    expected_pairs: int = 0
    succeeded_count: int = 0
    failed: dict[tuple[str, str], str] = field(default_factory=dict)
    broker_count: int = 0
    date_count: int = 0

    def summary(self) -> str:
        return (
            f"{self.succeeded_count}/{self.expected_pairs} broker-date pairs "
            f"fetched successfully; {len(self.failed)} failed."
        )

    def failed_by_broker(self) -> dict[str, list[tuple[str, str]]]:
        """Aggregate failures: broker_id -> list of (date, reason)."""
        by_broker: dict[str, list[tuple[str, str]]] = {}
        for (broker_id, day), reason in sorted(self.failed.items()):
            by_broker.setdefault(broker_id, []).append((day, reason))
        return by_broker


def load_config() -> OmsConfig:
    """Read and validate the data-retrieval OMS environment variables.

    Loads .env first (without overriding existing env vars). All problems
    are collected and reported together so the user can fix them in one
    pass. Login credentials are validated separately by oms_auth.
    """
    import os

    from oms_auth import load_env_file

    load_env_file()

    problems: list[str] = []

    base_url = os.environ.get("OMS_API_BASE_URL", "").strip()
    if not base_url:
        problems.append("OMS_API_BASE_URL is not set")
    elif not base_url.startswith(("http://", "https://")):
        problems.append("OMS_API_BASE_URL must start with http:// or https://")

    from_date = os.environ.get("OMS_FROM_DATE", "").strip()
    to_date = os.environ.get("OMS_TO_DATE", "").strip()
    for name, value in (("OMS_FROM_DATE", from_date), ("OMS_TO_DATE", to_date)):
        if not value:
            problems.append(f"{name} is not set")
        else:
            try:
                datetime.strptime(value, "%Y-%m-%d")
            except ValueError:
                problems.append(f"{name} must be in YYYY-MM-DD format (got {value!r})")

    if from_date and to_date and from_date > to_date:
        problems.append(
            f"OMS_FROM_DATE ({from_date}) must not be after OMS_TO_DATE ({to_date})"
        )

    raw_ids = os.environ.get("OMS_BROKER_IDS", "").strip()
    broker_ids: list[str] = []
    if not raw_ids:
        problems.append("OMS_BROKER_IDS is not set")
    else:
        # Split on comma, strip whitespace, drop empties, dedupe (preserve order).
        seen: set[str] = set()
        for part in raw_ids.split(","):
            broker_id = part.strip()
            if broker_id and broker_id not in seen:
                seen.add(broker_id)
                broker_ids.append(broker_id)
        if not broker_ids:
            problems.append("OMS_BROKER_IDS contains no valid broker IDs")

    stock_exchange = os.environ.get("OMS_STOCK_EXCHANGE", "").strip() or "1"

    if problems:
        raise OmsConfigError(
            "Invalid OMS configuration:\n"
            + "\n".join(f"  - {p}" for p in problems)
            + "\nSet these in your environment or local .env file "
            "(see .env.example)."
        )

    return OmsConfig(
        base_url=base_url.rstrip("/"),
        from_date=from_date,
        to_date=to_date,
        broker_ids=tuple(broker_ids),
        stock_exchange=stock_exchange,
    )


def iter_dates(from_date: str, to_date: str) -> Iterator[str]:
    """Yield every calendar date (YYYY-MM-DD) inclusive, start through end."""
    current = datetime.strptime(from_date, "%Y-%m-%d").date()
    end = datetime.strptime(to_date, "%Y-%m-%d").date()
    while current <= end:
        yield current.isoformat()
        current += timedelta(days=1)


def _request_json(
    cfg: OmsConfig, path: str, broker_id: str, day: str, auth: OmsAuthenticator
) -> dict[str, Any]:
    """Perform one authenticated GET for one broker-day and return 'data'.

    FromDate and ToDate are both set to `day` so the response represents a
    single day only. On HTTP 401/403 (token expired/rejected), refreshes the
    token once and retries the request once. A second auth failure raises
    OmsAuthError. All other failures raise OmsApiError with a sanitized
    message (broker ID and date included for diagnosability; never tokens).
    """
    url = f"{cfg.base_url}{path}"
    params = {
        "FromDate": day,
        "ToDate": day,
        "StockExchange": cfg.stock_exchange,
    }

    def do_get(token: str) -> requests.Response:
        headers = {
            "accept": "text/plain",
            "X-BrokerId": broker_id,
            "Authorization": f"Bearer {token}",
        }
        try:
            return requests.get(
                url, params=params, headers=headers, timeout=REQUEST_TIMEOUT
            )
        except requests.exceptions.Timeout:
            raise OmsApiError(
                f"request to {path} timed out (broker {broker_id}, {day})"
            ) from None
        except requests.exceptions.ConnectionError:
            raise OmsApiError(
                f"could not connect to {cfg.base_url} (broker {broker_id}, {day})"
            ) from None
        except requests.exceptions.RequestException as exc:
            raise OmsApiError(
                f"request to {path} failed: {exc.__class__.__name__} "
                f"(broker {broker_id}, {day})"
            ) from None

    token = auth.get_access_token()
    response = do_get(token)

    if response.status_code in (401, 403):
        # Access token expired or rejected: refresh once, retry once.
        # refresh_if_still(token) is concurrency-safe: when several workers
        # 401 at once, only the first triggers a refresh/re-login; the rest
        # reuse the rotated token. OmsAuthError propagates (bounded — no loop).
        new_token = auth.refresh_if_still(token)
        response = do_get(new_token)
        if response.status_code in (401, 403):
            auth.invalidate()
            raise OmsAuthError(
                f"authentication rejected by {path} (HTTP "
                f"{response.status_code}) after refreshing the token — the "
                "login may have expired; re-run login (update OMS_MFA_CODE) "
                "and restart."
            )

    if response.status_code == 404:
        raise OmsApiError(
            f"HTTP 404 from {path} (broker {broker_id}, {day}) — check the "
            "broker ID and API base URL"
        )
    if response.status_code >= 500:
        raise OmsApiError(
            f"server error (HTTP {response.status_code}) from {path} "
            f"(broker {broker_id}, {day})"
        )
    if response.status_code >= 400:
        raise OmsApiError(
            f"HTTP {response.status_code} from {path} (broker {broker_id}, {day})"
        )

    try:
        payload = response.json()
    except ValueError:
        raise OmsApiError(
            f"response from {path} is not valid JSON (content type "
            f"{response.headers.get('content-type', 'unknown')!r}, "
            f"broker {broker_id}, {day})"
        ) from None

    if not isinstance(payload, dict):
        raise OmsApiError(f"response from {path} is not a JSON object")

    # Validate API-level success if the payload exposes a success flag.
    for flag in ("success", "isSuccess", "Status", "status"):
        if flag in payload and isinstance(payload[flag], bool) and not payload[flag]:
            message = payload.get("message") or payload.get("Message") or "unknown error"
            raise OmsApiError(
                f"API reported failure from {path} (broker {broker_id}, {day}): "
                f"{message}"
            )

    data = payload.get("data")
    if not isinstance(data, dict):
        raise OmsApiError(f"response from {path} has no 'data' object")
    return data


def _extract_number(data: dict[str, Any], key: str, path: str) -> float:
    """Pull a required numeric field out of a 'data' object."""
    value = data.get(key)
    if value is None:
        raise OmsApiError(f"response from {path} is missing required field {key!r}")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise OmsApiError(
            f"field {key!r} from {path} is not numeric (got {value!r})"
        ) from None
    return number


def fetch_broker_day(
    broker_id: str, day: str, cfg: OmsConfig, auth: OmsAuthenticator
) -> dict[str, float]:
    """Fetch both endpoints for one broker on one calendar day.

    Returns a dict with keys: totalValue, total, totalTrade. A valid
    response with numeric zeros is a real observation (never zero-filled
    by us). Raises OmsApiError on data failures or OmsAuthError on auth
    failures.
    """
    execution = _request_json(cfg, ORDERS_EXECUTION_PATH, broker_id, day, auth)
    orders = _request_json(cfg, ORDERS_PATH, broker_id, day, auth)

    total_value = _extract_number(execution, "totalValue", ORDERS_EXECUTION_PATH)
    total_trade = _extract_number(execution, "totalTrade", ORDERS_EXECUTION_PATH)
    total = _extract_number(orders, "total", ORDERS_PATH)

    return {
        FEATURE_COL: total_value,
        TARGET_ORDER_COL: total,
        TARGET_TRADE_COL: total_trade,
    }


def build_dataset(
    cfg: OmsConfig, auth: OmsAuthenticator, max_workers: int = FETCH_MAX_WORKERS
) -> tuple[pd.DataFrame, FetchReport]:
    """Fetch every broker x day pair and assemble the in-memory dataset.

    One token (from the authenticator) serves all pairs. Pairs are fetched
    with a bounded thread pool; results are collected in submission order
    of completion but the final DataFrame is sorted by Date then BrokerId
    for determinism. Per-pair DATA failures are skipped and reported, never
    silently dropped or zero-filled. An AUTHENTICATION failure
    (OmsAuthError) aborts the whole build — no partial dataset is trained
    on an unauthenticated session. Nothing is written to disk.
    """
    dates = list(iter_dates(cfg.from_date, cfg.to_date))
    pairs = [(broker_id, day) for day in dates for broker_id in cfg.broker_ids]
    report = FetchReport(
        expected_pairs=len(pairs),
        broker_count=len(cfg.broker_ids),
        date_count=len(dates),
    )

    # Authenticate once in the main thread before fanning out, so an auth
    # failure surfaces before any pair is fetched.
    auth.get_access_token()

    rows: list[dict[str, Any]] = []
    auth_error: OmsAuthError | None = None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(fetch_broker_day, broker_id, day, cfg, auth): (
                broker_id,
                day,
            )
            for broker_id, day in pairs
        }
        for future in as_completed(future_map):
            broker_id, day = future_map[future]
            if auth_error is not None:
                # Auth already failed: stop consuming results; cancel whatever
                # has not started. Running requests finish on their own.
                future.cancel()
                continue
            try:
                observation = future.result()
            except OmsAuthError as exc:
                # Auth is globally invalid: stop dataset construction entirely.
                auth_error = exc
                for pending in future_map:
                    pending.cancel()
                continue
            except OmsApiError as exc:
                report.failed[(broker_id, day)] = str(exc)
                continue
            report.succeeded_count += 1
            rows.append(
                {
                    DATE_COL: day,
                    BROKER_COL: broker_id,
                    **observation,
                }
            )

    if auth_error is not None:
        raise auth_error

    df = pd.DataFrame(
        rows,
        columns=[DATE_COL, BROKER_COL, FEATURE_COL, TARGET_ORDER_COL, TARGET_TRADE_COL],
    )
    df = df.sort_values([DATE_COL, BROKER_COL]).reset_index(drop=True)
    return df, report


def print_failure_summary(report: FetchReport) -> None:
    """Print an aggregated per-broker failure summary (keeps long ranges readable)."""
    for broker_id, failures in report.failed_by_broker().items():
        reasons = sorted({reason for _, reason in failures})
        sample_dates = ", ".join(day for day, _ in failures[:3])
        more = "" if len(failures) <= 3 else ", ..."
        print(
            f"  FAILED {broker_id}: {len(failures)} date(s); "
            f"example: {reasons[0]} (e.g. {sample_dates}{more})"
        )


# --- CLI verification: authenticate, fetch, and report ---
if __name__ == "__main__":
    try:
        data_config = load_config()
        auth_config = load_auth_config()
    except (OmsConfigError, OmsAuthError) as exc:
        raise SystemExit(str(exc))

    authenticator = OmsAuthenticator(auth_config)

    n_days = sum(1 for _ in iter_dates(data_config.from_date, data_config.to_date))
    n_pairs = n_days * len(data_config.broker_ids)

    # Print config without secrets.
    print("--- OMS Configuration ---")
    print(f"Base URL:       {data_config.base_url}")
    print(f"Date range:     {data_config.from_date} to {data_config.to_date} "
          f"({n_days} days, inclusive)")
    print(f"Stock exchange: {data_config.stock_exchange}")
    print(f"Broker IDs:     {', '.join(data_config.broker_ids)}")
    print(f"Expected pairs: {n_pairs} "
          f"({len(data_config.broker_ids)} brokers x {n_days} days, "
          f"{FETCH_MAX_WORKERS} workers)")
    print("Credentials:    <redacted>")

    print("\n--- Authenticating ---")
    try:
        authenticator.get_access_token()
        print("Login succeeded (token in memory, not printed).")
    except OmsAuthError as exc:
        raise SystemExit(f"Authentication failed: {exc}")

    dataset, fetch_report = build_dataset(data_config, authenticator)

    print("\n--- Fetch Report ---")
    print(fetch_report.summary())
    print_failure_summary(fetch_report)

    print("\n--- Dataset ---")
    print(f"Rows: {len(dataset)}")
    if len(dataset) > 20:
        print(dataset.head(10).to_string(index=False))
        print(f"... ({len(dataset) - 20} more rows) ...")
        print(dataset.tail(10).to_string(index=False))
    else:
        print(dataset.to_string(index=False))
