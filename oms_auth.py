"""OMS API authentication and token lifecycle (spec 02 §3.1).

Centralized auth component, separate from data retrieval (oms_api.py) and
model training (model.py). Handles login (with MFA), in-memory token state,
access-token expiry, refresh, and bounded retry — one shared token serves
every configured broker ID.

Endpoints:

  POST /api/login             -> login with loginId/password/deviceId/mfaKey/mfaCode
  POST /api/login/refresh-token -> refresh with current access+refresh tokens

CRITICAL: the API returns HTTP 200 with top-level success=true even on
FAILED logins. The real auth outcome is the NESTED data.success flag plus
data.isAuthenticated / data.isAuthorised. An HTTP 200 alone never means
"authenticated".

Configuration (environment variables, never hardcoded; loaded from .env
when present, without overriding already-set env vars):

  OMS_LOGIN_ID      login identifier
  OMS_PASSWORD      password
  OMS_DEVICE_ID     device identifier (defaults to "Android-01")
  OMS_MFA_KEY       MFA key (optional — only for accounts with MFA enabled)
  OMS_MFA_CODE      current MFA code (optional; may be one-time/rotating — see below)

MFA note: MFA values are optional. Accounts without MFA leave them unset and
empty strings are sent; the API is the source of truth. For MFA-enabled
accounts the code is generated dynamically by an authenticator — a static
env value may stop being valid; when login is rejected, update
OMS_MFA_CODE and restart the app. The API's own errorMessage is surfaced
(sanitized) so the reason is visible.

No secrets (passwords, MFA codes, tokens, auth headers) are ever logged,
persisted to disk, or exposed to the UI.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import requests

# --- Endpoint paths (relative to the configured base URL) ---
LOGIN_PATH = "/api/login"
REFRESH_PATH = "/api/login/refresh-token"

# --- Request behavior ---
REQUEST_TIMEOUT = (5, 30)  # (connect, read) seconds
LOGIN_APP_TYPE = 1

# Refresh the access token this many seconds before it expires, to avoid
# issuing requests with a nearly-expired token. Kept generous because a
# long day-wise fetch can queue requests for minutes at low worker counts,
# and this API appears to require a still-valid access token to refresh.
EXPIRY_SAFETY_MARGIN = timedelta(seconds=300)


class OmsAuthError(RuntimeError):
    """Raised when authentication fails (login, refresh, or token rejection).

    The message is always sanitized: it never contains passwords, MFA
    codes, tokens, or authorization headers.
    """


@dataclass(frozen=True)
class OmsAuthConfig:
    """Validated OMS login configuration (secrets never printed/logged)."""

    base_url: str
    login_id: str
    password: str
    device_id: str
    mfa_key: str
    mfa_code: str


@dataclass
class TokenState:
    """In-memory authentication state (never written to disk)."""

    access_token: str
    refresh_token: str
    expiry_utc: Optional[datetime]  # None when the API omits/unparseably formats it
    authenticated: bool = True


def load_env_file(path: str = ".env") -> None:
    """Load KEY=VALUE lines from a local .env into os.environ.

    Values already present in the environment are NOT overridden (standard
    dotenv semantics: real env vars win). Malformed lines and comments are
    skipped. The .env file is gitignored; only .env.example is committed.
    """
    import os

    try:
        with open(path, encoding="utf-8") as fh:
            lines = fh.readlines()
    except FileNotFoundError:
        return

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_auth_config() -> OmsAuthConfig:
    """Read and validate login configuration from the environment.

    Loads .env first (without overriding existing env vars), then validates
    every required field, reporting all problems together. Secret VALUES are
    never included in error messages — only variable NAMES.
    """
    import os

    load_env_file()

    problems: list[str] = []

    base_url = os.environ.get("OMS_API_BASE_URL", "").strip()
    if not base_url:
        problems.append("OMS_API_BASE_URL is not set")
    elif not base_url.startswith(("http://", "https://")):
        problems.append("OMS_API_BASE_URL must start with http:// or https://")

    login_id = os.environ.get("OMS_LOGIN_ID", "").strip()
    if not login_id:
        problems.append("OMS_LOGIN_ID is not set")

    password = os.environ.get("OMS_PASSWORD", "").strip()
    if not password:
        problems.append("OMS_PASSWORD is not set")

    device_id = os.environ.get("OMS_DEVICE_ID", "").strip() or "Android-01"

    # MFA is OPTIONAL: only accounts with MFA enabled need these. When unset,
    # empty strings are sent; if the account actually requires MFA, the API's
    # own response (data.isMfaRequired / errorMessage) is surfaced by
    # _validate_auth_payload with guidance to set the vars and restart.
    mfa_key = os.environ.get("OMS_MFA_KEY", "").strip()
    mfa_code = os.environ.get("OMS_MFA_CODE", "").strip()

    if problems:
        raise OmsAuthError(
            "Invalid OMS authentication configuration:\n"
            + "\n".join(f"  - {p}" for p in problems)
            + "\nSet these in your environment or local .env file "
            "(see .env.example)."
        )

    return OmsAuthConfig(
        base_url=base_url.rstrip("/"),
        login_id=login_id,
        password=password,
        device_id=device_id,
        mfa_key=mfa_key,
        mfa_code=mfa_code,
    )


def parse_utc_expiry(value: Any) -> Optional[datetime]:
    """Parse an API-supplied expiry value into an aware UTC datetime.

    Handles ISO-8601 (with 'Z' or numeric offset), .NET epoch-milliseconds
    ("/Date(...)/" or bare numbers). Returns None when the value is missing
    or cannot be parsed — callers then rely on HTTP 401-triggered refresh
    instead of proactive expiry checks.
    """
    if value is None:
        return None

    # Numeric (epoch seconds or milliseconds).
    if isinstance(value, (int, float)):
        seconds = float(value)
        if seconds > 1e11:  # clearly milliseconds
            seconds /= 1000.0
        try:
            return datetime.fromtimestamp(seconds, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None

    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None

    # .NET "/Date(1730000000000)/" format.
    if text.startswith("/Date(") and text.endswith(")/"):
        inner = text[len("/Date(") : -len(")/")]
        if inner.isdigit() or (inner.startswith("-") and inner[1:].isdigit()):
            return parse_utc_expiry(int(inner))
        return None

    # ISO-8601: normalize a trailing Z to +00:00 for fromisoformat.
    iso = text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
    try:
        parsed = datetime.fromisoformat(iso)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


class OmsAuthenticator:
    """Logs in, holds token state in memory, refreshes, and gates requests.

    Thread-safe: a lock prevents concurrent refreshes (which could
    invalidate rotated refresh tokens) when Streamlit reruns overlap.
    """

    def __init__(self, config: OmsAuthConfig):
        self._config = config
        self._state: Optional[TokenState] = None
        self._lock = threading.Lock()

    # --- Public API ---

    def get_access_token(self) -> str:
        """Return a valid access token, logging in or refreshing as needed."""
        with self._lock:
            if self._state is None or not self._state.authenticated:
                self._login_locked()
            elif self._is_expiring_locked():
                self._refresh_locked()
            assert self._state is not None
            return self._state.access_token

    def refresh(self) -> str:
        """Force a token refresh (e.g. after an HTTP 401) and return the new token."""
        with self._lock:
            self._refresh_locked()
            assert self._state is not None
            return self._state.access_token

    def refresh_if_still(self, stale_token: str) -> str:
        """Refresh only if `stale_token` is still the current token.

        Used by the API client on HTTP 401: when several worker threads 401
        simultaneously, the first one refreshes (or re-logins) and the rest
        detect the rotated token and skip the redundant refresh, preventing
        refresh/login storms that could invalidate rotated refresh tokens.
        """
        with self._lock:
            if self._state is not None and self._state.access_token != stale_token:
                # Another thread already replaced the token: reuse it.
                return self._state.access_token
            self._refresh_locked()
            assert self._state is not None
            return self._state.access_token

    def invalidate(self) -> None:
        """Clear in-memory token state (auth definitively rejected)."""
        with self._lock:
            self._state = None

    # --- Internals (call only while holding the lock) ---

    def _is_expiring_locked(self) -> bool:
        if self._state is None or self._state.expiry_utc is None:
            return False  # unknown expiry: rely on 401-triggered refresh
        now = datetime.now(timezone.utc)
        return now >= (self._state.expiry_utc - EXPIRY_SAFETY_MARGIN)

    def _login_locked(self) -> None:
        url = f"{self._config.base_url}{LOGIN_PATH}"
        body = {
            "loginId": self._config.login_id,
            "password": self._config.password,
            "deviceId": self._config.device_id,
            "mfaKey": self._config.mfa_key,
            "mfaCode": self._config.mfa_code,
            "appType": LOGIN_APP_TYPE,
        }
        headers = {
            "accept": "text/plain",
            "Content-Type": "application/json-patch+json",
        }

        payload = self._post_json(url, headers, body, "login")
        self._state = self._validate_auth_payload(payload, "login")

    def _refresh_locked(self) -> None:
        if self._state is None:
            # No tokens to refresh: fall back to a full login.
            self._login_locked()
            return

        url = f"{self._config.base_url}{REFRESH_PATH}"
        body = {
            "accessToken": self._state.access_token,
            "refreshToken": self._state.refresh_token,
            "deviceId": self._config.device_id,
        }
        headers = {
            "accept": "text/plain",
            "Content-Type": "application/json-patch+json",
        }

        try:
            payload = self._post_json(url, headers, body, "token refresh")
            self._state = self._validate_auth_payload(payload, "token refresh")
        except OmsAuthError as refresh_error:
            # Refresh rejected — confirmed live against the test API when the
            # access token has already expired server-side: this API appears to
            # require a STILL-VALID access token to refresh, so refresh can fail
            # purely from timing out mid-run. Since we hold the credentials,
            # fall back to a full re-login ONCE (spec 3.1.G "Authenticate if no
            # valid token state exists"). This is bounded — no retry loop.
            self._state = None
            try:
                self._login_locked()
            except OmsAuthError as login_error:
                raise OmsAuthError(
                    f"token refresh failed ({refresh_error}); re-login also "
                    f"failed ({login_error})"
                ) from None

    def _post_json(
        self, url: str, headers: dict[str, str], body: dict[str, Any], context: str
    ) -> dict[str, Any]:
        """POST JSON and return the parsed object, or raise OmsAuthError."""
        try:
            response = requests.post(
                url, json=body, headers=headers, timeout=REQUEST_TIMEOUT
            )
        except requests.exceptions.Timeout:
            raise OmsAuthError(f"{context} request timed out") from None
        except requests.exceptions.ConnectionError:
            raise OmsAuthError(
                f"could not connect to {self._config.base_url} for {context}"
            ) from None
        except requests.exceptions.RequestException as exc:
            raise OmsAuthError(
                f"{context} request failed: {exc.__class__.__name__}"
            ) from None

        if response.status_code >= 400:
            raise OmsAuthError(
                f"{context} failed: HTTP {response.status_code} from the OMS API"
            )

        try:
            payload = response.json()
        except ValueError:
            raise OmsAuthError(
                f"{context} response is not valid JSON (content type "
                f"{response.headers.get('content-type', 'unknown')!r})"
            ) from None

        if not isinstance(payload, dict):
            raise OmsAuthError(f"{context} response is not a JSON object")
        return payload

    def _validate_auth_payload(
        self, payload: dict[str, Any], context: str
    ) -> TokenState:
        """Validate the 8 auth conditions (spec §3.1.C) and build TokenState.

        HTTP 200 + top-level success=true are NOT sufficient: the nested
        data.success / isAuthenticated / isAuthorised flags plus non-empty
        tokens are the real gate.
        """
        # 3. Top-level success.
        if payload.get("success") is not True:
            raise OmsAuthError(f"{context}: top-level 'success' is not true")

        data = payload.get("data")
        # 2/4. data must be an object with nested success=true.
        if not isinstance(data, dict):
            raise OmsAuthError(f"{context}: response has no 'data' object")
        if data.get("success") is not True:
            reason = data.get("errorMessage") or "unsuccessful authentication"
            raise OmsAuthError(f"{context} rejected by the OMS API: {reason}")

        # Explicit password-change / MFA branches (spec §3.1.C).
        if data.get("passwordChangeRequired") is True:
            raise OmsAuthError(
                f"{context}: a password change is required — complete the "
                "password change in the OMS, then update OMS_PASSWORD and restart."
            )

        access_token = data.get("accessToken")
        # 7. access token present and non-empty.
        if not isinstance(access_token, str) or not access_token.strip():
            # MFA not satisfied (or another cause) — surface the API's reason.
            if data.get("isMfaRequired") is True:
                reason = data.get("errorMessage") or "MFA code required"
                raise OmsAuthError(
                    f"{context}: MFA not satisfied — {reason}. Set "
                    "OMS_MFA_KEY and the current OMS_MFA_CODE (the code "
                    "rotates) and restart the app."
                )
            raise OmsAuthError(
                f"{context}: no access token returned"
                + (
                    f" ({data.get('errorMessage')})"
                    if data.get("errorMessage")
                    else ""
                )
            )

        refresh_token = data.get("refreshToken")
        # 8. refresh token present and non-empty.
        if not isinstance(refresh_token, str) or not refresh_token.strip():
            raise OmsAuthError(f"{context}: no refresh token returned")

        # 5/6. authentication/authorisation flags.
        if data.get("isAuthenticated") is not True:
            raise OmsAuthError(f"{context}: session is not authenticated")
        if data.get("isAuthorised") is not True:
            raise OmsAuthError(f"{context}: session is not authorised")

        expiry = parse_utc_expiry(data.get("accessTokenExpiryDateTimeUtc"))
        return TokenState(
            access_token=access_token.strip(),
            refresh_token=refresh_token.strip(),
            expiry_utc=expiry,
            authenticated=True,
        )
