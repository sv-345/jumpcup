"""SportsPredict REST client — the submission surface.

Confirmed working against https://api.sportspredict.com/api/v1 with a
`Authorization: Bearer sp_live_...` token. This lets the daemon run fully
standalone (no MCP). Handles auth, the 60 req/min rate limit (token bucket with
a safety margin), retries on transient errors, and the idempotency quirks:
re-POSTing an already-predicted market returns 409, so we recover existing
prediction ids via GET /predictions and PATCH thereafter.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests
from tenacity import (retry, retry_if_exception_type, stop_after_attempt,
                      wait_exponential)

from . import config
from .logging_setup import get_logger

log = get_logger("jumpbot.submission")

_BASE = "https://api.sportspredict.com/api/v1"
_RATE_MAX = 50          # stay under the 60/min cap with margin
_RATE_WINDOW = 60.0


class TransientError(Exception):
    pass


class _RateLimiter:
    """Simple sliding-window limiter shared across the client."""

    def __init__(self, max_calls: int, window: float):
        self.max_calls = max_calls
        self.window = window
        self._calls: deque = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            now = time.time()
            while self._calls and now - self._calls[0] > self.window:
                self._calls.popleft()
            if len(self._calls) >= self.max_calls:
                sleep_for = self.window - (now - self._calls[0]) + 0.05
                log.info("rate limit: sleeping %.1fs", sleep_for)
                time.sleep(max(sleep_for, 0))
                now = time.time()
                while self._calls and now - self._calls[0] > self.window:
                    self._calls.popleft()
            self._calls.append(time.time())


@dataclass
class BatchResult:
    total: int
    succeeded: int
    failed: int
    # market_id -> (success, prediction_id_or_None, error_or_None)
    per_market: Dict[str, tuple]


class SportsPredictClient:
    def __init__(self, api_key: Optional[str] = None, base: str = _BASE):
        self.key = api_key or config.SPORTPREDICT_API_KEY
        if not self.key:
            raise RuntimeError("SPORTPREDICT_API_KEY not set")
        self.base = base
        self.limiter = _RateLimiter(_RATE_MAX, _RATE_WINDOW)
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": config.USER_AGENT,
        })

    # --- low-level ---------------------------------------------------------
    @retry(retry=retry_if_exception_type(TransientError),
           stop=stop_after_attempt(config.HTTP_RETRIES),
           wait=wait_exponential(multiplier=config.HTTP_BACKOFF_BASE, min=1, max=30),
           reraise=True)
    def _request(self, method: str, path: str, *, params=None, json=None) -> requests.Response:
        self.limiter.acquire()
        url = f"{self.base}{path}"
        try:
            resp = self.session.request(method, url, params=params, json=json,
                                        timeout=config.HTTP_TIMEOUT)
        except requests.RequestException as exc:
            log.warning("%s %s network error: %s", method, path, exc)
            raise TransientError(str(exc)) from exc
        if resp.status_code == 429 or resp.status_code >= 500:
            log.warning("%s %s -> %s (transient)", method, path, resp.status_code)
            raise TransientError(f"status {resp.status_code}")
        return resp

    def _json(self, resp: requests.Response):
        try:
            return resp.json()
        except ValueError:
            return None

    # --- discovery ---------------------------------------------------------
    def get_events(self) -> list:
        return self._json(self._request("GET", "/events")) or []

    def get_lobbies(self, event_id: str) -> list:
        return self._json(self._request("GET", "/lobbies", params={"event_id": event_id})) or []

    def join_lobby(self, lobby_id: str) -> bool:
        resp = self._request("POST", f"/lobbies/{lobby_id}/join")
        if resp.status_code in (200, 201):
            return True
        if resp.status_code == 409:
            return True  # already joined
        log.warning("join_lobby -> %s: %s", resp.status_code, resp.text[:120])
        return False

    def get_matches(self, event_id: str, lobby_id: Optional[str] = None) -> list:
        params = {"event_id": event_id}
        if lobby_id:
            params["lobby_id"] = lobby_id
        return self._json(self._request("GET", "/matches", params=params)) or []

    def get_markets(self, lobby_id: str, match_id: Optional[str] = None) -> list:
        """All open markets in the lobby (one call) or scoped to a match."""
        params = {"lobby_id": lobby_id}
        if match_id:
            params["match_id"] = match_id
        return self._json(self._request("GET", "/markets", params=params)) or []

    # --- predictions -------------------------------------------------------
    def get_predictions(self, lobby_id: Optional[str] = None) -> list:
        params = {"lobby_id": lobby_id} if lobby_id else None
        return self._json(self._request("GET", "/predictions", params=params)) or []

    def get_results(self, lobby_id: Optional[str] = None) -> list:
        params = {"lobby_id": lobby_id} if lobby_id else None
        return self._json(self._request("GET", "/results", params=params)) or []

    def submit_one(self, market_id: str, lobby_id: str, probability: int):
        """POST a single prediction. Returns (ok, prediction_id, error)."""
        body = {"market_id": market_id, "lobby_id": lobby_id, "probability": int(probability)}
        resp = self._request("POST", "/predictions", json=body)
        data = self._json(resp)
        if resp.status_code in (200, 201) and data:
            return True, data.get("id"), None
        if resp.status_code == 409:
            return False, None, "already_predicted"
        return False, None, f"{resp.status_code}: {(data or {}).get('message', resp.text[:80])}"

    def submit_batch(self, predictions: List[dict]) -> BatchResult:
        """POST up to 50 predictions. predictions: [{market_id, lobby_id, probability}]."""
        body = {"predictions": predictions[:50]}
        resp = self._request("POST", "/predictions/batch", json=body)
        data = self._json(resp) or {}
        per: Dict[str, tuple] = {}
        for r in data.get("results", []):
            mid = r.get("market_id")
            if r.get("success"):
                per[mid] = (True, (r.get("trade") or {}).get("id"), None)
            else:
                per[mid] = (False, None, r.get("error", "unknown"))
        return BatchResult(
            total=data.get("total", len(predictions)),
            succeeded=data.get("succeeded", 0),
            failed=data.get("failed", 0),
            per_market=per,
        )

    def patch_prediction(self, prediction_id: str, probability: int):
        """PATCH an existing prediction. Returns (ok, error)."""
        resp = self._request("PATCH", f"/predictions/{prediction_id}",
                             json={"probability": int(probability)})
        if resp.status_code in (200, 201):
            return True, None
        data = self._json(resp) or {}
        return False, f"{resp.status_code}: {data.get('message', resp.text[:80])}"
