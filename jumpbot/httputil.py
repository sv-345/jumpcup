"""Shared HTTP helper: retry with exponential backoff + a simple disk cache.

Free API tiers are tiny, so caching is essential — we serve a recent response
rather than burning quota on every loop tick.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import requests
from tenacity import (retry, retry_if_exception_type, stop_after_attempt,
                      wait_exponential)

from . import config
from .logging_setup import get_logger

log = get_logger("jumpbot.http")


class TransientHTTPError(Exception):
    pass


@retry(
    retry=retry_if_exception_type(TransientHTTPError),
    stop=stop_after_attempt(config.HTTP_RETRIES),
    wait=wait_exponential(multiplier=config.HTTP_BACKOFF_BASE, min=1, max=30),
    reraise=True,
)
def get_json(url: str, params: Optional[dict] = None, headers: Optional[dict] = None) -> dict:
    """GET returning parsed JSON, retrying transient failures with backoff."""
    hdrs = {"User-Agent": config.USER_AGENT, "Accept": "application/json"}
    if headers:
        hdrs.update(headers)
    try:
        resp = requests.get(url, params=params, headers=hdrs, timeout=config.HTTP_TIMEOUT)
    except requests.RequestException as exc:
        log.warning("GET %s network error: %s", url, exc)
        raise TransientHTTPError(str(exc)) from exc
    if resp.status_code in (429, 500, 502, 503, 504):
        log.warning("GET %s -> %s (transient)", url, resp.status_code)
        raise TransientHTTPError(f"status {resp.status_code}")
    if resp.status_code >= 400:
        log.warning("GET %s -> %s (permanent): %s", url, resp.status_code, resp.text[:200])
        resp.raise_for_status()
    return resp.json()


@retry(
    retry=retry_if_exception_type(TransientHTTPError),
    stop=stop_after_attempt(config.HTTP_RETRIES),
    wait=wait_exponential(multiplier=config.HTTP_BACKOFF_BASE, min=1, max=30),
    reraise=True,
)
def get_json_with_headers(url: str, params: Optional[dict] = None,
                          headers: Optional[dict] = None):
    """Like get_json but also returns response headers (for quota tracking)."""
    hdrs = {"User-Agent": config.USER_AGENT, "Accept": "application/json"}
    if headers:
        hdrs.update(headers)
    try:
        resp = requests.get(url, params=params, headers=hdrs, timeout=config.HTTP_TIMEOUT)
    except requests.RequestException as exc:
        raise TransientHTTPError(str(exc)) from exc
    if resp.status_code in (429, 500, 502, 503, 504):
        raise TransientHTTPError(f"status {resp.status_code}")
    if resp.status_code >= 400:
        resp.raise_for_status()
    return resp.json(), resp.headers


def cache_path(name: str) -> Path:
    return config.CACHE_DIR / name


def read_cache(name: str, max_age_s: float) -> Optional[dict]:
    """Return cached payload if it exists and is younger than max_age_s."""
    p = cache_path(name)
    if not p.exists():
        return None
    age = time.time() - p.stat().st_mtime
    if age > max_age_s:
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def write_cache(name: str, payload: dict) -> None:
    p = cache_path(name)
    tmp = p.with_suffix(p.suffix + ".tmp")
    try:
        with open(tmp, "w") as f:
            json.dump(payload, f)
        tmp.replace(p)
    except OSError as exc:
        log.warning("cache write failed for %s: %s", name, exc)
