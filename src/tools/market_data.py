"""Alpha Vantage quotes with a 30-min cache and a mock fallback.

Order: cache → live API → data/mock/quotes.json. The demo must never die
because the 25 req/day quota ran out.
"""

import json
import logging
import threading
import time
from typing import Any

import requests

from src.config import settings
from src.tools.cache import ttl_cache

logger = logging.getLogger(__name__)

BASE_URL = "https://www.alphavantage.co/query"
TIMEOUT_S = 10
MIN_INTERVAL_S = 1.1  # free tier rejects bursts faster than 1 request/second

_throttle_lock = threading.Lock()
_last_call = 0.0


def _throttle() -> None:
    """Space live calls out so multi-ticker lookups don't trip the burst limit."""
    global _last_call
    with _throttle_lock:
        wait = _last_call + MIN_INTERVAL_S - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()


@ttl_cache("alpha_vantage_quote")
def _fetch_quote(symbol: str) -> dict[str, Any] | None:
    _throttle()
    resp = requests.get(
        BASE_URL,
        params={
            "function": "GLOBAL_QUOTE",
            "symbol": symbol,
            "apikey": settings.alpha_vantage_api_key,
        },
        timeout=TIMEOUT_S,
    )
    resp.raise_for_status()
    payload = resp.json()
    # Quota/limit responses come back as 200 with "Note" or "Information".
    raw = payload.get("Global Quote")
    if not raw:
        logger.warning("Alpha Vantage returned no quote for %s: %s", symbol, payload)
        return None
    return {
        "symbol": raw["01. symbol"],
        "price": float(raw["05. price"]),
        "open": float(raw["02. open"]),
        "high": float(raw["03. high"]),
        "low": float(raw["04. low"]),
        "volume": int(raw["06. volume"]),
        "previous_close": float(raw["08. previous close"]),
        "change": float(raw["09. change"]),
        "change_percent": float(raw["10. change percent"].rstrip("%")),
        "latest_trading_day": raw["07. latest trading day"],
    }


def _mock_quote(symbol: str) -> dict[str, Any] | None:
    quotes = json.loads((settings.mock_dir / "quotes.json").read_text())
    quote = quotes.get(symbol)
    return {**quote, "symbol": symbol} if quote else None


def get_quote(symbol: str) -> dict[str, Any] | None:
    """Return a normalized quote dict with a `source` of "live" or "mock"."""
    symbol = symbol.upper().strip()
    if not settings.use_mock_data and settings.alpha_vantage_api_key:
        try:
            quote = _fetch_quote(symbol)
            if quote:
                return {**quote, "source": "live"}
        except (requests.RequestException, KeyError, ValueError) as exc:
            logger.warning("Alpha Vantage failed for %s, using mock: %s", symbol, exc)
    quote = _mock_quote(symbol)
    return {**quote, "source": "mock"} if quote else None


def get_quotes(symbols: list[str]) -> dict[str, dict[str, Any]]:
    """Fetch several quotes; symbols with no data are omitted."""
    return {s: q for s in symbols if (q := get_quote(s))}
