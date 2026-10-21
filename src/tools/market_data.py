"""Alpha Vantage quotes and daily price history, cached, with mock fallbacks.

Order: cache → live API → mock data. The demo must never die because the
25 req/day quota ran out.
"""

import json
import logging
import random
import threading
import time
import zlib
from typing import Any

import pandas as pd
import requests

from src.config import settings
from src.tools.cache import ttl_cache

logger = logging.getLogger(__name__)

BASE_URL = "https://www.alphavantage.co/query"
TIMEOUT_S = 10
MIN_INTERVAL_S = 1.1  # free tier rejects bursts faster than 1 request/second
HISTORY_TTL_S = 12 * 3600  # daily bars change once a day; a long TTL spares the quota

# Typical daily volatility for synthetic sample history; anything unlisted gets the default.
MOCK_DAILY_VOL = {"BND": 0.003, "SPY": 0.009, "VTI": 0.009, "VXUS": 0.009, "QQQ": 0.012}
MOCK_DEFAULT_VOL = 0.018
MOCK_HISTORY_DAYS = 100

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


@ttl_cache("alpha_vantage_daily", ttl=HISTORY_TTL_S)
def _fetch_history(symbol: str) -> list[dict[str, Any]] | None:
    _throttle()
    resp = requests.get(
        BASE_URL,
        params={
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "outputsize": "compact",  # last 100 trading days; "full" is premium-only
            "apikey": settings.alpha_vantage_api_key,
        },
        timeout=TIMEOUT_S,
    )
    resp.raise_for_status()
    payload = resp.json()
    series = payload.get("Time Series (Daily)")
    if not series:
        logger.warning("Alpha Vantage returned no history for %s: %s", symbol, payload)
        return None
    return [{"date": day, "close": float(bar["4. close"])} for day, bar in sorted(series.items())]


def _mock_history(symbol: str) -> list[dict[str, Any]] | None:
    """Synthetic random walk ending at the mock price; seeded so every run draws the same."""
    quote = _mock_quote(symbol)
    if not quote:
        return None
    rng = random.Random(zlib.crc32(symbol.encode()))
    vol = MOCK_DAILY_VOL.get(symbol, MOCK_DEFAULT_VOL)
    walk = [1.0]
    for _ in range(MOCK_HISTORY_DAYS - 1):
        walk.append(walk[-1] * (1 + rng.gauss(0.0004, vol)))
    scale = quote["price"] / walk[-1]
    days = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=MOCK_HISTORY_DAYS)
    return [
        {"date": day.strftime("%Y-%m-%d"), "close": round(price * scale, 2)}
        for day, price in zip(days, walk, strict=True)
    ]


def get_history(symbol: str) -> dict[str, Any] | None:
    """Return {symbol, source, bars: [{date, close}, ...]} oldest first, "live" or "mock"."""
    symbol = symbol.upper().strip()
    if not settings.use_mock_data and settings.alpha_vantage_api_key:
        try:
            bars = _fetch_history(symbol)
            if bars:
                return {"symbol": symbol, "source": "live", "bars": bars}
        except (requests.RequestException, KeyError, ValueError) as exc:
            logger.warning("Alpha Vantage history failed for %s, using mock: %s", symbol, exc)
    bars = _mock_history(symbol)
    return {"symbol": symbol, "source": "mock", "bars": bars} if bars else None


def get_histories(symbols: list[str]) -> dict[str, dict[str, Any]]:
    """Fetch several histories; symbols with no data are omitted."""
    return {s: h for s in symbols if (h := get_history(s))}
