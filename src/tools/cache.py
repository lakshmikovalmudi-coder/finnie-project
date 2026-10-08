"""Disk-backed TTL cache for external API calls.

Alpha Vantage allows only 25 requests/day, so every external call is cached
(default 30 min) and survives Streamlit reruns and app restarts.
"""

import functools
import hashlib
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.config import settings

logger = logging.getLogger(__name__)


def _key(namespace: str, args: tuple, kwargs: dict) -> str:
    raw = json.dumps([namespace, args, kwargs], sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def cache_get(namespace: str, key: str, ttl: int, cache_dir: Path) -> Any | None:
    path = cache_dir / namespace / f"{key}.json"
    if not path.exists():
        return None
    entry = json.loads(path.read_text())
    if time.time() - entry["ts"] >= ttl:
        return None
    return entry["value"]


def cache_set(namespace: str, key: str, value: Any, cache_dir: Path) -> None:
    path = cache_dir / namespace / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"ts": time.time(), "value": value}, default=str))


def ttl_cache(namespace: str, ttl: int | None = None, cache_dir: Path | None = None) -> Callable:
    """Cache a function's JSON-serializable result on disk. `None` results are not cached."""

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            ttl_s = ttl if ttl is not None else settings.cache_ttl_seconds
            directory = cache_dir or settings.cache_dir
            key = _key(namespace, args, kwargs)
            hit = cache_get(namespace, key, ttl_s, directory)
            if hit is not None:
                logger.debug("cache hit %s %s", namespace, args)
                return hit
            value = fn(*args, **kwargs)
            if value is not None:
                cache_set(namespace, key, value, directory)
            return value

        return wrapper

    return decorator
