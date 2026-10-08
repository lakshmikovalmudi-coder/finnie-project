"""Run coroutines from sync code (Streamlit, tests) on one long-lived event loop."""

import asyncio
import threading
from collections.abc import Coroutine
from functools import lru_cache
from typing import Any


@lru_cache(maxsize=1)
def _loop() -> asyncio.AbstractEventLoop:
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, name="finnie-asyncio", daemon=True).start()
    return loop


def run_sync[T](coro: Coroutine[Any, Any, T]) -> T:
    """Use instead of asyncio.run: cached HTTP clients (OpenAI) bind to the loop that
    first used them, so a fresh loop per call breaks every call after the first."""
    return asyncio.run_coroutine_threadsafe(coro, _loop()).result()
