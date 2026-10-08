"""Market Analysis agent: fetches quotes and explains what the numbers mean."""

import asyncio
import json
import re

from src.agents.base import explain
from src.state import FinnieState
from src.tools.market_data import get_quotes

SYSTEM_PROMPT = """\
You explain stock/ETF quotes to a beginner: what price, change, % change,
day range, and volume mean. Do not predict future prices. If a quote's
source is "mock", say the data is sample data, not live.
"""

NAMES_TO_TICKERS = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "amazon": "AMZN",
    "nvidia": "NVDA",
    "tesla": "TSLA",
    "s&p 500": "SPY",
    "s&p": "SPY",
    "nasdaq": "QQQ",
    "total market": "VTI",
}
NOT_TICKERS = {"I", "A", "ETF", "IRA", "US", "USA", "THE", "AND", "OR", "PE", "EPS", "CEO", "FAQ"}


def extract_tickers(query: str) -> list[str]:
    found = [t for name, t in NAMES_TO_TICKERS.items() if name in query.lower()]
    found += [t for t in re.findall(r"\$?\b([A-Z]{1,5})\b", query) if t not in NOT_TICKERS]
    return list(dict.fromkeys(found))[:5]  # dedupe, keep order, respect rate limits


async def run(state: FinnieState) -> dict:
    tickers = extract_tickers(state["query"]) or ["SPY"]
    quotes = await asyncio.to_thread(get_quotes, tickers)
    if not quotes:
        return {"answer": f"I couldn't find market data for {', '.join(tickers)}.", "data": {}}
    answer = await explain(SYSTEM_PROMPT, state["query"], json.dumps(quotes, indent=2))
    sources = [
        f"Alpha Vantage ({q['symbol']})"
        if q["source"] == "live"
        else f"Sample data ({q['symbol']})"
        for q in quotes.values()
    ]
    return {"answer": answer, "sources": sources, "data": {"quotes": quotes}}
