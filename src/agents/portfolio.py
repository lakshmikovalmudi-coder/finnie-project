"""Portfolio Analysis agent: diversification and risk review of uploaded holdings."""

import asyncio
import json

import pandas as pd

from src.agents.base import explain
from src.state import FinnieState
from src.tools import portfolio_math
from src.tools.market_data import get_quotes

SYSTEM_PROMPT = """\
You review a beginner's portfolio analysis. Explain the allocation, any
concentration or diversification flags, and what HHI / effective holdings
mean. Describe trade-offs; do not tell the user to buy or sell anything.
"""


async def analyze_portfolio(holdings: list[dict]) -> dict:
    """Price holdings and compute analytics; shared by the agent and the Portfolio tab."""
    df = portfolio_math.load_holdings(pd.DataFrame(holdings))
    quotes = await asyncio.to_thread(get_quotes, df["symbol"].tolist())
    prices = {s: q["price"] for s, q in quotes.items()}
    return portfolio_math.analyze(df, prices)


async def run(state: FinnieState) -> dict:
    holdings = state.get("portfolio")
    if not holdings:
        return {
            "answer": "Upload a portfolio CSV in the **Portfolio** tab "
            "(columns: symbol, shares, optional asset_class) and ask again.",
            "data": {},
        }
    analysis = await analyze_portfolio(holdings)
    answer = await explain(SYSTEM_PROMPT, state["query"], json.dumps(analysis, indent=2))
    return {"answer": answer, "sources": ["Portfolio analysis"], "data": {"portfolio": analysis}}
