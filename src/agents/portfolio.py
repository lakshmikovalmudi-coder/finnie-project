"""Portfolio Analysis agent: diversification and risk review of uploaded holdings."""

import asyncio
import json

import pandas as pd

from src.agents.base import explain
from src.state import FinnieState
from src.tools import portfolio_math
from src.tools.market_data import get_histories, get_quotes

SYSTEM_PROMPT = """\
You review a beginner's portfolio analysis. Explain the allocation, any
concentration or diversification flags, and what HHI / effective holdings
mean. If risk metrics are present, explain annualized volatility (how much the
value typically swings in a year), max drawdown (the worst peak-to-trough drop
in the period) and the period return, and note the period is short. Mention the
listed considerations as things to learn about. Describe trade-offs; do not
tell the user to buy or sell anything.
"""


async def analyze_portfolio(holdings: list[dict]) -> dict:
    """Price holdings and compute analytics; shared by the agent and the Portfolio tab."""
    df = portfolio_math.load_holdings(pd.DataFrame(holdings))
    quotes = await asyncio.to_thread(get_quotes, df["symbol"].tolist())
    prices = {s: q["price"] for s, q in quotes.items()}
    analysis = portfolio_math.analyze(df, prices)
    weights = {h["symbol"]: h["weight"] for h in analysis["holdings"]}
    histories = await asyncio.to_thread(get_histories, list(weights))
    bars = {s: h["bars"] for s, h in histories.items()}
    analysis["risk"] = portfolio_math.risk_metrics(weights, bars)
    if analysis["risk"]:
        analysis["risk"]["sample_data"] = any(h["source"] == "mock" for h in histories.values())
        if analysis["risk"]["volatility"] >= 0.18:
            analysis["considerations"].append(
                f"A volatility of {analysis['risk']['volatility']:.0%} means a typical year "
                "can swing that much either way: worth comparing with how soon you need the money."
            )
    return analysis


async def run(state: FinnieState) -> dict:
    holdings = state.get("portfolio")
    if not holdings:
        return {
            "answer": "Upload a portfolio CSV in the **Portfolio** tab "
            "(columns: symbol, shares, optional asset_class) and ask again.",
            "data": {},
        }
    analysis = await analyze_portfolio(holdings)
    # The chart series is for the UI; the LLM only needs the summary numbers.
    risk = {k: v for k, v in analysis["risk"].items() if k != "growth_of_100"}
    context = json.dumps({**analysis, "risk": risk}, indent=2)
    answer = await explain(SYSTEM_PROMPT, state["query"], context)
    return {"answer": answer, "sources": ["Portfolio analysis"], "data": {"portfolio": analysis}}
