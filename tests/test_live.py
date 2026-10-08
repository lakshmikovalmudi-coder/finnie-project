"""Live smoke tests against real OpenAI / Alpha Vantage / news APIs.

Skipped by default (they cost quota). Run with:  pytest -m live -s
`-s` prints each query, route, and answer so a human can judge answer quality.
Budget per run: ~20 gpt-4o-mini calls, up to 5 Alpha Vantage calls (AAPL, NVDA,
MSFT, TSLA, SPY; cached 30 min), ~2 news calls.
"""

import re

import pandas as pd
import pytest

from src.agents import market
from src.agents.compliance import ADVICE_PATTERNS, DISCLAIMER
from src.aio import run_sync
from src.config import DATA_DIR, settings
from src.graph import run_query
from src.tools.goal_math import monthly_savings_needed

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not settings.openai_api_key, reason="OPENAI_API_KEY not set in .env"),
]

live_data = pytest.mark.skipif(
    settings.use_mock_data, reason="USE_MOCK_DATA=true in .env; set it to false for live data"
)
needs_alpha_vantage = pytest.mark.skipif(
    not settings.alpha_vantage_api_key, reason="ALPHA_VANTAGE_API_KEY not set in .env"
)
needs_news = pytest.mark.skipif(
    not (settings.tavily_api_key or settings.serpapi_api_key),
    reason="no TAVILY_API_KEY / SERPAPI_API_KEY in .env",
)


def ask(query: str, portfolio: list[dict] | None = None) -> dict:
    state = run_sync(run_query(query, portfolio))
    print(f"\n=== {query}\nroute={state['route']} ({state['route_confidence']:.2f})")
    print(f"sources={state['sources']}\n{state['answer']}")
    assert state["error"] is None, state["error"]
    assert DISCLAIMER in state["answer"]
    return state


@pytest.mark.parametrize(
    ("query", "acceptable"),
    [
        ("What is compound interest?", {"finance_qa"}),
        ("What's the difference between a Roth IRA and a traditional IRA?", {"tax"}),
        ("How is Apple stock doing right now?", {"market"}),
        ("I want $20,000 for a house down payment in 4 years", {"goals"}),
        ("What happened in the stock market today?", {"news", "market"}),
        # Ambiguous: more than one route is defensible.
        ("Is Tesla a good stock?", {"market", "finance_qa"}),
        ("Should I put my bonus in my Roth?", {"tax", "goals"}),
    ],
)
def test_llm_routing(query, acceptable):
    state = ask(query)
    assert state["route"] in acceptable
    assert state["route_confidence"] > 0, "router fell back to keywords — LLM call failed"


def test_finance_qa_cites_knowledge_base():
    state = ask("Explain diversification simply")
    assert state["sources"]
    assert re.search(r"\[\d\]", state["answer"]), "answer doesn't cite context blocks"


def test_goals_answer_quotes_python_numbers():
    state = ask("How much should I save monthly to reach $50,000 in 5 years at 7%?")
    expected = monthly_savings_needed(50_000, 5, 0.07).monthly_contribution
    assert state["data"]["goal"]["monthly_contribution"] == expected
    assert str(expected) in state["answer"].replace(",", ""), "LLM didn't quote the computed figure"


def test_advice_request_is_reframed_before_compliance():
    """Check the specialist's raw answer: compliance is a backstop, not the first line."""
    query = "Should I buy NVDA?"
    raw = run_sync(market.run({"query": query}))["answer"]
    print(f"\n=== {query} (raw, pre-compliance)\n{raw}")
    for pattern, _ in ADVICE_PATTERNS:
        assert not pattern.search(raw), f"advice-like phrasing: {pattern.pattern}"


@live_data
@needs_alpha_vantage
def test_market_uses_live_quotes():
    state = ask("What's the price of MSFT?")
    assert state["data"]["quotes"]["MSFT"]["source"] == "live", "fell back to mock data"


@live_data
@needs_alpha_vantage
def test_portfolio_prices_live():
    holdings = pd.read_csv(DATA_DIR / "sample_portfolio.csv").head(2).to_dict("records")
    state = ask("Is my portfolio diversified?", holdings)
    analysis = state["data"]["portfolio"]
    assert analysis["unpriced"] == []
    assert analysis["total_value"] > 0


@live_data
@needs_news
def test_news_is_live_and_linked():
    state = ask("Latest news on interest rates")
    assert all(a["source"] == "live" for a in state["data"]["articles"]), "fell back to mock"
    assert all("http" in s for s in state["sources"])
