"""End-to-end: run real queries through router → specialist → compliance, offline.

The LLM is replaced by a deterministic fake and external APIs are forced to mock
data, so these tests exercise the graph wiring, tools, RAG, and compliance
without API keys. Every test asserts `error is None`, because the graph's
`_safe` wrapper otherwise hides agent crashes behind a friendly answer.
"""

import asyncio
import dataclasses
import json

import pandas as pd
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.agents import base, goals, router
from src.agents.base import NO_SOURCE_NOTE
from src.agents.compliance import DISCLAIMER
from src.agents.goals import GoalParams
from src.agents.router import RouteDecision, keyword_route
from src.aio import run_sync
from src.config import DATA_DIR
from src.graph import AGENT_NODES, run_query
from src.tools import market_data, news_search
from src.tools.goal_math import monthly_savings_needed

SAMPLE_PORTFOLIO = pd.read_csv(DATA_DIR / "sample_portfolio.csv").to_dict("records")


def _text(message) -> str:
    return message[1] if isinstance(message, tuple) else str(message.content)


class FakeLLM:
    """Stands in for ChatOpenAI: echoes its prompt so tests can check grounding."""

    def __init__(self, goal_params: GoalParams, fail: bool = False):
        self.goal_params = goal_params
        self.fail = fail
        self.standalone_query = ""  # what the router "rewrites" a follow-up to
        self.router_calls: list[list] = []

    async def ainvoke(self, messages):
        if self.fail:
            raise RuntimeError("LLM unavailable")
        return AIMessage(content=f"FAKE ANSWER\n{_text(messages[-1])}")

    def with_structured_output(self, schema):
        llm = self

        class Structured:
            async def ainvoke(self, messages):
                if llm.fail:
                    raise RuntimeError("LLM unavailable")
                if schema is RouteDecision:
                    llm.router_calls.append(messages)
                    query = llm.standalone_query or _text(messages[-1])
                    return RouteDecision(
                        route=keyword_route(query),
                        confidence=0.9,
                        standalone_query=llm.standalone_query,
                    )
                if schema is GoalParams:
                    return llm.goal_params
                raise AssertionError(f"Unexpected structured schema {schema}")

        return Structured()


@pytest.fixture
def fake_llm(monkeypatch):
    llm = FakeLLM(GoalParams(target=50_000, years=5, annual_return=0.07))
    for module in (base, router, goals):
        monkeypatch.setattr(module, "get_llm", lambda temperature=0.2: llm)
    for module in (market_data, news_search):
        monkeypatch.setattr(
            module, "settings", dataclasses.replace(module.settings, use_mock_data=True)
        )
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")  # embedding model must come from local cache
    return llm


def ask(query: str, portfolio: list[dict] | None = None, thread_id: str | None = None) -> dict:
    state = run_sync(run_query(query, portfolio, thread_id))
    assert state["error"] is None, state["error"]
    assert DISCLAIMER in state["answer"]
    return state


@pytest.mark.parametrize(
    ("query", "route"),
    [
        ("What is compound interest?", "finance_qa"),
        ("Explain Roth IRA vs 401k", "tax"),
    ],
)
def test_rag_routes_answer_from_knowledge_base(fake_llm, query, route):
    state = ask(query)
    assert state["route"] == route
    assert state["sources"], "RAG returned no sources"
    assert "[1]" in state["answer"]  # retrieved context reached the LLM prompt


def test_off_topic_question_is_labelled_unsourced(fake_llm):
    state = ask("What is the best pizza topping?")
    assert state["route"] == "finance_qa"
    assert state["answer"].startswith(NO_SOURCE_NOTE)
    assert state["sources"] == []
    assert state["data"]["grounded"] is False


def test_follow_up_sees_earlier_turns_in_same_thread(fake_llm):
    ask("What is a Roth IRA?", thread_id="t-memory")
    fake_llm.standalone_query = "What are the Roth IRA contribution limits?"
    state = ask("What about the limits?", thread_id="t-memory")

    conversation = fake_llm.router_calls[-1][1:]  # drop the system prompt
    assert [type(m) for m in conversation] == [HumanMessage, AIMessage, HumanMessage]
    assert conversation[0].content == "What is a Roth IRA?"
    assert DISCLAIMER not in conversation[1].content  # history stores the bare answer
    # Specialists answer the rewritten question, so retrieval finds Roth limits.
    assert state["query"] == "What are the Roth IRA contribution limits?"
    assert state["route"] == "tax"


def test_first_question_is_never_rewritten(fake_llm):
    fake_llm.standalone_query = "Something the user didn't ask"
    state = ask("What is compound interest?", thread_id="t-first")
    assert state["query"] == "What is compound interest?"


def test_threads_do_not_share_history(fake_llm):
    ask("What is a Roth IRA?", thread_id="t-a")
    ask("What is compound interest?", thread_id="t-b")
    conversation = fake_llm.router_calls[-1][1:]
    assert [m.content for m in conversation] == ["What is compound interest?"]


def test_market_route_uses_mock_quotes(fake_llm):
    state = ask("What's the stock price of $AAPL")
    assert state["route"] == "market"
    assert state["data"]["quotes"]["AAPL"]["source"] == "mock"
    assert state["sources"] == ["Sample data (AAPL)"]


def test_news_route_cites_articles(fake_llm):
    state = ask("Any market news today?")
    assert state["route"] == "news"
    assert state["data"]["articles"]
    assert all("http" in s for s in state["sources"])


def test_goals_route_computes_numbers_in_python(fake_llm):
    state = ask("How much should I save monthly to reach $50,000 in 5 years at 7%?")
    expected = monthly_savings_needed(50_000, 5, 0.07)
    assert state["route"] == "goals"
    assert state["data"]["goal"]["monthly_contribution"] == expected.monthly_contribution
    assert str(expected.monthly_contribution) in state["answer"]
    assert len(state["data"]["projection"]) > 0


def test_goals_route_asks_for_missing_params(fake_llm):
    fake_llm.goal_params = GoalParams()
    state = ask("I want to save for a goal")
    assert state["route"] == "goals"
    assert "target amount" in state["answer"]


def test_portfolio_route_analyzes_uploaded_holdings(fake_llm):
    state = ask("Is my portfolio diversified?", SAMPLE_PORTFOLIO)
    analysis = state["data"]["portfolio"]
    assert state["route"] == "portfolio"
    assert analysis["unpriced"] == []
    assert analysis["total_value"] > 0
    assert analysis["flags"]  # sample is concentrated in single stocks
    assert analysis["considerations"]
    assert json.dumps(analysis["total_value"]) in state["answer"]
    risk = analysis["risk"]
    assert risk["sample_data"] is True
    assert set(risk["holding_volatility"]) == {h["symbol"] for h in analysis["holdings"]}
    assert '"growth_of_100"' not in state["answer"]  # chart series kept out of the prompt


def test_portfolio_route_without_upload_prompts_for_csv(fake_llm):
    state = ask("Is my portfolio diversified?")
    assert state["route"] == "portfolio"
    assert "Upload a portfolio CSV" in state["answer"]


def test_router_falls_back_to_keywords_when_llm_fails(fake_llm):
    fake_llm.fail = True
    state = run_sync(run_query("Explain Roth IRA vs 401k"))
    assert state["route"] == "tax"
    assert state["route_confidence"] == 0.0
    # The specialist's LLM call also fails; the graph must still return a safe answer.
    assert state["error"].startswith("tax:")
    assert DISCLAIMER in state["answer"]


def test_agent_crash_is_reported_not_raised(fake_llm, monkeypatch):
    async def boom(state):
        raise ValueError("kaboom")

    monkeypatch.setitem(AGENT_NODES, "market", boom)
    from src.graph import build_graph, get_graph

    get_graph.cache_clear()
    monkeypatch.setattr("src.graph.get_graph", build_graph)
    state = run_sync(run_query("What's the stock price of $AAPL"))
    get_graph.cache_clear()
    assert state["error"] == "market: kaboom"
    assert "Sorry" in state["answer"]
    assert DISCLAIMER in state["answer"]


def test_run_sync_reuses_one_event_loop():
    """Regression: asyncio.run per turn closed the loop the cached OpenAI client was bound to."""

    async def current_loop():
        return asyncio.get_running_loop()

    assert run_sync(current_loop()) is run_sync(current_loop())
