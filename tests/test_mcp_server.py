"""MCP server: tools are registered and callable in-process (no Claude Desktop needed)."""

import dataclasses
import json

import pytest
from mcp import Client

from src import mcp_server
from src.tools import market_data
from src.tools.goal_math import monthly_savings_needed

EXPECTED_TOOLS = {
    "ask_finnie",
    "get_stock_quote",
    "analyze_portfolio",
    "plan_savings_goal",
    "search_knowledge_base",
    "get_financial_news",
}


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(
        market_data, "settings", dataclasses.replace(market_data.settings, use_mock_data=True)
    )


async def call(name: str, args: dict):
    async with Client(mcp_server.mcp) as client:
        result = await client.call_tool(name, args)
    assert not result.is_error, result.content
    return json.loads(result.content[0].text)


@pytest.mark.asyncio
async def test_all_tools_are_listed_with_descriptions():
    async with Client(mcp_server.mcp) as client:
        tools = (await client.list_tools()).tools
    assert {t.name for t in tools} == EXPECTED_TOOLS
    assert all(t.description for t in tools)


@pytest.mark.asyncio
async def test_plan_savings_goal_matches_calculator():
    plan = await call("plan_savings_goal", {"target": 50_000, "years": 5, "annual_return": 0.07})
    assert (
        plan["monthly_contribution"] == monthly_savings_needed(50_000, 5, 0.07).monthly_contribution
    )


@pytest.mark.asyncio
async def test_get_stock_quote_includes_history_stats():
    quote = await call("get_stock_quote", {"symbol": "vti"})
    assert quote["symbol"] == "VTI"
    assert quote["source"] == "mock"
    assert {"volatility", "max_drawdown", "period_return"} <= quote["history"].keys()
    assert "growth_of_100" not in quote["history"]


@pytest.mark.asyncio
async def test_get_stock_quote_unknown_symbol_returns_error_field():
    assert "error" in await call("get_stock_quote", {"symbol": "ZZZZ"})


@pytest.mark.asyncio
async def test_analyze_portfolio_validates_and_analyzes():
    result = await call(
        "analyze_portfolio",
        {"holdings": [{"symbol": "AAPL", "shares": 10}, {"symbol": "BND", "shares": 50}]},
    )
    assert result["total_value"] > 0
    assert result["risk"]["volatility"] > 0


@pytest.mark.asyncio
async def test_analyze_portfolio_rejects_negative_shares():
    async with Client(mcp_server.mcp) as client:
        result = await client.call_tool(
            "analyze_portfolio", {"holdings": [{"symbol": "AAPL", "shares": -1}]}
        )
    assert result.is_error
