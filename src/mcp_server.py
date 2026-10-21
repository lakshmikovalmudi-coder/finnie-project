"""MCP server: exposes Finnie's agents and tools to Claude Desktop over stdio.

Run with the project venv: `python src/mcp_server.py` (README shows the Claude Desktop
config). Stdout carries the MCP protocol, so logging goes to stderr.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

# Claude Desktop launches this file directly, so the project root may not be on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp.server.mcpserver import MCPServer  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from src.agents.portfolio import analyze_portfolio as _analyze_portfolio  # noqa: E402
from src.graph import run_query  # noqa: E402
from src.tools.goal_math import monthly_savings_needed  # noqa: E402
from src.tools.market_data import get_history, get_quote  # noqa: E402
from src.tools.news_search import search_news  # noqa: E402
from src.tools.portfolio_math import risk_metrics  # noqa: E402

logger = logging.getLogger(__name__)

mcp = MCPServer(
    "finnie",
    instructions=(
        "Finnie is a finance tutor for beginners. Its tools return education, not "
        "personalized financial, investment, or tax advice; keep that framing in answers."
    ),
)


class Holding(BaseModel):
    symbol: str = Field(description="Ticker, e.g. VTI")
    shares: float = Field(ge=0, description="Number of shares held")
    asset_class: str = Field("equity", description="equity, bond, cash, …")


@mcp.tool()
async def ask_finnie(question: str, conversation_id: str | None = None) -> str:
    """Ask Finnie's multi-agent tutor a finance question. A router picks the specialist
    (concepts, markets, goals, news, tax) and the answer carries sources and a disclaimer.
    Pass the same conversation_id on later calls so follow-up questions keep context."""
    state = await run_query(question, thread_id=conversation_id)
    sources = "\n".join(f"- {s}" for s in state.get("sources", []))
    footer = f"\n\nSources:\n{sources}" if sources else ""
    return f"[{state.get('route')} agent]\n{state['answer']}{footer}"


@mcp.tool()
async def get_stock_quote(symbol: str) -> dict[str, Any]:
    """Latest quote for a stock or ETF, plus 100-day return, volatility and max drawdown.
    `source` is "live" (Alpha Vantage, cached) or "mock" (sample data)."""
    quote, history = await asyncio.gather(
        asyncio.to_thread(get_quote, symbol), asyncio.to_thread(get_history, symbol)
    )
    if not quote:
        return {"error": f"No market data for {symbol.upper()}"}
    if history:
        stats = risk_metrics({quote["symbol"]: 1.0}, {quote["symbol"]: history["bars"]})
        stats.pop("growth_of_100", None)
        quote["history"] = stats
    return quote


@mcp.tool()
async def analyze_portfolio(holdings: list[Holding]) -> dict[str, Any]:
    """Diversification and risk review of holdings: weights, concentration (HHI), asset mix,
    volatility, max drawdown, warning flags and educational things to consider."""
    analysis = await _analyze_portfolio([h.model_dump() for h in holdings])
    analysis.get("risk", {}).pop("growth_of_100", None)
    return analysis


@mcp.tool()
def plan_savings_goal(
    target: float, years: float, annual_return: float = 0.06, current_savings: float = 0.0
) -> dict[str, Any]:
    """Monthly saving needed to reach `target` dollars in `years`, at a constant
    `annual_return` (decimal, 0.06 = 6%) compounded monthly."""
    return monthly_savings_needed(target, years, annual_return, current_savings).to_dict()


@mcp.tool()
async def search_knowledge_base(query: str, k: int = 4) -> list[dict[str, Any]]:
    """Search Finnie's curated finance articles (IRS, CFPB and more). Returns the best
    matching passages with title and a 0–1 relevance score; below ~0.45 is off-topic."""
    from src.rag.retriever import asearch  # lazy: loads the embedding model on first use

    return await asearch(query, k)


@mcp.tool()
async def get_financial_news(topic: str = "stock market") -> list[dict[str, Any]]:
    """Recent financial news on a topic, each with title, url and snippet.
    `source` is "mock" when no news API key is configured."""
    return await asyncio.to_thread(search_news, f"finance {topic}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    mcp.run("stdio")


if __name__ == "__main__":
    main()
