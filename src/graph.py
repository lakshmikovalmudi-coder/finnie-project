"""LangGraph wiring: router → one specialist → compliance → END, with per-thread memory."""

import functools
import logging
from collections.abc import Awaitable, Callable
from uuid import uuid4

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from src.agents import compliance, finance_qa, goals, market, news, portfolio, router, tax
from src.state import FinnieState, Route

logger = logging.getLogger(__name__)

Node = Callable[[FinnieState], Awaitable[dict]]

AGENT_NODES: dict[Route, Node] = {
    "finance_qa": finance_qa.run,
    "portfolio": portfolio.run,
    "market": market.run,
    "goals": goals.run,
    "news": news.run,
    "tax": tax.run,
}


def _safe(name: str, node: Node) -> Node:
    """Turn a specialist crash into a friendly answer so the UI never shows a traceback."""

    @functools.wraps(node)
    async def wrapper(state: FinnieState) -> dict:
        try:
            return await node(state)
        except Exception as exc:
            logger.exception("Agent %s failed", name)
            return {
                "answer": "Sorry — I hit a problem answering that. Please try rephrasing.",
                "error": f"{name}: {exc}",
            }

    return wrapper


def build_graph(checkpointer: BaseCheckpointSaver | None = None):
    graph = StateGraph(FinnieState)
    graph.add_node("router", router.run)
    for name, node in AGENT_NODES.items():
        graph.add_node(name, _safe(name, node))
        graph.add_edge(name, "compliance")
    graph.add_node("compliance", compliance.run)

    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", lambda s: s["route"], {r: r for r in AGENT_NODES})
    graph.add_edge("compliance", END)
    return graph.compile(checkpointer=checkpointer)


@functools.lru_cache(maxsize=1)
def get_graph():
    # The checkpointer keeps each thread's messages between turns, in process memory.
    return build_graph(checkpointer=InMemorySaver())


async def run_query(
    query: str, portfolio_rows: list[dict] | None = None, thread_id: str | None = None
) -> FinnieState:
    """Run one user turn. Turns that share a thread_id share conversation history;
    without one, the turn starts a fresh conversation."""
    config = {"configurable": {"thread_id": thread_id or uuid4().hex}}
    return await get_graph().ainvoke(
        {
            "query": query,
            "messages": [HumanMessage(content=query)],
            "portfolio": portfolio_rows,
            "sources": [],
            "data": {},
            "error": None,
        },
        config,
    )
