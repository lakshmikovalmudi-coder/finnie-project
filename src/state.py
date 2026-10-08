"""Shared LangGraph state passed between the router and specialist agents."""

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

Route = Literal["finance_qa", "portfolio", "market", "goals", "news", "tax"]
ROUTES: tuple[Route, ...] = ("finance_qa", "portfolio", "market", "goals", "news", "tax")


class FinnieState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    query: str
    route: Route
    route_confidence: float
    portfolio: list[dict[str, Any]] | None
    answer: str
    sources: list[str]
    data: dict[str, Any]
    error: str | None
