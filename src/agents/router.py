"""Router: classify the latest user message into one specialist route, using the
conversation so follow-ups ("what about Roth?") are rewritten to stand alone."""

import logging
import re

from pydantic import BaseModel, Field

from src.config import settings
from src.llm import get_llm
from src.state import ROUTES, FinnieState, Route

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
Classify the user's finance question into exactly one route:
- finance_qa: general concepts (compound interest, ETFs, diversification, what is X)
- portfolio: the user's own holdings, allocation, or uploaded portfolio
- market: live prices, quotes, or how a specific ticker/index is doing
- goals: saving toward a target amount, retirement, or a date
- news: current events, headlines, what happened in the market today
- tax: 401(k), IRA, Roth, capital gains, tax-advantaged accounts

If the question names a tax-advantaged account (401(k), IRA, Roth, HSA) or a tax
topic, choose tax, even for "what is" or "what's the difference" questions.

Earlier messages are conversation context. Classify the LAST user message. Also
return standalone_query: that message rewritten so it makes sense on its own,
filling in what a follow-up refers to (e.g. after a question about Roth IRAs,
"what about the limits?" -> "What are the Roth IRA contribution limits?").
If it already stands alone, return it unchanged.
"""

HISTORY_MESSAGES = 6  # the last three exchanges are enough to resolve a follow-up

# Checked in order; first match wins. Used when the LLM is unavailable.
KEYWORD_ROUTES: list[tuple[Route, re.Pattern[str]]] = [
    ("tax", re.compile(r"\b(401\(?k\)?|ira|roth|tax(es)?|capital gains?|hsa)\b", re.I)),
    ("portfolio", re.compile(r"\b(my (portfolio|holdings|stocks)|allocation|rebalanc\w*)\b", re.I)),
    ("goals", re.compile(r"\b(save|saving|retire\w*|goal|target|by age|down payment)\b", re.I)),
    ("news", re.compile(r"\b(news|headlines?|today|this week|latest)\b", re.I)),
    ("market", re.compile(r"\b(price|quote|ticker|trading at|stock price)\b|\$[A-Z]{1,5}\b", re.I)),
]


class RouteDecision(BaseModel):
    route: Route
    confidence: float = Field(ge=0.0, le=1.0)
    standalone_query: str = Field("", description="Last user message, rewritten to stand alone")


def keyword_route(query: str) -> Route:
    for route, pattern in KEYWORD_ROUTES:
        if pattern.search(query):
            return route
    return "finance_qa"


async def run(state: FinnieState) -> dict:
    query = state["query"]
    conversation = state.get("messages", [])[-HISTORY_MESSAGES:] or [("human", query)]
    try:
        classifier = get_llm(temperature=0).with_structured_output(RouteDecision)
        decision: RouteDecision = await classifier.ainvoke(
            [("system", SYSTEM_PROMPT), *conversation]
        )
        route, confidence = decision.route, decision.confidence
        if len(conversation) > 1:  # only a follow-up needs rewriting; keep first questions verbatim
            query = decision.standalone_query.strip() or query
    except Exception as exc:  # LLM down / no key: degrade, don't crash
        logger.warning("Router LLM failed, using keyword fallback: %s", exc)
        route, confidence = keyword_route(query), 0.0

    if route not in ROUTES or (0 < confidence < settings.router_min_confidence):
        route = "finance_qa"
    # Specialists read `query`, so they answer the rewritten, self-contained question.
    return {"route": route, "route_confidence": confidence, "query": query}
