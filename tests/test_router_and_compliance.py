import asyncio

import pytest

from src.agents.compliance import DISCLAIMER
from src.agents.compliance import run as compliance_run
from src.agents.router import keyword_route
from src.rag.retriever import relevant


@pytest.mark.parametrize(
    ("query", "route"),
    [
        ("Explain Roth IRA vs 401k", "tax"),
        ("Is my portfolio diversified?", "portfolio"),
        ("How much to save for retirement?", "goals"),
        ("Any market news today?", "news"),
        ("What's the stock price of $AAPL", "market"),
        ("What is an index fund?", "finance_qa"),
    ],
)
def test_keyword_route(query, route):
    assert keyword_route(query) == route


def test_compliance_adds_disclaimer_and_softens_advice():
    out = asyncio.run(compliance_run({"answer": "You should buy index funds."}))
    assert DISCLAIMER in out["answer"]
    assert "You should buy" not in out["answer"]


def test_compliance_keeps_bare_answer_in_history():
    out = asyncio.run(compliance_run({"answer": "Index funds hold many companies."}))
    assert out["messages"][0].content == "Index funds hold many companies."


def test_relevant_drops_hits_below_threshold():
    hits = [
        {"title": "on", "score": 0.62},
        {"title": "edge", "score": 0.45},
        {"title": "off", "score": 0.3},
    ]
    assert [h["title"] for h in relevant(hits, min_score=0.45)] == ["on", "edge"]
    assert relevant([], min_score=0.45) == []
