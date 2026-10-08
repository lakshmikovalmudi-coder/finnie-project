import asyncio

import pytest

from src.agents.compliance import DISCLAIMER
from src.agents.compliance import run as compliance_run
from src.agents.router import keyword_route


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
