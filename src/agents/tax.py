"""Tax Education agent: explains account types and rules without giving tax advice."""

from src.agents.base import explain
from src.rag.retriever import asearch, format_context
from src.state import FinnieState

SYSTEM_PROMPT = """\
You explain US tax-advantaged accounts (401(k), traditional IRA, Roth IRA/401(k),
HSA) and basic tax concepts. Compare options in a small table when useful.
Contribution limits change yearly: point to IRS.gov rather than stating a
figure not in the context. Always suggest a tax professional for the user's
specific situation. End every paragraph with the context block(s) it draws on,
e.g. [1] or [2][3].
"""


async def run(state: FinnieState) -> dict:
    hits = await asearch(f"tax {state['query']}")
    answer = await explain(SYSTEM_PROMPT, state["query"], format_context(hits))
    return {"answer": answer, "sources": list(dict.fromkeys(h["title"] for h in hits))}
