"""Finance Q&A agent: explains concepts grounded in the knowledge base."""

from src.agents.base import explain
from src.rag.retriever import asearch, format_context
from src.state import FinnieState

SYSTEM_PROMPT = """\
You explain personal-finance and investing concepts. Use a short example with
numbers when it helps.

Citations are required: end every paragraph with the context block(s) it
draws on, e.g. [1] or [2][3]. Even a short, simple answer cites at least once.
"""


async def run(state: FinnieState) -> dict:
    hits = await asearch(state["query"])
    answer = await explain(SYSTEM_PROMPT, state["query"], format_context(hits))
    return {"answer": answer, "sources": list(dict.fromkeys(h["title"] for h in hits))}
