"""Shared helpers for specialist agents."""

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from src.llm import get_llm
from src.rag.retriever import format_context

logger = logging.getLogger(__name__)

EDUCATION_RULES = """\
You are Finnie, a patient finance tutor for beginners.
- Educate; do not give personalized financial, investment, or tax advice.
- Never say "you should buy/sell" or promise returns. Explain trade-offs instead.
- Use plain language, define jargon the first time, keep it under ~250 words.
- Use ONLY the numbers and facts in the provided context. If the context is
  empty or insufficient, say so plainly instead of guessing.
"""


async def explain(system_prompt: str, query: str, context: str) -> str:
    """Ask the LLM to answer `query` grounded in `context`."""
    messages = [
        SystemMessage(content=f"{EDUCATION_RULES}\n{system_prompt}"),
        HumanMessage(content=f"Context:\n{context or '(none)'}\n\nQuestion: {query}"),
    ]
    response = await get_llm().ainvoke(messages)
    return str(response.content)


NO_SOURCE_NOTE = (
    "_I don't have an article on this in my knowledge base, so this is a general "
    "explanation without sources._"
)

NO_SOURCE_PROMPT = """\
No knowledge-base article matches this question, so this overrides the
citation rule: give a brief, general explanation of well-established concepts
only, with no [n] citations and no figures that change over time (contribution
limits, tax rates, brackets, prices).
"""


async def answer_from_knowledge_base(system_prompt: str, query: str, hits: list[dict]) -> dict:
    """Answer from retrieved chunks, or label the answer as unsourced when none matched."""
    if not hits:
        answer = await explain(f"{system_prompt}\n{NO_SOURCE_PROMPT}", query, "")
        return {
            "answer": f"{NO_SOURCE_NOTE}\n\n{answer}",
            "sources": [],
            "data": {"grounded": False},
        }
    answer = await explain(system_prompt, query, format_context(hits))
    sources = list(dict.fromkeys(h["title"] for h in hits))
    return {"answer": answer, "sources": sources, "data": {"grounded": True}}
