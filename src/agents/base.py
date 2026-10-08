"""Shared helpers for specialist agents."""

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from src.llm import get_llm

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
