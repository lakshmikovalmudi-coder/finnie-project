"""News Synthesizer agent: summarizes current financial news with source links."""

import asyncio

from src.agents.base import explain
from src.state import FinnieState
from src.tools.news_search import search_news

SYSTEM_PROMPT = """\
Summarize the articles for a beginner in 3-5 bullets. Each bullet must cite
its article as [n]. Explain any jargon. If articles are marked "mock", say
these are sample headlines, not live news.
"""


async def run(state: FinnieState) -> dict:
    articles = await asyncio.to_thread(search_news, f"finance {state['query']}")
    context = "\n\n".join(
        f"[{n}] ({a['source']}) {a['title']}\n{a['snippet']}" for n, a in enumerate(articles, 1)
    )
    answer = await explain(SYSTEM_PROMPT, state["query"], context)
    sources = [f"[{n}] {a['title']} — {a['url']}" for n, a in enumerate(articles, 1)]
    return {"answer": answer, "sources": sources, "data": {"articles": articles}}
