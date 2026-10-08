"""LLM factory — the only place a chat model is constructed."""

from functools import lru_cache

from langchain_openai import ChatOpenAI

from src.config import settings


@lru_cache(maxsize=4)
def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    """Return a shared chat model so agents don't each open their own client."""
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Copy .env.example to .env.")
    return ChatOpenAI(
        model=settings.openai_model,
        temperature=temperature,
        api_key=settings.openai_api_key,
    )
