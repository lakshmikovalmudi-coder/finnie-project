"""Financial news search via Tavily or SerpApi, cached, with a mock fallback."""

import json
import logging
from typing import Any

import requests

from src.config import settings
from src.tools.cache import ttl_cache

logger = logging.getLogger(__name__)

TIMEOUT_S = 15


@ttl_cache("tavily_news")
def _tavily(query: str, max_results: int) -> list[dict[str, Any]] | None:
    resp = requests.post(
        "https://api.tavily.com/search",
        headers={"Authorization": f"Bearer {settings.tavily_api_key}"},
        json={"query": query, "topic": "news", "max_results": max_results},
        timeout=TIMEOUT_S,
    )
    resp.raise_for_status()
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("content", ""),
            "published": r.get("published_date", ""),
        }
        for r in resp.json().get("results", [])
    ] or None


@ttl_cache("serpapi_news")
def _serpapi(query: str, max_results: int) -> list[dict[str, Any]] | None:
    resp = requests.get(
        "https://serpapi.com/search.json",
        params={"engine": "google_news", "q": query, "api_key": settings.serpapi_api_key},
        timeout=TIMEOUT_S,
    )
    resp.raise_for_status()
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("link", ""),
            "snippet": r.get("snippet", ""),
            "published": r.get("date", ""),
        }
        for r in resp.json().get("news_results", [])[:max_results]
    ] or None


def _mock_news() -> list[dict[str, Any]]:
    return json.loads((settings.mock_dir / "news.json").read_text())


def search_news(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Return news articles as {title, url, snippet, published, source}."""
    providers = {
        "tavily": (_tavily, settings.tavily_api_key),
        "serpapi": (_serpapi, settings.serpapi_api_key),
    }
    fetch, key = providers.get(settings.news_provider, providers["tavily"])
    if not settings.use_mock_data and key:
        try:
            results = fetch(query, max_results)
            if results:
                return [{**r, "source": "live"} for r in results]
        except requests.RequestException as exc:
            logger.warning("News search failed, using mock: %s", exc)
    return [{**r, "source": "mock"} for r in _mock_news()[:max_results]]
