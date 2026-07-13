"""Web search integration — fetches external information via SearXNG."""

import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class WebSearcher:
    """Lightweight web search via configurable backend.

    Defaults to local SearXNG instance. Falls back gracefully
    if no search backend is configured.

    Usage:
        searcher = WebSearcher(search_url="http://localhost:8080")
        results = searcher.search("latest LLM benchmarks")
        context = searcher.format_for_prompt(results)
    """

    def __init__(self, search_url: Optional[str] = None, api_key: Optional[str] = None):
        self.search_url = search_url
        self.api_key = api_key

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        """Search the web. Returns list of {title, url, snippet} dicts."""
        if not self.search_url:
            logger.info("No search backend configured — returning empty results")
            return []

        try:
            return self._search_searxng(query, max_results)
        except Exception as e:
            logger.warning("Web search failed: %s", e)
            return []

    def _search_searxng(self, query: str, max_results: int) -> list[dict]:
        resp = requests.get(
            f"{self.search_url}/search",
            params={"q": query, "format": "json", "language": "en"},
            timeout=10,
        )
        if resp.status_code != 200:
            raise Exception(f"SearXNG returned {resp.status_code}")

        data = resp.json()
        results = []
        for r in data.get("results", [])[:max_results]:
            results.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content", ""),
            })
        return results

    @staticmethod
    def format_for_prompt(results: list[dict]) -> str:
        """Format search results as context for an LLM prompt."""
        if not results:
            return "No search results available."
        parts = []
        for r in results:
            parts.append(f"- {r['title']}: {r['snippet']} ({r['url']})")
        return "\n".join(parts)
