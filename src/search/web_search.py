"""Web search integration — fetches external information when the query is beyond the source of truth."""

import json
import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class WebSearcher:
    """Lightweight web search via configurable backend.

    Falls back gracefully if no search backend is configured.
    """

    def __init__(self, search_url: Optional[str] = None, api_key: Optional[str] = None):
        self.search_url = search_url
        self.api_key = api_key

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        """Search the web for a query.

        Returns list of {title, url, snippet} dicts.
        Falls back to a simple response if no backend configured.
        """
        if not self.search_url:
            logger.info("No search backend configured — returning empty results")
            return [{"title": "(search unavailable)", "url": "", "snippet": "No web search backend configured. Configure SEARCH_URL in .env"}]

        try:
            return self._search_searxng(query, max_results)
        except Exception as e:
            logger.warning(f"Web search failed: {e}")
            return [{"title": "(search error)", "url": "", "snippet": f"Search error: {e}"}]

    def _search_searxng(self, query: str, max_results: int) -> list[dict]:
        """Search via local SearXNG instance."""
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

    def format_for_prompt(self, results: list[dict]) -> str:
        """Format search results as context for an LLM prompt."""
        if not results:
            return "No search results available."
        parts = []
        for r in results:
            parts.append(f"- {r['title']}: {r['snippet']} ({r['url']})")
        return "\n".join(parts)


_searcher: Optional[WebSearcher] = None


def get_searcher(search_url: Optional[str] = None, api_key: Optional[str] = None) -> WebSearcher:
    global _searcher
    if _searcher is None:
        _searcher = WebSearcher(search_url=search_url, api_key=api_key)
    return _searcher
