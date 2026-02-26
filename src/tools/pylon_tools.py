"""Support article search and retrieval via Tavily."""

from __future__ import annotations

import json
import logging
from typing import Any

import requests
from langchain.tools import tool

from src.tools.tavily_tools import tavily_search

logger = logging.getLogger(__name__)

DEFAULT_MAX_RESULTS = 5
MAX_RESULTS_LIMIT = 10
SUPPORT_DOMAIN = "support.langchain.com"
USER_AGENT = "Chat-LangChain-SupportFetcher/1.0"


def _format_support_results(results: list[dict[str, Any]], query: str) -> str:
    if not results:
        return json.dumps(
            {"query": query, "total": 0, "articles": [], "note": "No results found."},
            indent=2,
        )

    articles = []
    for result in results:
        url = result.get("url") or ""
        title = result.get("title") or "Untitled"
        snippet = result.get("content") or ""

        if not url:
            continue

        articles.append(
            {
                "id": url,
                "title": title,
                "url": url,
                "snippet": snippet,
                "source": SUPPORT_DOMAIN,
            }
        )

    return json.dumps(
        {
            "query": query,
            "total": len(articles),
            "articles": articles,
            "note": "Article IDs are URLs. Use get_article_content to fetch full HTML when needed.",
        },
        indent=2,
    )


@tool
def search_support_articles(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> str:
    """Search the public LangChain support site using Tavily.

    Args:
        query: Search query string (error message, feature, or product name).
        max_results: Number of results to return (default: 5, max: 10).

    Returns:
        JSON string with article IDs (URLs), titles, snippets, and URLs.
    """
    try:
        limit = max(1, min(int(max_results), MAX_RESULTS_LIMIT))
        results = tavily_search(
            query,
            max_results=limit,
            include_domains=[SUPPORT_DOMAIN],
        )
        return _format_support_results(results, query)
    except Exception as exc:
        logger.warning("Support search failed: %s", exc)
        return json.dumps({"error": str(exc), "query": query}, indent=2)


@tool
def get_article_content(article_id: str, max_chars: int = 5000) -> str:
    """Fetch the full HTML content for a support article by URL.

    Args:
        article_id: The article URL returned by search_support_articles.
        max_chars: Maximum characters of HTML to return (default: 5000).

    Returns:
        HTML content string with URL and title (if available).
    """
    url = (article_id or "").strip()
    if not url.startswith("http"):
        return "Error: article_id must be a support.langchain.com URL."

    try:
        response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
        response.raise_for_status()
        content = response.text or ""
        truncated = content[:max_chars]
        return f"URL: {url}\n\nContent:\n{truncated}"
    except Exception as exc:
        logger.warning("Failed to fetch support article %s: %s", url, exc)
        return f"Error fetching article: {exc}"
