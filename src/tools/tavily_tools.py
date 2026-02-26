"""Tavily search helpers for LangChain docs and support lookups."""

from __future__ import annotations

import logging
import os
from typing import Any, Iterable

import requests

logger = logging.getLogger(__name__)

TAVILY_API_URL = os.getenv("TAVILY_API_URL", "https://api.tavily.com/search")
DEFAULT_SEARCH_DEPTH = os.getenv("TAVILY_SEARCH_DEPTH", "basic")
DEFAULT_MAX_RESULTS = int(os.getenv("TAVILY_MAX_RESULTS", "5"))
MAX_RESULTS_LIMIT = 20


def _get_api_key() -> str:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY not found in environment")
    return api_key


def _normalize_domains(domains: Iterable[str] | None) -> list[str]:
    if not domains:
        return []
    normalized = []
    for domain in domains:
        if not domain:
            continue
        normalized.append(domain.strip())
    return [d for d in normalized if d]


def tavily_search(
    query: str,
    *,
    max_results: int | None = None,
    include_domains: Iterable[str] | None = None,
    search_depth: str | None = None,
    include_raw_content: bool = False,
) -> list[dict[str, Any]]:
    """Execute a Tavily Search API request and return result objects."""
    if not query or not query.strip():
        return []

    api_key = _get_api_key()
    results_limit = max(1, min(int(max_results or DEFAULT_MAX_RESULTS), MAX_RESULTS_LIMIT))
    domains = _normalize_domains(include_domains)

    payload: dict[str, Any] = {
        "query": query.strip(),
        "max_results": results_limit,
        "search_depth": (search_depth or DEFAULT_SEARCH_DEPTH),
        "include_answer": False,
        "include_raw_content": include_raw_content,
    }

    if domains:
        payload["include_domains"] = domains

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    response = requests.post(TAVILY_API_URL, json=payload, headers=headers, timeout=30)
    response.raise_for_status()

    data = response.json()
    results = data.get("results", []) if isinstance(data, dict) else []
    if not isinstance(results, list):
        logger.warning("Unexpected Tavily response shape: %s", type(results))
        return []

    return results
