"""
Serper.dev web search tool for use with swarms Agent.

Wraps the Serper Google Search API as a simple callable that can be
passed to Agent(tools=[serper_search]).

Supports multiple API keys for round-robin rotation:
  - SERPER_API_KEYS: Comma-separated list of keys (preferred)
  - SERPER_API_KEY: Single key fallback
"""

import itertools
import json
import os
import threading

import httpx

# Module-level key rotation pool (initialized on first call)
_serper_key_cycle = None
_serper_key_lock = threading.Lock()
_serper_initialized = False


def _get_serper_key() -> str:
    """Get the next Serper API key from the round-robin pool."""
    global _serper_key_cycle, _serper_initialized

    if not _serper_initialized:
        with _serper_key_lock:
            if not _serper_initialized:
                keys = []
                env_keys = os.environ.get("SERPER_API_KEYS", "")
                if env_keys:
                    keys = [k.strip() for k in env_keys.split(",") if k.strip()]
                if not keys:
                    single = os.environ.get("SERPER_API_KEY", "")
                    if single:
                        keys = [single]
                if keys:
                    _serper_key_cycle = itertools.cycle(keys)
                _serper_initialized = True

    if _serper_key_cycle is None:
        return ""
    with _serper_key_lock:
        return next(_serper_key_cycle)


def serper_search(
    query: str,
    num_results: int = 10,
    search_type: str = "search",
) -> str:
    """
    Search the web using Serper.dev Google Search API.

    Args:
        query: The search query string.
        num_results: Number of results to return (default 10, max 100).
        search_type: Type of search - 'search', 'news', 'images', or 'places'.

    Returns:
        str: JSON string containing search results with titles, links, and snippets.
    """
    api_key = _get_serper_key()
    if not api_key:
        return json.dumps({"error": "Set SERPER_API_KEYS or SERPER_API_KEY env var"})

    url = f"https://google.serper.dev/{search_type}"
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "q": query,
        "num": num_results,
    }

    try:
        response = httpx.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()

        # Extract the most useful parts for the agent
        results = []
        for item in data.get("organic", []):
            results.append({
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", ""),
            })

        # Include knowledge graph if present
        kg = data.get("knowledgeGraph")

        # Include answer box if present
        answer_box = data.get("answerBox")

        output = {"results": results}
        if kg:
            output["knowledgeGraph"] = kg
        if answer_box:
            output["answerBox"] = answer_box

        return json.dumps(output, indent=2)

    except httpx.HTTPStatusError as e:
        return json.dumps({"error": f"HTTP {e.response.status_code}: {e.response.text}"})
    except Exception as e:
        return json.dumps({"error": str(e)})
