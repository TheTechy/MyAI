"""
myai_skills/wikipedia_skill.py
==============================
Wikipedia skill — instant encyclopaedia lookups, no API key required.

API used
--------
  Wikipedia REST API (summary endpoint)
  https://en.wikipedia.org/api/rest_v1/page/summary/{title}

  - No authentication needed
  - Returns a clean plain-text extract (no markup to strip)
  - Also returns the canonical page URL for attribution
  - Rate limit: polite use only — one request per query is fine

Flow
----
  1. Extract the search subject from the user query
  2. Attempt a direct REST summary lookup (fast path)
  3. If ambiguous or not found, fall back to the OpenSearch API
     to find the best matching article title, then look that up
  4. Return a structured plain-text context for the LLM

The LLM uses this context to give a conversational answer —
it should cite the topic and can mention the Wikipedia URL.
"""

from __future__ import annotations

import re
import urllib.parse

import requests

from .base import BaseSkill

# ── Constants ──────────────────────────────────────────────────────────────────

_USER_AGENT  = "MyAI-WikipediaSkill/1.0"
_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"
_SEARCH_URL  = "https://en.wikipedia.org/w/api.php"
_TIMEOUT     = 8   # seconds
_MAX_CHARS   = 1800  # cap extract length to protect context window

# Phrases to strip from the front of the query before searching
_QUERY_STRIP = re.compile(
    r'^(what\s+is\s+(a\s+|an\s+|the\s+)?|'
    r'who\s+(is|was)\s+(a\s+|an\s+|the\s+)?|'
    r'tell\s+me\s+about\s+(a\s+|an\s+|the\s+)?|'
    r'explain\s+(a\s+|an\s+|the\s+)?|'
    r'describe\s+(a\s+|an\s+|the\s+)?|'
    r'define\s+(a\s+|an\s+|the\s+)?|'
    r'what\s+are\s+(a\s+|an\s+|the\s+)?|'
    r'wikipedia\s+(article\s+)?(on|about|for)\s+|'
    r'look\s+up\s+)',
    re.IGNORECASE,
)

# Trailing noise to strip
_QUERY_TAIL = re.compile(
    r'\s*(on\s+wikipedia|\?+|please|thanks|thank\s+you)$',
    re.IGNORECASE,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _extract_subject(query: str) -> str:
    """Strip question preamble and return the core search subject."""
    subject = _QUERY_STRIP.sub("", query.strip())
    subject = _QUERY_TAIL.sub("", subject).strip()
    return subject or query.strip()


def _fetch_summary(title: str) -> dict | None:
    """
    Fetch the Wikipedia REST summary for a given title.
    Returns the JSON dict on success, None if not found (404).
    Raises requests.RequestException on network error.
    """
    encoded = urllib.parse.quote(title.replace(" ", "_"), safe="")
    url     = _SUMMARY_URL.format(encoded)
    r = requests.get(
        url,
        headers={"User-Agent": _USER_AGENT},
        timeout=_TIMEOUT,
        allow_redirects=True,
    )
    if r.status_code == 404:
        return None
    r.raise_for_status()
    data = r.json()
    # Disambiguation pages are not useful — treat as not found
    if data.get("type") == "disambiguation":
        return None
    return data


def _opensearch(query: str) -> str | None:
    """
    Use Wikipedia's OpenSearch API to find the best matching article title.
    Returns the first result title, or None if nothing found.
    """
    r = requests.get(
        _SEARCH_URL,
        params={
            "action":    "opensearch",
            "search":    query,
            "limit":     3,
            "namespace": 0,
            "format":    "json",
        },
        headers={"User-Agent": _USER_AGENT},
        timeout=_TIMEOUT,
    )
    r.raise_for_status()
    results = r.json()
    # OpenSearch returns [query, [titles], [descriptions], [urls]]
    titles = results[1] if len(results) > 1 else []
    return titles[0] if titles else None


def _build_context(data: dict) -> str:
    """Format the Wikipedia API response into plain-text LLM context."""
    title    = data.get("title", "Unknown")
    extract  = data.get("extract", "No summary available.")
    page_url = data.get("content_urls", {}).get("desktop", {}).get("page", "")

    # Cap extract length to protect context window
    if len(extract) > _MAX_CHARS:
        # Trim at a sentence boundary if possible
        trimmed = extract[:_MAX_CHARS]
        last_dot = trimmed.rfind(". ")
        if last_dot > _MAX_CHARS // 2:
            trimmed = trimmed[: last_dot + 1]
        extract = trimmed + " [summary truncated]"

    lines = [
        f"Wikipedia article: {title}",
        f"URL: {page_url}" if page_url else "",
        "",
        extract,
    ]
    return "\n".join(l for l in lines if l is not None)


# ── Skill class ────────────────────────────────────────────────────────────────

class WikipediaSkill(BaseSkill):
    """
    Encyclopaedia lookup skill using the Wikipedia REST API.

    Returns a plain-text summary for the LLM to use when answering
    factual questions about people, places, concepts, and events.
    Falls back gracefully when an article is not found.
    """

    name        = "wikipedia"
    description = "Wikipedia article lookup — definitions, people, places, events"

    def execute(self, query: str) -> str:
        subject = _extract_subject(query)

        try:
            # ── Fast path: direct title lookup ─────────────────────────────
            data = _fetch_summary(subject)

            # ── Fallback: OpenSearch to find the right title ────────────────
            if data is None:
                best_title = _opensearch(subject)
                if best_title:
                    data = _fetch_summary(best_title)

            if data is None:
                return (
                    f"No Wikipedia article found for '{subject}'. "
                    "The topic may not have a Wikipedia page, or the spelling "
                    "may be different. Let the user know and suggest they check "
                    "Wikipedia directly if needed."
                )

            return _build_context(data)

        except requests.Timeout:
            return f"Wikipedia request timed out while looking up '{subject}'."
        except requests.RequestException as e:
            return f"Wikipedia lookup failed: {e}"
