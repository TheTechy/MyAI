"""
skills/web_search.py
====================
Web search skill — DuckDuckGo + intelligent content extraction.

Enhanced scraping pipelinea
  1. DuckDuckGo → top 5 candidate URLs
  2. Score and filter URLs (drop low-quality sources)
  3. Fetch each page → extract main content zones
  4. Score paragraphs by keyword relevance
  5. Return top-ranked content, cleanly formatted

Dependencies
------------
    pip install duckduckgo-search beautifulsoup4 requests
"""

from __future__ import annotations

import os
import re
import time
from urllib.parse import urlparse

from .base import BaseSkill

# ── Soft imports ──────────────────────────────────────────────────────────────
try:
    from ddgs import DDGS
    _DDGS_AVAILABLE = True
except ImportError:
    _DDGS_AVAILABLE = False

# ── Search region ──────────────────────────────────────────────────────────────
# Loaded once at startup from .env
# Common region codes:
#   wt-wt  → worldwide (default)
#   uk-en  → United Kingdom
#   us-en  → United States
#   de-de  → Germany
#   fr-fr  → France
#   jp-jp  → Japan
#   cn-zh  → China
_SEARCH_REGION = os.getenv("SEARCH_REGION", "wt-wt").strip()

try:
    import requests
    from bs4 import BeautifulSoup, Tag
    _SCRAPE_AVAILABLE = True
except ImportError:
    _SCRAPE_AVAILABLE = False


# ── Constants ─────────────────────────────────────────────────────────────────
CANDIDATE_RESULTS  = 5      # fetch this many from DuckDuckGo
MAX_RESULTS        = 3      # return this many to the LLM
MAX_CHARS_PAGE     = 2_500  # max chars per page in final output
MAX_PARAGRAPHS     = 8      # max paragraphs extracted per page
MIN_PARA_LEN       = 40     # ignore paragraphs shorter than this
REQUEST_TIMEOUT    = 6
FETCH_HEADERS      = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# ── Tags that are pure noise ───────────────────────────────────────────────────
TAGS_TO_DROP = [
    "script", "style", "head", "nav", "footer", "header",
    "aside", "form", "iframe", "noscript", "svg", "img",
    "button", "input", "select", "textarea", "meta", "link",
    "figure", "figcaption", "picture", "video", "audio",
    "advertisement", "cookie", "popup",
]

# ── Content zone selectors — tried in order, first match wins ─────────────────
CONTENT_SELECTORS = [
    "article",
    "main",
    '[role="main"]',
    ".article-body",
    ".article-content",
    ".post-content",
    ".entry-content",
    ".story-body",
    ".content-body",
    ".article__body",
    "#article-body",
    "#main-content",
    "#content",
]

# ── Low-quality domain signals — skip these ────────────────────────────────────
LOW_QUALITY_DOMAINS = {
    "pinterest.com", "instagram.com", "facebook.com", "twitter.com",
    "tiktok.com", "youtube.com", "reddit.com", "quora.com",
    "amazon.com", "ebay.com", "etsy.com",
}

# ── High-quality domain signals — boost score ─────────────────────────────────
HIGH_QUALITY_PATTERNS = re.compile(
    r"\b(bbc\.|reuters\.|apnews\.|theguardian\.|nytimes\.|ft\.|"
    r"economist\.|nature\.|science\.|gov\.|ac\.uk|edu\.)\b",
    re.I
)


class WebSearchSkill(BaseSkill):
    """
    Handles queries requiring live / real-time information.
    Uses DuckDuckGo for URLs and intelligent BS4 extraction for content.
    """

    name        = "web_search"
    description = "Search the web for current information using DuckDuckGo"




    # ── Main execution ────────────────────────────────────────────────────────
    def execute(self, query: str) -> str:
        if not _DDGS_AVAILABLE:
            return "Web search unavailable: duckduckgo-search not installed."

        # Step 1 — get candidates from DuckDuckGo
        try:
            with DDGS() as ddgs:
                candidates = list(ddgs.text(
                    query,
                    region=_SEARCH_REGION,
                    max_results=CANDIDATE_RESULTS,
                ))
        except Exception as exc:
            return f"Search error: {exc}"

        if not candidates:
            return "No search results found."

        # Step 2 — score and filter candidates
        scored = self._score_results(candidates, query)
        top    = scored[:MAX_RESULTS]

        # Step 3 — fetch and extract content from each
        parts = []
        for result in top:
            title   = result.get("title", "No title")
            url     = result.get("href", "")
            snippet = result.get("body", "").strip()

            content = self._extract_content(url, query) if (_SCRAPE_AVAILABLE and url) else None
            text    = content if content else f"[Snippet only]\n{snippet}"

            parts.append(
                f"[Source] {title}\n"
                f"[URL] {url}\n"
                f"{text}"
            )

        return "\n\n─────\n\n".join(parts)

    # ── Result scoring ────────────────────────────────────────────────────────
    def _score_results(self, results: list[dict], query: str) -> list[dict]:
        """Score DuckDuckGo results by quality signals, return sorted list."""
        query_words = set(re.findall(r'\w+', query.lower()))
        scored = []

        for r in results:
            url     = r.get("href", "")
            title   = r.get("title", "")
            snippet = r.get("body", "")
            domain  = urlparse(url).netloc.lower().replace("www.", "")

            score = 1.0

            # Penalise low-quality domains
            if any(d in domain for d in LOW_QUALITY_DOMAINS):
                score -= 0.8

            # Boost high-quality sources
            if HIGH_QUALITY_PATTERNS.search(domain):
                score += 0.5

            # Boost if query keywords appear in title
            title_words = set(re.findall(r'\w+', title.lower()))
            overlap = len(query_words & title_words) / max(len(query_words), 1)
            score += overlap * 0.4

            # Boost if snippet is substantial
            if len(snippet) > 100:
                score += 0.2

            # Penalise very short snippets (usually paywalled or blocked)
            if len(snippet) < 30:
                score -= 0.3

            scored.append((score, r))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored]

    # ── Content extraction ────────────────────────────────────────────────────
    def _extract_content(self, url: str, query: str) -> str | None:
        """
        Fetch the page and extract the most relevant content.
        Returns clean text or None on failure.
        """
        try:
            resp = requests.get(
                url,
                timeout=REQUEST_TIMEOUT,
                headers=FETCH_HEADERS,
                allow_redirects=True,
            )
            resp.raise_for_status()

            # Only parse HTML content
            ct = resp.headers.get("content-type", "")
            if "text/html" not in ct:
                return None

            soup = BeautifulSoup(resp.text, "html.parser")

            # Drop noise tags
            for tag_name in TAGS_TO_DROP:
                for tag in soup.find_all(tag_name):
                    tag.decompose()

            # Try to find the main content zone
            zone = self._find_content_zone(soup)

            # Extract and score paragraphs
            paragraphs = self._extract_paragraphs(zone or soup, query)

            if not paragraphs:
                return None

            text = "\n\n".join(paragraphs[:MAX_PARAGRAPHS])

            # Final truncation
            if len(text) > MAX_CHARS_PAGE:
                text = text[:MAX_CHARS_PAGE] + "…"

            return text

        except Exception:
            return None

    def _find_content_zone(self, soup: BeautifulSoup) -> Tag | None:
        """
        Find the main content container using common selectors.
        Returns the best matching tag, or None if not found.
        """
        for selector in CONTENT_SELECTORS:
            try:
                zone = soup.select_one(selector)
                if zone and len(zone.get_text(strip=True)) > 200:
                    return zone
            except Exception:
                continue

        # Fallback — find the <div> with the most paragraph text
        best_div  = None
        best_len  = 0
        for div in soup.find_all("div"):
            text_len = len(div.get_text(strip=True))
            if text_len > best_len:
                best_len = text_len
                best_div = div

        return best_div if best_len > 300 else None

    def _extract_paragraphs(self, zone: Tag, query: str) -> list[str]:
        """
        Extract paragraphs from the content zone, scored by relevance to query.
        Returns paragraphs sorted by relevance score, longest/best first.
        """
        query_words = set(re.findall(r'\w+', query.lower()))
        scored      = []

        for tag in zone.find_all(["p", "h2", "h3", "h4", "li"]):
            text = tag.get_text(separator=" ", strip=True)

            # Skip short or noisy paragraphs
            if len(text) < MIN_PARA_LEN:
                continue
            if self._is_boilerplate(text):
                continue

            # Score by keyword overlap
            words   = set(re.findall(r'\w+', text.lower()))
            overlap = len(query_words & words) / max(len(query_words), 1)

            # Boost headings
            boost = 0.3 if tag.name in ("h2", "h3", "h4") else 0.0

            # Boost longer paragraphs (more informative)
            length_bonus = min(len(text) / 1000, 0.3)

            score = overlap + boost + length_bonus
            scored.append((score, text))

        # Sort by score descending, deduplicate
        scored.sort(key=lambda x: x[0], reverse=True)
        seen   = set()
        result = []
        for _, text in scored:
            key = text[:60]
            if key not in seen:
                seen.add(key)
                result.append(text)

        return result

    def _is_boilerplate(self, text: str) -> bool:
        """Return True if this paragraph looks like navigation/cookie/ad noise."""
        boilerplate_signals = re.compile(
            r'\b(cookie|privacy policy|terms of service|subscribe|newsletter|'
            r'sign up|log in|sign in|advertisement|click here|read more|'
            r'share this|follow us|all rights reserved|copyright ©|'
            r'location not found|weather unavailable|enable javascript|'
            r'please enable cookies|ad blocker|disable adblock)\b',
            re.I
        )
        # Short text with boilerplate signal
        if boilerplate_signals.search(text) and len(text) < 200:
            return True
        # Mostly uppercase (navigation text)
        if sum(1 for c in text if c.isupper()) / max(len(text), 1) > 0.5:
            return True
        return False