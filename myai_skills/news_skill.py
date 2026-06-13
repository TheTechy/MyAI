"""
myai_skills/news_skill.py
=========================
News headlines skill — fetches current headlines from RSS feeds.
No API key required.

Sources
-------
  BBC News        https://feeds.bbci.co.uk/news/rss.xml
  BBC Technology  https://feeds.bbci.co.uk/news/technology/rss.xml
  BBC UK          https://feeds.bbci.co.uk/news/uk/rss.xml
  The Guardian    https://www.theguardian.com/uk/rss
  Reuters         https://feeds.reuters.com/reuters/topNews

Flow
----
  1. Parse the user query to determine desired source / topic
  2. Fetch the appropriate RSS feed(s)
  3. Parse with xml.etree.ElementTree (stdlib — no extra deps)
  4. Return a structured plain-text summary for the LLM

The LLM uses this context to present the headlines conversationally.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import requests

from .base import BaseSkill

# ── Feed definitions ───────────────────────────────────────────────────────────

_FEEDS: dict[str, dict] = {
    "bbc_top": {
        "label": "BBC News — Top Stories",
        "url":   "https://feeds.bbci.co.uk/news/rss.xml",
    },
    "bbc_uk": {
        "label": "BBC News — UK",
        "url":   "https://feeds.bbci.co.uk/news/uk/rss.xml",
    },
    "bbc_tech": {
        "label": "BBC News — Technology",
        "url":   "https://feeds.bbci.co.uk/news/technology/rss.xml",
    },
    "bbc_science": {
        "label": "BBC News — Science & Environment",
        "url":   "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
    },
    "bbc_business": {
        "label": "BBC News — Business",
        "url":   "https://feeds.bbci.co.uk/news/business/rss.xml",
    },
    "bbc_sport": {
        "label": "BBC Sport",
        "url":   "https://feeds.bbci.co.uk/sport/rss.xml",
    },
    "guardian": {
        "label": "The Guardian — UK",
        "url":   "https://www.theguardian.com/uk/rss",
    },
    "guardian_world": {
        "label": "The Guardian — World",
        "url":   "https://www.theguardian.com/world/rss",
    },
    "guardian_tech": {
        "label": "The Guardian — Technology",
        "url":   "https://www.theguardian.com/uk/technology/rss",
    },
    "sky_news": {
        "label": "Sky News",
        "url":   "https://feeds.skynews.com/feeds/rss/home.xml",
    },
    "daily_mail": {
        "label": "Daily Mail",
        "url":   "https://www.dailymail.com/home/index.rss",
    },
    "independent": {
        "label": "The Independent",
        "url":   "https://www.independent.co.uk/rss",
    },
    "express": {
        "label": "Daily Express",
        "url":   "https://www.express.co.uk/posts/rss/77/news",
    },
    "metro": {
        "label": "Metro",
        "url":   "https://metro.co.uk/feed/",
    },
}

# Default feeds when no specific source/topic is requested
_DEFAULT_FEEDS = ["bbc_top", "sky_news"]

# ── Topic → feed mapping ───────────────────────────────────────────────────────

_TOPIC_MAP: list[tuple[re.Pattern, list[str]]] = [
    (re.compile(r'\b(tech(nology)?|AI|artificial intelligence|software|app(s)?)\b', re.I), ["bbc_tech", "guardian_tech"]),
    (re.compile(r'\b(science|environment|climate|nature|space)\b',                  re.I), ["bbc_science"]),
    (re.compile(r'\b(business|economy|finance|market(s)?|stocks?)\b',               re.I), ["bbc_business"]),
    (re.compile(r'\b(sport(s)?|football|cricket|tennis|rugby|F1)\b',                re.I), ["bbc_sport"]),
    (re.compile(r'\b(world|international|global|foreign)\b',                         re.I), ["guardian_world"]),
    (re.compile(r'\b(guardian)\b',                                                   re.I), ["guardian"]),
    (re.compile(r'\b(sky(\s*news)?)\b',                                              re.I), ["sky_news"]),
    (re.compile(r'\b(daily\s*mail|mail\s*online)\b',                                 re.I), ["daily_mail"]),
    (re.compile(r'\b(independent)\b',                                                re.I), ["independent"]),
    (re.compile(r'\b(express|daily\s*express)\b',                                    re.I), ["express"]),
    (re.compile(r'\b(metro)\b',                                                      re.I), ["metro"]),
    (re.compile(r'\b(bbc)\b',                                                        re.I), ["bbc_top", "bbc_uk"]),
    (re.compile(r'\b(uk|britain|british|england|scotland|wales)\b',                  re.I), ["bbc_uk", "sky_news"]),
]

# ── Constants ──────────────────────────────────────────────────────────────────

_USER_AGENT = "MyAI-NewsSkill/1.0"
_TIMEOUT    = 8
_MAX_ITEMS  = 3   # headlines per feed
_MAX_FEEDS  = 4   # max feeds to fetch per query

# ── Helpers ────────────────────────────────────────────────────────────────────

def _pick_feeds(query: str) -> list[str]:
    """Return a list of feed keys based on the query topic/source."""
    for pattern, feeds in _TOPIC_MAP:
        if pattern.search(query):
            return feeds[:_MAX_FEEDS]
    return _DEFAULT_FEEDS


def _fetch_feed(url: str) -> list[dict]:
    """
    Fetch and parse an RSS feed. Returns a list of item dicts with
    keys: title, description, link, published.
    Returns [] on any error — skill degrades gracefully.
    """
    try:
        r = requests.get(
            url,
            headers={"User-Agent": _USER_AGENT},
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
    except requests.RequestException:
        return []

    try:
        root = ET.fromstring(r.content)
    except ET.ParseError:
        return []

    items = []
    # RSS 2.0: <rss><channel><item>...</item></channel></rss>
    for item in root.findall(".//item")[:_MAX_ITEMS]:
        title = item.findtext("title", "").strip()
        desc  = item.findtext("description", "").strip()
        link  = item.findtext("link", "").strip()
        pub   = item.findtext("pubDate", "").strip()

        # Strip any HTML tags from description
        desc = re.sub(r'<[^>]+>', '', desc).strip()

        # Parse publish date to a friendly string
        pub_str = ""
        if pub:
            try:
                dt     = parsedate_to_datetime(pub)
                now    = datetime.now(timezone.utc)
                delta  = now - dt
                mins   = int(delta.total_seconds() // 60)
                if mins < 60:
                    pub_str = f"{mins}m ago"
                elif mins < 1440:
                    pub_str = f"{mins // 60}h ago"
                else:
                    pub_str = dt.strftime("%-d %b")
            except Exception:
                pub_str = pub[:16]

        if title:
            items.append({
                "title":       title,
                "description": desc,
                "link":        link,
                "published":   pub_str,
            })

    return items


def _format_items(items: list[dict]) -> str:
    """Format headlines as a numbered markdown list with clickable links."""
    lines = []
    for i, item in enumerate(items, 1):
        age      = f" · {item['published']}" if item["published"] else ""
        title    = item["title"]
        link     = item["link"]
        headline = f"[{title}]({link})" if link else title
        line     = f"{i}. {headline}{age}"
        if item["description"] and len(item["description"]) < 180:
            line += f"\n   *{item['description']}*"
        lines.append(line)
    return "\n".join(lines)


# ── Skill class ────────────────────────────────────────────────────────────────

class NewsSkill(BaseSkill):
    """
    News headlines skill using RSS feeds.
    No API key required — uses BBC and Guardian public RSS.
    """

    name        = "news"
    description = "Latest news headlines from BBC and The Guardian via RSS"

    def execute(self, query: str) -> str:
        feed_keys = _pick_feeds(query)
        sections  = []
        fetched   = 0

        for key in feed_keys:
            feed = _FEEDS.get(key)
            if not feed:
                continue
            items = _fetch_feed(feed["url"])
            if not items:
                continue
            sections.append(f"**{feed['label']}**\n{_format_items(items)}")
            fetched += 1

        if not sections:
            return (
                "Could not fetch news headlines at this time. "
                "The RSS feeds may be temporarily unavailable. "
                "Please let the user know and suggest they check BBC News or The Guardian directly."
            )

        header = f"News headlines fetched from {fetched} source(s):\n"
        return header + "\n\n".join(sections)
