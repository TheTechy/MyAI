"""
test_suite/helpers.py
=====================
Shared test utilities imported by individual test modules.
"""
from __future__ import annotations

import json


def parse_sse(data: bytes) -> list[dict]:
    """
    Parse a raw SSE byte stream into a list of {event, data} dicts.

    Each blank-line-delimited block becomes one entry.  The 'data' field
    is JSON-decoded when possible, left as a raw string otherwise.
    """
    events: list[dict] = []
    current: dict = {}

    for raw_line in data.decode("utf-8", errors="replace").splitlines():
        line = raw_line.strip()

        if not line:
            if current:
                events.append(current)
                current = {}
        elif line.startswith("event:"):
            current["event"] = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            payload = line.split(":", 1)[1].strip()
            try:
                current["data"] = json.loads(payload)
            except json.JSONDecodeError:
                current["data"] = payload

    if current:
        events.append(current)

    return events


def event_names(events: list[dict]) -> list[str]:
    """Return just the event-name strings from a parsed SSE list."""
    return [e.get("event", "") for e in events]


def token_text(events: list[dict]) -> str:
    """Concatenate all token payloads from a parsed SSE list."""
    return "".join(
        e["data"].get("text", "")
        for e in events
        if e.get("event") == "token" and isinstance(e.get("data"), dict)
    )
