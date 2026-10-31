"""
core/inference/router.py — single-pass reason-then-emit router (MyAI v2, F3 S1)

Replaces v1's GBNF-constrained two-pass classifier, which forced a category
BEFORE the model could reason and so misrouted on reasoning models (Qwen emitted
a label before thinking → weather routed to wikipedia). Here the model reasons
FREELY, then commits a structured route on its final line — the ordering the
classifier couldn't do. Validated in armB_routing_probe / armB_ambiguity_probe:
100% parse, stable consistency, clean keyword-trap sweep on the Gemma 4 12B
daily-driver.

This module is F3 S1 T1: the reason-then-emit prompt + reliable parse of the
final selection. It reuses streaming.strip_reasoning to separate reasoning from
the emit (T3). The grammar-lock on the final line (T2) plugs in as a repair on
parse failure — parse_route already returns a safe DIRECT fallback that T2 will
pre-empt.

Menu and policy are the F3-settled decisions:
  • 11 routes; wikipedia fires ONLY on an explicit Wikipedia request; news is
    distinct from web_search.
  • Volatility is the line: timeless/known → direct; recent/changeable → web_search.
  • Numeric unit conversions → calculator; definitional ratios → direct.
  • Multi-skill → the single most central tool. Route on meaning, not keywords.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .base import OutputConstraint     # T2: grammar-locked repair emit
from .streaming import strip_reasoning   # T3 reuse: same reasoning strip as chat


# ── The production route menu (F3 decisions, confirmed) ─────────────────────────
# Route names MUST match skill names in the dispatch (myai_skills registry).
ROUTES: dict[str, str] = {
    "calculator": "arithmetic, percentages, and numeric unit/measure conversions "
                  "(e.g. 100 miles to km, 15% of 240)",
    "currency":   "live currency exchange rates and money conversions",
    "weather":    "current weather or forecast for a place",
    "datetime":   "current time, date, or timezone conversions",
    "web_search": "anything recent or that may have changed — current facts, "
                  "prices, live results, latest versions, who currently holds a role",
    "news":       "current news headlines and top stories",
    "image":      "adjusting or manipulating an uploaded image (resize, crop, "
                  "rotate, convert, greyscale)",
    "directions": "driving directions or routes between two places",
    "wikipedia":  "ONLY when the user explicitly asks to look something up on "
                  "Wikipedia",
    "memory":     "ONLY when the user explicitly asks you to remember, recall, "
                  "or forget a fact about them — e.g. 'remember that…', 'what do "
                  "you remember about me', 'forget that…'",
}

DIRECT = "direct"                       # the no-tool route (emitted as {"route": null})
_VALID_ROUTES = set(ROUTES) | {DIRECT}


def build_route_grammar() -> str:
    """GBNF constraining output to EXACTLY one valid emit: {"route": "<name>"}
    for a known route, or {"route": null} for direct. Built from ROUTES so it
    can never drift from the menu.

    Used only by the repair path (T2) — the grammar constrains the re-emitted
    final line, never the free reasoning that produced it (decision #3).
    """
    # Each route → the GBNF literal for its JSON string, e.g. "\"weather\"".
    route_literals = [f'"\\"{name}\\""' for name in ROUTES]
    value_alts = " | ".join(route_literals + ['"null"'])
    return (
        'root ::= "{\\"route\\": " value "}"\n'
        f"value ::= {value_alts}\n"
    )


def build_router_prompt(query: str) -> str:
    """The reason-then-emit router prompt: menu + policy + final-line JSON contract.
    Ported from the validated probe, extended to the full 11-route menu and the
    committed policy."""
    menu = "\n".join(f"  {name:<11}— {desc}" for name, desc in ROUTES.items())
    return (
        "You are a router deciding how to handle a user's message.\n\n"
        "Available tools:\n"
        f"{menu}\n\n"
        "How to choose:\n"
        "- Answer directly (route null) for anything you already know that does not "
        "change over time — timeless facts, definitions, explanations, simple "
        "conversions you can do yourself, recommendations, and casual messages.\n"
        "- Use web_search when the answer is recent or could have changed since your "
        "training — current office-holders, prices, latest products, live results.\n"
        "- If a message needs several tools, choose the single most central one.\n"
        "- Route on what the user actually means, not on keywords: 'convert my life "
        "around' is not a calculation, and 'directions in life' is not a route. "
        "Mentioning a person's name or a plan ('the date with Sarah on Friday') is "
        "not a request to remember anything — answer directly.\n\n"
        "Think first if it helps. When you are done, the LAST line of your reply must "
        "be a single JSON object and nothing else, in exactly one of these forms:\n"
        '  {"route": "<tool name from the list>"}   — to use a tool\n'
        '  {"route": null}                          — to answer directly, no tool\n\n'
        f"User message: {query}"
    )


@dataclass
class RouteDecision:
    route: str          # a tool name from ROUTES, or DIRECT
    parsed: bool        # did a usable {"route": ...} emit parse at all?
    valid: bool         # was the emitted route a known route (or null)?
    raw_tail: str       # last non-empty line of the cleaned reply (debug/eval)
    reasoning: str      # cleaned reply (reasoning + emit) for inspection/eval
    repaired: bool = False   # was this decided by the grammar-locked repair?


# Flat JSON object (route emits are flat — no nested braces). Matches the probe.
_JSON_OBJ_RE = re.compile(r"\{[^{}]*\}")


def parse_route(text: str, emits_think: bool) -> RouteDecision:
    """Parse a router reply into a RouteDecision.

    Strips reasoning (no-op on non-thinking models), then takes the LAST
    parseable {"route": ...} object — the model's committed decision, after any
    it may have mused about mid-reasoning. null/"null"/"none"/"" → DIRECT. An
    emitted route that isn't in the menu falls back to DIRECT with valid=False.
    No usable emit → DIRECT with parsed=False (the safe fallback T2 pre-empts).
    """
    text = strip_reasoning(text or "", emits_think)
    stripped = text.strip()
    tail = stripped.splitlines()[-1].strip() if stripped else ""

    for blob in reversed(_JSON_OBJ_RE.findall(text)):
        try:
            obj = json.loads(blob)
        except Exception:
            continue
        if isinstance(obj, dict) and "route" in obj:
            raw = obj["route"]
            if raw in (None, "null", "none", ""):
                return RouteDecision(DIRECT, True, True, tail, stripped)
            route_name = str(raw).strip().lower()
            if route_name in _VALID_ROUTES:
                return RouteDecision(route_name, True, True, tail, stripped)
            # Parsed a route, but it isn't one we know — safe-fallback to direct.
            return RouteDecision(DIRECT, True, False, tail, stripped)

    return RouteDecision(DIRECT, False, False, tail, stripped)


def repair_route(backend: Any, query: str, *, reasoning: str = "",
                 max_tokens: int = 64) -> RouteDecision:
    """Grammar-locked repair: force a valid {"route": ...} emit.

    Feeds the free pass's own reasoning back as context so the constrained choice
    still reflects it — the grammar governs only the re-emitted final line, never
    the reasoning (decision #3). Deterministic (temperature 0). Only reached when
    the free pass didn't yield a clean, valid route.
    """
    messages = [{"role": "user", "content": build_router_prompt(query)}]
    if reasoning:
        messages.append({"role": "assistant", "content": reasoning})
        messages.append({"role": "user",
                         "content": "Output only your routing decision now, "
                                    "as the JSON object."})
    out = backend.create_chat_completion(
        messages,
        max_tokens=max_tokens,
        temperature=0.0,
        stream=False,
        constraint=OutputConstraint(gbnf=build_route_grammar()),
    )
    text = out["choices"][0]["message"]["content"] or ""
    return parse_route(text, backend.emits_think_reasoning)


def route(backend: Any, query: str, *, temperature: float = 0.2,
          max_tokens: int = 512, grammar_repair: bool = True) -> RouteDecision:
    """Run one reason-then-emit pass and return the routing decision.

    Fast path: the free pass parses to a valid route → returned directly (no
    grammar, one call). Repair path: on a parse failure OR an unknown route, a
    grammar-locked emit (T2) re-attempts the final line before falling back to
    DIRECT. Never raises.
    """
    out = backend.create_chat_completion(
        [{"role": "user", "content": build_router_prompt(query)}],
        max_tokens=max_tokens,
        temperature=temperature,
        stream=False,
    )
    text = out["choices"][0]["message"]["content"] or ""
    decision = parse_route(text, backend.emits_think_reasoning)

    if decision.parsed and decision.valid:
        return decision                          # fast path — no grammar needed

    if grammar_repair:
        repaired = repair_route(backend, query, reasoning=decision.reasoning)
        if repaired.parsed and repaired.valid:
            repaired.repaired = True
            return repaired                      # grammar-locked valid emit

    return decision                              # safe DIRECT fallback
