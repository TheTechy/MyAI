"""
core/inference/dispatch.py — route → skill execution (MyAI v2, F4 S1)

The bridge from a router RouteDecision to a running skill. Replaces v1's
_classify_intent + _CATEGORY_TO_SKILL: the router (F3) decides the route, this
maps that route to a BaseSkill and runs it with the right arguments.

Per-skill calling conventions carry over from v1, but are satisfied by SIGNATURE
INSPECTION rather than hard-coded name checks: each skill's execute() is passed
only the context keys it actually declares. So image gets file_contexts, memory
gets user_id + history, directions gets memories, and a plain skill gets just the
query — with no route-specific branching here, and a future skill that declares a
new dependency works without editing dispatch.

This module only RUNS the skill and returns its text. Interpreting that text —
SKILL_DELIVER file payloads, memory-final replies, the Wikipedia→web fallback,
feeding context to an answer pass — is the pipeline's job (later F4 task), kept
out of here so dispatch stays small and testable.

Skills are injected (not imported): the pipeline passes the real
REGISTERED_SKILLS, tests pass fakes, and this module stays import-light with no
skill dependencies (Pillow, requests, …) pulled in at import.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .router import DIRECT, ROUTES


@dataclass
class DispatchResult:
    route: str                    # the route dispatched (a skill name, or DIRECT)
    ran_skill: bool               # did a skill actually execute?
    context: str                  # the skill's returned text ("" for direct/none)
    error: Optional[str] = None   # error string if the skill raised (belt-and-braces)


def build_skill_map(skills: List[Any]) -> Dict[str, Any]:
    """Index a list of skills by their .name (as the router routes by name)."""
    return {s.name: s for s in skills}


def _match_kwargs(execute: Callable, context: Dict[str, Any]) -> Dict[str, Any]:
    """Return the subset of `context` that this skill's execute() declares.

    The query is passed positionally, so the first parameter is never matched
    here (guards against double-passing it). A skill whose execute() accepts
    **kwargs receives the whole context.
    """
    try:
        sig = inspect.signature(execute)
    except (ValueError, TypeError):
        return {}
    params = sig.parameters
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return dict(context)

    names = list(params)
    first = names[0] if names else None          # the positional query slot
    matched: Dict[str, Any] = {}
    for name, value in context.items():
        if name == first:
            continue
        p = params.get(name)
        if p is not None and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY):
            matched[name] = value
    return matched


def dispatch(route: str, query: str, context: Dict[str, Any],
             skill_map: Dict[str, Any]) -> DispatchResult:
    """Execute the skill for `route`, or signal a direct answer.

    Args:
        route: the router's chosen route (a skill name, or DIRECT).
        query: the raw user input handed to the skill (NOT file-prepended — skills
            receive the user's own words, matching v1).
        context: available request context, e.g. {"user_id", "history",
            "memories", "file_contexts"}. Each skill gets only what it declares.
        skill_map: name → skill (from build_skill_map).

    Never raises: a skill error is captured into DispatchResult.error so the
    pipeline can report gracefully, mirroring the BaseSkill "never raises" contract.
    """
    if route == DIRECT:
        return DispatchResult(DIRECT, ran_skill=False, context="")

    skill = skill_map.get(route)
    if skill is None:
        return DispatchResult(route, ran_skill=False, context="",
                              error=f"no skill registered for route {route!r}")

    kwargs = _match_kwargs(skill.execute, context)
    try:
        out = skill.execute(query, **kwargs)
    except Exception as e:                                    # noqa: BLE001
        return DispatchResult(route, ran_skill=True, context="",
                              error=f"{type(e).__name__}: {e}")
    return DispatchResult(route, ran_skill=True, context=out or "")


def missing_skills(skill_map: Dict[str, Any]) -> List[str]:
    """Route names in the menu with no registered skill — a menu↔skills drift
    check the pipeline can assert at startup. DIRECT is not a skill, so excluded."""
    return [name for name in ROUTES if name not in skill_map]
