"""
myai_skills/memory_skill.py
===========================
User memory skill — stores explicit, user-stated facts and recalls them.

Memories are added only when the user explicitly asks ("remember I'm allergic
to peanuts"). They are injected into the system prompt by core/prompts.py so the
model can use them naturally in any conversation.

Sub-operations (classified here, by regex — no extra LLM call):
  add        "Remember I'm allergic to peanuts"
  list       "What do you remember about me?"
  forget one "Forget that I'm allergic to peanuts"
  clear all  "Forget everything"  → two-turn confirmation

Per-user isolation is enforced in core/db.py — every helper is scoped by user_id.

Unlike most skills, the strings returned here are already the final, user-facing
reply. core/llm_inference.py delivers them directly (no second LLM pass) so that
confirmations and the memory list are exact and never reworded or hallucinated.
"""

from __future__ import annotations

import re

from .base import BaseSkill
from core.db import (
    add_memory,
    list_memories,
    delete_memories_matching,
    clear_memories,
)

# ── Sub-classification patterns ────────────────────────────────────────────────
# A bare "forget everything" — wipe the lot (handled via two-turn confirmation).
_CLEAR_ALL_RE = re.compile(
    r"\bforget\s+(?:it\s+)?(?:everything|all|the\s+lot)\b"
    r"|\b(?:forget|delete|clear|wipe|erase)\s+(?:all\s+)?(?:of\s+)?(?:my\s+)?(?:memories|notes|everything)\b",
    re.IGNORECASE,
)

# "What do you remember about me?" / "list my memories"
_LIST_RE = re.compile(
    r"\bwhat\s+(?:do|can)\s+you\s+(?:remember|know)\b"
    r"|\bwhat\s+have\s+you\s+remembered\b"
    r"|\b(?:list|show)\s+(?:me\s+)?(?:my\s+)?(?:memories|notes)\b"
    r"|\bwhat(?:'s| is)\s+in\s+(?:my\s+)?memory\b",
    re.IGNORECASE,
)

# Leading verb for an add ("remember that …", "note that …")
_ADD_LEAD_RE = re.compile(
    r"^(?:please\s+)?(?:can\s+you\s+|could\s+you\s+|would\s+you\s+)?"
    r"(?:remember|note|keep\s+in\s+mind|don'?t\s+forget|do\s+not\s+forget|make\s+a\s+note)"
    r"(?:\s+(?:that|this|the\s+fact\s+that|about))?\s*[:,\-]?\s*",
    re.IGNORECASE,
)

# Leading verb for a single forget ("forget that …", "forget about …")
_FORGET_LEAD_RE = re.compile(
    r"^(?:please\s+)?(?:can\s+you\s+|could\s+you\s+|would\s+you\s+)?"
    r"forget(?:\s+(?:that|about|the\s+fact\s+that))?\s*[:,\-]?\s*",
    re.IGNORECASE,
)

_AFFIRMATIVE_RE = re.compile(
    r"^\s*(?:yes|yep|yeah|yup|sure|ok|okay|go\s+ahead|do\s+it|confirm(?:ed)?|"
    r"i'?m\s+sure|please\s+do|absolutely|definitely|correct|that'?s\s+right)\b",
    re.IGNORECASE,
)

_NEGATIVE_RE = re.compile(
    r"^\s*(?:no|nope|nah|cancel|stop|don'?t|do\s+not|never\s*mind|leave\s+it|"
    r"keep\s+(?:them|it))\b",
    re.IGNORECASE,
)


class MemorySkill(BaseSkill):
    name        = "memory"
    description = "Stores and recalls explicit facts the user asks to be remembered."

    # Exact text returned when a clear-all is requested. core/llm_inference.py
    # delivers it verbatim, so awaiting_clear_confirmation() can recognise it as
    # the previous model turn and complete the two-turn confirmation.
    CLEAR_CONFIRM_PROMPT = (
        "⚠️ This will permanently delete **everything** I remember about you. "
        "Reply **yes** to confirm, or **no** to keep your memories."
    )

    # ── Confirmation helpers (used by llm_inference for two-turn routing) ───────
    @staticmethod
    def awaiting_clear_confirmation(history: list[dict] | None) -> bool:
        """True if the previous model turn was the clear-all confirmation prompt."""
        if not history:
            return False
        last = history[-1]
        return (
            last.get("role") == "model"
            and last.get("content", "").strip() == MemorySkill.CLEAR_CONFIRM_PROMPT
        )

    @staticmethod
    def is_confirmation_response(query: str) -> bool:
        """True if *query* reads as a yes/no answer to a confirmation prompt."""
        return bool(_AFFIRMATIVE_RE.match(query) or _NEGATIVE_RE.match(query))

    # ── Main entry ─────────────────────────────────────────────────────────────
    def execute(self, query: str, user_id: str | None = None,
                history: list[dict] | None = None) -> str:
        if not user_id:
            return "I can't access your memories right now — no user is signed in."

        query = query.strip()

        # ── Two-turn clear-all: completing a pending confirmation ──────────────
        if self.awaiting_clear_confirmation(history):
            if _AFFIRMATIVE_RE.match(query):
                removed = clear_memories(user_id)
                if removed:
                    return f"Done — I've cleared all {removed} memories I had about you."
                return "There was nothing saved, so nothing to clear."
            if _NEGATIVE_RE.match(query):
                return "Okay, I've left your memories untouched."
            # Force-routed here only on yes/no, but stay safe.
            return self.CLEAR_CONFIRM_PROMPT

        # ── Clear-all request: ask for confirmation (turn one) ─────────────────
        if _CLEAR_ALL_RE.search(query):
            return self.CLEAR_CONFIRM_PROMPT

        # ── List ───────────────────────────────────────────────────────────────
        if _LIST_RE.search(query):
            return self._render_list(user_id)

        # ── Forget one ──────────────────────────────────────────────────────────
        if _FORGET_LEAD_RE.match(query):
            term = _FORGET_LEAD_RE.sub("", query).strip().rstrip(".!?")
            if not term:
                return "What would you like me to forget? For example: \"forget that I'm allergic to peanuts\"."
            removed = delete_memories_matching(user_id, term)
            if removed:
                joined = "; ".join(m["content"] for m in removed)
                return f"Done — I've forgotten that: {joined}."
            return (
                f"I couldn't find anything matching \"{term}\" in your memories.\n\n"
                + self._render_list(user_id)
            )

        # ── Add ─────────────────────────────────────────────────────────────────
        if _ADD_LEAD_RE.match(query):
            content = _ADD_LEAD_RE.sub("", query).strip().rstrip(".!")
            if not content:
                return "What would you like me to remember? For example: \"remember I'm allergic to peanuts\"."
            add_memory(user_id, content)
            return f"Got it — I'll remember that {content}."

        # ── Fallback ────────────────────────────────────────────────────────────
        return self._render_list(user_id)

    # ── Rendering ──────────────────────────────────────────────────────────────
    @staticmethod
    def _render_list(user_id: str) -> str:
        memories = list_memories(user_id)
        if not memories:
            return (
                "I don't have any memories saved about you yet. Tell me something "
                "with \"remember …\" and I'll keep it."
            )
        lines = "\n".join(f"- {m['content']}" for m in memories)
        return f"Here's what I remember about you:\n\n{lines}"
