"""
core/inference/messages.py — OpenAI-shaped message assembler (MyAI v2, F2 S1 T1)

Turns a conversation into the `messages` array that InferenceBackend.
create_chat_completion() consumes, letting each model's OWN embedded chat
template build the actual prompt (v2 decision #2). Replaces v1's
_messages_from_history + hand-built Gemma turn markers.

TWO ENTRY POINTS
----------------
  assemble_messages(...)        PURE — takes already-fetched pieces, no DB. This
                                is the testable core and where all the logic is.
  build_messages_for_turn(...)  thin wrapper — does the get_prompt_for_user +
                                get_messages fetches, then delegates. DB imports
                                are lazy so the pure core stays import-light.

SYSTEM-ROLE STRATEGY (settled in F2 planning, from the real GGUF templates)
---------------------------------------------------------------------------
Default: emit a proper {"role": "system"} message at position 0. Verified against
the templates actually in rotation — Gemma 4 12B, Gemma 4 31B and Qwen 3.5 all
render a real system turn, so this is correct for the whole current lineup. This
is the forward-compatible shape: models converging on OpenAI semantics use their
system channel as designed.

Fallback: `supports_system_role=False` reproduces v1's approach EXACTLY — no
system message, the instruction fused into the latest user turn. This is a safety
net for older/other templates that lack a system slot (Gemma 3, Llama-2-era,
etc.); none of the current models need it, so it is unit-tested but has not yet
met a real model that requires it. Auto-detecting the flag from a trial render is
deferred to F2 S1 T2.

HARD RULE the templates impose: a system message MUST be first. Qwen's template
raises "System message must be at the beginning"; Gemma silently drops a
mis-placed one. So system lives at index 0, always — never fused mid-history.

Notes:
  • Role mapping is mandatory: the DB stores assistant turns as role "model" (a
    Gemma-ism); the OpenAI shape wants "assistant". Miss this and history breaks.
  • History rows that are neither "user" nor "model" are dropped (as v1 did) —
    this also stops a stray stored "system" row from violating the system-first
    invariant.
  • Message `files` are ignored here — attachments/multimodal are F2 S3 / F6. The
    content-part array shape is reserved, not built (see the fuse guard below).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# DB role string → OpenAI role string.
_ROLE_MAP = {"user": "user", "model": "assistant"}


def _map_history(history: List[dict]) -> List[Dict[str, Any]]:
    """Map stored turns to OpenAI-shaped messages, dropping unknown roles."""
    out: List[Dict[str, Any]] = []
    for turn in history:
        role = turn.get("role")
        mapped = _ROLE_MAP.get(role)
        if mapped is None:
            continue  # skip anything not user/model (v1 behaviour)
        out.append({"role": mapped, "content": turn.get("content", "")})
    return out


def assemble_messages(
    system_prompt: str,
    history: List[dict],
    new_user_message: Any,
    *,
    supports_system_role: bool = True,
) -> List[Dict[str, Any]]:
    """Assemble the OpenAI-shaped `messages` array for one turn.

    Args:
        system_prompt: the fully-composed system instruction (personality +
            rules + memories). Passed through opaquely — this function does not
            build or edit it. Falsy → no system content is emitted at all.
        history: prior turns as stored, each {"role": "user"|"model",
            "content": str, ...}. Extra keys (msg_id, files) are ignored.
        new_user_message: the current user turn's content. A str today; a
            content-part list (multimodal) is accepted only in system-role mode
            (see the fuse guard).
        supports_system_role: True (default) → emit {"role":"system"} at index 0.
            False → fuse the system prompt into the latest user turn (v1's exact
            approach), emitting no system message.

    Returns:
        A list of {"role", "content"} dicts, system-first when applicable.
    """
    messages: List[Dict[str, Any]] = _map_history(history)

    if supports_system_role:
        if system_prompt:
            # System-first invariant: prepend, never fuse. Both current template
            # families require the system message at position 0.
            messages.insert(0, {"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": new_user_message})
        return messages

    # ── fuse fallback (v1 parity) ───────────────────────────────────────────────
    # No system role. The instruction rides on the latest user turn so the rules
    # stay adjacent to the live question — v1's fix for long-chat rule-drift on
    # templates without a system slot.
    fused = new_user_message
    if system_prompt:
        if not isinstance(new_user_message, str):
            # A content-part array can't be string-fused. When multimodal lands
            # (F6), fuse by PREPENDING a {"type":"text","text": system} part
            # here. Until then, refuse loudly rather than mangle the turn.
            raise NotImplementedError(
                "fuse fallback (supports_system_role=False) requires a string "
                "user message; content-part arrays arrive with multimodal (F6)"
            )
        fused = f"{system_prompt}\n\nUser question: {new_user_message}"
    messages.append({"role": "user", "content": fused})
    return messages


def build_messages_for_turn(
    user_id: str,
    conv_id: str,
    new_user_message: Any,
    *,
    supports_system_role: bool = True,
    max_turns: int = 10,
) -> List[Dict[str, Any]]:
    """Fetch this user's system prompt + recent history, then assemble.

    Thin convenience wrapper over assemble_messages. DB/prompt imports are lazy
    so importing this module costs nothing and needs no database present.
    """
    from core.db import get_messages          # lazy: keeps the core import-light
    from core.prompts import get_prompt_for_user

    system_prompt = get_prompt_for_user(user_id)
    history = get_messages(conv_id, max_turns=max_turns)
    return assemble_messages(
        system_prompt,
        history,
        new_user_message,
        supports_system_role=supports_system_role,
    )
