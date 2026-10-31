"""
core/inference/engine.py — live wiring + app entry (MyAI v2, F4 stage 3a)

Wires the tested pipeline (pipeline.run_turn) to the REAL periphery and exposes
`ask_stream(user_input, user_id, conv_id)` — the exact entry point app.py already
imports from core.llm_inference, so cutover is a one-line import swap:

    from core.inference.engine import ask_stream, warmup

Everything model- or skill-heavy is imported LAZILY inside functions, so importing
this module (and pipeline.py) stays cheap and the logic stays unit-tested. The two
helpers relocated here from the retiring llm_inference.py — _title_sse_if_updated
and _persist_message_files — are not model-tied; they just needed a home before v1
is deleted (ADR-0002).
"""
from __future__ import annotations

import os
from typing import Any, Dict, Iterator, Optional

from .dispatch import build_skill_map
from .pipeline import PipelineDeps, _sse, run_turn

# ── backend singleton (replaces v1's import-time global Llama) ───────────────────
_backend: Optional[Any] = None


def get_backend() -> Any:
    """Load the configured backend once, then reuse it. Does NOT page weights —
    that's warmup()'s job, so a lazy first-prompt load (WARMUP=0) stays cheap."""
    global _backend
    if _backend is None:
        from .config import load_config
        from .registry import create_backend
        b = create_backend(load_config())
        b.load()
        _backend = b
    return _backend


def warmup() -> bool:
    """Startup hook (app.py imports this) — mirrors v1's contract: honour the
    WARMUP env var and return True only when the model was actually warmed, so
    app.py's 'ready' vs 'warmup skipped' status line is correct."""
    if os.getenv("WARMUP", "1") != "1":
        return False
    get_backend().warmup()
    return True


# ── relocated periphery helpers (were in llm_inference.py) ───────────────────────
def _title_sse_if_updated(conv_id: str) -> Optional[str]:
    """A 'title' SSE once the conversation has been auto-titled."""
    from core.db import get_conversation_title
    title = get_conversation_title(conv_id)
    if title and title != "New conversation":
        return _sse("title", {"conv_id": conv_id, "title": title})
    return None


def _persist_message_files(msg_id: str, files: list) -> None:
    """Link generated files to msg_id so download cards survive a reload."""
    from core.db import store_message_file
    for f in files:
        if not f.get("disk_path"):
            continue
        try:
            store_message_file(
                msg_id=msg_id, filename=f["filename"], mime_type=f["mime_type"],
                size=f["size"], disk_path=f["disk_path"],
            )
        except Exception as e:                                # noqa: BLE001
            print(f"[engine] failed to persist message_file "
                  f"'{f.get('filename')}': {e}")


# ── deps + skills wiring ────────────────────────────────────────────────────────
def build_default_deps(skill_map: Dict[str, Any]) -> PipelineDeps:
    """Assemble the real PipelineDeps. Memory predicates come off the registered
    memory skill (they're @staticmethod on its class), so no extra import and no
    coupling to the skill's module path."""
    from core import db
    from core.prompts import get_prompt_for_user
    from core.file_generation import (
        FILE_GENERATION_SYSTEM_SNIPPET, extract_and_generate, has_file_block,
        strip_file_blocks,
    )

    mem = skill_map.get("memory")
    if mem is not None:
        awaiting_clear = type(mem).awaiting_clear_confirmation
        is_confirmation = type(mem).is_confirmation_response
    else:
        awaiting_clear = lambda history: False          # noqa: E731
        is_confirmation = lambda query: False           # noqa: E731

    return PipelineDeps(
        get_messages=db.get_messages,
        get_prompt_for_user=get_prompt_for_user,
        get_file_contexts=db.get_file_contexts,
        append_message=db.append_message,
        list_memories=db.list_memories,
        auto_title=db.auto_title_from_first_message,
        title_event=_title_sse_if_updated,
        awaiting_clear=awaiting_clear,
        is_confirmation=is_confirmation,
        has_file_block=has_file_block,
        extract_and_generate=extract_and_generate,
        strip_file_blocks=strip_file_blocks,
        persist_files=_persist_message_files,
        file_gen_snippet=FILE_GENERATION_SYSTEM_SNIPPET,
    )


_deps: Optional[PipelineDeps] = None
_skill_map: Optional[Dict[str, Any]] = None


def _get_skill_map() -> Dict[str, Any]:
    global _skill_map
    if _skill_map is None:
        from myai_skills import REGISTERED_SKILLS
        _skill_map = build_skill_map(REGISTERED_SKILLS)
    return _skill_map


def _get_deps() -> PipelineDeps:
    global _deps
    if _deps is None:
        _deps = build_default_deps(_get_skill_map())
    return _deps


# ── app-facing entry ────────────────────────────────────────────────────────────
def ask_stream(user_input: str, user_id: str, conv_id: str) -> Iterator[str]:
    """Stream SSE for one turn (the v2 replacement for llm_inference.ask_stream).

    Wraps run_turn with the live backend/skills/deps and a top-level guard so an
    unexpected failure becomes a clean 'error' event rather than a broken stream.
    """
    try:
        yield from run_turn(
            user_input, user_id, conv_id,
            backend=get_backend(), skill_map=_get_skill_map(), deps=_get_deps(),
        )
    except Exception as e:                                    # noqa: BLE001
        print(f"[engine] unhandled error in ask_stream: {type(e).__name__}: {e}")
        yield _sse("error",
                   {"message": "Something went wrong generating a response."})
