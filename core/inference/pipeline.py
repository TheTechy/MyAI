"""
core/inference/pipeline.py — router-driven turn pipeline (MyAI v2, F4)

The v2 replacement for v1's _ask_stream_impl. Built in stages:
  1. PLANNING (this file, now): route → dispatch → decide the ACTION for the turn.
     Pure decision logic — no model, no SSE, no DB — so every branch is testable.
  2. EXECUTION (next): a generator that runs a Plan — streams the answer via
     stream_clean, emits SSE events, persists, handles [FILE:] blocks.

Planning reproduces the branch logic of v1's ask_stream, mapping each turn to one
of a few actions:
  • ANSWER_DIRECT        — no skill; the model answers the user directly.
  • ANSWER_WITH_CONTEXT  — a skill ran; its text feeds the model's answer pass.
  • FINAL_TEXT           — a fixed answer, no model pass (memory replies must be
                           exact).
  • FILE_DELIVER         — a skill produced a file to emit directly (image's
                           SKILL_DELIVER protocol), no model pass.
  • ERROR                — a skill errored; report it.

The Wikipedia→web-search fallback (a Wikipedia miss transparently retries as a web
search) lives in resolve_skill, so planning downstream sees the effective route.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .dispatch import DispatchResult, dispatch
from .router import DIRECT

# ── turn actions ────────────────────────────────────────────────────────────────
ANSWER_DIRECT = "answer_direct"
ANSWER_WITH_CONTEXT = "answer_with_context"
FINAL_TEXT = "final_text"
FILE_DELIVER = "file_deliver"
ERROR = "error"

_SKILL_DELIVER = "SKILL_DELIVER|"
_WIKI_MISS = "No Wikipedia article found"


@dataclass
class Plan:
    action: str
    route: str
    text: str = ""                       # FINAL_TEXT answer / FILE_DELIVER blurb
    context: str = ""                    # ANSWER_WITH_CONTEXT skill text to enrich
    file: Optional[Dict[str, Any]] = None  # FILE_DELIVER payload (SSE 'file' shape)
    error: Optional[str] = None


def resolve_skill(route: str, query: str, context: Dict[str, Any],
                  skill_map: Dict[str, Any]) -> DispatchResult:
    """Run the skill for `route`, applying the Wikipedia→web-search fallback.

    A Wikipedia miss transparently retries the query as a web search (v1
    behaviour), so the returned DispatchResult carries the EFFECTIVE route —
    downstream planning never sees the dead-end. DIRECT passes straight through.
    """
    result = dispatch(route, query, context, skill_map)

    if (route == "wikipedia" and result.ran_skill and not result.error
            and result.context.startswith(_WIKI_MISS)
            and "web_search" in skill_map):
        return dispatch("web_search", query, context, skill_map)

    return result


def _parse_skill_deliver(route: str, context: str) -> Plan:
    """Parse an image SKILL_DELIVER payload into a FILE_DELIVER plan.

    Format: 'SKILL_DELIVER|filename|mime|<base64>[|<disk_path>]\\n---\\n<desc>'.
    The disk_path (5th field, F91 #1) is optional — a 4-field payload is still
    accepted (delivered in-session, not persisted). On a malformed payload, fall
    back to a plain text notice (v1 behaviour).
    """
    try:
        header, description = context.split("\n---\n", 1)
        parts = header.split("|", 4)                 # [tag, filename, mime, b64, disk_path?]
        filename, mime, b64 = parts[1], parts[2], parts[3]
        disk_path = parts[4] if len(parts) > 4 else ""
        file = {
            "filename": filename,
            "mime_type": mime,
            "encoding": "base64",
            "data": b64,
            "size": len(b64) * 3 // 4,
        }
        if disk_path:
            file["disk_path"] = disk_path            # enables persistence downstream
        return Plan(FILE_DELIVER, route=route, text=description.strip(), file=file)
    except Exception as e:                                    # noqa: BLE001
        return Plan(FINAL_TEXT, route=route,
                    text=f"Image processed but could not deliver file: {e}")


def plan_action(result: DispatchResult) -> Plan:
    """Decide the turn's action from a (resolved) dispatch result.

    memory → FINAL_TEXT (exact reply, no model). SKILL_DELIVER → FILE_DELIVER.
    other skills → ANSWER_WITH_CONTEXT. no skill (direct) → ANSWER_DIRECT.
    """
    if result.error:
        return Plan(ERROR, route=result.route, error=result.error)

    if not result.ran_skill:
        return Plan(ANSWER_DIRECT, route=DIRECT)

    if result.route == "memory":
        return Plan(FINAL_TEXT, route="memory", text=result.context.strip())

    if result.context.startswith(_SKILL_DELIVER):
        return _parse_skill_deliver(result.route, result.context)

    return Plan(ANSWER_WITH_CONTEXT, route=result.route, context=result.context)


def plan_turn(route: str, query: str, context: Dict[str, Any],
              skill_map: Dict[str, Any]) -> Plan:
    """Full planning step: resolve the skill (with fallback), then decide the
    action. The executor calls this after the router has chosen `route`."""
    return plan_action(resolve_skill(route, query, context, skill_map))


# ═══════════════════════════════════════════════════════════════════════════════
# EXECUTION (F4 pipeline stage 2): run a turn, yield SSE events.
#
# run_turn is the v2 replacement for v1's _ask_stream_impl. Periphery (db, prompts,
# file-gen, memory predicates) is injected via PipelineDeps so every SSE branch is
# testable with fakes — no model, no database. The thin ask_stream() wrapper that
# wires the real backend + REGISTERED_SKILLS is stage 3 (cutover).
# ═══════════════════════════════════════════════════════════════════════════════
import json as _json
from typing import Callable, Iterator, List

from .messages import assemble_messages
from .filegen import prepare_files
from .router import route as _default_route
from .streaming import stream_clean

MAX_HISTORY_TURNS = 10


def _sse(event: str, data: dict) -> str:
    """Format a Server-Sent Event string (matches v1's _sse)."""
    return f"event: {event}\ndata: {_json.dumps(data)}\n\n"


def _file_payload_for_client(f: dict) -> dict:
    """Drop server-only fields (disk_path) before sending a file dict to the browser."""
    return {k: v for k, v in f.items() if k != "disk_path"}


def _enrich_answer_message(user_message: str, search_context: str) -> str:
    return (
        "Use the web search results below to answer the user's question.\n\n"
        f"Web search results:\n{search_context}\n\n"
        f"User question: {user_message}"
    )


_SKILL_LABELS = {
    "calculator": "calculation result", "weather": "weather data",
    "datetime": "date and time data", "currency": "currency conversion data",
    "directions": "driving directions", "wikipedia": "Wikipedia article summary",
    "news": "news headlines",
}


def _enrich_skill_message(user_message: str, skill_name: str, context: str) -> str:
    if skill_name == "news":
        return (
            "Below are the latest news headlines in markdown format with clickable "
            "links. Present them to the user exactly as provided — do not rewrite, "
            "summarise, or remove any links or descriptions. Add a brief one-sentence "
            "introduction mentioning the sources, then output the headlines verbatim.\n\n"
            f"{context}\n\n"
            f"User question: {user_message}"
        )
    label = _SKILL_LABELS.get(skill_name, "data")
    return (
        f"Use the {label} below to answer the user's question directly and "
        f"concisely.\n\n"
        f"{label.capitalize()}:\n{context}\n\n"
        f"User question: {user_message}"
    )


@dataclass
class PipelineDeps:
    """Injected periphery — the reused, non-model-tied functions. build_default_deps()
    (stage 3) supplies the real ones; tests supply fakes."""
    get_messages: Callable            # (conv_id, max_turns) -> history
    get_prompt_for_user: Callable     # (user_id) -> str
    get_file_contexts: Callable       # (conv_id) -> list[dict]
    append_message: Callable          # (conv_id, role, content) -> msg_id | None
    list_memories: Callable           # (user_id) -> list
    auto_title: Callable              # (conv_id, user_input) -> None
    title_event: Callable             # (conv_id) -> Optional[str]  (pre-formatted SSE)
    awaiting_clear: Callable          # (history) -> bool
    is_confirmation: Callable         # (user_input) -> bool
    has_file_block: Callable          # (text) -> bool
    extract_and_generate: Callable    # (text) -> list[file dict]
    strip_file_blocks: Callable       # (text) -> str
    persist_files: Callable           # (msg_id, files) -> None
    file_gen_snippet: str = ""        # appended to the system prompt (interim [FILE:])


def _attach_file_blocks(user_input: str, file_contexts: List[dict]) -> str:
    if not file_contexts:
        return user_input
    blocks = "\n\n".join(
        f"[Attached file: {fc['filename']}]\n{fc['extracted_text']}\n[/Attached file]"
        for fc in file_contexts
    )
    return f"{blocks}\n\nUser question: {user_input}"


def _system_prompt_for(user_id: str, deps: PipelineDeps) -> str:
    system_prompt = deps.get_prompt_for_user(user_id)
    if deps.file_gen_snippet:
        system_prompt = f"{system_prompt}\n\n{deps.file_gen_snippet}"
    return system_prompt


def context_usage(user_id: str, conv_id: Optional[str], *,
                  backend: Any, deps: PipelineDeps) -> Optional[dict]:
    """How full the context will be when this chat's NEXT message is sent: the
    system prompt, the history that turn will carry and any attached files,
    counted with the model's tokenizer. No conv_id means a new, empty chat.

    Measured from the chat itself rather than the engine's live token count,
    which reflects whichever turn ran last, for any user. None when the backend
    can't count tokens.
    """
    history = deps.get_messages(conv_id, MAX_HISTORY_TURNS) if conv_id else []
    file_contexts = deps.get_file_contexts(conv_id) if conv_id else []
    messages = assemble_messages(
        _system_prompt_for(user_id, deps), history,
        _attach_file_blocks("", file_contexts),
        supports_system_role=backend.supports_system_role,
    )
    used = backend.count_tokens(messages)
    if used is None:
        return None
    max_ctx = backend.context_window()
    return {"used": used, "max": max_ctx, "remaining": max_ctx - used,
            "pct": round(used / max_ctx * 100, 1)}


def _ctx_event(user_id: str, conv_id: str, backend: Any,
               deps: PipelineDeps) -> Iterator[str]:
    usage = context_usage(user_id, conv_id, backend=backend, deps=deps)
    if usage is not None:
        yield _sse("ctx", usage)


def run_turn(user_input: str, user_id: str, conv_id: str, *,
             backend: Any, skill_map: Dict[str, Any], deps: PipelineDeps,
             route_fn: Callable = _default_route) -> Iterator[str]:
    """Generate SSE events for one turn (the v2 _ask_stream_impl).

    route → plan → act. FINAL_TEXT/FILE_DELIVER deliver directly; ANSWER_* stream
    the model via stream_clean. [FILE:] blocks are generated on the direct path
    only (skill answers strip them), per ADR-0002's interim.
    """
    history = deps.get_messages(conv_id, MAX_HISTORY_TURNS)
    system_prompt = _system_prompt_for(user_id, deps)
    file_contexts = deps.get_file_contexts(conv_id)
    prompt_input = _attach_file_blocks(user_input, file_contexts)

    context = {
        "user_id": user_id,
        "history": history,
        "memories": deps.list_memories(user_id),
        "file_contexts": file_contexts,
    }

    # ── pre-route: two-turn memory-clear confirmation (bare yes/no won't route) ──
    if deps.awaiting_clear(history) and deps.is_confirmation(user_input):
        route = "memory"
    else:
        yield _sse("status", {"message": "Thinking…"})
        route = route_fn(backend, user_input).route

    plan = plan_turn(route, user_input, context, skill_map)

    # ── ERROR ───────────────────────────────────────────────────────────────────
    if plan.action == ERROR:
        yield _sse("error", {"message": f"Skill error: {plan.error}"})
        return

    # ── FINAL_TEXT (memory) / FILE_DELIVER (image): no model pass ────────────────
    if plan.action in (FINAL_TEXT, FILE_DELIVER):
        if plan.action == FILE_DELIVER and plan.file:
            yield _sse("file", _file_payload_for_client(plan.file))
        answer = plan.text
        deps.append_message(conv_id, "user", user_input)
        msg_id = deps.append_message(conv_id, "model", answer)
        # F91 #1: persist a disk-backed delivered image so it survives a reload,
        # exactly like generated files do (no-op when there's no disk_path).
        if (plan.action == FILE_DELIVER and plan.file
                and plan.file.get("disk_path") and msg_id):
            deps.persist_files(msg_id, [plan.file])
        deps.auto_title(conv_id, user_input)
        yield from _ctx_event(user_id, conv_id, backend, deps)
        if answer:
            yield _sse("token", {"text": answer})
        yield _sse("done", {"truncated": False})
        ev = deps.title_event(conv_id)
        if ev:
            yield ev
        return

    # ── ANSWER_WITH_CONTEXT / ANSWER_DIRECT: model answer + v1-parity file handling
    # Direct answers BUFFER (like v1) so the raw [FILE:] block never streams; skill/
    # search answers stream live (like v1's search path). Both run prepare_files().
    if plan.action == ANSWER_WITH_CONTEXT:
        yield _sse("status", {"message": "Generating answer…"})
        if plan.route == "web_search":
            answer_input = _enrich_answer_message(user_input, plan.context)
        else:
            answer_input = _enrich_skill_message(user_input, plan.route, plan.context)
        stream_live = True
    else:  # ANSWER_DIRECT
        answer_input = prompt_input
        stream_live = False

    messages = assemble_messages(
        system_prompt, history, answer_input,
        supports_system_role=backend.supports_system_role,
    )

    meta = {"truncated": False}
    collected: List[str] = []
    for piece in stream_clean(backend, messages, meta=meta):
        collected.append(piece)
        if stream_live:
            yield _sse("token", {"text": piece})
    raw_answer = "".join(collected)

    # v1-parity: correct/wrap/extract/strip/suppress (see filegen.prepare_files)
    uploaded_names = {fc["filename"] for fc in file_contexts}
    answer, files = prepare_files(raw_answer, user_input, history, uploaded_names, deps)
    for f in files:
        yield _sse("file", _file_payload_for_client(f))

    deps.append_message(conv_id, "user", user_input)
    msg_id = deps.append_message(conv_id, "model", answer)
    if files and msg_id:
        deps.persist_files(msg_id, files)
    deps.auto_title(conv_id, user_input)
    yield from _ctx_event(user_id, conv_id, backend, deps)

    # Direct path buffered, so emit its (cleaned, single) token now; the context
    # path already streamed its tokens live.
    if not stream_live and answer:
        yield _sse("token", {"text": answer})

    yield _sse("done", {"truncated": meta["truncated"]})
    ev = deps.title_event(conv_id)
    if ev:
        yield ev
