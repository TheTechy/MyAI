"""
core/inference/streaming.py — reasoning strip + clean streaming (MyAI v2, F2 S2)

Ports v1's _strip_reasoning / _stream_chat off the module-global `llm` and onto
an InferenceBackend, and closes the truncation-leak that lived in v1's stream
flush.

WHAT THIS HANDLES (and what it deliberately doesn't)
----------------------------------------------------
Thinking models (Qwen 3.5, etc.) wrap deliberation in <think>…</think> that must
never reach the user. Two shapes occur:
  1. Full pair:   <think> … </think> … answer
  2. Opener-less: … reasoning … </think> … answer
     Qwen's template injects the OPENING <think> into the prompt, so the
     completion carries only the reasoning + a CLOSING </think>. Observed on
     Qwen 3.5.

Only <think> is handled — NOT Gemma's <|channel>thought markers. Gemma's thinking
is off by default in this deployment, so its channel reasoning never appears; if
it is ever enabled, add a channel closer alongside the </think> logic (a one-line
change in both functions). Gated by backend.emits_think_reasoning, so on a
non-thinking model every path here is a hard no-op.

No hard-coded stop tokens: v1's Gemma/channel-specific RESPONSE_TOKENS list is
dropped. create_chat_completion + the model's embedded template stop on the
model's own eos, which is correct for every model, not just Gemma.

THE TRUNCATION GUARD (the bug this fixes)
-----------------------------------------
If a thinking model hits the token limit BEFORE emitting </think>, the buffered
text is unclosed REASONING, not an answer. v1's stream flush yielded it anyway
("an answer with no reasoning block") — which leaks raw chain-of-thought on a
truncated Qwen turn. Here: a still-open buffer is flushed ONLY on a natural stop
(the genuine no-reasoning-answer case); on a length-truncation it is SUPPRESSED
and the turn is marked truncated. An empty-but-flagged reply beats leaking the
model's private reasoning. The same rule guards the non-streamed cleaner.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterator, List, Optional

_THINK_CLOSE = "</think>"

# Full <think>…</think> pairs, and opener-less leading reasoning up to the first
# </think>. DOTALL so they span newlines; IGNORECASE for tag-case variance.
_REASONING_PAIRED_RE = re.compile(r"<think>.*?</think>\s*", re.IGNORECASE | re.DOTALL)
_REASONING_LEADING_RE = re.compile(r"^\s*.*?</think>\s*", re.IGNORECASE | re.DOTALL)

# Belt-and-braces: strip stray Gemma control tokens that might leak into finalised
# text. Largely redundant now (create_chat_completion decodes without special
# tokens), kept as cheap defence.
_CONTROL_TOKEN_RE = re.compile(r"<(eos|end_of_turn|start_of_turn|/?turn)>", re.IGNORECASE)


def strip_reasoning(text: str, emits_think: bool) -> str:
    """Remove <think> reasoning from FINALISED text. No-op when the model doesn't
    think or the text is empty. Handles both the full-pair and opener-less shapes.
    """
    if not text or not emits_think:
        return text
    text = _REASONING_PAIRED_RE.sub("", text)    # full <think>…</think>
    text = _REASONING_LEADING_RE.sub("", text)   # opener-less leading reasoning
    return text


def clean_text(text: str, emits_think: bool, *, truncated: bool = False) -> str:
    """Clean a FINALISED (non-streamed) completion for display.

    Truncation guard: if the model thinks, the turn was truncated, and there is
    no </think> in the text, then the whole text is unclosed reasoning — suppress
    it (return "") rather than leak it. Otherwise strip reasoning + stray control
    tokens as normal.
    """
    if emits_think and truncated and _THINK_CLOSE not in text:
        return ""   # cut-off reasoning with no closer — never leak it
    text = strip_reasoning(text, emits_think)
    text = _CONTROL_TOKEN_RE.sub("", text)
    return text.strip()


def stream_clean(
    backend: Any,
    messages: List[dict],
    *,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Iterator[str]:
    """Stream CLEANED text pieces from `backend`, suppressing reasoning.

    For a thinking model, buffers until the first </think> (so a closer split
    across chunk boundaries — "</thi" + "nk>" — is still caught), then streams the
    answer through. For a non-thinking model it's a straight passthrough. Reads
    the OpenAI streaming delta shape (choices[0].delta.content).

    If `meta` is given, sets meta["truncated"] = True when the model stops on
    length. On a length-truncation while still inside the reasoning block, the
    buffered reasoning is SUPPRESSED, not yielded.
    """
    thinking = backend.emits_think_reasoning
    buffer = ""
    started = False                    # have we emitted the first visible piece yet?
    finish: Optional[str] = None

    for chunk in backend.create_chat_completion(
        messages, stream=True, temperature=temperature, max_tokens=max_tokens
    ):
        choice = chunk["choices"][0]
        fr = choice.get("finish_reason")
        if fr:
            finish = fr
            if fr == "length" and meta is not None:
                meta["truncated"] = True

        piece = (choice.get("delta") or {}).get("content", "") or ""
        if not piece:
            continue

        if thinking:
            buffer += piece
            idx = buffer.find(_THINK_CLOSE)
            if idx == -1:
                continue                      # still inside the reasoning block
            piece = buffer[idx + len(_THINK_CLOSE):]   # answer text after the closer
            thinking = False
            buffer = ""
            if not piece:
                continue                      # answer arrives in a later delta

        # Emit path: the post-reasoning answer, or a non-thinking passthrough.
        # Trim leading whitespace ONLY until the first visible piece, so the blank
        # line a template leaves between </think> and the answer never shows —
        # while whitespace INSIDE the answer is left untouched.
        if not started:
            piece = piece.lstrip()
            if not piece:
                continue
            started = True
        yield piece

    # End-of-stream flush decision.
    if thinking and buffer:
        if finish == "length":
            # Buffer is truncated reasoning — NEVER leak it.
            if meta is not None:
                meta["truncated"] = True
        else:
            # Natural stop with no </think> ever seen: assume it was an answer
            # that carried no reasoning block, and flush it (v1's rescue).
            flushed = buffer.lstrip() if not started else buffer
            if flushed:
                yield flushed
