"""
core/inference/base.py — the pluggable inference-backend seam (MyAI v2, F1).

WHY THIS EXISTS
---------------
v2 replaces the Gemma-shackled model core with a seam that any backend can plug
into, as long as it speaks the OpenAI-shaped chat-completions *data shape*
in-process (v2 decision #2). "OpenAI-shaped" means the message/response DATA
FORMAT only — NOT the OpenAI company, NOT their servers, NOT HTTP, NOT an API
key. The default backend runs entirely in-process via llama-cpp-python; nothing
leaves the machine.

The good news the v1 spike already banked: llm_inference.py's `_run_chat_silent`
and `_stream_chat` ALREADY call `create_chat_completion(messages=...)` and let
the model's own embedded template build the prompt. This ABC just formalises the
surface those two functions rely on, so the concrete backend (F1 Story 2) is a
thin wrapper and a future HTTP backend (F6, SME tier — NOT built in milestone 1)
slots in behind the same contract without a core rewrite.

WHAT MAPS TO WHAT (every method traces to a real v1 call site)
--------------------------------------------------------------
  create_chat_completion()  ← llm.create_chat_completion() in _run_chat_silent /
                              _stream_chat / _classify_chat
  context_window()          ← llm.n_ctx()   (the 'ctx' SSE event)
  context_used()            ← llm.n_tokens  (the 'ctx' SSE event)
  emits_think_reasoning     ← _MODEL_EMITS_THINK, sniffed from the model's
                              embedded chat template at load

DELIBERATELY NOT IN THE SEAM
----------------------------
  • save_state()/load_state() — v1's Arm A classifier-prefix KV caching. It only
    served the standalone classifier, which v2 deletes (decision #3). Forcing it
    onto every backend would burden the HTTP backend with a llama-cpp-only trick.
  • grammar=LlamaGrammar — llama-cpp-specific and un-honourable by an HTTP
    backend. Replaced by the backend-neutral OutputConstraint below.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, List, Optional, TypedDict, Union


# ── OpenAI-shaped response fragments (only the fields MyAI actually reads) ──────
# These document, for a future backend author, the EXACT access paths the
# pipeline depends on. The non-stream vs stream split is the classic porting
# gotcha the v1 code calls out in comments: content lives at a different path in
# each, and a new backend must reproduce both faithfully.
class _ChatMessageBase(TypedDict):
    role: str


class ChatMessage(_ChatMessageBase, total=False):
    # `content` is None on a turn that emits tool calls instead of text.
    # `tool_calls` is passed through UNMODIFIED — MyAI never rewrites it.
    content: Optional[str]
    tool_calls: List[dict]


class NonStreamChoice(TypedDict, total=False):
    message: ChatMessage
    finish_reason: Optional[str]


class ChatCompletion(TypedDict):
    # Non-stream answer text: choices[0]["message"]["content"]
    choices: List[NonStreamChoice]


class StreamDelta(TypedDict, total=False):
    role: str
    content: str
    tool_calls: List[dict]   # streamed incrementally; passed through unmodified


class StreamChoice(TypedDict, total=False):
    delta: StreamDelta
    finish_reason: Optional[str]


class ChatCompletionChunk(TypedDict):
    # Stream text: choices[0]["delta"]["content"]
    # Truncation:  choices[0]["finish_reason"] == "length"
    choices: List[StreamChoice]


# ── Backend-neutral output constraint ──────────────────────────────────────────
@dataclass(frozen=True)
class OutputConstraint:
    """A backend-neutral request to constrain generation to a grammar/schema.

    Used ONLY for the final routing-selection line (v2 decision #3), never for
    the free reasoning that precedes it. Each backend honours whichever field it
    can:
      • in-process llama-cpp        → `gbnf` (compiled to LlamaGrammar internally)
      • future OpenAI-shaped HTTP    → `json_schema` (structured outputs)

    A backend handed a constraint it cannot honour MUST raise, not silently emit
    unconstrained text — a silent fallback would let a malformed routing line
    through, which is exactly the brittleness v2 is removing.
    """
    gbnf: Optional[str] = None
    json_schema: Optional[dict] = None


# ── The seam ───────────────────────────────────────────────────────────────────
class InferenceBackend(ABC):
    """The single seam every MyAI v2 model call goes through.

    Contract in one sentence: given OpenAI-shaped `messages`, produce an
    OpenAI-shaped completion (or a stream of OpenAI-shaped chunks) by applying
    the MODEL'S OWN embedded chat template — never hand-built turn markers.

    `messages` items are {"role": "user"|"assistant"|"system", "content": ...}.
    `content` is a plain str today; the content-part array shape (text+image) is
    reserved for multimodal (F2 Story 3 / F6) and must not be designed out here.
    """

    # ── lifecycle ──────────────────────────────────────────────────────────────
    @abstractmethod
    def load(self) -> None:
        """Bring the backend to a ready state (load weights / open the session).

        Idempotent: load() on an already-loaded backend is a no-op. v2 makes
        this explicit so models can be swapped at runtime, replacing v1's
        import-time module-global Llama().
        """

    @abstractmethod
    def unload(self) -> None:
        """Release the model and its resources.

        After unload(), load() must be callable again — a model swap is
        unload-old then load-new.
        """

    # ── the one generation call ─────────────────────────────────────────────────
    @abstractmethod
    def create_chat_completion(
        self,
        messages: List[dict],
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
        stream: bool = False,
        constraint: Optional[OutputConstraint] = None,
        tools: Optional[List[dict]] = None,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        """Generate a chat completion from OpenAI-shaped `messages`.

        The backend applies the model's OWN embedded chat template to turn
        `messages` into a prompt (no hand-built <start_of_turn> markers).

        stream=False → returns one ChatCompletion; text at
            choices[0]["message"]["content"].
        stream=True  → returns an iterator of ChatCompletionChunk; incremental
            text at choices[0]["delta"]["content"], and a final
            choices[0]["finish_reason"] == "length" signals truncation.

        `constraint`, when given, restricts output to a grammar/schema. Callers
        apply it ONLY to the final routing-selection line, never the reasoning.

        `tools` is an OpenAI-shaped tool/function schema array, forwarded to the
        model UNMODIFIED. Any resulting `tool_calls` are returned UNMODIFIED at
        choices[0]["message"]["tool_calls"] (or, streaming, in the delta), with
        finish_reason == "tool_calls". MyAI does not parse, rewrite, or invent
        this shape — that passthrough is what lets local tool-calling/actions
        land later without touching the seam. Nothing in v2's first milestone
        passes `tools`; backends must simply not strip it.
        """

    # ── introspection the pipeline needs ────────────────────────────────────────
    @property
    @abstractmethod
    def emits_think_reasoning(self) -> bool:
        """True if the loaded model's chat template opens a <think> reasoning
        block that must be stripped from output.

        Decided once, from the model's embedded template — not guessed from the
        filename (robust across renamed/re-quantised GGUFs). Gates MyAI's
        reasoning-strip so it is a HARD no-op on non-thinking models like Gemma.
        """

    @abstractmethod
    def context_window(self) -> int:
        """Maximum context length in tokens (v1's llm.n_ctx())."""

    @abstractmethod
    def context_used(self) -> Optional[int]:
        """Tokens currently resident in context, for the 'ctx' status event
        (v1's llm.n_tokens).

        Return None when the backend cannot report it (an HTTP backend generally
        cannot); callers degrade the ctx readout gracefully rather than failing.
        """

    # ── optional, non-abstract ──────────────────────────────────────────────────
    @property
    def supports_system_role(self) -> bool:
        """Whether the loaded model's chat template honours a {"role":"system"}
        message at position 0.

        Drives the assembler: True → emit a proper system message; False → fuse
        the instruction into the latest user turn instead. Default True — a
        backend speaking clean OpenAI semantics (the future HTTP backend) supports
        system by definition. Only a backend facing arbitrary local GGUF templates
        needs to detect this and possibly override to False. Concrete, not
        abstract, so it breaks no existing backend.
        """
        return True

    def trained_context_window(self) -> Optional[int]:
        """The context length the MODEL WAS TRAINED FOR, if the backend can tell.

        Distinct from context_window(), which reports what was actually
        REQUESTED/allocated. The two differ when configuration asks for more (or
        less) than the model was trained on; asking for more is a legitimate
        experiment (RoPE scaling and friends) but degrades quality silently, so
        callers surface the discrepancy rather than guessing.

        Default None — a remote backend generally cannot know. Concrete, not
        abstract, so adding it breaks no existing backend.
        """
        return None

    def warmup(self) -> bool:
        """Optional: page weights in / JIT kernels so the first real prompt isn't
        slow. Default is a no-op returning False (not warmed). The in-process
        backend overrides this; an HTTP backend may leave it as-is.

        NOTE: v1's warmup also primed the deleted classifier prefix — that half
        does NOT carry over.
        """
        return False
