"""
core/inference/llama_cpp_backend.py — the default in-process backend
(MyAI v2, F1 Story 2 Task 1)

Implements InferenceBackend against llama-cpp-python, in-process. No server, no
HTTP, no API key — the model is loaded into this process and nothing leaves the
machine. This is the zero-dependency backend MyAI-Family ships on.

Verified against llama-cpp-python 0.3.33 source (the installed version):
  • Llama.create_chat_completion(..., grammar=LlamaGrammar, stream=, stop=,
    max_tokens=, temperature=)                            — grammar still accepted
  • LlamaGrammar.from_string(gbnf, verbose=) / .from_json_schema(json_str, verbose=)
  • Llama.n_ctx() -> int          → context_window()
  • Llama.n_tokens (attribute)    → context_used()
  • Llama.metadata["tokenizer.chat_template"] → emits_think_reasoning
  • Llama.close()                 → unload()
  • type_k/type_v are Optional[int]; verbose defaults True (we pass False)

NOTE on flash_attn: 0.3.33 still accepts the bool, but internally maps it onto
context_params.flash_attn_type (llama.cpp moved to an enum). Works today; expect
the bool to be deprecated eventually.

llama_cpp is imported LAZILY inside load() so this module stays importable — and
unit-testable — on a machine with no llama_cpp installed, matching base.py and
config.py.
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, Iterator, List, Optional, Union

from .base import (
    ChatCompletion,
    ChatCompletionChunk,
    InferenceBackend,
    OutputConstraint,
)
from .config import BackendConfig, LlamaCppParams


class BackendNotLoadedError(RuntimeError):
    """Raised when a generation/introspection call arrives before load()."""


class ConstraintError(ValueError):
    """Raised when an OutputConstraint cannot be honoured.

    Per ADR-0001 a backend MUST raise rather than silently emit unconstrained
    text — a silent fallback would let a malformed routing selection through,
    which is the exact brittleness v2 exists to remove.
    """


class LlamaCppBackend(InferenceBackend):
    """In-process llama-cpp-python backend (the Family default)."""

    def __init__(self, config: BackendConfig) -> None:
        if config.llama_cpp is None:
            raise ValueError(
                "BackendConfig.llama_cpp params are required for LlamaCppBackend"
            )
        self._config: BackendConfig = config
        self._params: LlamaCppParams = config.llama_cpp
        self._llm: Optional[Any] = None          # llama_cpp.Llama once loaded
        self._emits_think: bool = False
        self._trained_ctx: Optional[int] = None
        self._supports_system_role: bool = True

    # ── lifecycle ──────────────────────────────────────────────────────────────
    @property
    def is_loaded(self) -> bool:
        return self._llm is not None

    def load(self) -> None:
        """Load the model. Idempotent — a second call is a no-op.

        Replaces v1's import-time module-global Llama(), so a model swap is now
        an explicit unload()/load() rather than a process restart.
        """
        if self._llm is not None:
            return

        from llama_cpp import Llama  # lazy: keeps this module import-light

        p = self._params
        self._llm = Llama(
            model_path=self._config.model_path,
            n_ctx=p.n_ctx,
            n_gpu_layers=p.n_gpu_layers,
            verbose=p.verbose,
            n_threads=p.n_threads,
            n_threads_batch=p.n_threads_batch,
            n_batch=p.n_batch,
            n_ubatch=p.n_ubatch,
            flash_attn=p.flash_attn,
            use_mmap=p.use_mmap,
            type_k=p.type_k,
            type_v=p.type_v,
        )

        # Decide ONCE, from the model's OWN embedded chat template, whether it
        # emits <think> reasoning. Sniffing the template rather than guessing
        # from the filename stays robust across renamed / re-quantised GGUFs.
        # Predicate reproduced verbatim from v1.
        metadata = getattr(self._llm, "metadata", None) or {}
        template = (metadata.get("tokenizer.chat_template", "") or "")
        self._emits_think = "<think>" in template.lower()
        print(f"[llama_cpp] model emits <think> reasoning: {self._emits_think}")

        # Does this model's template honour a system role? Decided ONCE at load
        # by rendering the model's OWN template (via llama-cpp's own formatter, so
        # it matches what create_chat_completion will actually do) with a system
        # sentinel. Drives whether the assembler emits a system message or fuses.
        self._supports_system_role = self._detect_system_role_support(template)
        print(f"[llama_cpp] model supports system role: {self._supports_system_role}")

        # Context window: what the MODEL was trained for vs what we ASKED for.
        self._trained_ctx = self._read_trained_ctx()
        allocated = int(self._llm.n_ctx())
        if self._trained_ctx:
            print(
                f"[llama_cpp] context: {allocated} allocated / "
                f"{self._trained_ctx} trained"
            )
            if allocated > self._trained_ctx:
                # Deliberately a WARNING, not an error: running past the trained
                # length is a valid experiment, it just degrades quality quietly.
                # Refusing to boot would block that; saying nothing hides it.
                print(
                    f"[llama_cpp] WARNING: CTX_SIZE={allocated} exceeds this "
                    f"model's trained context of {self._trained_ctx} — output "
                    f"quality may degrade beyond that point."
                )
        else:
            print(f"[llama_cpp] context: {allocated} allocated / trained unknown")

    def _read_trained_ctx(self) -> Optional[int]:
        """Trained context length, or None if it can't be determined.

        Preferred path is llama.cpp's own architecture-independent accessor, so
        this works across Gemma/Qwen/whatever without per-family metadata keys.
        Falls back to the internal wrapper, then to the metadata dict, then
        gives up quietly — an unreadable context length must never stop a model
        from loading.
        """
        llm = self._llm
        try:
            import llama_cpp as _llama_cpp
            return int(_llama_cpp.llama_model_n_ctx_train(llm.model))
        except Exception:
            pass
        try:
            return int(llm._model.n_ctx_train())
        except Exception:
            pass
        try:
            metadata = getattr(llm, "metadata", None) or {}
            arch = metadata.get("general.architecture", "")
            if arch:
                value = metadata.get(f"{arch}.context_length")
                if value is not None:
                    return int(value)
        except Exception:
            pass
        return None

    # System-role probe sentinel: plain ASCII alnum so no template transforms it.
    _SYS_ROLE_SENTINEL = "SYSROLEPROBE7F3A9C"

    def _token_text(self, token_id: int) -> str:
        """Best-effort string for a special token id; '' if it can't decode.

        eos/bos barely affect system-role detection (they only appear where the
        template inserts them literally), so an empty fallback is harmless.
        """
        try:
            return self._llm.detokenize([token_id]).decode("utf-8", errors="ignore")
        except Exception:
            return ""

    def _detect_system_role_support(self, template: str) -> bool:
        """Render the model's OWN template with a system sentinel to decide
        whether a {"role":"system"} message is honoured.

        Uses llama-cpp's Jinja2ChatFormatter — the SAME renderer
        create_chat_completion uses — so detection matches real behaviour.

        Returns:
          True  — sentinel survives into the rendered prompt (real system turn).
          False — template raises on a system message, or silently drops it;
                  either way the assembler should fuse instead.
          True  — safe modern default when there's no template to probe or the
                  formatter can't be imported (can't prove absence of support).

        Known limitation: a template that SILENTLY MERGES system into the first
        user turn reports True (sentinel present, just repositioned). The
        instruction still reaches the model; the only cost is v1's long-chat
        drift. No current model does this, so it's documented, not guarded.
        """
        if not template:
            return True
        try:
            from llama_cpp.llama_chat_format import Jinja2ChatFormatter
        except Exception:
            return True

        eos = self._token_text(self._llm.token_eos())
        bos = self._token_text(self._llm.token_bos())
        probe = [
            {"role": "system", "content": self._SYS_ROLE_SENTINEL},
            {"role": "user", "content": "hi"},
        ]
        try:
            fmt = Jinja2ChatFormatter(template=template, eos_token=eos, bos_token=bos)
            rendered = fmt(messages=probe).prompt
        except Exception:
            return False   # template rejects a system message
        return self._SYS_ROLE_SENTINEL in rendered

    def unload(self) -> None:
        """Free the model. Safe to call when not loaded; load() works again after."""
        if self._llm is None:
            return
        try:
            self._llm.close()
        finally:
            self._llm = None
            self._emits_think = False
            self._trained_ctx = None
            self._supports_system_role = True

    def _require(self) -> Any:
        if self._llm is None:
            raise BackendNotLoadedError(
                "LlamaCppBackend.load() must be called before use"
            )
        return self._llm

    # ── constraint translation ─────────────────────────────────────────────────
    def _to_grammar(self, constraint: OutputConstraint) -> Any:
        """Translate a backend-neutral OutputConstraint into a LlamaGrammar.

        Honours BOTH fields: `gbnf` directly, and `json_schema` via llama-cpp's
        own schema→GBNF conversion. Raises when neither is usable — never falls
        back to unconstrained generation.
        """
        from llama_cpp import LlamaGrammar  # lazy, as above

        if constraint.gbnf:
            return LlamaGrammar.from_string(constraint.gbnf, verbose=False)

        if constraint.json_schema is not None:
            try:
                schema_str = json.dumps(constraint.json_schema)
            except (TypeError, ValueError) as err:
                raise ConstraintError(
                    f"json_schema is not JSON-serialisable: {err}"
                ) from err
            return LlamaGrammar.from_json_schema(schema_str, verbose=False)

        raise ConstraintError(
            "OutputConstraint carried neither gbnf nor json_schema — refusing to "
            "generate unconstrained output where a constraint was requested"
        )

    # ── the one generation call ────────────────────────────────────────────────
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
        """Forward to llama-cpp, letting the model's own chat template build the
        prompt. Returns the OpenAI-shaped result unmodified.

        Optional args are only forwarded when set, so llama-cpp's own defaults
        (temperature=0.2, stop=[]) apply rather than being overwritten with None.
        """
        llm = self._require()

        kwargs: Dict[str, Any] = {"messages": messages, "stream": stream}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if temperature is not None:
            kwargs["temperature"] = temperature
        if stop is not None:
            kwargs["stop"] = stop
        if constraint is not None:
            kwargs["grammar"] = self._to_grammar(constraint)
        if tools is not None:
            # Verified present in llama-cpp-python 0.3.33's signature. Forwarded
            # untouched; any tool_calls come back untouched.
            kwargs["tools"] = tools

        return llm.create_chat_completion(**kwargs)

    # ── introspection ──────────────────────────────────────────────────────────
    @property
    def emits_think_reasoning(self) -> bool:
        """Whether the loaded model wraps deliberation in <think> blocks.

        False when nothing is loaded — the reasoning strip is then a hard no-op,
        which is the safe direction.
        """
        return self._emits_think

    @property
    def supports_system_role(self) -> bool:
        """Whether the loaded model's template honours a system role (detected at
        load). False → the assembler should fuse. False when nothing is loaded —
        the caller then can't have assembled with a system message anyway."""
        return self._supports_system_role

    def context_window(self) -> int:
        return int(self._require().n_ctx())

    def context_used(self) -> Optional[int]:
        """Tokens resident in context, for the 'ctx' status event.

        Returns None rather than raising if llama-cpp ever stops exposing
        n_tokens — the ctx readout degrades, the chat does not break.
        """
        n = getattr(self._require(), "n_tokens", None)
        return int(n) if n is not None else None

    def trained_context_window(self) -> Optional[int]:
        """Context length this model was trained for, read at load. None if the
        build doesn't expose it."""
        self._require()
        return self._trained_ctx

    # ── optional ───────────────────────────────────────────────────────────────
    def warmup(self) -> bool:
        """Page weights in / JIT kernels so the first real prompt isn't slow.

        v1's warmup also primed the standalone classifier prefix; that half does
        NOT carry over — the classifier is deleted in v2.
        """
        if not self._config.warmup:
            return False
        llm = self._require()
        t0 = time.time()
        llm.create_chat_completion(
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
        )
        print(f"[llama_cpp] warmup complete in {time.time() - t0:.1f}s")
        return True
