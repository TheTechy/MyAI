"""
core/inference/config.py — backend selection & config (MyAI v2, F1 Story 1 Task 2)

Decides WHICH InferenceBackend to build and with WHAT params, read from the
environment. Dependency-free on purpose (no llama_cpp import), same as base.py,
so it loads anywhere and the registry/factory (F1 Story 3) can consume it
without dragging a concrete backend in.

THE "NO CONFIG" FAMILY PATH
---------------------------
Set MODEL and you get the in-process llama-cpp backend with the proven M4 Pro
defaults below — there is no backend-selection variable to learn. The SME/HTTP
backend (F6) later adds kind="http" plus its OWN param block WITHOUT touching any
llama-cpp field. That non-collision is the seam working as intended.

WHAT MAPS TO WHAT (grounded in v1 llm_inference.py)
---------------------------------------------------
  model_path            ← os.getenv("MODEL")
  llama_cpp.n_ctx       ← os.getenv("CTX_SIZE")   (v1: int(ctx_size))
  llama_cpp.*           ← the exact Llama(...) tuning knobs v1 ships
  warmup                ← os.getenv("WARMUP", "1") == "1"
  kind                  ← os.getenv("MYAI_BACKEND")  [NEW; defaults in-process]

Config stays deliberately dumb about which backend kinds exist — it just reads
the string. The registry (Story 3) is what rejects an unknown kind, so that
knowledge lives in exactly one place.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Optional


# ── defaults ────────────────────────────────────────────────────────────────────
DEFAULT_BACKEND = "llama_cpp"   # Family never has to set MYAI_BACKEND
DEFAULT_N_CTX   = 0             # 0 = use the MODEL'S OWN trained maximum.
                                # llama-cpp resolves 0 to n_ctx_train at load, so
                                # the no-config Family path adapts to whatever
                                # GGUF is loaded instead of guessing a number.


# ── llama-cpp tuning (verbatim from v1's proven Llama(...) on the M4 Pro) ────────
@dataclass(frozen=True)
class LlamaCppParams:
    """The in-process backend's construction params. Defaults reproduce v1's
    Llama(...) exactly, so the proven M4 Pro tuning carries over unchanged; only
    n_ctx is environment-driven in practice."""
    n_ctx: int = DEFAULT_N_CTX
    n_gpu_layers: int = -1
    n_threads: int = 6
    n_threads_batch: int = 12
    n_batch: int = 512
    n_ubatch: int = 512
    flash_attn: bool = True
    use_mmap: bool = True
    type_k: int = 8            # GGML_TYPE_Q8_0 KV-cache quant (v1 value)
    type_v: int = 8
    verbose: bool = False


# ── top-level backend config ────────────────────────────────────────────────────
@dataclass(frozen=True)
class BackendConfig:
    """Which backend to build, plus the params it needs.

    `kind` selects the backend (registry maps it to a class). `model_path` is
    common to every backend. Backend-specific params live in their OWN nested
    block (llama_cpp here) so a later HTTP backend adds an `http` block without
    disturbing anything above — the whole point of the seam.
    """
    kind: str
    model_path: str
    warmup: bool = True
    llama_cpp: Optional[LlamaCppParams] = None
    # future: http: Optional[HttpParams] = None   # added with the F6 backend


class ConfigError(ValueError):
    """Raised when required configuration is missing or malformed."""


def _get(env: Mapping[str, str], key: str) -> Optional[str]:
    v = env.get(key)
    if v is None:
        return None
    v = v.strip()
    return v or None


def load_config(env: Optional[Mapping[str, str]] = None) -> BackendConfig:
    """Build a BackendConfig from the environment.

    Pass an explicit `env` mapping for tests; defaults to os.environ. Raises
    ConfigError with an actionable message when MODEL is missing (Family still
    has to say which model to run — silently guessing a path would be worse).
    """
    env = os.environ if env is None else env

    kind = _get(env, "MYAI_BACKEND") or DEFAULT_BACKEND

    model_path = _get(env, "MODEL")
    if not model_path:
        raise ConfigError(
            "MODEL is not set — point it at a GGUF model file, e.g. "
            "MODEL=/path/to/model.gguf. This is the only variable MyAI-Family "
            "needs to boot."
        )

    warmup = (_get(env, "WARMUP") or "1") == "1"

    llama_cpp: Optional[LlamaCppParams] = None
    if kind == DEFAULT_BACKEND:
        ctx_raw = _get(env, "CTX_SIZE")
        if ctx_raw is None:
            n_ctx = DEFAULT_N_CTX
            print("[config] CTX_SIZE unset — using the model's trained maximum")
        else:
            try:
                n_ctx = int(ctx_raw)
            except ValueError:
                raise ConfigError(f"CTX_SIZE must be an integer, got {ctx_raw!r}")
            if n_ctx < 0:
                raise ConfigError(
                    f"CTX_SIZE must be 0 (= model's trained maximum) or a "
                    f"positive integer, got {n_ctx}"
                )
        llama_cpp = LlamaCppParams(n_ctx=n_ctx)

    return BackendConfig(
        kind=kind,
        model_path=model_path,
        warmup=warmup,
        llama_cpp=llama_cpp,
    )
