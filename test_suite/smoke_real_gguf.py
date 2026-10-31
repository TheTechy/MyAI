#!/usr/bin/env python3
"""
scripts/smoke_real_gguf.py — F1 Story 2 Task 3

Everything up to now was verified against a FAKE mimicking llama-cpp-python
0.3.33's API. That proves the wiring; it does not prove a real 31B GGUF loads and
completes. This script is the real thing, and it must run on the Mac Mini.

USAGE
    python scripts/smoke_real_gguf.py                     # model from .env
    python scripts/smoke_real_gguf.py --model /path/x.gguf
    python scripts/smoke_real_gguf.py --model /path/x.gguf --ctx 0   # trained max
    python scripts/smoke_real_gguf.py --skip-tools        # skip the tools probe

Run it once PER spike model (Gemma 3 27B, Gemma 4 12B, Gemma 4 31B, Qwen 3.5 27B).
Expect exactly one model — the Qwen — to report `emits <think> reasoning: True`.

Exit code 0 = all required checks passed. Non-zero = something is genuinely wrong.
The tools probe is advisory only and never fails the run: small local models are
inconsistent at emitting tool_calls, and that is a MODEL property, not a backend
bug. The backend's job is only to not strip the field, which is already unit-tested.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Make `core.inference` importable when run from anywhere in the repo.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.inference.base import OutputConstraint          # noqa: E402
from core.inference.config import load_config             # noqa: E402
from core.inference.llama_cpp_backend import (            # noqa: E402
    ConstraintError,
    LlamaCppBackend,
)

PASS, FAIL, INFO, WARN = "PASS", "FAIL", "····", "WARN"
_failures: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> bool:
    tag = PASS if ok else FAIL
    print(f"  [{tag}] {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        _failures.append(label)
    return ok


def info(label: str, detail: str = "") -> None:
    print(f"  [{INFO}] {label}" + (f" — {detail}" if detail else ""))


def warn(label: str, detail: str = "") -> None:
    print(f"  [{WARN}] {label}" + (f" — {detail}" if detail else ""))


def section(title: str) -> None:
    print(f"\n─── {title} " + "─" * max(0, 60 - len(title)))


def load_dotenv_if_present() -> None:
    """Populate os.environ from .env without requiring python-dotenv."""
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)   # real env wins over .env
    print(f"[env] loaded {env_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="GGUF path (overrides MODEL from .env)")
    ap.add_argument("--ctx", help="context size; 0 = the model's trained maximum")
    ap.add_argument("--skip-tools", action="store_true")
    args = ap.parse_args()

    load_dotenv_if_present()
    if args.model:
        os.environ["MODEL"] = args.model
    if args.ctx is not None:
        os.environ["CTX_SIZE"] = args.ctx

    cfg = load_config()
    model_name = Path(cfg.model_path).name

    print("=" * 68)
    print(f"MyAI v2 — real GGUF smoke test")
    print(f"model : {model_name}")
    print(f"path  : {cfg.model_path}")
    print(f"ctx   : {cfg.llama_cpp.n_ctx} (0 = trained maximum)")
    print("=" * 68)

    if not Path(cfg.model_path).exists():
        print(f"\n[{FAIL}] model file not found: {cfg.model_path}")
        return 2

    backend = LlamaCppBackend(cfg)

    # ── 1. load ────────────────────────────────────────────────────────────────
    section("1. load")
    t0 = time.time()
    backend.load()
    load_s = time.time() - t0
    check(backend.is_loaded, "model loaded", f"{load_s:.1f}s")

    allocated = backend.context_window()
    trained = backend.trained_context_window()
    check(allocated > 0, "context_window() positive", f"{allocated} allocated")
    if trained is None:
        warn("trained_context_window() unreadable",
             "not fatal — this build may not expose it")
    else:
        info("trained context", f"{trained}")
        if allocated > trained:
            warn("allocated exceeds trained", "expect quality degradation past "
                 f"{trained} tokens")

    thinks = backend.emits_think_reasoning
    info("emits <think> reasoning", str(thinks))
    info("expectation", "True for Qwen-style models, False for Gemma")

    # ── 2. warmup ──────────────────────────────────────────────────────────────
    section("2. warmup")
    t0 = time.time()
    warmed = backend.warmup()
    check(warmed is True, "warmup ran", f"{time.time() - t0:.1f}s")

    # ── 3. non-stream completion ───────────────────────────────────────────────
    section("3. non-stream completion")
    messages = [{"role": "user", "content": "Reply with exactly: hello"}]
    t0 = time.time()
    out = backend.create_chat_completion(messages, max_tokens=32, temperature=0.0)
    dt = time.time() - t0

    ok = check(isinstance(out, dict) and "choices" in out, "returned a dict with choices")
    if ok:
        choice = out["choices"][0]
        content = choice.get("message", {}).get("content")
        check(isinstance(content, str) and content.strip() != "",
              "content is a non-empty string at choices[0].message.content",
              f"{len(content or '')} chars in {dt:.1f}s")
        info("finish_reason", str(choice.get("finish_reason")))
        info("reply", repr((content or "")[:120]))

    # ── 4. streamed completion ─────────────────────────────────────────────────
    section("4. streamed completion")
    t0 = time.time()
    stream = backend.create_chat_completion(
        [{"role": "user", "content": "Count from 1 to 5, numbers only."}],
        max_tokens=64, temperature=0.0, stream=True,
    )
    pieces, chunk_count, finish = [], 0, None
    for chunk in stream:
        chunk_count += 1
        ch = chunk["choices"][0]
        piece = ch.get("delta", {}).get("content")
        if piece:
            pieces.append(piece)
        if ch.get("finish_reason"):
            finish = ch["finish_reason"]
    dt = time.time() - t0
    text = "".join(pieces)

    check(chunk_count > 1, "received multiple chunks", f"{chunk_count} chunks")
    check(text.strip() != "",
          "text accumulated from choices[0].delta.content",
          f"{len(text)} chars in {dt:.1f}s")
    info("finish_reason", str(finish))
    info("streamed", repr(text[:120]))

    # ── 5. context accounting ──────────────────────────────────────────────────
    section("5. context accounting")
    used = backend.context_used()
    check(used is not None and used > 0,
          "context_used() reports resident tokens", f"{used} / {allocated}")

    # ── 6. grammar constraint (this is what routing depends on) ────────────────
    section("6. grammar-constrained decoding")
    gbnf = 'root ::= "direct" | "web_search" | "calculator"'
    try:
        out = backend.create_chat_completion(
            [{"role": "user", "content":
              "Reply with one word only: which tool suits 'what is 12 * 9'?"}],
            max_tokens=8, temperature=0.0,
            constraint=OutputConstraint(gbnf=gbnf),
        )
        picked = (out["choices"][0]["message"]["content"] or "").strip()
        check(picked in {"direct", "web_search", "calculator"},
              "GBNF constrained output to the allowed set", repr(picked))
    except Exception as err:                       # noqa: BLE001
        check(False, "GBNF constraint", f"{type(err).__name__}: {err}")

    # JSON-schema path (0.3.33 converts schema → GBNF internally)
    try:
        out = backend.create_chat_completion(
            [{"role": "user", "content": "Reply in JSON with a 'route' field."}],
            max_tokens=48, temperature=0.0,
            constraint=OutputConstraint(json_schema={
                "type": "object",
                "properties": {"route": {"type": "string"}},
                "required": ["route"],
            }),
        )
        raw = (out["choices"][0]["message"]["content"] or "").strip()
        import json as _json
        parsed = _json.loads(raw)
        check(isinstance(parsed, dict) and "route" in parsed,
              "json_schema constraint produced valid matching JSON", repr(raw[:80]))
    except Exception as err:                       # noqa: BLE001
        check(False, "json_schema constraint", f"{type(err).__name__}: {err}")

    # The rule that protects routing: an empty constraint must RAISE.
    try:
        backend.create_chat_completion([{"role": "user", "content": "hi"}],
                                       max_tokens=4,
                                       constraint=OutputConstraint())
        check(False, "empty constraint raises", "it did not raise")
    except ConstraintError:
        check(True, "empty constraint raises instead of silently unconstrained")

    # ── 7. tools passthrough (advisory) ────────────────────────────────────────
    if not args.skip_tools:
        section("7. tools passthrough (advisory — never fails the run)")
        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }]
        try:
            out = backend.create_chat_completion(
                [{"role": "user", "content": "What's the weather in Manchester?"}],
                max_tokens=128, temperature=0.0, tools=tools,
            )
            choice = out["choices"][0]
            calls = choice.get("message", {}).get("tool_calls")
            if calls:
                info("model emitted tool_calls", str(calls)[:160])
                info("finish_reason", str(choice.get("finish_reason")))
            else:
                warn("model answered in prose rather than calling the tool",
                     "a model behaviour, not a backend fault")
        except Exception as err:                   # noqa: BLE001
            warn("tools call errored", f"{type(err).__name__}: {err}")

    # ── 8. unload / reload — the model-swap path ───────────────────────────────
    section("8. unload / reload (model swap)")
    backend.unload()
    check(not backend.is_loaded, "unloaded")
    backend.unload()
    check(True, "second unload is safe")
    t0 = time.time()
    backend.load()
    check(backend.is_loaded and backend.context_window() == allocated,
          "reloaded with the same context window", f"{time.time() - t0:.1f}s")
    backend.unload()

    # ── verdict ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    if _failures:
        print(f"FAILED ({len(_failures)}): " + "; ".join(_failures))
        print("=" * 68)
        return 1
    print(f"ALL CHECKS PASSED — {model_name}")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
