#!/usr/bin/env python3
"""
scripts/manage_template_fixtures.py — keep the vendored chat-template fixtures current

The cross-family templating test (tests/smoke_cross_family_templating.py) renders
against VENDORED copies of each model's chat template so it can run offline. Those
copies can go stale if a model is re-quantised. This is the (Mini-only) tool that
writes and verifies them. Needs llama_cpp + the real GGUFs, so it runs on the box
with the models — never in lightweight CI.

    # add / update a fixture from a model
    python scripts/manage_template_fixtures.py refresh ./models/Qwen3.5-27B-UD-Q8_K_XL.gguf

    # check every model against its vendored fixture (warns on drift)
    python scripts/manage_template_fixtures.py check ./models/*.gguf

To DELETE a fixture, just remove the .jinja file — the test auto-discovers what's
present, so nothing else to do.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "chat_templates"


def _read_live_template(model_path: str) -> str:
    """Load a GGUF just far enough to read its embedded chat template."""
    from llama_cpp import Llama
    # Tiny + CPU: metadata only, no real inference.
    llm = Llama(model_path=model_path, n_ctx=512, n_gpu_layers=0, verbose=False)
    try:
        return (llm.metadata or {}).get("tokenizer.chat_template", "") or ""
    finally:
        llm.close()


def _fixture_path(model_path: str) -> Path:
    return FIXTURES / f"{Path(model_path).stem}.jinja"


def cmd_refresh(models: list[str]) -> int:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    rc = 0
    for m in models:
        if not Path(m).exists():
            print(f"[skip] not found: {m}")
            rc = 1
            continue
        tmpl = _read_live_template(m)
        if not tmpl:
            print(f"[skip] no embedded chat template in {Path(m).name}")
            rc = 1
            continue
        fp = _fixture_path(m)
        existed = fp.exists()
        fp.write_text(tmpl, encoding="utf-8")
        print(f"[{'updated' if existed else 'wrote'}] {fp.name} ({len(tmpl)} chars)")
    return rc


def cmd_check(models: list[str]) -> int:
    rc = 0
    for m in models:
        fp = _fixture_path(m)
        name = Path(m).name
        if not Path(m).exists():
            print(f"[skip] model not found: {name}")
            continue
        if not fp.exists():
            print(f"[MISS] no fixture for {name} — run: refresh {m}")
            rc = 1
            continue
        live = _read_live_template(m)
        vendored = fp.read_text(encoding="utf-8")
        if live.strip() == vendored.strip():
            print(f"[ok]   {fp.name} matches the live template")
        else:
            print(f"[DRIFT] {fp.name} differs from {name}'s live template — "
                  f"run: refresh {m}")
            rc = 1
    return rc


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("refresh", "check"):
        p = sub.add_parser(c)
        p.add_argument("models", nargs="+", help="GGUF path(s)")
    args = ap.parse_args()
    return cmd_refresh(args.models) if args.cmd == "refresh" else cmd_check(args.models)


if __name__ == "__main__":
    sys.exit(main())
