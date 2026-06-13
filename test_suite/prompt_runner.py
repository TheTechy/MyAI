#!/usr/bin/env python3
"""
test_suite/prompt_runner.py
===========================
Live-server integration runner — sends every prompt from TEST_PROMPTS.md
to a running MyAI server and reports a pass/fail result for each one.

This script complements the pytest unit tests by exercising the full stack
(real model, real skills, real network) against human-authored prompts.

Usage
-----
    # 1. Start the MyAI server first.
    # 2. Run:
    python test_suite/prompt_runner.py --url http://localhost:8080 --user Alice --pin 1234

    # Run only specific sets:
    python test_suite/prompt_runner.py ... --sets "Calculator,Weather,Currency"

    # Slow down between prompts (default 1 s):
    python test_suite/prompt_runner.py ... --delay 2.0

Exit code
---------
    0  — all prompts passed
    1  — one or more prompts failed

Pass / Fail criteria
--------------------
  PASS: response stream ends with a 'done' event and contains either
        token text or a 'file' event (and no 'error' event).
  FAIL: an 'error' event was received, the stream timed out, or a
        set-specific assertion failed (see _validate()).

Set-specific checks
-------------------
  File Generation  — response must include a 'file' event
  Calculator       — token text must contain at least one digit
  Weather          — token text must be non-empty
  Currency         — token text must be non-empty
  Web Search       — at least one status message must mention "search"
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import requests
except ImportError:
    print("ERROR: 'requests' is not installed.  Run: pip install requests")
    sys.exit(1)


# ── Data structures ────────────────────────────────────────────────────────────
@dataclass
class Prompt:
    set_name: str
    text: str
    turn: Optional[int] = None


@dataclass
class SSEResult:
    events:   list[dict] = field(default_factory=list)
    done:     bool       = False
    error:    bool       = False
    statuses: list[str]  = field(default_factory=list)
    has_file: bool       = False
    tokens:   list[str]  = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "".join(self.tokens)


# ── TEST_PROMPTS.md parser ─────────────────────────────────────────────────────
def parse_prompts(md_path: Path) -> list[Prompt]:
    """Parse TEST_PROMPTS.md into a flat list of Prompt objects."""
    text = md_path.read_text(encoding="utf-8")
    prompts: list[Prompt] = []
    current_set = "Unknown"
    current_turn: Optional[int] = None
    buffer: list[str] = []

    set_re    = re.compile(r'^###\s+SET\s+\d+\s*[-–]\s*(.+)', re.IGNORECASE)
    turn_re   = re.compile(r'\(Turn\s+(\d+)\)', re.IGNORECASE)
    bullet_re = re.compile(r'^\s*[\*\-]\s+')
    goal_re   = re.compile(r'^\*Goal:', re.IGNORECASE)

    def flush(lines: list[str], set_name: str, turn: Optional[int]):
        # Filter out meta-lines (Turn markers, *Goal: …)
        content_lines = [
            l for l in lines
            if not turn_re.search(l) and not goal_re.match(l.strip())
        ]
        raw = "\n".join(content_lines).strip()
        # Strip leading bullet characters
        raw = bullet_re.sub("", raw).strip()
        if raw:
            prompts.append(Prompt(set_name=set_name, text=raw, turn=turn))

    for line in text.splitlines():
        if set_re.match(line):
            if buffer:
                flush(buffer, current_set, current_turn)
                buffer = []
                current_turn = None
            current_set = set_re.match(line).group(1).strip()
            continue

        if line.strip() == "---":
            if buffer:
                flush(buffer, current_set, current_turn)
                buffer = []
                current_turn = None
            continue

        tm = turn_re.search(line)
        if tm:
            if buffer:
                flush(buffer, current_set, current_turn)
                buffer = []
            current_turn = int(tm.group(1))
            continue

        buffer.append(line)

    if buffer:
        flush(buffer, current_set, current_turn)

    return prompts


# ── SSE collector ──────────────────────────────────────────────────────────────
def send_prompt(
    session: "requests.Session",
    base_url: str,
    prompt_text: str,
    user_id: str,
    conv_id: Optional[str],
    timeout: int,
) -> tuple[SSEResult, Optional[str]]:
    """Stream a prompt and return (SSEResult, updated_conv_id)."""
    payload: dict = {"prompt": prompt_text, "name": user_id}
    if conv_id:
        payload["conv_id"] = conv_id

    result   = SSEResult()
    new_conv = conv_id
    current_event = "message"

    try:
        with session.post(
            f"{base_url}/prompt",
            json=payload,
            stream=True,
            timeout=timeout,
        ) as resp:
            if resp.status_code == 401:
                result.error = True
                result.events.append({"event": "http_error", "status": 401, "detail": "Unauthorized"})
                return result, new_conv

            if resp.status_code != 200:
                result.error = True
                result.events.append({"event": "http_error", "status": resp.status_code})
                return result, new_conv

            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                if raw_line.startswith("event:"):
                    current_event = raw_line.split(":", 1)[1].strip()
                elif raw_line.startswith("data:"):
                    try:
                        data = json.loads(raw_line.split(":", 1)[1].strip())
                    except json.JSONDecodeError:
                        data = {}

                    result.events.append({"event": current_event, "data": data})

                    if current_event == "conv_id":
                        new_conv = data.get("conv_id", new_conv)
                    elif current_event == "status":
                        result.statuses.append(data.get("message", ""))
                    elif current_event == "token":
                        result.tokens.append(data.get("text", ""))
                    elif current_event == "done":
                        result.done = True
                    elif current_event == "error":
                        result.error = True
                    elif current_event == "file":
                        result.has_file = True

    except requests.exceptions.Timeout:
        result.error = True
        result.events.append({"event": "timeout"})
    except Exception as exc:
        result.error = True
        result.events.append({"event": "exception", "detail": str(exc)})

    return result, new_conv


# ── Validation ─────────────────────────────────────────────────────────────────
def validate(prompt: Prompt, result: SSEResult) -> tuple[bool, str]:
    """Return (passed, reason_string)."""
    if result.error:
        err = next((e for e in result.events if e.get("event") in ("error", "http_error", "timeout", "exception")), {})
        return False, f"Error event: {err}"
    if not result.done:
        return False, "Stream ended without a 'done' event"
    if not result.full_text and not result.has_file:
        return False, "No content: no tokens and no file event"

    s = prompt.set_name.lower()

    if "file generation" in s:
        if not result.has_file:
            return False, "Expected a file event — none received"

    if "calculator" in s:
        if not any(ch.isdigit() for ch in result.full_text):
            return False, "Calculator response contains no digits"

    if "web search" in s:
        if not any("search" in st.lower() for st in result.statuses):
            return False, "No 'search' status message seen"

    return True, "OK"


# ── Console output ─────────────────────────────────────────────────────────────
RESET  = "\033[0m"
GREEN  = "\033[32m"
RED    = "\033[31m"
YELLOW = "\033[33m"
BOLD   = "\033[1m"


def print_result(
    prompt: Prompt,
    result: SSEResult,
    passed: bool,
    reason: str,
    index: int,
):
    badge   = f"{GREEN}✓{RESET}" if passed else f"{RED}✗{RESET}"
    routing = " › ".join(result.statuses) if result.statuses else "—"
    file_tag = f" {YELLOW}[FILE]{RESET}" if result.has_file else ""
    short   = prompt.text[:70].replace("\n", " ").strip()
    turn_tag = f" (Turn {prompt.turn})" if prompt.turn else ""

    print(f"  {badge} #{index:02d}{turn_tag}{file_tag}  {BOLD}{prompt.set_name}{RESET}")
    print(f"     Prompt  : {short!r}")
    print(f"     Routing : {routing}")
    if not passed:
        print(f"     {RED}FAIL{RESET}    : {reason}")
    print()


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Run TEST_PROMPTS.md against a live MyAI server"
    )
    parser.add_argument("--url",     default="http://localhost:8080", help="Server base URL")
    parser.add_argument("--user",    required=True,  help="Username")
    parser.add_argument("--pin",     required=True,  help="4-digit PIN")
    parser.add_argument("--sets",    default="",     help="Comma-separated set names to run")
    parser.add_argument("--delay",   type=float, default=1.0, help="Seconds between prompts")
    parser.add_argument("--timeout", type=int,   default=90,  help="Per-prompt timeout (seconds)")
    args = parser.parse_args()

    md_path = Path(__file__).parent / "TEST_PROMPTS.md"
    if not md_path.exists():
        print(f"ERROR: {md_path} not found")
        sys.exit(1)

    prompts = parse_prompts(md_path)

    if args.sets:
        wanted = {s.strip().lower() for s in args.sets.split(",")}
        prompts = [p for p in prompts if any(w in p.set_name.lower() for w in wanted)]

    if not prompts:
        print("No prompts matched.  Check --sets spelling.")
        sys.exit(0)

    # ── Authenticate ──────────────────────────────────────────────────────────
    session = requests.Session()
    r = session.post(f"{args.url}/auth", json={"user_id": args.user, "pin": args.pin})
    if r.status_code != 200:
        print(f"Authentication failed ({r.status_code}): {r.text}")
        sys.exit(1)
    print(f"Authenticated as '{args.user}' at {args.url}\n")

    # ── Run prompts ────────────────────────────────────────────────────────────
    total  = 0
    passed = 0
    current_set: Optional[str] = None
    conv_id: Optional[str] = None

    for idx, prompt in enumerate(prompts, start=1):
        # Reset conversation between sets; keep same conv for multi-turn sets
        if prompt.set_name != current_set:
            r = session.post(f"{args.url}/conversations", json={"user_id": args.user})
            conv_id = r.json().get("conv_id")
            current_set = prompt.set_name
            print(f"{BOLD}{'─' * 60}{RESET}")
            print(f"{BOLD}SET: {prompt.set_name}{RESET}")
            print(f"{BOLD}{'─' * 60}{RESET}")

        result, conv_id = send_prompt(
            session, args.url, prompt.text, args.user, conv_id, args.timeout
        )
        ok, reason = validate(prompt, result)
        print_result(prompt, result, ok, reason, idx)

        total  += 1
        passed += int(ok)

        if idx < len(prompts):
            time.sleep(args.delay)

    # ── Summary ────────────────────────────────────────────────────────────────
    colour = GREEN if passed == total else RED
    print(f"{colour}{BOLD}Results: {passed}/{total} passed{RESET}")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
