"""
installer/tui.py — the look and feel of the MyAI setup app.

Standard library only. Every screen is a framed panel: a header with the step
and a progress bar, the content, and a footer of key hints. Menus use the arrow
keys (number keys jump straight to an option); long-running jobs show a spinner
while their output goes to a log file instead of the screen.

Key handling uses termios (macOS/Linux). Anything that isn't an interactive
terminal — pipes, CI, tests — gets plain numbered prompts instead, so the setup
app still works, just without the animation.

Drawing and key decoding are pure functions of their inputs; only the TUI
methods do I/O.
"""

from __future__ import annotations

import getpass
import os
import re
import select
import shutil
import subprocess
import sys
import textwrap
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

MAX_WIDTH = 76          # wide enough to breathe, narrow enough to read
MIN_WIDTH = 40
SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
TAIL_LINES = 12         # log lines shown when a task fails

_ANSI = re.compile(r"\033\[[0-9;?]*[A-Za-z]")
_CODES = {"bold": "1", "dim": "2", "red": "31", "green": "32",
          "yellow": "33", "cyan": "36"}

CLEAR = "\033[H\033[2J"
HIDE_CURSOR, SHOW_CURSOR = "\033[?25l", "\033[?25h"
CLEAR_LINE = "\r\033[K"

HINTS_MENU = "↑↓ move · Enter select · Ctrl-C quit"
HINTS_MENU_BACK = "↑↓ move · Enter select · Esc back · Ctrl-C quit"
HINTS_ASK = "Type below · Enter to confirm · Ctrl-C quit"
HINTS_SHOW = "Enter to continue · Ctrl-C quit"
HINTS_RUN = "This can take a while · Ctrl-C to stop"


# ── Text measuring (pure) ─────────────────────────────────────────────────────

def vis_len(s: str) -> int:
    """Columns `s` occupies on screen: ignores colour codes, counts wide chars as 2."""
    n = 0
    for ch in _ANSI.sub("", s):
        if unicodedata.combining(ch):
            continue
        n += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return n


def fit(s: str, width: int) -> str:
    """Pad `s` to exactly `width` columns, truncating with '…' if it's too long."""
    if vis_len(s) <= width:
        return s + " " * (width - vis_len(s))
    out, used, i = [], 0, 0
    while i < len(s):
        m = _ANSI.match(s, i)
        if m:                                   # colour codes take no space
            out.append(m.group()); i = m.end(); continue
        w = vis_len(s[i])
        if used + w > width - 1:
            break
        out.append(s[i]); used += w; i += 1
    reset = "\033[0m" if _ANSI.search(s) else ""
    return "".join(out) + "…" + reset + " " * (width - 1 - used)


def wrap(lines: Sequence[str], width: int) -> list[str]:
    """Word-wrap plain lines to `width`, keeping their indent. Styled lines
    (containing colour codes) are left as-is and truncated by fit() if needed."""
    out: list[str] = []
    for line in lines:
        if not line.strip() or _ANSI.search(line):
            out.append(line)
            continue
        indent = line[: len(line) - len(line.lstrip())]
        out.extend(textwrap.wrap(line, width, subsequent_indent=indent) or [""])
    return out


# ── Keys and menus (pure) ─────────────────────────────────────────────────────

def decode_key(seq: str) -> str:
    """Map raw terminal input to a key name: up/down/enter/esc/backspace, or the char."""
    if seq in ("\r", "\n"):
        return "enter"
    if seq in ("\x1b[A", "\x1bOA"):
        return "up"
    if seq in ("\x1b[B", "\x1bOB"):
        return "down"
    if seq == "\x1b":
        return "esc"
    if seq in ("\x7f", "\x08"):
        return "backspace"
    return seq


def menu_step(index: int, key: str, count: int) -> tuple[int, bool]:
    """Apply one key press to a menu. Returns (new index, chosen?).
    Arrows wrap round; a number key 1-9 picks that option outright."""
    if key == "up":
        return (index - 1) % count, False
    if key == "down":
        return (index + 1) % count, False
    if key == "enter":
        return index, True
    if key.isdigit() and 1 <= int(key) <= count:
        return int(key) - 1, True
    return index, False


@dataclass(frozen=True)
class TaskResult:
    ok: bool
    returncode: int
    seconds: float
    tail: list[str]         # last lines this task wrote to the log


def _elapsed(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


# ── The TUI ───────────────────────────────────────────────────────────────────

class TUI:
    """One per setup run. Call step() as you move through the wizard, then
    show()/menu()/ask()/confirm()/run() to draw each screen."""

    def __init__(self, title: str = "MyAI Setup", total_steps: int = 0, *,
                 color: bool | None = None, interactive: bool | None = None,
                 width: int | None = None, stream=None):
        self.title = title
        self.total_steps = total_steps
        self.stream = stream or sys.stdout
        tty_out = hasattr(self.stream, "isatty") and self.stream.isatty()
        if color is None:
            color = (tty_out and not os.environ.get("NO_COLOR")
                     and os.environ.get("TERM") != "dumb")
        if interactive is None:
            interactive = tty_out and sys.stdin.isatty() and _has_termios()
        self.color = color
        self.interactive = interactive
        self._width = width
        self._step = 0
        self._step_title = ""

    # ── Styling ──
    def paint(self, text: str, *styles: str) -> str:
        if not self.color or not styles:
            return text
        return f"\033[{';'.join(_CODES[s] for s in styles)}m{text}\033[0m"

    def ok(self, text: str) -> str:
        return f"{self.paint('✔', 'green')} {text}"

    def warn(self, text: str) -> str:
        return f"{self.paint('⚠', 'yellow')} {text}"

    def err(self, text: str) -> str:
        return f"{self.paint('✘', 'red')} {text}"

    def dim(self, text: str) -> str:
        return self.paint(text, "dim")

    def bold(self, text: str) -> str:
        return self.paint(text, "bold")

    # ── Layout ──
    @property
    def width(self) -> int:
        if self._width:
            return self._width
        cols = shutil.get_terminal_size((80, 24)).columns
        return max(MIN_WIDTH, min(MAX_WIDTH, cols - 2))

    def step(self, number: int, title: str) -> None:
        """Move to step `number` of total_steps (0 = an unnumbered screen, e.g. welcome)."""
        self._step, self._step_title = number, title

    def frame(self, body: Sequence[str], hints: str) -> list[str]:
        """The full panel for `body` as a list of lines, each exactly `width` columns."""
        w = self.width
        inner = w - 4
        b = lambda s: self.paint(s, "dim")        # borders stay quiet

        # ┌─ MyAI Setup ─────────── Step 2 of 7 ─┐
        title = f" {self.paint(self.title, 'bold', 'cyan')} "
        step = f" Step {self._step} of {self.total_steps} " if self._step else ""
        fill = w - 4 - vis_len(title) - vis_len(step)
        lines = [b("┌─") + title + b("─" * max(fill, 0)) + step + b("─┐")]

        row = lambda s: b("│") + " " + fit(s, inner) + " " + b("│")
        rule = b("├" + "─" * (w - 2) + "┤")

        if self._step:
            bar_w = min(24, inner // 2)
            done = round(bar_w * self._step / max(self.total_steps, 1))
            bar = self.paint("█" * done, "cyan") + self.dim("░" * (bar_w - done))
            lines.append(row(f"{bar}  {self.bold(self._step_title)}"))
        else:
            lines.append(row(self.bold(self._step_title)))
        lines.append(rule)
        lines.append(row(""))
        lines.extend(row(s) for s in wrap(body, inner - 1))
        lines.append(row(""))
        if hints:
            lines.append(rule)
            lines.append(row(self.dim(hints)))
        lines.append(b("└" + "─" * (w - 2) + "┘"))
        return lines

    def draw(self, body: Sequence[str], hints: str = HINTS_SHOW) -> None:
        out = (CLEAR if self.interactive else "\n") + "\n".join(self.frame(body, hints)) + "\n"
        self.stream.write(out)
        self.stream.flush()

    # ── Screens ──
    def show(self, body: Sequence[str], *, wait: bool = True) -> None:
        """An information screen. With wait=True, carries on when Enter is pressed."""
        self.draw(body, HINTS_SHOW if wait else "")
        if not wait:
            return
        if self.interactive:
            with _hidden_cursor(self.stream):
                while self._read_key() != "enter":
                    pass
        else:
            _input("  Press Enter to continue… ")

    def menu(self, question: str, options: Sequence[str], *, default: int = 0,
             intro: Sequence[str] = (), allow_back: bool = False) -> int | None:
        """Pick one of `options`. Returns its index, or None if Esc was pressed
        and allow_back is set."""
        if not options:
            raise ValueError("menu() needs at least one option")
        index = default
        hints = HINTS_MENU_BACK if allow_back else HINTS_MENU

        if not self.interactive:
            body = [*intro, *([""] if intro else []), question, ""]
            body += [f"  [{i}]  {o}" for i, o in enumerate(options, 1)]
            self.draw(body, "Type a number · Enter to confirm")
            while True:
                raw = _input(f"  Choice [{default + 1}]: ").strip() or str(default + 1)
                if raw.isdigit() and 1 <= int(raw) <= len(options):
                    return int(raw) - 1
                self.stream.write(f"  Please enter a number from 1 to {len(options)}.\n")

        with _hidden_cursor(self.stream):
            while True:
                body = [*intro, *([""] if intro else []), question, ""]
                for i, o in enumerate(options):
                    if i == index:
                        body.append(self.paint(f"▶ {o}", "bold", "cyan"))
                    else:
                        body.append(f"  {o}")
                self.draw(body, hints)
                key = self._read_key()
                if key == "esc" and allow_back:
                    return None
                index, chosen = menu_step(index, key, len(options))
                if chosen:
                    return index

    def confirm(self, question: str, *, default: bool = True,
                intro: Sequence[str] = ()) -> bool:
        return self.menu(question, ["Yes", "No"], default=0 if default else 1,
                         intro=intro) == 0

    def ask(self, question: str, *, default: str = "", intro: Sequence[str] = (),
            secret: bool = False,
            validate: Callable[[str], str | None] | None = None) -> str:
        """Free-text answer, typed below the panel. `validate` returns an error
        message to show (and ask again), or None to accept."""
        error = ""
        while True:
            body = [*intro, *([""] if intro else []), question]
            if error:
                body += ["", self.err(error)]
            self.draw(body, HINTS_ASK)
            hint = f" [{default}]" if default and not secret else ""
            label = f"  › {question.rstrip(':?')}{hint}: "
            value = (_getpass(label) if secret else _input(label)).strip() or default
            error = validate(value) if validate else None
            if not error:
                return value

    def run(self, body: Sequence[str], status: str, cmd: Sequence[str], log_path: Path, *,
            env: dict | None = None, cwd: str | Path | None = None) -> TaskResult:
        """Run `cmd` with its output going to `log_path`, showing a spinner and
        the elapsed time. Returns how it went, including the log's last lines."""
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.draw(body, HINTS_RUN)
        start = time.monotonic()
        with open(log_path, "a", encoding="utf-8") as log:
            log.write(f"\n=== {datetime.now():%Y-%m-%d %H:%M:%S}  {status}\n$ {' '.join(cmd)}\n")
            log.flush()
            offset = log.tell()
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                    env=env, cwd=cwd)
            try:
                if self.interactive:
                    with _hidden_cursor(self.stream):
                        frame = 0
                        while proc.poll() is None:
                            spin = self.paint(SPINNER[frame % len(SPINNER)], "cyan")
                            self.stream.write(f"{CLEAR_LINE}  {spin} {status}  "
                                              f"{self.dim(_elapsed(time.monotonic() - start))}")
                            self.stream.flush()
                            frame += 1
                            time.sleep(0.1)
                else:
                    self.stream.write(f"  … {status}\n")
                    self.stream.flush()
                    proc.wait()
            except KeyboardInterrupt:
                proc.terminate()
                proc.wait()
                raise
        seconds = time.monotonic() - start
        ok = proc.returncode == 0
        mark = self.ok(status) if ok else self.err(status)
        self.stream.write(f"{CLEAR_LINE if self.interactive else ''}  {mark}  "
                          f"{self.dim(_elapsed(seconds))}\n")
        self.stream.flush()
        with open(log_path, encoding="utf-8", errors="replace") as log:
            log.seek(offset)
            tail = log.read().splitlines()[-TAIL_LINES:]
        return TaskResult(ok, proc.returncode, seconds, tail)

    # ── Input ──
    def _read_key(self) -> str:
        import termios
        import tty
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)          # no echo, no line buffering; Ctrl-C still works
            seq = os.read(fd, 1)
            if seq == b"\x1b":         # an arrow key arrives as ESC [ A — gather the rest
                while len(seq) < 3 and select.select([fd], [], [], 0.03)[0]:
                    seq += os.read(fd, 1)
            return decode_key(seq.decode("utf-8", errors="ignore"))
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _has_termios() -> bool:
    try:
        import termios  # noqa: F401
        return True
    except ImportError:
        return False


def _input(prompt: str) -> str:
    # Closed stdin is the user going away, same as Ctrl-C: one way out for callers.
    try:
        return input(prompt)
    except EOFError:
        raise KeyboardInterrupt from None


def _getpass(prompt: str) -> str:
    try:
        return getpass.getpass(prompt)
    except EOFError:
        raise KeyboardInterrupt from None


class _hidden_cursor:
    def __init__(self, stream):
        self.stream = stream

    def __enter__(self):
        self.stream.write(HIDE_CURSOR); self.stream.flush()

    def __exit__(self, *exc):
        self.stream.write(SHOW_CURSOR); self.stream.flush()
        return False


# ── Demo: `python -m installer.tui` to see the screens ────────────────────────

def _demo() -> None:
    import tempfile
    ui = TUI(total_steps=4)
    try:
        ui.step(0, "Welcome")
        ui.show(["Welcome to MyAI — a private AI assistant that runs entirely on this Mac.",
                 "", "This demo shows each kind of screen. Nothing is installed."])
        ui.step(1, "Choose a model")
        pick = ui.menu("Which model should MyAI use?", [
            f"gemma-4-12b-it-Q5_K_XL     8.4 GB  {ui.paint('fits', 'green')}",
            f"gemma-4-12b-it-Q8_K_XL    12.9 GB  {ui.paint('tight', 'yellow')}",
            f"Swift-Qwen3.8-27B-Q5_K_M  20.2 GB  {ui.paint('too big', 'red')}",
            "Enter a path manually…"], allow_back=True)
        ui.step(2, "Your name")
        name = ui.ask("What should MyAI call you?", default="George",
                      validate=lambda v: None if v else "A name is required.")
        ui.step(3, "Building")
        log = Path(tempfile.gettempdir()) / "myai-tui-demo.log"
        res = ui.run(["Pretending to build the AI engine.", "",
                      ui.dim(f"Full output: {log}")], "Building the AI engine",
                     [sys.executable, "-c", "import time\nfor i in range(30): print(i); time.sleep(0.1)"],
                     log)
        ui.step(4, "Done")
        ui.show([ui.ok(f"Picked option {pick}"), ui.ok(f"Hello, {name}"),
                 ui.ok(f"Task finished in {_elapsed(res.seconds)}")], wait=False)
    except KeyboardInterrupt:
        ui.stream.write(SHOW_CURSOR + "\n  Cancelled.\n")


if __name__ == "__main__":
    _demo()
