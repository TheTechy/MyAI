import builtins, io, sys, tempfile
from pathlib import Path

import installer.tui as T

def ui_for(width=60, color=False, steps=7):
    return T.TUI(total_steps=steps, color=color, interactive=False,
                 width=width, stream=io.StringIO())

def feed(*answers):
    """Make input() return `answers` in turn, then behave like closed stdin."""
    it = iter(answers)
    def fake(prompt=""):
        try:
            return next(it)
        except StopIteration:
            raise EOFError
    builtins.input = fake

_real_input = builtins.input

# 1) Every frame line is exactly `width` columns — whatever the width, colour,
#    or content (long text wraps, over-long/styled lines truncate, wide chars count 2).
for width in (40, 60, 76):
    for color in (False, True):
        ui = ui_for(width, color)
        for step, title in [(0, "Welcome"), (3, "Choose a model")]:
            ui.step(step, title)
            body = ["word " * 40, "", "x" * 200, ui.paint("y" * 200, "bold", "cyan"),
                    ui.warn("wide 🎉 chars"), "  indented " * 12]
            lines = ui.frame(body, "↑↓ move · Enter select · Ctrl-C quit")
            assert {T.vis_len(l) for l in lines} == {width}, (width, color, step)
            if not color:
                assert not any("\033" in l for l in lines)
print("ok  frame lines exactly `width` cols (40/60/76, colour on/off, wrap/truncate/wide)")

# 2) Header shows the step only on numbered steps; the progress bar tracks it.
ui = ui_for(60); ui.step(0, "Welcome")
assert "Step" not in ui.frame([], "")[0]
ui.step(7, "Done")
top, bar = ui.frame([], "")[:2]
assert "Step 7 of 7" in top and "░" not in bar and "█" in bar
print("ok  header step label + progress bar")

# 3) fit(): pads, truncates with an ellipsis, and resets colour if it cut a styled line.
assert T.fit("abc", 5) == "abc  "
assert T.fit("abcdef", 4) == "abc…"
cut = T.fit("\033[1m" + "z" * 10 + "\033[0m", 5)
assert T.vis_len(cut) == 5 and cut.endswith("…\033[0m")
print("ok  fit pads / truncates / resets colour")

# 4) Key decoding and menu movement.
assert [T.decode_key(k) for k in ("\r", "\n", "\x1b[A", "\x1bOB", "\x1b", "\x7f", "3")] == \
       ["enter", "enter", "up", "down", "esc", "backspace", "3"]
assert T.menu_step(0, "up", 3) == (2, False)          # wraps to the bottom
assert T.menu_step(2, "down", 3) == (0, False)        # wraps to the top
assert T.menu_step(1, "enter", 3) == (1, True)
assert T.menu_step(0, "3", 3) == (2, True)            # number jumps + selects
assert T.menu_step(0, "4", 3) == (0, False)           # out of range ignored
assert T.menu_step(0, "x", 3) == (0, False)
print("ok  decode_key + menu_step")

try:
    # 5) Non-interactive menu: numbered, re-asks on bad input, Enter = default.
    ui = ui_for(); ui.step(1, "Pick")
    feed("9", "abc", "2")
    assert ui.menu("Which?", ["a", "b", "c"]) == 1
    assert "[3]  c" in ui.stream.getvalue()
    feed("")
    assert ui.menu("Which?", ["a", "b", "c"], default=2) == 2
    feed("2")
    assert ui.confirm("Sure?") is False
    print("ok  numbered menu fallback (re-asks, default, confirm)")

    # 6) ask(): default on empty, validation re-asks and shows the error.
    ui = ui_for()
    feed("")
    assert ui.ask("Port", default="8080") == "8080"
    feed("abc", "8888")
    v = lambda s: None if s.isdigit() else "Digits only, please."
    assert ui.ask("Port", validate=v) == "8888"
    assert "Digits only, please." in ui.stream.getvalue()
    print("ok  ask default + validation")

    # 7) Closed stdin is treated like Ctrl-C — one exit path for the setup app.
    feed()
    try:
        ui.ask("Name"); raise AssertionError("expected KeyboardInterrupt")
    except KeyboardInterrupt:
        pass
    print("ok  EOF → KeyboardInterrupt")
finally:
    builtins.input = _real_input

# 8) run(): output goes to the log (appended, not the screen); failures return
#    the tail of *this* task's output.
with tempfile.TemporaryDirectory() as tmp:
    log = Path(tmp) / "logs" / "install.log"
    ui = ui_for()
    r = ui.run(["Working"], "Saying hello", [sys.executable, "-c", "print('hello-from-task')"], log)
    assert r.ok and r.returncode == 0 and r.tail == ["hello-from-task"], r
    assert "hello-from-task" not in ui.stream.getvalue()
    bad = ui.run(["Working"], "Failing", [sys.executable, "-c",
                 "import sys\nfor i in range(30): print('line', i)\nsys.exit(3)"], log)
    assert not bad.ok and bad.returncode == 3
    assert len(bad.tail) == T.TAIL_LINES and bad.tail[-1] == "line 29"
    text = log.read_text()
    assert "hello-from-task" in text and "line 29" in text and text.count("=== ") == 2
    print("ok  run → log file, ok/fail, per-task tail")

print("\nALL GREEN")
