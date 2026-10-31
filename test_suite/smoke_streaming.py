import ast, sys
ast.parse(open("core/inference/streaming.py").read()); print("ok  ast.parse")

from core.inference.streaming import strip_reasoning, clean_text, stream_clean

# ── a minimal fake backend: only what stream_clean touches ──────────────────────
class FakeBackend:
    def __init__(self, chunks, emits_think):
        self._chunks = chunks
        self.emits_think_reasoning = emits_think
        self.seen_kwargs = None
    def create_chat_completion(self, messages, **kw):
        self.seen_kwargs = kw
        for text, fr in self._chunks:
            delta = {"content": text} if text is not None else {}
            yield {"choices": [{"delta": delta, "finish_reason": fr}]}

def run(chunks, emits_think, meta=None):
    b = FakeBackend(chunks, emits_think)
    out = list(stream_clean(b, [{"role":"user","content":"x"}], meta=meta))
    return "".join(out), b

# ══ strip_reasoning (finalised) ═════════════════════════════════════════════════
assert strip_reasoning("<think>reasoning</think>answer", True) == "answer"
assert strip_reasoning("mulling it over</think>answer", True) == "answer"   # opener-less
assert strip_reasoning("<think>x</think>hi", False) == "<think>x</think>hi" # no-op when off
assert strip_reasoning("plain answer", True) == "plain answer"             # nothing to strip
print("ok  strip_reasoning: paired, opener-less, gated-off, passthrough")

# ══ clean_text (finalised) + truncation guard ═══════════════════════════════════
assert clean_text("reasoning</think>  answer ", True) == "answer"
assert clean_text("answer<end_of_turn>", True) == "answer"                 # control-token clean
# truncation guard: thinking + truncated + NO closer → suppress
assert clean_text("Thinking Process: 1. Analyze the...", True, truncated=True) == ""
# truncated but HAS closer → strip normally (reasoning closed before cut-off)
assert clean_text("reason</think>partial ans", True, truncated=True) == "partial ans"
# natural stop, no closer, thinking → assume no-reasoning answer, keep
assert clean_text("just an answer", True, truncated=False) == "just an answer"
# non-thinking model truncated → keep partial (nothing to leak)
assert clean_text("partial answer", False, truncated=True) == "partial answer"
print("ok  clean_text: control clean + truncation guard (both directions)")

# ══ stream_clean: non-thinking passthrough ══════════════════════════════════════
text, b = run([("Hel", None), ("lo", None), (None, "stop")], emits_think=False)
assert text == "Hello"
print("ok  stream: non-thinking straight passthrough")

# ══ stream_clean: thinking, closer split across chunks ══════════════════════════
text, _ = run([("reasoning here ", None), ("</thi", None), ("nk>", None),
               ("the answer", None), (None, "stop")], emits_think=True)
assert text == "the answer", repr(text)      # reasoning suppressed, split closer caught
print("ok  stream: reasoning suppressed, </think> split across chunks caught")

# ══ stream_clean: thinking, answer trails the closer in same chunk ══════════════
text, _ = run([("mull</think>hi", None), (" there", None), (None, "stop")], emits_think=True)
assert text == "hi there", repr(text)
print("ok  stream: answer after closer streams through")

# ══ THE FIX: truncated mid-think → nothing leaks, truncated flagged ═════════════
meta = {}
text, _ = run([("Thinking Process:", None), (" 1. Analyze", None),
               (None, "length")], emits_think=True, meta=meta)
assert text == "", f"LEAK: {text!r}"          # <<< raw CoT must NOT appear
assert meta.get("truncated") is True
print("ok  stream: truncated-mid-think SUPPRESSED (no CoT leak) + truncated flag")

# ══ natural stop, thinking, no closer → flush (v1 no-reasoning rescue) ══════════
meta = {}
text, _ = run([("direct answer no think", None), (None, "stop")],
              emits_think=True, meta=meta)
assert text == "direct answer no think"
assert meta.get("truncated") is not True
print("ok  stream: natural-stop no-closer buffer flushed (answer rescued)")

# ══ non-thinking truncated → partial shown + flagged ════════════════════════════
meta = {}
text, _ = run([("partial ans", None), (None, "length")], emits_think=False, meta=meta)
assert text == "partial ans" and meta.get("truncated") is True
print("ok  stream: non-thinking truncation shows partial + flags truncated")

# ══ Decision 2: NO hard-coded stop tokens passed (rely on eos) ══════════════════
_, b = run([("x", None), (None, "stop")], emits_think=False)
assert b.seen_kwargs.get("stop") is None, b.seen_kwargs
assert b.seen_kwargs.get("stream") is True
print("ok  no hard-coded stops forwarded (eos handles stopping)")

print("\nALL GREEN")

# ══ leading-newline trim (the real Qwen \n\n case) ══════════════════════════════
# closer, then "\n\n" as its OWN delta, then the answer → no leading blank line.
text, _ = run([("reason ", None), ("</think>", None), ("\n\n", None),
               ("The answer", None), (None, "stop")], emits_think=True)
assert text == "The answer", repr(text)
print("ok  stream: \\n\\n as separate delta after </think> is trimmed")

# closer + "\n\n" + answer all in ONE delta → also trimmed.
text, _ = run([("reason</think>\n\nThe answer", None), (None, "stop")], emits_think=True)
assert text == "The answer", repr(text)
print("ok  stream: </think>\\n\\nanswer in one delta is trimmed")

# leading whitespace on a NON-thinking answer's first piece is trimmed too.
text, _ = run([("\n\n  Hello", None), (" world", None), (None, "stop")], emits_think=False)
assert text == "Hello world", repr(text)
print("ok  stream: leading whitespace trimmed on non-thinking first piece")

# whitespace INSIDE the answer must be preserved (only the LEADING is trimmed).
text, _ = run([("</think>", None), ("Line one", None), ("\n\nLine two", None),
               (None, "stop")], emits_think=True)
assert text == "Line one\n\nLine two", repr(text)
print("ok  stream: internal whitespace preserved (only leading trimmed)")

# natural-stop flush (no closer) is leading-trimmed as well.
text, _ = run([("\n\ndirect answer", None), (None, "stop")], emits_think=True)
assert text == "direct answer", repr(text)
print("ok  stream: natural-stop flush leading-trimmed")

print("\nALL GREEN (leading-trim)")
