import ast, sys
ast.parse(open("core/inference/router.py").read()); print("ok  ast.parse")

from core.inference.router import (
    build_router_prompt, parse_route, route, RouteDecision, ROUTES, DIRECT)

# ── prompt ──────────────────────────────────────────────────────────────────────
p = build_router_prompt("what's the weather in Rochdale?")
assert all(name in p for name in ROUTES), "all routes must appear in the menu"
assert '"route": null' in p and '"route": "<tool name from the list>"' in p
assert "what's the weather in Rochdale?" in p
assert "ONLY when the user explicitly asks" in p          # wikipedia policy present
print("ok  prompt carries full menu + emit contract + policy")

# ── parse: clean emits (non-thinking model) ─────────────────────────────────────
def d(text, emits_think=False): return parse_route(text, emits_think)

r = d('{"route": "weather"}')
assert (r.route, r.parsed, r.valid) == ("weather", True, True)
r = d('{"route": null}')
assert (r.route, r.parsed, r.valid) == (DIRECT, True, True)
for null_form in ('"null"', '"none"', '""'):
    assert d('{"route": %s}' % null_form).route == DIRECT
print("ok  parse: tool route, null, and null-like forms → correct")

# ── parse: reasoning THEN emit (Gemma-style prose, last line JSON) ──────────────
gemma = ('The user wants current conditions, which change constantly, so a tool '
         'is needed.\n{"route": "weather"}')
assert d(gemma).route == "weather"
print("ok  parse: prose reasoning + final-line emit (non-thinking)")

# ── parse: Qwen-style <think> reasoning is stripped before parsing ──────────────
qwen = '<think>\nThey asked the time in Tokyo — that is a datetime tool.\n</think>\n{"route": "datetime"}'
assert d(qwen, emits_think=True).route == "datetime"
# and with emits_think=False the <think> text stays but the LAST json still wins
assert d(qwen, emits_think=False).route == "datetime"
print("ok  parse: <think> stripped when thinking; last-JSON still wins otherwise")

# ── parse: LAST object wins (mid-reasoning JSON must not hijack) ────────────────
musing = ('Maybe {"route": "calculator"} at first glance, but it is timeless.\n'
          '{"route": null}')
assert d(musing).route == DIRECT, "must take the committed LAST emit, not the musing"
print("ok  parse: last emit wins over mid-reasoning JSON")

# ── parse: unknown route → safe direct, flagged invalid ─────────────────────────
r = d('{"route": "banana"}')
assert (r.route, r.parsed, r.valid) == (DIRECT, True, False)
print("ok  parse: unknown route → direct, valid=False")

# ── parse: case/whitespace normalised ───────────────────────────────────────────
assert d('{"route": "  WEATHER "}').route == "weather"
print("ok  parse: route normalised (case/space)")

# ── parse: no JSON at all → direct, parsed=False (the T2 target) ────────────────
r = d('I think you should just ask me directly, no tool needed.')
assert (r.route, r.parsed, r.valid) == (DIRECT, False, False)
print("ok  parse: no emit → direct, parsed=False (grammar-repair target)")

# ── parse: extra keys in the emit are tolerated ─────────────────────────────────
assert d('{"route": "currency", "why": "fx rate"}').route == "currency"
print("ok  parse: extra keys tolerated")

# ── route(): drives the backend with the right kwargs, returns a decision ───────
class FakeBackend:
    emits_think_reasoning = False
    def __init__(self, content): self._c = content; self.seen = None
    def create_chat_completion(self, messages, **kw):
        self.seen = (messages, kw)
        return {"choices":[{"message":{"role":"assistant","content":self._c}}]}

b = FakeBackend('reasoning...\n{"route": "directions"}')
dec = route(b, "how do I drive to Leeds")
assert dec.route == "directions" and dec.parsed
msgs, kw = b.seen
assert msgs[0]["role"] == "user" and "how do I drive to Leeds" in msgs[0]["content"]
assert kw["max_tokens"] == 512 and kw["temperature"] == 0.2 and kw["stream"] is False
print("ok  route(): single user-message pass, correct kwargs, decision returned")

# ── route(): never raises on empty/garbage content ──────────────────────────────
assert route(FakeBackend(""), "x").route == DIRECT
assert route(FakeBackend(None), "x").route == DIRECT
print("ok  route(): empty/None content → safe direct, no raise")

print("\nALL GREEN")

# ══════════════════════════ T2: grammar-lock repair ════════════════════════════
from core.inference.router import build_route_grammar, repair_route
from core.inference.base import OutputConstraint

# grammar carries every route + null, and the {"route": ...} shape
g = build_route_grammar()
for name in ROUTES:
    assert f'\\"{name}\\"' in g, f"{name} missing from grammar"
assert '"null"' in g and 'root ::= "{\\"route\\": " value "}"' in g
print("ok  grammar: all routes + null, correct emit shape")

# every emit the grammar can produce parses back to the right route
import json
for name in list(ROUTES) + [None]:
    emit = '{"route": %s}' % (json.dumps(name) if name else 'null')
    exp = name if name else DIRECT
    assert parse_route(emit, False).route == exp
print("ok  grammar: every producible emit parses to the right route")

# ── a fake backend that records calls and can script per-call replies ───────────
class ScriptBackend:
    def __init__(self, replies, emits_think=False):
        self._replies = list(replies)
        self.emits_think_reasoning = emits_think
        self.calls = []                       # list of (messages, kwargs)
    def create_chat_completion(self, messages, **kw):
        self.calls.append((messages, kw))
        content = self._replies.pop(0)
        return {"choices": [{"message": {"role": "assistant", "content": content}}]}

# repair_route: passes the grammar constraint + feeds reasoning back
b = ScriptBackend(['{"route": "weather"}'])
dec = repair_route(b, "weather in Leeds", reasoning="They want current conditions.")
assert dec.route == "weather" and dec.valid
msgs, kw = b.calls[0]
assert isinstance(kw.get("constraint"), OutputConstraint)
assert kw["constraint"].gbnf == build_route_grammar()
assert kw["temperature"] == 0.0
# reasoning fed back as an assistant turn
assert any(m["role"] == "assistant" and "current conditions" in m["content"] for m in msgs)
print("ok  repair_route: grammar constraint applied + reasoning fed back")

# repair with no reasoning → single user turn, still constrained
b = ScriptBackend(['{"route": "datetime"}'])
dec = repair_route(b, "time in Tokyo")
assert dec.route == "datetime"
assert len(b.calls[0][0]) == 1 and b.calls[0][0][0]["role"] == "user"
print("ok  repair_route: works with no prior reasoning")

# ══════════════════════════ route(): fast path vs repair ═══════════════════════
# FAST PATH: free pass already valid → exactly ONE call, no grammar
b = ScriptBackend(['reasoning...\n{"route": "currency"}'])
dec = route(b, "50 usd to gbp")
assert dec.route == "currency" and dec.valid
assert len(b.calls) == 1, "fast path must not make a second call"
assert "constraint" not in b.calls[0][1], "fast path must not use grammar"
print("ok  route(): valid free pass → one call, no grammar (fast path)")

# REPAIR PATH: free pass parse-fails → second, grammar-locked call rescues it
b = ScriptBackend(['I cannot decide, no json here', '{"route": "web_search"}'])
dec = route(b, "who is the current PM")
assert dec.route == "web_search" and dec.valid and dec.parsed
assert len(b.calls) == 2, "parse-fail must trigger the repair call"
assert isinstance(b.calls[1][1].get("constraint"), OutputConstraint)
print("ok  route(): parse-fail → grammar repair rescues (two calls)")

# REPAIR PATH via INVALID route: free pass emits unknown route → repair
b = ScriptBackend(['{"route": "banana"}', '{"route": "calculator"}'])
dec = route(b, "what is 2+2 times 5")
assert dec.route == "calculator" and dec.valid
assert len(b.calls) == 2
print("ok  route(): unknown route → grammar repair corrects it")

# REPAIR ALSO FAILS → safe DIRECT fallback (never raises)
b = ScriptBackend(['no json', 'still no json'])
dec = route(b, "hmm")
assert dec.route == DIRECT and not dec.valid
assert len(b.calls) == 2
print("ok  route(): repair also fails → safe direct fallback")

# grammar_repair=False → no repair attempted
b = ScriptBackend(['no json at all'])
dec = route(b, "hi", grammar_repair=False)
assert dec.route == DIRECT and len(b.calls) == 1
print("ok  route(): grammar_repair=False disables the repair call")

# a valid {"route": null} free pass is NOT 'repaired' (direct is a real decision)
b = ScriptBackend(['timeless fact.\n{"route": null}'])
dec = route(b, "capital of France")
assert dec.route == DIRECT and dec.valid and len(b.calls) == 1
print("ok  route(): explicit null (direct) is final, no repair")

print("\nALL GREEN (T2)")

# ── repaired flag is set on the repair path only (standing guard) ───────────────
assert route(ScriptBackend(['x\n{"route": "weather"}']), 'q').repaired is False
assert route(ScriptBackend(['no json', '{"route": "web_search"}']), 'q').repaired is True
print("ok  repaired flag: False on fast path, True on repair path")
