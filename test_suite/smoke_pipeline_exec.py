import json, base64
from core.inference.pipeline import run_turn, PipelineDeps
from core.inference.dispatch import build_skill_map
from core.inference.router import RouteDecision, DIRECT

# ── fake backend ────────────────────────────────────────────────────────────────
class FakeBackend:
    def __init__(self, answer="Answer text.", truncate=False, think=False):
        self.emits_think_reasoning = think
        self.supports_system_role = True
        self._answer = answer
        self._truncate = truncate
        self.seen_messages = None
    def create_chat_completion(self, messages, **kw):
        self.seen_messages = messages
        def gen():
            for ch in self._answer:
                yield {"choices":[{"delta":{"content":ch},"finish_reason":None}]}
            yield {"choices":[{"delta":{},"finish_reason":"length" if self._truncate else "stop"}]}
        return gen()
    def count_tokens(self, messages): return 100
    def context_window(self): return 1000

# ── fake deps ────────────────────────────────────────────────────────────────────
def make_deps(**over):
    calls = {"appended": [], "titled": 0, "persisted": []}
    d = dict(
        get_messages=lambda c, n: [],
        get_prompt_for_user=lambda u: "SYS",
        get_file_contexts=lambda c: [],
        append_message=lambda c, r, t: calls["appended"].append((r, t)) or f"msg_{r}",
        list_memories=lambda u: [],
        auto_title=lambda c, u: calls.__setitem__("titled", calls["titled"]+1),
        title_event=lambda c: None,
        awaiting_clear=lambda h: False,
        is_confirmation=lambda u: False,
        has_file_block=lambda t: False,
        extract_and_generate=lambda t: [],
        strip_file_blocks=lambda t: t,
        persist_files=lambda m, f: calls["persisted"].append((m, f)),
        file_gen_snippet="",
    )
    d.update(over)
    return PipelineDeps(**d), calls

def route_to(name):
    return lambda backend, q: RouteDecision(name, True, True, "", "")

def run(user_input, *, backend, skill_map, deps, route_fn):
    evs = []
    for raw in run_turn(user_input, "u1", "c1", backend=backend, skill_map=skill_map,
                        deps=deps, route_fn=route_fn):
        assert raw.startswith("event: ") and raw.endswith("\n\n")
        head, _, body = raw.partition("\n")
        ev = head[len("event: "):]
        data = json.loads(body[len("data: "):])
        evs.append((ev, data))
    return evs

class Weather:
    name="weather"
    def execute(self, q): return "sunny 20C"
class Memory:
    name="memory"
    def execute(self, q, user_id=None, history=None): return "Saved."
class Image:
    name="image"
    def __init__(self, p): self.p = p
    def execute(self, q, file_contexts=None): return self.p
class Boom:
    name="news"
    def execute(self, q): raise RuntimeError("down")

SM = build_skill_map([Weather(), Memory(), Boom()])

# 1) ANSWER_DIRECT: status → tokens → ctx → done, persists both turns
b = FakeBackend("Hi.")
deps, calls = make_deps()
evs = run("capital of France", backend=b, skill_map=SM, deps=deps, route_fn=route_to(DIRECT))
names = [e for e,_ in evs]
assert names[0] == "status" and evs[0][1]["message"] == "Thinking…"
assert "".join(d["text"] for e,d in evs if e=="token") == "Hi."
assert "ctx" in names and names[-1] == "done" and evs[-1][1]["truncated"] is False
assert ("user","capital of France") in calls["appended"] and ("model","Hi.") in calls["appended"]
print("ok  ANSWER_DIRECT: status→tokens→ctx→done, both turns persisted")

# 2) ANSWER_WITH_CONTEXT (weather): extra 'Generating…' status, enriched input reaches model
b = FakeBackend("It is sunny.")
deps, calls = make_deps()
evs = run("weather in Leeds", backend=b, skill_map=SM, deps=deps, route_fn=route_to("weather"))
statuses = [d["message"] for e,d in evs if e=="status"]
assert statuses == ["Thinking…", "Generating answer…"]
# the skill context reached the model prompt
assert any("sunny 20C" in m["content"] for m in b.seen_messages)
assert "".join(d["text"] for e,d in evs if e=="token") == "It is sunny."
print("ok  ANSWER_WITH_CONTEXT: two statuses, skill context enriched into prompt")

# 3) FINAL_TEXT (memory): token+done, NO model stream (backend untouched)
b = FakeBackend("SHOULD_NOT_APPEAR")
deps, calls = make_deps()
evs = run("remember X", backend=b, skill_map=SM, deps=deps, route_fn=route_to("memory"))
assert b.seen_messages is None, "memory must not invoke the model"
assert ("token", {"text":"Saved."}) in evs and evs[-1][0]=="done"
print("ok  FINAL_TEXT (memory): exact reply, model never called")

# 4) FILE_DELIVER (image): file event then description token then done
b64 = base64.b64encode(b"IMG").decode()
img = Image(f"SKILL_DELIVER|out.png|image/png|{b64}\n---\nDone.")
b = FakeBackend("NO")
deps, calls = make_deps()
evs = run("resize", backend=b, skill_map=build_skill_map([img]), deps=deps, route_fn=route_to("image"))
file_evs = [d for e,d in evs if e=="file"]
assert len(file_evs)==1 and file_evs[0]["filename"]=="out.png" and "disk_path" not in file_evs[0]
assert ("token",{"text":"Done."}) in evs and b.seen_messages is None
# file event comes after the routing status, before the description token
names=[e for e,_ in evs]; assert names.index("file") < names.index("token")
print("ok  FILE_DELIVER (image): file event + description, model never called")

# 5) ERROR: skill raises → single error event, nothing persisted
b = FakeBackend()
deps, calls = make_deps()
evs = run("headlines", backend=b, skill_map=SM, deps=deps, route_fn=route_to("news"))
assert evs == [("status",{"message":"Thinking…"}), ("error",{"message":"Skill error: RuntimeError: down"})]
assert calls["appended"] == []
print("ok  ERROR: single error event, no persistence")

# 6) memory-clear pre-route: forces memory, SKIPS router + the 'Thinking…' status
b = FakeBackend()
deps, calls = make_deps(awaiting_clear=lambda h: True, is_confirmation=lambda u: True)
# route_fn would send elsewhere, but pre-route must win and not call it
called = {"n":0}
def rf(backend,q): called["n"]+=1; return RouteDecision(DIRECT,True,True,"","")
evs = run("yes", backend=b, skill_map=SM, deps=deps, route_fn=rf)
assert called["n"]==0, "router must be skipped on memory-clear confirmation"
assert not any(e=="status" for e,_ in evs)   # no 'Thinking…'
assert ("token",{"text":"Saved."}) in evs
print("ok  memory-clear pre-route: router skipped, no status, memory delivered")

# 7) [FILE:] on direct path: file generated + stripped; skill path would NOT
b = FakeBackend("Here you go [FILE:a.txt]hi[/FILE] enjoy")
gen_files = [{"filename":"a.txt","mime_type":"text/plain","encoding":"base64","data":"aGk=","size":2,"disk_path":"/x"}]
deps, calls = make_deps(
    file_gen_snippet="SNIPPET",
    has_file_block=lambda t: "[FILE:" in t,
    extract_and_generate=lambda t: gen_files,
    strip_file_blocks=lambda t: "Here you go  enjoy",
)
evs = run("write a file", backend=b, skill_map=SM, deps=deps, route_fn=route_to(DIRECT))
file_evs = [d for e,d in evs if e=="file"]
assert len(file_evs)==1 and file_evs[0]["filename"]=="a.txt" and "disk_path" not in file_evs[0]
assert ("model","Here you go  enjoy") in calls["appended"]   # persisted stripped
assert calls["persisted"] and calls["persisted"][0][1]==gen_files
# snippet appended to system prompt
assert any("SNIPPET" in m["content"] for m in b.seen_messages if m["role"]=="system")
print("ok  direct [FILE:]: file generated, stripped answer persisted, files linked")

# 8) truncated flag propagates to done
b = FakeBackend("cut", truncate=True)
deps, calls = make_deps()
evs = run("x", backend=b, skill_map=SM, deps=deps, route_fn=route_to(DIRECT))
assert evs[-1] == ("done", {"truncated": True})
print("ok  truncated stream → done.truncated True")

# 9) title event appended after done when present
b = FakeBackend("hi")
deps, calls = make_deps(title_event=lambda c: 'event: title\ndata: {"conv_id":"c1","title":"T"}\n\n')
evs = run("x", backend=b, skill_map=SM, deps=deps, route_fn=route_to(DIRECT))
assert evs[-1][0]=="title" and evs[-2][0]=="done"
print("ok  title event follows done when present")

print("\nALL GREEN")

# ══════════════ F4 Option A: [FILE:] parity behaviours ══════════════════════════
import re as _re
def file_deps(**over):
    base = dict(
        get_messages=lambda c,n: [], get_prompt_for_user=lambda u: "SYS",
        get_file_contexts=lambda c: [], append_message=lambda c,r,t: f"m_{r}",
        list_memories=lambda u: [], auto_title=lambda c,u: None, title_event=lambda c: None,
        awaiting_clear=lambda h: False, is_confirmation=lambda u: False,
        has_file_block=lambda t: "[FILE:" in t,
        extract_and_generate=lambda t: [{"filename":m.group(1),"mime_type":"text/plain",
            "encoding":"base64","data":"eA==","size":1,"disk_path":"/tmp/"+m.group(1)}
            for m in _re.finditer(r'\[FILE:([^\]]+)\]', t)],
        strip_file_blocks=lambda t: _re.sub(r'\[FILE:[^\]]+\]\n?.*?\[/FILE\]','',t,flags=_re.DOTALL).strip(),
        persist_files=lambda m,f: None, file_gen_snippet="SNIP",
    )
    base.update(over)
    return PipelineDeps(**base)

# 1) THE LEAK FIX: model streams a raw [FILE:] block on the direct path →
#    NO token event may contain the raw block; a file event is emitted instead.
b = FakeBackend("Here are tips.\n[FILE:tips.txt]\nSleep more.\nDrink water.[/FILE]")
deps = file_deps()
evs = run("write me a text file of tips", backend=b, skill_map=SM, deps=deps, route_fn=route_to(DIRECT))
tokens = [d["text"] for e,d in evs if e=="token"]
assert all("[FILE:" not in t and "[/FILE]" not in t for t in tokens), tokens
assert any(e=="file" and d["filename"]=="tips.txt" for e,d in evs)
print("ok  direct [FILE:] block NEVER leaks into a token event (buffered)")

# 2) WRAP: file requested but model returns prose with NO block → wrapped into a file
b = FakeBackend("def greet():\n    return 'hi'")
deps = file_deps()
evs = run("write me a python script that greets", backend=b, skill_map=SM, deps=deps, route_fn=route_to(DIRECT))
file_evs = [d for e,d in evs if e=="file"]
assert file_evs and file_evs[0]["filename"].endswith(".py"), file_evs
print("ok  file requested, model wrote prose → WRAPPED into a .py file")

# 3) SUPPRESS: long prose + a delivered file → persisted answer suppressed
b = FakeBackend("Here is a long explanation line one.\nAnd line two.\nAnd three.\n"
                "[FILE:notes.txt]\nx[/FILE]")
appended = []
deps = file_deps(append_message=lambda c,r,t: appended.append((r,t)) or "mid")
evs = run("save notes as a text file", backend=b, skill_map=SM, deps=deps, route_fn=route_to(DIRECT))
model_persist = [t for r,t in appended if r=="model"][0]
# long prose suppressed → falls back to the clean file description (v1 behaviour)
assert "[FILE:" not in model_persist and "long explanation" not in model_persist
assert "ready to download" in model_persist, repr(model_persist)
assert any(e=="file" for e,d in evs)
print("ok  file delivered + long prose → prose suppressed, description used")

# 4) context path STILL streams live (unchanged) and strips any block for persistence
b = FakeBackend("It is sunny today.")
deps = file_deps()
evs = run("weather in Leeds", backend=b, skill_map=SM, deps=deps, route_fn=route_to("weather"))
# two statuses = the context path; tokens streamed live
assert [d["message"] for e,d in evs if e=="status"] == ["Thinking…","Generating answer…"]
assert "".join(d["text"] for e,d in evs if e=="token") == "It is sunny today."
print("ok  context path still streams live (parity with v1 search path)")

print("\nALL GREEN (file parity)")
