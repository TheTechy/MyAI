import sys, types
import core.inference.engine as E

# ── stub the heavy periphery the lazy imports reach for ─────────────────────────
db = types.ModuleType("core.db")
db.get_messages = lambda c, n: ["MSGS"]
db.get_file_contexts = lambda c: ["FC"]
db.append_message = lambda c, r, t: "MSGID"
db.list_memories = lambda u: ["MEM"]
db.auto_title_from_first_message = lambda c, u: None
db.get_conversation_title = lambda c: "A Real Title"
db.store_message_file = lambda **k: "FID"
sys.modules["core.db"] = db

prompts = types.ModuleType("core.prompts")
prompts.get_prompt_for_user = lambda u: "SYSTEM PROMPT"
sys.modules["core.prompts"] = prompts

fg = types.ModuleType("core.file_generation")
fg.FILE_GENERATION_SYSTEM_SNIPPET = "FILE-SNIPPET"
fg.has_file_block = lambda t: False
fg.extract_and_generate = lambda t: []
fg.strip_file_blocks = lambda t: t
sys.modules["core.file_generation"] = fg

# fake memory skill exposing the static predicates
class FakeMemory:
    name = "memory"
    @staticmethod
    def awaiting_clear_confirmation(history): return history == ["CLEAR"]
    @staticmethod
    def is_confirmation_response(query): return query.lower() == "yes"
    def execute(self, query, user_id=None, history=None): return "mem"
class FakeWeather:
    name = "weather"
    def execute(self, query): return "w"

from core.inference.dispatch import build_skill_map
smap = build_skill_map([FakeMemory(), FakeWeather()])

# 1) build_default_deps wires every field to the right function
deps = E.build_default_deps(smap)
assert deps.get_messages("c", 10) == ["MSGS"]
assert deps.get_prompt_for_user("u") == "SYSTEM PROMPT"
assert deps.get_file_contexts("c") == ["FC"]
assert deps.append_message("c", "user", "x") == "MSGID"
assert deps.list_memories("u") == ["MEM"]
assert deps.file_gen_snippet == "FILE-SNIPPET"
print("ok  build_default_deps wires db/prompts/file-gen correctly")

# 2) memory predicates come off the registered memory skill
assert deps.awaiting_clear(["CLEAR"]) is True and deps.awaiting_clear([]) is False
assert deps.is_confirmation("yes") is True and deps.is_confirmation("nope") is False
print("ok  memory predicates sourced from the memory skill class")

# 3) no memory skill → safe no-op predicates
deps2 = E.build_default_deps(build_skill_map([FakeWeather()]))
assert deps2.awaiting_clear(["CLEAR"]) is False and deps2.is_confirmation("yes") is False
print("ok  missing memory skill → safe no-op predicates")

# 4) title_event uses the real title read
ev = deps.title_event("c")
assert ev.startswith("event: title") and "A Real Title" in ev
db.get_conversation_title = lambda c: "New conversation"
assert deps.title_event("c") is None       # default title → no event
print("ok  title_event emits only when titled")

# 5) persist_files skips no-disk files, calls store for real ones
stored = []
db.store_message_file = lambda **k: stored.append(k["filename"])
deps.persist_files("m", [{"disk_path": None, "filename": "skip"},
                         {"disk_path": "/x", "filename": "keep.txt",
                          "mime_type": "text/plain", "size": 3}])
assert stored == ["keep.txt"]
print("ok  persist_files: skips diskless, stores real")

# 6) ask_stream drives run_turn with the singletons (backend/skills/deps)
captured = {}
def fake_run_turn(ui, uid, cid, *, backend, skill_map, deps):
    captured.update(ui=ui, backend=backend, skill_map=skill_map, deps=deps)
    yield "event: token\ndata: {}\n\n"
E.run_turn = fake_run_turn
E.get_backend = lambda: "BACKEND"
E._skill_map = smap
E._deps = deps
out = list(E.ask_stream("hello", "u1", "c1"))
assert captured["ui"] == "hello" and captured["backend"] == "BACKEND"
assert captured["skill_map"] is smap and captured["deps"] is deps
assert out == ["event: token\ndata: {}\n\n"]
print("ok  ask_stream wires backend+skills+deps into run_turn")

# 7) top-level guard: an unexpected failure becomes a clean error event
def boom(*a, **k):
    raise RuntimeError("kaboom")
    yield
E.run_turn = boom
out = list(E.ask_stream("x", "u", "c"))
assert len(out) == 1 and out[0].startswith("event: error") and "went wrong" in out[0]
print("ok  ask_stream guard → clean error event on unexpected failure")

print("\nALL GREEN")

# ── warmup() contract: returns bool, respects WARMUP env (cutover parity) ────────
import os as _os
# reset backend singleton + stub get_backend to avoid a real load
E._backend = None
_warmed = {"n": 0}
class _FakeB:
    def warmup(self): _warmed["n"] += 1
E.get_backend = lambda: _FakeB()

_os.environ["WARMUP"] = "0"
assert E.warmup() is False and _warmed["n"] == 0, "WARMUP=0 must skip + return False"
_os.environ["WARMUP"] = "1"
assert E.warmup() is True and _warmed["n"] == 1, "WARMUP=1 must warm + return True"
del _os.environ["WARMUP"]
assert E.warmup() is True and _warmed["n"] == 2, "default (unset) warms + returns True"
print("ok  warmup(): bool return + honours WARMUP env (v1 contract)")
print("\nALL GREEN (engine + warmup)")
