import ast, sys
ast.parse(open("core/inference/dispatch.py").read()); print("ok  ast.parse")

from core.inference.dispatch import (
    dispatch, build_skill_map, missing_skills, DispatchResult)
from core.inference.router import DIRECT, ROUTES

# ── fake skills mirroring the REAL per-skill signatures ─────────────────────────
class Weather:      # plain: execute(query)
    name = "weather"
    def execute(self, query): return f"weather for {query}"
class Image:        # execute(query, file_contexts=...)
    name = "image"
    def execute(self, query, file_contexts=None): return f"image|fc={file_contexts}"
class Memory:       # execute(query, user_id=..., history=...)
    name = "memory"
    def execute(self, query, user_id=None, history=None): return f"mem|{user_id}|{history}"
class Directions:   # execute(query, memories=...)
    name = "directions"
    def execute(self, query, memories=None): return f"dir|{memories}"
class Kw:           # execute(query, **kwargs)
    name = "calculator"
    def execute(self, query, **kw): return f"calc|{sorted(kw)}"
class Boom:         # raises despite the contract — dispatch must catch
    name = "news"
    def execute(self, query): raise RuntimeError("kaboom")

smap = build_skill_map([Weather(), Image(), Memory(), Directions(), Kw(), Boom()])
CTX = {"user_id": "u1", "history": ["h"], "memories": ["m"], "file_contexts": ["f"]}

# 1) plain skill: query only, no extra kwargs leak in
r = dispatch("weather", "London", CTX, smap)
assert r.ran_skill and r.context == "weather for London" and r.error is None
print("ok  plain skill runs with query only")

# 2) image gets ONLY file_contexts
r = dispatch("image", "crop", CTX, smap)
assert r.context == "image|fc=['f']", r.context
print("ok  image ← file_contexts only")

# 3) memory gets ONLY user_id + history (not memories/file_contexts)
r = dispatch("memory", "remember", CTX, smap)
assert r.context == "mem|u1|['h']", r.context
print("ok  memory ← user_id + history only")

# 4) directions gets ONLY memories
r = dispatch("directions", "to Leeds", CTX, smap)
assert r.context == "dir|['m']", r.context
print("ok  directions ← memories only")

# 5) **kwargs skill receives the whole context
r = dispatch("calculator", "2+2", CTX, smap)
assert r.context == "calc|['file_contexts', 'history', 'memories', 'user_id']", r.context
print("ok  **kwargs skill ← full context")

# 6) DIRECT → no skill runs
r = dispatch(DIRECT, "capital of France", CTX, smap)
assert not r.ran_skill and r.route == DIRECT and r.context == "" and r.error is None
print("ok  direct → no skill executed")

# 7) unknown route → error, no skill run, no raise
r = dispatch("banana", "x", CTX, smap)
assert not r.ran_skill and r.error and "no skill registered" in r.error
print("ok  unknown route → error, no raise")

# 8) a skill that raises → captured into error, ran_skill True, no propagation
r = dispatch("news", "headlines", CTX, smap)
assert r.ran_skill and r.context == "" and "RuntimeError" in r.error and "kaboom" in r.error
print("ok  skill exception captured, not propagated")

# 9) query never double-passed even if context collides with the query param name
r = dispatch("weather", "Q", {"query": "COLLIDE"}, smap)
assert r.context == "weather for Q"      # positional query wins; 'query' key ignored
print("ok  query passed positionally; colliding context key ignored")

# 10) empty skill output normalised to ""
class Empty:
    name = "wikipedia"
    def execute(self, query): return None
r = dispatch("wikipedia", "x", CTX, build_skill_map([Empty()]))
assert r.ran_skill and r.context == ""
print("ok  None skill output → empty string")

# 11) missing_skills detects menu↔skills drift
full = build_skill_map([type("S",(),{"name":n})() for n in ROUTES])
assert missing_skills(full) == []
partial = build_skill_map([Weather()])
missing = missing_skills(partial)
assert set(missing) == set(ROUTES) - {"weather"} and "direct" not in missing
print("ok  missing_skills flags drift; excludes direct")

# 12) T1 — a minimal BaseSkill-shaped object is enough (name + execute)
class Minimal:
    name = "datetime"
    description = "d"
    def execute(self, query): return "now"
assert dispatch("datetime", "time?", {}, build_skill_map([Minimal()])).context == "now"
print("ok  BaseSkill contract (name + execute) is sufficient for dispatch")

print("\nALL GREEN")
