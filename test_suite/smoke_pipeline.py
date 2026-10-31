import ast
ast.parse(open("core/inference/pipeline.py").read()); print("ok  ast.parse")

from core.inference.pipeline import (
    plan_turn, plan_action, resolve_skill, Plan,
    ANSWER_DIRECT, ANSWER_WITH_CONTEXT, FINAL_TEXT, FILE_DELIVER, ERROR)
from core.inference.dispatch import build_skill_map, DispatchResult
from core.inference.router import DIRECT
import base64

# ── fake skills ─────────────────────────────────────────────────────────────────
class Weather:
    name = "weather"
    def execute(self, query): return f"forecast: sunny in {query}"
class Memory:
    name = "memory"
    def execute(self, query, user_id=None, history=None): return "  Saved: no peanuts.  "
class Image:
    name = "image"
    def __init__(self, payload): self._p = payload
    def execute(self, query, file_contexts=None): return self._p
class Wikipedia:
    name = "wikipedia"
    def __init__(self, out): self._o = out
    def execute(self, query): return self._o
class WebSearch:
    name = "web_search"
    def execute(self, query): return f"web results for {query}"
class Boom:
    name = "news"
    def execute(self, query): raise ValueError("feed down")

CTX = {"user_id": "u1", "history": [], "memories": [], "file_contexts": []}

# 1) DIRECT route → ANSWER_DIRECT, no skill
p = plan_turn(DIRECT, "capital of France", CTX, build_skill_map([Weather()]))
assert p.action == ANSWER_DIRECT and p.route == DIRECT
print("ok  direct → ANSWER_DIRECT")

# 2) context skill → ANSWER_WITH_CONTEXT carrying the skill text
p = plan_turn("weather", "Leeds", CTX, build_skill_map([Weather()]))
assert p.action == ANSWER_WITH_CONTEXT and p.route == "weather"
assert p.context == "forecast: sunny in Leeds"
print("ok  weather → ANSWER_WITH_CONTEXT with context")

# 3) memory → FINAL_TEXT, trimmed, no model pass
p = plan_turn("memory", "remember no peanuts", CTX, build_skill_map([Memory()]))
assert p.action == FINAL_TEXT and p.route == "memory" and p.text == "Saved: no peanuts."
print("ok  memory → FINAL_TEXT (trimmed)")

# 4) image SKILL_DELIVER → FILE_DELIVER with parsed payload
b64 = base64.b64encode(b"PNGDATA").decode()
payload = f"SKILL_DELIVER|out.png|image/png|{b64}\n---\nHere is your resized image."
p = plan_turn("image", "resize", CTX, build_skill_map([Image(payload)]))
assert p.action == FILE_DELIVER and p.route == "image"
assert p.file["filename"] == "out.png" and p.file["mime_type"] == "image/png"
assert p.file["data"] == b64 and p.file["encoding"] == "base64"
assert p.text == "Here is your resized image."
print("ok  image SKILL_DELIVER → FILE_DELIVER (payload parsed)")

# 5) malformed SKILL_DELIVER → graceful FINAL_TEXT notice
p = plan_turn("image", "resize", CTX, build_skill_map([Image("SKILL_DELIVER|garbage")]))
assert p.action == FINAL_TEXT and "could not deliver file" in p.text
print("ok  malformed SKILL_DELIVER → FINAL_TEXT fallback")

# 6) Wikipedia MISS → transparently resolves to web_search context
smap = build_skill_map([Wikipedia("No Wikipedia article found for 'xyz'"), WebSearch()])
p = plan_turn("wikipedia", "xyz", CTX, smap)
assert p.action == ANSWER_WITH_CONTEXT and p.route == "web_search"
assert p.context == "web results for xyz"
print("ok  wikipedia miss → web_search fallback (effective route swapped)")

# 7) Wikipedia HIT → stays wikipedia
smap = build_skill_map([Wikipedia("The Eiffel Tower is..."), WebSearch()])
p = plan_turn("wikipedia", "Eiffel Tower", CTX, smap)
assert p.action == ANSWER_WITH_CONTEXT and p.route == "wikipedia"
print("ok  wikipedia hit → stays wikipedia")

# 8) Wikipedia miss but NO web_search registered → no fallback, keeps wiki text
smap = build_skill_map([Wikipedia("No Wikipedia article found")])
p = plan_turn("wikipedia", "x", CTX, smap)
assert p.route == "wikipedia" and p.action == ANSWER_WITH_CONTEXT
print("ok  wiki miss w/o web_search → no fallback (graceful)")

# 9) skill error → ERROR plan
p = plan_turn("news", "headlines", CTX, build_skill_map([Boom()]))
assert p.action == ERROR and "feed down" in p.error
print("ok  skill error → ERROR plan")

# 10) unknown route → ERROR (dispatch reports no skill)
p = plan_turn("banana", "x", CTX, build_skill_map([Weather()]))
assert p.action == ERROR and "no skill registered" in p.error
print("ok  unknown route → ERROR plan")

# 11) resolve_skill returns EFFECTIVE route on fallback (unit check)
r = resolve_skill("wikipedia", "xyz",
                  CTX, build_skill_map([Wikipedia("No Wikipedia article found"), WebSearch()]))
assert r.route == "web_search" and r.context == "web results for xyz"
print("ok  resolve_skill surfaces effective route after fallback")

print("\nALL GREEN")
