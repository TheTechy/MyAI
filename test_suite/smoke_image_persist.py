# F91 #1 — delivered image persists (disk_path → message_files). Pipeline side.
import sys, base64
from pathlib import Path
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from core.inference.pipeline import (
    _parse_skill_deliver, run_turn, PipelineDeps, FILE_DELIVER, FINAL_TEXT)
from core.inference.router import RouteDecision, DIRECT

b64 = base64.b64encode(b"IMGBYTES").decode()

# 1) 5-field payload → disk_path carried into plan.file
new = f"SKILL_DELIVER|photo_400x485.jpeg|image/jpeg|{b64}|/tmp/out/photo_400x485.jpeg\n---\nResized to 400×485 px."
plan = _parse_skill_deliver("image", new)
assert plan.action == FILE_DELIVER
assert plan.file["disk_path"] == "/tmp/out/photo_400x485.jpeg"
assert plan.file["filename"] == "photo_400x485.jpeg"
print("ok  5-field payload → disk_path in plan.file")

# 2) legacy 4-field payload → still works, no disk_path
old = f"SKILL_DELIVER|old.png|image/png|{b64}\n---\nlegacy"
plan_old = _parse_skill_deliver("image", old)
assert plan_old.action == FILE_DELIVER and "disk_path" not in plan_old.file
print("ok  legacy 4-field payload still parses (no disk_path)")

# 3) empty disk_path (skill's disk-save failed) → treated as not-persistable
empty = f"SKILL_DELIVER|x.png|image/png|{b64}|\n---\nno-disk"
pe = _parse_skill_deliver("image", empty)
assert pe.action == FILE_DELIVER and "disk_path" not in pe.file
print("ok  empty disk_path field → no disk_path key (won't persist)")

# ── run_turn FILE_DELIVER persistence ──────────────────────────────────────────
class FakeBackend:
    emits_think_reasoning = False; supports_system_role = True
    def create_chat_completion(self, *a, **k): return iter([])
    def count_tokens(self, messages): return None
    def context_window(self): return 1000

def deps_with(persisted):
    return PipelineDeps(
        get_messages=lambda c,n: [], get_prompt_for_user=lambda u: "S",
        get_file_contexts=lambda c: [], append_message=lambda c,r,t: "MID",
        list_memories=lambda u: [], auto_title=lambda c,u: None, title_event=lambda c: None,
        awaiting_clear=lambda h: False, is_confirmation=lambda u: False,
        has_file_block=lambda t: False, extract_and_generate=lambda t: [],
        strip_file_blocks=lambda t: t,
        persist_files=lambda m,f: persisted.append((m,f)), file_gen_snippet="",
    )

# an image skill returning the NEW 5-field payload
class Image5:
    name="image"
    def execute(self, q, file_contexts=None):
        return f"SKILL_DELIVER|r.jpg|image/jpeg|{b64}|/tmp/out/r.jpg\n---\nDone."
from core.inference.dispatch import build_skill_map
def route_to(n): return lambda backend,q: RouteDecision(n,True,True,"","")

def run(skill, deps):
    evs=[]
    for raw in run_turn("resize it","u","c",backend=FakeBackend(),
                        skill_map=build_skill_map([skill]),deps=deps,route_fn=route_to("image")):
        head,_,body=raw.partition("\n"); import json
        evs.append((head[len('event: '):], json.loads(body[len('data: '):])))
    return evs

# 4) disk-backed image → persist_files called with the file (incl disk_path)
persisted=[]
evs = run(Image5(), deps_with(persisted))
assert any(e=="file" and d["filename"]=="r.jpg" and "disk_path" not in d for e,d in evs), "file event should be sent w/o disk_path"
assert len(persisted)==1, f"expected 1 persist call, got {len(persisted)}"
mid, files = persisted[0]
assert mid=="MID" and files[0]["disk_path"]=="/tmp/out/r.jpg"
print("ok  FILE_DELIVER persists the disk-backed image (message_files)")

# 5) legacy image (no disk_path) → NOT persisted (parity with old behaviour)
class ImageOld:
    name="image"
    def execute(self, q, file_contexts=None):
        return f"SKILL_DELIVER|r.jpg|image/jpeg|{b64}\n---\nDone."
persisted2=[]
run(ImageOld(), deps_with(persisted2))
assert persisted2==[], "legacy no-disk_path image must not attempt persistence"
print("ok  legacy image (no disk_path) → not persisted (graceful)")

# 6) the live file event carries the stored file_id, so the card can open the
#    file from MyAI's own address (route maps need that for their map tiles)
def persist_with_ids(m, files):
    for f in files: f["file_id"] = "FID-9"
import dataclasses
evs = run(Image5(), dataclasses.replace(deps_with([]), persist_files=persist_with_ids))
assert [d.get("file_id") for e,d in evs if e=="file"] == ["FID-9"], evs
print("ok  live file event carries the stored file_id")

print("\nALL GREEN")
