# F91 #2 — image resize size-parsing guard (no model, no Pillow needed).
# Stubs PIL so it runs headless in CI; exercises _parse_resize across every
# phrasing, including the "400px wide" case that used to silently halve.
import sys, types
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

# stub PIL (we only test parsing/aspect logic, not real pixel ops)
pil = types.ModuleType("PIL")
pil.Image = types.SimpleNamespace(LANCZOS=1)
pil.ImageOps = object()
sys.modules.setdefault("PIL", pil)

from myai_skills.image_skill import ImageSkill

class Img:
    def __init__(self, w, h): self.width, self.height = w, h
    @property
    def size(self): return (self.width, self.height)
    def resize(self, size, *a): return Img(*size)

sk = ImageSkill()
port = Img(2210, 2678)   # portrait (the original bug image)
land = Img(1000, 500)    # landscape
def p(q, img=port): return sk._parse_resize(q.lower(), img)

def ok(label): print(f"ok  {label}")

# THE REGRESSION GUARD: single-side 'wide' must scale proportionally, never halve.
assert p("resize this to 400px wide") == (400, round(400*2678/2210))
assert p("resize this to 400px wide") != (1105, 1339)   # the old bug value
ok("'400px wide' scales proportionally (not 50% halving)")

assert p("make it 800 wide") == (800, round(800*2678/2210))
assert p("width 600") == (600, round(600*2678/2210))
assert p("resize to 300 tall")[1] == 300
assert p("height 500")[1] == 500
ok("single-dimension wide/width/tall/height proportional")

assert p("resize to 800x600") == (800, 600)
assert p("800×600") == (800, 600)
ok("exact WxH unchanged (incl. unicode ×)")

assert p("resize to 50%") == (1105, 1339)
assert p("25%") == (round(2210*.25), round(2678*.25))
ok("percentage scaling")

assert p("resize to half") == (1105, 1339)
assert p("make it a quarter") == (round(2210*.25), round(2678*.25))
assert p("three quarters") == (round(2210*.75), round(2678*.75))
assert p("double the size") == (4420, 5356)
ok("natural-language half/quarter/three quarters/double")

assert p("resize to 400", port)[1] == 400          # portrait: long edge = height
assert p("resize to 400", land)[0] == 400          # landscape: long edge = width
ok("lone number -> longest edge, aspect preserved")

assert p("resize this image") is None and p("scale it") is None
ok("no size -> None")

class _S(ImageSkill):
    def _deliver(self, *a): return "D"
msg = _S()._apply_resize(port, "p", "JPEG", "resize this image")
assert "What size" in msg and "800x600" in msg and "50%" in msg and "half" in msg
ok("no-size -> asks the user (not silent halving)")

print("\nALL GREEN")
