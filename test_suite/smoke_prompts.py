import ast, sys, os, tempfile, uuid
from pathlib import Path
_REPO = Path(__file__).resolve().parent.parent

ast.parse(open(_REPO / "core" / "prompts.py").read()); print("ok  prompts.py ast.parse")
import json
json.load(open(_REPO / "data" / "age_band_prompts.json")); print("ok  age_band_prompts.json valid JSON")

# Self-contained: a throwaway DB with our own users, so the test never depends on
# users that happen to exist in the live DB (and never touches family data).
# MYAIDB must be set BEFORE core.db is imported (it binds DB_PATH at import).
os.environ["MYAIDB"] = str(Path(tempfile.mkdtemp(prefix="smoke_prompts_")) / "t.db")
from core import db
db.init_db()
from core.prompts import get_prompt_for_user, _date_preamble, _AGE_BAND_PERSONALITIES

# JSON loaded the four bands.
assert set(_AGE_BAND_PERSONALITIES) == {"adult","child_young","child_teen","senior"}, _AGE_BAND_PERSONALITIES
print("ok  four bands loaded")

def _mkuser(band: str) -> str:
    return db.create_user(display_id=f"_{band}_{uuid.uuid4().hex[:8]}", pin="0", age_band=band)

bands   = {b: _mkuser(b) for b in ("adult", "child_young", "child_teen", "senior")}
prompts = {b: get_prompt_for_user(uid) for b, uid in bands.items()}

# 1) Identity renamed everywhere; 'Gemma' gone entirely.
for b, p in prompts.items():
    assert "You are MyAI" in p, f"{b}: missing MyAI identity"
    assert "Gemma" not in p, f"{b}: Gemma still present"
print("ok  every band says 'You are MyAI'; no 'Gemma' anywhere")

# 2) Model-specific cutoff claim removed; model-agnostic date anchoring kept.
for b, p in prompts.items():
    assert "training cutoff" not in p.lower(), f"{b}: cutoff claim survived"
    assert "Today is" in p, f"{b}: lost date anchoring"
    assert "this year" in p, f"{b}: lost relative-date anchoring"
print("ok  cutoff claim gone; date anchoring preserved")

# 3) Date line appears EXACTLY once (factored out, not duplicated per band).
for b, p in prompts.items():
    assert p.count("Today is") == 1, f"{b}: date line duplicated ({p.count('Today is')}×)"
print("ok  date line appears once (no duplication)")

# 4) Child-safety guardrail preserved verbatim in child_young.
cy = prompts["child_young"]
assert "avoid scary topics, graphic detail, and adult content of any kind" in cy.lower()
print("ok  child_young safety guardrail intact")

# 5) Descriptive rewrites landed; the prescriptive originals are gone.
teen = prompts["child_teen"]
assert "kind of helper who shows the reasoning" in teen
assert "For schoolwork, prefer" not in teen
snr = prompts["senior"]
assert "rarely lean on jargon" in snr and "Avoid jargon" not in snr
print("ok  prescriptive lines rewritten as descriptive")

# 6) Composition order: personality → date → rules → (memories).
p = prompts["adult"]
i_id  = p.index("You are MyAI")
i_dat = p.index("Today is")
i_rul = p.index("How to handle knowledge")
assert i_id < i_dat < i_rul, (i_id, i_dat, i_rul)
print("ok  composition order personality → date → rules")

# 7) Memories still inject when present, after the rules.
mem_uid = _mkuser("adult")
db.add_memory(mem_uid, "Has a dog named Rex")
pm = get_prompt_for_user(mem_uid)      # adult band + a memory
assert "Has a dog named Rex" in pm
assert pm.index("Has a dog named Rex") > pm.index("How to handle knowledge")
print("ok  memory injection still works, after rules")

# 8) Unknown user → adult fallback, still MyAI, still no Gemma/cutoff.
pf = get_prompt_for_user("does_not_exist")
assert "You are MyAI" in pf and "Gemma" not in pf and "training cutoff" not in pf.lower()
print("ok  unknown-user fallback → MyAI, clean")

print("\nALL GREEN")
