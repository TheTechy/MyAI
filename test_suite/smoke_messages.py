import ast, sys

ast.parse(open("core/inference/messages.py").read()); print("ok  ast.parse")

from core.inference.messages import assemble_messages, _map_history

SYS = "You are Gemma. Rules: be helpful."
HIST = [
    {"msg_id": "1", "role": "user",  "content": "hi",        "files": []},
    {"msg_id": "2", "role": "model", "content": "hello!",    "files": []},
    {"msg_id": "3", "role": "user",  "content": "2+2?",      "files": []},
    {"msg_id": "4", "role": "model", "content": "4",         "files": []},
]

# 1) System-role mode (default): system FIRST, history mapped, new turn last.
m = assemble_messages(SYS, HIST, "what's the capital of France?")
assert m[0] == {"role": "system", "content": SYS}, m[0]
assert m[-1] == {"role": "user", "content": "what's the capital of France?"}
roles = [x["role"] for x in m]
assert roles == ["system", "user", "assistant", "user", "assistant", "user"], roles
print("ok  system-role: system@0, history mapped, current turn last")

# 2) THE mandatory mapping — "model" → "assistant", never leaks through.
assert all(x["role"] != "model" for x in m)
assert m[2] == {"role": "assistant", "content": "hello!"}
assert m[4] == {"role": "assistant", "content": "4"}
print("ok  role map: model→assistant, no 'model' leaks")

# 3) System-first invariant holds — exactly one system, at index 0.
sys_positions = [i for i, x in enumerate(m) if x["role"] == "system"]
assert sys_positions == [0], sys_positions
print("ok  exactly one system message, at index 0")

# 4) Unknown roles in history are dropped (protects the invariant).
dirty = [{"role": "system", "content": "STRAY"},        # must not survive
         {"role": "tool",   "content": "junk"},          # must not survive
         {"role": "user",   "content": "real"}]
m2 = assemble_messages(SYS, dirty, "q")
assert [x["role"] for x in m2] == ["system", "user", "user"], [x["role"] for x in m2]
assert all(x["content"] != "STRAY" for x in m2)
assert all(x["content"] != "junk" for x in m2)
print("ok  stray system/tool history rows dropped")

# 5) Empty system_prompt → no system message emitted.
m3 = assemble_messages("", HIST, "q")
assert not any(x["role"] == "system" for x in m3)
assert m3[-1] == {"role": "user", "content": "q"}
print("ok  empty system_prompt → no system message")

# 6) Empty history → just [system, user].
m4 = assemble_messages(SYS, [], "first message")
assert m4 == [{"role": "system", "content": SYS},
              {"role": "user", "content": "first message"}]
print("ok  empty history → [system, user]")

# 7) FUSE fallback: no system message; instruction fused into the last user turn
#    — byte-for-byte v1's format.
mf = assemble_messages(SYS, HIST, "capital of France?", supports_system_role=False)
assert not any(x["role"] == "system" for x in mf), "fuse mode must emit no system"
assert mf[-1]["role"] == "user"
assert mf[-1]["content"] == f"{SYS}\n\nUser question: capital of France?"
# history still present and mapped
assert [x["role"] for x in mf] == ["user", "assistant", "user", "assistant", "user"]
print("ok  fuse fallback: v1-exact fusion, no system role, history intact")

# 8) Fuse with empty system → plain user turn, no fusion prefix.
mf2 = assemble_messages("", HIST, "bare", supports_system_role=False)
assert mf2[-1] == {"role": "user", "content": "bare"}
print("ok  fuse + empty system → unmodified user turn")

# 9) System-role mode PASSES THROUGH a content-part list (multimodal-ready).
parts = [{"type": "text", "text": "describe"},
         {"type": "image_url", "image_url": {"url": "x"}}]
mm = assemble_messages(SYS, [], parts)          # allowed in system-role mode
assert mm[-1]["content"] is parts
print("ok  system-role mode passes content-part arrays through untouched")

# 10) Fuse mode REFUSES a content-part list rather than mangling it.
try:
    assemble_messages(SYS, [], parts, supports_system_role=False)
    print("FAIL fuse accepted a content-part array"); sys.exit(1)
except NotImplementedError as e:
    assert "multimodal" in str(e).lower()
print("ok  fuse mode refuses content-parts loudly (points to F6)")

# 11) _map_history in isolation.
assert _map_history([{"role":"model","content":"a"}]) == [{"role":"assistant","content":"a"}]
assert _map_history([{"role":"nope","content":"x"}]) == []
assert _map_history([{"role":"user"}]) == [{"role":"user","content":""}]  # missing content tolerated
print("ok  _map_history maps, drops, and tolerates missing content")

print("\nALL GREEN")
