import sys, types

# Stub core.db and core.prompts BEFORE importing the wrapper (it imports lazily,
# but we register the fakes up front so the lazy import finds them).
fake_db = types.ModuleType("core.db")
fake_prompts = types.ModuleType("core.prompts")

_CALLS = {}
def get_messages(conv_id, max_turns=10):
    _CALLS["get_messages"] = (conv_id, max_turns)
    return [{"msg_id":"1","role":"user","content":"earlier q","files":[]},
            {"msg_id":"2","role":"model","content":"earlier a","files":[]}]
def get_prompt_for_user(user_id):
    _CALLS["get_prompt_for_user"] = (user_id,)
    return f"SYSTEM for {user_id}"

fake_db.get_messages = get_messages
fake_prompts.get_prompt_for_user = get_prompt_for_user
sys.modules["core.db"] = fake_db
sys.modules["core.prompts"] = fake_prompts

from core.inference.messages import build_messages_for_turn

# 1) Wrapper fetches from both sources and assembles correctly.
m = build_messages_for_turn("user-uuid-123", "conv-abc", "current question")
assert _CALLS["get_prompt_for_user"] == ("user-uuid-123",)
assert _CALLS["get_messages"] == ("conv-abc", 10)      # default window
assert m[0] == {"role":"system","content":"SYSTEM for user-uuid-123"}
assert m[1] == {"role":"user","content":"earlier q"}
assert m[2] == {"role":"assistant","content":"earlier a"}   # model→assistant
assert m[-1] == {"role":"user","content":"current question"}
print("ok  wrapper: fetches prompt+history, maps, assembles system-first")

# 2) max_turns is forwarded to get_messages.
build_messages_for_turn("u", "c", "q", max_turns=25)
assert _CALLS["get_messages"] == ("c", 25)
print("ok  wrapper: max_turns forwarded")

# 3) supports_system_role=False flows through to fuse behaviour.
mf = build_messages_for_turn("u2", "c2", "hello", supports_system_role=False)
assert not any(x["role"]=="system" for x in mf)
assert mf[-1]["content"] == "SYSTEM for u2\n\nUser question: hello"
print("ok  wrapper: fuse flag flows through")

print("\nALL GREEN")
