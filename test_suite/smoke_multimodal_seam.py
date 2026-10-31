# F2 S3 T1: the multimodal content-parts SEAM. No image handling is built — this
# only proves the assembler carries an OpenAI content-part array as message
# content, unchanged, so multimodal slots in at F6 without an assembler refactor.
import sys
from core.inference.messages import assemble_messages, _map_history

# An OpenAI-shaped multimodal user turn: text + image parts.
PARTS = [
    {"type": "text", "text": "What's in this picture?"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
]

# 1) System-role mode: the content-part array passes through UNTOUCHED (identity).
m = assemble_messages("SYS", [], PARTS)          # supports_system_role=True default
assert m[0] == {"role": "system", "content": "SYS"}
assert m[-1]["role"] == "user"
assert m[-1]["content"] is PARTS, "content-part array must pass through by identity"
print("ok  content-part array carried through untouched (system-role mode)")

# 2) It also survives with prior history in place (position + mapping intact).
hist = [{"role": "user", "content": "hi"}, {"role": "model", "content": "hello"}]
m2 = assemble_messages("SYS", hist, PARTS)
assert [x["role"] for x in m2] == ["system", "user", "assistant", "user"]
assert m2[-1]["content"] is PARTS
print("ok  content-parts survive alongside mapped history")

# 3) A content-part array in HISTORY (a prior image turn) is preserved too.
hist_mm = [{"role": "user", "content": PARTS}, {"role": "model", "content": "a cat"}]
mapped = _map_history(hist_mm)
assert mapped[0]["content"] is PARTS and mapped[0]["role"] == "user"
assert mapped[1] == {"role": "assistant", "content": "a cat"}
print("ok  content-part arrays preserved through history mapping")

# 4) Text-only turns still behave exactly as before (no regression).
m3 = assemble_messages("SYS", [], "plain text question")
assert m3[-1] == {"role": "user", "content": "plain text question"}
print("ok  plain-string turns unchanged (no regression)")

# 5) The DELIBERATE boundary: fuse mode CANNOT string-fuse an array, and must
#    refuse loudly rather than mangle it — until multimodal fusion lands (F6).
try:
    assemble_messages("SYS", [], PARTS, supports_system_role=False)
    print("FAIL fuse mode silently accepted a content-part array"); sys.exit(1)
except NotImplementedError as e:
    assert "multimodal" in str(e).lower() and "F6" in str(e)
print("ok  fuse mode refuses content-parts loudly, points to F6")

# 6) Seam is shape-agnostic: an arbitrary future part type is carried, not parsed.
future = [{"type": "text", "text": "hi"}, {"type": "audio", "audio": {"id": "x"}}]
m4 = assemble_messages("SYS", [], future)
assert m4[-1]["content"] is future
print("ok  unknown future part types pass through (assembler doesn't parse parts)")

print("\nALL GREEN")
