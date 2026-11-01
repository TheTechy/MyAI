"""How full the context will be on a chat's next message (the context bar)."""
import sys, json, types, io, contextlib
import fake_llama_cpp as F
sys.modules["llama_cpp"] = F

from core.inference.base import InferenceBackend
from core.inference.config import load_config
from core.inference.llama_cpp_backend import LlamaCppBackend, BackendNotLoadedError
from core.inference.pipeline import (
    MAX_HISTORY_TURNS, PipelineDeps, context_usage, run_turn)
from core.inference.dispatch import build_skill_map
from core.inference.router import RouteDecision

def loaded_backend():
    b = LlamaCppBackend(load_config({"MODEL": "/m.gguf", "CTX_SIZE": "1000"}))
    with contextlib.redirect_stdout(io.StringIO()):
        b.load()
    return b

# 1) Backend: counting needs a loaded model.
try:
    LlamaCppBackend(load_config({"MODEL": "/m.gguf"})).count_tokens([]); sys.exit("FAIL pre-load")
except BackendNotLoadedError: pass
print("ok  count_tokens raises before load")

# 2) Backend without a chat template: counts the message text itself.
b = loaded_backend()
msgs = [{"role": "system", "content": "be kind"}, {"role": "user", "content": "hi there friend"}]
assert b.count_tokens(msgs) == 5, b.count_tokens(msgs)
print("ok  no template → counts the message contents")

# 3) Backend with a chat template: counts the prompt the model would really see,
#    rendered by llama-cpp's own formatter, tokenized with its special tokens.
class Rendered:
    def __init__(self, prompt): self.prompt = prompt
class FakeFormatter:
    def __init__(self, template, eos_token, bos_token): FakeFormatter.bos = bos_token
    def __call__(self, messages):
        return Rendered(" ".join(f"<{m['role']}> {m['content']} <end>" for m in messages))
sys.modules["llama_cpp.llama_chat_format"] = types.SimpleNamespace(Jinja2ChatFormatter=FakeFormatter)
class Templated(F.Llama):
    def __init__(self, **kw):
        super().__init__(**kw); self.metadata = {"tokenizer.chat_template": "{{ messages }}"}
    def token_eos(self): return 1
    def token_bos(self): return 2
    def detokenize(self, ids): return b""     # like llama-cpp: special tokens render as ''
plain = F.Llama
F.Llama = Templated
b = loaded_backend()
assert b.count_tokens(msgs) == 5 + 4, b.count_tokens(msgs)    # + a role tag and <end> per message
assert b._llm.tokenize_calls[-1][1:] == (True, True)           # as llama-cpp's chat handler tokenizes it
Rendered.added_special = True
b.count_tokens(msgs)
assert b._llm.tokenize_calls[-1][1:] == (False, True)          # formatter already added BOS
print("ok  template → counts the rendered prompt, tokenized as llama-cpp does")

# 3b) The template gets the real BOS/EOS text, as llama-cpp gives it (not detokenize's '').
class WithTokenText(Templated):
    def __init__(self, **kw):
        super().__init__(**kw)
        self._model.token_get_text = lambda i: {1: "<eos>", 2: "<bos>"}[i]
F.Llama = WithTokenText
loaded_backend()
assert FakeFormatter.bos == "<bos>", FakeFormatter.bos
print("ok  template is given the special-token text llama-cpp uses")
F.Llama = plain
del sys.modules["llama_cpp.llama_chat_format"]

# 4) Seam: optional, defaults to None so no backend breaks.
assert "count_tokens" not in InferenceBackend.__abstractmethods__
class Min(InferenceBackend):
    def load(self): pass
    def unload(self): pass
    def create_chat_completion(self, messages, **kw): ...
    @property
    def emits_think_reasoning(self): return False
    def context_window(self): return 1
    def context_used(self): return None
assert Min().count_tokens([]) is None
print("ok  count_tokens is optional on the seam, defaults to None")

# ── pipeline ────────────────────────────────────────────────────────────────────
class WordBackend:
    """Counts one token per word of content, and records what it was asked."""
    supports_system_role = True
    emits_think_reasoning = False
    def __init__(self, answer="Paris it is."): self.answer = answer; self.counted = []
    def count_tokens(self, messages):
        self.counted.append(messages)
        return sum(len(str(m["content"]).split()) for m in messages)
    def context_window(self): return 1000
    def context_used(self): return 999       # the engine's live count must NOT be used
    def create_chat_completion(self, messages, **kw):
        def gen():
            yield {"choices": [{"delta": {"content": self.answer}, "finish_reason": None}]}
            yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}
        return gen()

def make_deps(store, files=()):
    return PipelineDeps(
        get_messages=lambda c, n: store[-n * 2:],
        get_prompt_for_user=lambda u: "you are kind",
        get_file_contexts=lambda c: list(files),
        append_message=lambda c, r, t: store.append({"role": r, "content": t}) or f"m{len(store)}",
        list_memories=lambda u: [],
        auto_title=lambda c, u: None,
        title_event=lambda c: None,
        awaiting_clear=lambda h: False,
        is_confirmation=lambda u: False,
        has_file_block=lambda t: False,
        extract_and_generate=lambda t: [],
        strip_file_blocks=lambda t: t,
        persist_files=lambda m, f: None,
        file_gen_snippet="files ok",
    )

# 5) Next-turn usage = system prompt (+ file snippet) + recent history + attached files.
store = [{"role": "user", "content": "old question"}, {"role": "model", "content": "old answer"}]
files = [{"filename": "a.txt", "extracted_text": "alpha beta gamma"}]
be = WordBackend()
u = context_usage("u1", "c1", backend=be, deps=make_deps(store, files))
text = " ".join(str(m["content"]) for m in be.counted[-1])
for part in ("you are kind", "files ok", "old question", "old answer", "alpha beta gamma"):
    assert part in text, part
assert u == {"used": u["used"], "max": 1000, "remaining": 1000 - u["used"],
             "pct": round(u["used"] / 1000 * 100, 1)}, u
assert u["used"] == be.count_tokens(be.counted[-1])
print("ok  usage covers system prompt, history and attached files")

# 6) Only the history the next turn will actually send is counted.
long = []
for i in range(MAX_HISTORY_TURNS + 5):
    long += [{"role": "user", "content": f"q{i}"}, {"role": "model", "content": f"a{i}"}]
be = WordBackend()
context_usage("u1", "c1", backend=be, deps=make_deps(long))
sent = [m["content"] for m in be.counted[-1]]
assert "q0" not in sent and f"q{MAX_HISTORY_TURNS + 4}" in sent
print("ok  history beyond MAX_HISTORY_TURNS is not counted")

# 7) A backend that can't count → None (the bar shows "—", the chat carries on).
class Blind(WordBackend):
    def count_tokens(self, messages): return None
assert context_usage("u1", "c1", backend=Blind(), deps=make_deps([])) is None
print("ok  backend can't count → None")

def events(user_input, *, backend, deps, skill_map, route):
    out = []
    for raw in run_turn(user_input, "u1", "c1", backend=backend, skill_map=skill_map,
                        deps=deps, route_fn=lambda b, q: RouteDecision(route, True, True, "", "")):
        head, _, body = raw.partition("\n")
        out.append((head[len("event: "):], json.loads(body[len("data: "):])))
    return out

# 8) After a model answer, the ctx event is the NEXT turn's usage (answer included).
store = []
be = WordBackend()
deps = make_deps(store)
evs = events("capital of France?", backend=be, deps=deps, skill_map={}, route="direct")
ctx = [d for e, d in evs if e == "ctx"]
assert len(ctx) == 1 and [e for e, _ in evs][-1] == "done"
assert ctx[0] == context_usage("u1", "c1", backend=be, deps=deps), ctx
assert ctx[0]["used"] != 999
print("ok  answer turn: ctx = next turn's usage, not the engine's live count")

# 9) Turns with no model pass (memory replies) still update the bar.
class Memory:
    name = "memory"
    def execute(self, q, user_id=None, history=None): return "Saved."
store = []
be = WordBackend()
deps = make_deps(store)
evs = events("remember I like tea", backend=be, deps=deps,
             skill_map=build_skill_map([Memory()]), route="memory")
names = [e for e, _ in evs]
assert "ctx" in names and names[-1] == "done", names
assert dict(evs)["ctx"] == context_usage("u1", "c1", backend=be, deps=deps)
print("ok  memory turn (no model pass) still sends ctx")

print("\nALL GREEN")
