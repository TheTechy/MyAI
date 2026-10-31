import sys, json, importlib
import fake_llama_cpp
sys.modules["llama_cpp"] = fake_llama_cpp          # inject before lazy import

from core.inference.config import load_config, LlamaCppParams
from core.inference.base import InferenceBackend, OutputConstraint
from core.inference.llama_cpp_backend import (
    LlamaCppBackend, BackendNotLoadedError, ConstraintError)

MODEL = "/PATH/MyAi/models/gemma-4-31B-it-UD-Q8_K_XL.gguf"
ENV = {"MODEL": MODEL, "CTX_SIZE": "32768", "WARMUP": "1"}   # from the live .env

# 1) Satisfies the ABC — instantiable, so no abstract method is missing.
cfg = load_config(ENV)
b = LlamaCppBackend(cfg)
assert isinstance(b, InferenceBackend)
print("ok  implements InferenceBackend (ABC satisfied)")

# 2) Calls before load() raise clearly, not AttributeError.
for fn in (lambda: b.context_window(), lambda: b.context_used(),
           lambda: b.create_chat_completion([{"role":"user","content":"x"}])):
    try:
        fn(); print("FAIL pre-load call allowed"); sys.exit(1)
    except BackendNotLoadedError: pass
assert b.emits_think_reasoning is False        # safe default, no raise
print("ok  pre-load calls raise BackendNotLoadedError")

# 3) load() passes v1's EXACT tuning params through, with the live ctx size.
b.load()
kw = fake_llama_cpp.Llama.instances[-1].init_kwargs
assert kw["model_path"] == MODEL
assert kw["n_ctx"] == 32768                                   # from .env
assert kw["n_gpu_layers"] == -1 and kw["verbose"] is False
assert kw["n_threads"] == 6 and kw["n_threads_batch"] == 12
assert kw["n_batch"] == 512 and kw["n_ubatch"] == 512
assert kw["flash_attn"] is True and kw["use_mmap"] is True
assert kw["type_k"] == 8 and kw["type_v"] == 8
print("ok  v1 Llama() params forwarded verbatim, n_ctx=32768 from .env")

# 4) load() is idempotent.
n = len(fake_llama_cpp.Llama.instances)
b.load()
assert len(fake_llama_cpp.Llama.instances) == n
assert b.is_loaded
print("ok  load() idempotent")

# 5) Introspection.
assert b.context_window() == 32768
assert b.context_used() == 42
print("ok  context_window/context_used")

# 6) Non-stream returns the OpenAI shape at the documented path.
out = b.create_chat_completion([{"role":"user","content":"hi"}])
assert out["choices"][0]["message"]["content"] == "hello"
print("ok  non-stream shape: choices[0].message.content")

# 7) Stream yields chunks at the OTHER documented path + finish_reason.
chunks = list(b.create_chat_completion([{"role":"user","content":"hi"}], stream=True))
assert "".join(c["choices"][0]["delta"]["content"] for c in chunks) == "hello"
assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
print("ok  stream shape: choices[0].delta.content + finish_reason")

# 8) None-valued optionals are NOT forwarded (llama-cpp defaults must survive).
llm = fake_llama_cpp.Llama.instances[-1]; llm.calls.clear()
b.create_chat_completion([{"role":"user","content":"hi"}])
sent = llm.calls[-1]
assert "temperature" not in sent and "stop" not in sent and "max_tokens" not in sent
print("ok  unset optionals not forwarded (llama-cpp defaults preserved)")

# 9) Set optionals ARE forwarded.
llm.calls.clear()
b.create_chat_completion([{"role":"user","content":"hi"}],
                         max_tokens=64, temperature=0.0, stop=["\n"])
sent = llm.calls[-1]
assert sent["max_tokens"] == 64 and sent["temperature"] == 0.0 and sent["stop"] == ["\n"]
print("ok  set optionals forwarded")

# 10) GBNF constraint → LlamaGrammar.from_string.
llm.calls.clear()
gbnf = 'root ::= "direct" | "web_search"'
b.create_chat_completion([{"role":"user","content":"hi"}],
                         constraint=OutputConstraint(gbnf=gbnf))
g = llm.calls[-1]["grammar"]
assert g.origin == ("gbnf", gbnf, False)
print("ok  gbnf constraint → LlamaGrammar.from_string")

# 11) json_schema constraint → from_json_schema (0.3.33 bonus: both honoured).
llm.calls.clear()
schema = {"type":"object","properties":{"route":{"type":"string"}}}
b.create_chat_completion([{"role":"user","content":"hi"}],
                         constraint=OutputConstraint(json_schema=schema))
g = llm.calls[-1]["grammar"]
assert g.origin[0] == "schema" and json.loads(g.origin[1]) == schema
print("ok  json_schema constraint → LlamaGrammar.from_json_schema")

# 12) THE CRITICAL RULE: empty constraint raises, never silently unconstrained.
llm.calls.clear()
try:
    b.create_chat_completion([{"role":"user","content":"hi"}],
                             constraint=OutputConstraint())
    print("FAIL empty constraint silently allowed"); sys.exit(1)
except ConstraintError: pass
assert not llm.calls, "must not have called the model at all"
print("ok  empty constraint RAISES and never reaches the model")

# 13) Non-serialisable schema also raises.
try:
    b.create_chat_completion([{"role":"user","content":"hi"}],
                             constraint=OutputConstraint(json_schema={"x": {1,2}}))
    print("FAIL bad schema allowed"); sys.exit(1)
except ConstraintError: pass
print("ok  non-serialisable json_schema raises")

# 14) warmup() honours config and runs a 1-token completion.
llm.calls.clear()
assert b.warmup() is True
assert llm.calls[-1]["max_tokens"] == 1
b2 = LlamaCppBackend(load_config({**ENV, "WARMUP":"0"})); b2.load()
assert b2.warmup() is False
print("ok  warmup respects WARMUP flag")

# 15) unload() closes, is safe twice, and load() works again (model swap path).
b.unload()
assert llm.closed is True and not b.is_loaded
b.unload()                                   # no raise
b.load()
assert b.is_loaded and b.context_window() == 32768
print("ok  unload/reload (model-swap) path")

# 16) think sniff — Gemma (no <think>) vs Qwen-style template.
class G(fake_llama_cpp.Llama):
    def __init__(self, **kw): super().__init__(**kw); self.metadata = {"tokenizer.chat_template": "<start_of_turn>user"}
class Q(fake_llama_cpp.Llama):
    def __init__(self, **kw): super().__init__(**kw); self.metadata = {"tokenizer.chat_template": "...<think>\n..."}
class N(fake_llama_cpp.Llama):
    def __init__(self, **kw): super().__init__(**kw); self.metadata = {}

for cls, expect, label in ((G, False, "gemma"), (Q, True, "qwen"), (N, False, "no-metadata")):
    fake_llama_cpp.Llama_orig = fake_llama_cpp.Llama
    fake_llama_cpp.Llama = cls
    t = LlamaCppBackend(load_config(ENV)); t.load()
    assert t.emits_think_reasoning is expect, label
    fake_llama_cpp.Llama = fake_llama_cpp.Llama_orig
print("ok  <think> sniff: gemma=False, qwen=True, missing-metadata=False")

print("\nALL GREEN")
