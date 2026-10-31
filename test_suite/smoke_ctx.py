import sys, io, contextlib
import fake_llama_cpp as F
sys.modules["llama_cpp"] = F

from core.inference.base import InferenceBackend
from core.inference.config import load_config, DEFAULT_N_CTX, ConfigError
from core.inference.llama_cpp_backend import LlamaCppBackend, BackendNotLoadedError

def fresh(trained=8192, api_ok=True, **env):
    class L(F.Llama): TRAINED = trained; API_OK = api_ok
    F.Llama = L
    e = {"MODEL": "/m.gguf"}; e.update(env)
    b = LlamaCppBackend(load_config(e))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        b.load()
    return b, buf.getvalue()

# 1) CTX_SIZE unset → 0 → resolved to the model's trained maximum.
assert DEFAULT_N_CTX == 0
b, log = fresh(trained=131072)
assert b.context_window() == 131072, b.context_window()
assert b.trained_context_window() == 131072
print("ok  CTX_SIZE unset → model's trained maximum (131072)")

# 2) Explicit CTX_SIZE is honoured, trained reported alongside it.
b, log = fresh(trained=131072, CTX_SIZE="32768")
assert b.context_window() == 32768 and b.trained_context_window() == 131072
assert "32768 allocated / 131072 trained" in log
print("ok  explicit CTX_SIZE honoured; both values logged")

# 3) Over-request WARNS but still loads (a valid experiment, not an error).
b, log = fresh(trained=8192, CTX_SIZE="32768")
assert "WARNING" in log and "exceeds" in log
assert b.context_window() == 32768          # not clamped
print("ok  CTX_SIZE > trained → warns loudly, still loads, not clamped")

# 4) Under-request is silent (perfectly normal).
b, log = fresh(trained=131072, CTX_SIZE="8192")
assert "WARNING" not in log
print("ok  CTX_SIZE < trained → no warning")

# 5) Fallback chain: public C API fails → private wrapper still works.
b, log = fresh(trained=40960, api_ok=False, CTX_SIZE="1024")
assert b.trained_context_window() == 40960
print("ok  falls back to _model.n_ctx_train() when C API path fails")

# 6) Both fail → metadata key path.
class NoAPI(F.Llama):
    TRAINED = None; API_OK = False
    def __init__(self, **kw):
        super().__init__(**kw)
        self.metadata = {"general.architecture": "gemma3",
                         "gemma3.context_length": "131072",
                         "tokenizer.chat_template": ""}
F.Llama = NoAPI
b = LlamaCppBackend(load_config({"MODEL":"/m.gguf","CTX_SIZE":"1024"}))
with contextlib.redirect_stdout(io.StringIO()): b.load()
assert b.trained_context_window() == 131072
print("ok  falls back to {arch}.context_length metadata key")

# 7) All paths fail → None, and the model STILL loads.
class Blind(F.Llama):
    TRAINED = None; API_OK = False
    def __init__(self, **kw):
        super().__init__(**kw); self.metadata = {}
F.Llama = Blind
b = LlamaCppBackend(load_config({"MODEL":"/m.gguf","CTX_SIZE":"4096"}))
buf = io.StringIO()
with contextlib.redirect_stdout(buf): b.load()
assert b.trained_context_window() is None
assert b.context_window() == 4096
assert "trained unknown" in buf.getvalue()
print("ok  unreadable trained ctx → None, load still succeeds")

# 8) Seam default is None and NON-abstract (no backend breaks).
assert "trained_context_window" not in InferenceBackend.__abstractmethods__
class Min(InferenceBackend):
    def load(self): pass
    def unload(self): pass
    def create_chat_completion(self, messages, *, max_tokens=None, temperature=None,
                               stop=None, stream=False, constraint=None, tools=None): ...
    @property
    def emits_think_reasoning(self): return False
    def context_window(self): return 1
    def context_used(self): return None
assert Min().trained_context_window() is None
print("ok  trained_context_window concrete on seam, defaults None")

# 9) Pre-load raises; cleared on unload.
F.Llama = F.Llama if not hasattr(F.Llama,'TRAINED') else F.Llama
b2, _ = fresh(trained=8192, CTX_SIZE="8192")
b2.unload()
try:
    b2.trained_context_window(); print("FAIL"); sys.exit(1)
except BackendNotLoadedError: pass
print("ok  trained_context_window raises pre-load / post-unload")

# 10) Config: 0 explicitly allowed, negatives still rejected.
assert load_config({"MODEL":"/m","CTX_SIZE":"0"}).llama_cpp.n_ctx == 0
for bad in ("-1","-4096"):
    try:
        load_config({"MODEL":"/m","CTX_SIZE":bad}); print("FAIL", bad); sys.exit(1)
    except ConfigError: pass
try:
    load_config({"MODEL":"/m","CTX_SIZE":"abc"}); print("FAIL abc"); sys.exit(1)
except ConfigError: pass
print("ok  CTX_SIZE=0 allowed, negatives + non-int rejected")

print("\nALL GREEN")
