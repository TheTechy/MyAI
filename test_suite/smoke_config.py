import ast, importlib.util, sys
from dataclasses import fields, FrozenInstanceError
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent          # test_suite/ -> repo root

#SRC = "config.py"
SRC = str(_REPO / "core" / "inference" / "config.py")
ast.parse(open(SRC).read()); print("ok  ast.parse")

spec = importlib.util.spec_from_file_location("myai_inf_config", SRC)
mod = importlib.util.module_from_spec(spec); sys.modules["myai_inf_config"] = mod
spec.loader.exec_module(mod); print("ok  import (no llama_cpp needed)")

load = mod.load_config
BC, LP, CErr = mod.BackendConfig, mod.LlamaCppParams, mod.ConfigError

# 1) Family "no config" path: only MODEL set → in-process default, ctx default.
c = load({"MODEL": "/models/gemma4-12b.gguf"})
assert c.kind == "llama_cpp", c.kind
assert c.model_path == "/models/gemma4-12b.gguf"
assert c.warmup is True
assert c.llama_cpp is not None and c.llama_cpp.n_ctx == mod.DEFAULT_N_CTX
print("ok  family no-config default → in-process, ctx default")

# 2) v1 tuning knobs preserved verbatim in the defaults.
lp = LP()
assert (lp.n_gpu_layers, lp.n_threads, lp.n_threads_batch) == (-1, 6, 12)
assert (lp.n_batch, lp.n_ubatch) == (512, 512)
assert lp.flash_attn and lp.use_mmap
assert (lp.type_k, lp.type_v) == (8, 8)
assert lp.verbose is False
print("ok  v1 M4 Pro tuning preserved in defaults")

# 3) MODEL missing → actionable ConfigError.
for bad in ({}, {"MODEL": ""}, {"MODEL": "   "}):
    try:
        load(bad); print("FAIL missing MODEL accepted", bad); sys.exit(1)
    except CErr as e:
        assert "MODEL" in str(e)
print("ok  MODEL required → ConfigError")

# 4) CTX_SIZE honoured + validated.
assert load({"MODEL":"m","CTX_SIZE":"16384"}).llama_cpp.n_ctx == 16384
assert load({"MODEL":"m","CTX_SIZE":"0"}).llama_cpp.n_ctx == 0   # 0 = trained max
for bad in ("notanint", "-1"):
    try:
        load({"MODEL":"m","CTX_SIZE":bad}); print("FAIL bad CTX_SIZE", bad); sys.exit(1)
    except CErr:
        pass
print("ok  CTX_SIZE parsed + validated")

# 5) WARMUP toggle.
assert load({"MODEL":"m"}).warmup is True
assert load({"MODEL":"m","WARMUP":"0"}).warmup is False
print("ok  WARMUP toggle")

# 6) Unknown backend kind passes through untouched (registry judges it, not config),
#    and carries NO llama_cpp block.
u = load({"MODEL":"m","MYAI_BACKEND":"http"})
assert u.kind == "http" and u.llama_cpp is None
print("ok  unknown kind passes through, no llama_cpp block")

# 7) Everything frozen.
for obj in (BC(kind="llama_cpp", model_path="m"), LP()):
    try:
        object.__setattr__  # noqa
        setattr(obj, "model_path" if isinstance(obj, BC) else "n_ctx", 1)
        print("FAIL not frozen", type(obj)); sys.exit(1)
    except FrozenInstanceError:
        pass
print("ok  configs frozen")

print("\nALL GREEN")
