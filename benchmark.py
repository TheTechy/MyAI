"""Quick token-per-second benchmark for the current MODEL."""
# USAGE: python3 benchmark.py
import os
import time
from pathlib import Path
from llama_cpp import Llama
from dotenv import load_dotenv

load_dotenv()
model_path = os.getenv("MODEL")
ctx_size   = int(os.getenv("CTX_SIZE", "4096"))

print(f"Loading {Path(model_path).name}...")
llm = Llama(
    model_path=model_path,
    n_ctx=ctx_size,
    n_gpu_layers=-1,
    verbose=False,
    flash_attn=True,
    use_mmap=True,
)

# Warm up so JIT / page-in costs don't skew the result
print("Warming up...")
llm("Hi", max_tokens=1, echo=False)

# Real measurement
prompt = "<start_of_turn>user\nWrite a one-paragraph explanation of how photosynthesis works.<end_of_turn>\n<start_of_turn>model\n"

print("Measuring...")
start = time.perf_counter()
result = llm(
    prompt,
    max_tokens=200,
    stop=["<end_of_turn>"],
    echo=False,
    stream=False,
    temperature=0.7,
)
elapsed = time.perf_counter() - start

tokens_generated = result["usage"]["completion_tokens"]
tps = tokens_generated / elapsed

print(f"\n── Results ──")
print(f"Model:       {Path(model_path).name}")
print(f"Generated:   {tokens_generated} tokens")
print(f"Time:        {elapsed:.2f}s")
print(f"Throughput:  {tps:.1f} tok/s")