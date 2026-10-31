#!/usr/bin/env python3
"""
scripts/router_eval.py — F3 S1 T4

Ports the Arm B routing/ambiguity probes into a standing eval for the production
router (full 11-route menu + grammar-repair). Measures the numbers the plan calls
the crux, now on the real router:

  1. routing correctness   parsed route == expected, over a labelled set
  2. free-pass vs repair    how often the free reason-then-emit pass was clean vs
                            needed the grammar-locked repair (T2), vs fell back
  3. consistency            same query, N runs → stable route (the 9/9 baseline)
  4. decision latency       wall time per decision (vs the ~16s classifier)

Target: hold Gemma 4 12B at ~100% parse / stable consistency. Run Gemma first
(the baseline), then Qwen (stresses the grammar-repair + thinking interaction).

    python scripts/router_eval.py --model ./models/gemma-4-12b-it-UD-Q8_K_XL.gguf
    python scripts/router_eval.py --model ./models/Qwen3.5-27B-UD-Q8_K_XL.gguf --consistency-runs 9
    python scripts/router_eval.py --model ... --quick        # correctness only, no consistency

Recommendation queries are scored on CONSISTENCY, not correctness (policy: either
answer is defensible). Running the real model also implicitly validates that the
route grammar compiles in llama-cpp (the repair path exercises it).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

CONSISTENCY = "consistency"   # sentinel: score stability, not correctness

# ── labelled test set: (query, expected, group) ────────────────────────────────
TEST_SET = [
    # clear tool routes
    ("what's the weather in Rochdale tomorrow?", "weather", "tools"),
    ("will it rain in Leeds today?", "weather", "tools"),
    ("what's 15% of 240", "calculator", "tools"),
    ("what is 12345 * 678", "calculator", "tools"),
    ("convert 50 usd to gbp", "currency", "tools"),
    ("how much is $250 in pounds", "currency", "tools"),
    ("what time is it in Tokyo?", "datetime", "tools"),
    ("how many days until Christmas", "datetime", "tools"),
    ("what's in the news today", "news", "tools"),
    ("who won the football match last night", "web_search", "tools"),
    ("driving directions from Manchester to Leeds", "directions", "tools"),
    ("resize this image to 800x600", "image", "tools"),
    ("remember I'm allergic to peanuts", "memory", "tools"),
    ("look up the Eiffel Tower on Wikipedia", "wikipedia", "tools"),
    # volatility: timeless → direct, changeable → web_search
    ("what's the capital of France", "direct", "volatility"),
    ("when did World War 2 end", "direct", "volatility"),
    ("who wrote Pride and Prejudice", "direct", "volatility"),
    ("who was Isaac Newton", "direct", "volatility"),          # tests the wikipedia decision
    ("who is the current Prime Minister", "web_search", "volatility"),
    ("what's the population of France", "web_search", "volatility"),
    ("what's the latest iPhone", "web_search", "volatility"),
    # unit-conversion nuance
    ("how many centimetres in an inch", "direct", "conversion"),   # definitional
    ("convert 5 miles to km", "calculator", "conversion"),         # numeric
    # multi-skill → core ask
    ("convert 100 USD to EUR and is that a good rate", "currency", "multiskill"),
    ("what's the weather in Paris and what time is it there", "weather", "multiskill"),
    # keyword traps → semantic (direct)
    ("how do I convert my life around", "direct", "trap"),
    ("what's the date with Sarah on Friday", "direct", "trap"),
    ("directions in life for a lost soul", "direct", "trap"),
    # degenerate → direct
    ("hi", "direct", "degenerate"),
    ("thanks", "direct", "degenerate"),
    ("???", "direct", "degenerate"),
    # recommendations → consistency only
    ("what's the best laptop to buy", CONSISTENCY, "recommendation"),
    ("what should I cook for dinner", CONSISTENCY, "recommendation"),
]

# queries to repeat for the consistency check
CONSISTENCY_SET = [
    "what's the weather in Rochdale tomorrow?",
    "who is the current Prime Minister",
    "who was Isaac Newton",
    "what's the best laptop to buy",
    "how do I convert my life around",
]


def _format_row(row):
    outcome, group, expected, got, rep, dt, query = row
    tag = {"ok": "  ok ", "WRONG": "WRONG", "consist": " ~   "}[outcome]
    r = " (repaired)" if rep else ""
    return (f"  [{tag}] {group:<13} exp={expected:<11} got={got:<11}{r} "
            f"{dt:5.2f}s | {query[:40]!r}")


def run_eval(decide, test_set=TEST_SET, consistency_set=CONSISTENCY_SET,
             consistency_runs=5, quick=False, progress=False):
    """Run the eval using `decide(query) -> RouteDecision`. Pure of IO/model so it
    can be unit-tested with an injected decider. With progress=True, prints each
    per-query line as it is computed (so a long Mini run isn't a black box)."""
    rows, latencies = [], []
    correct = graded = 0
    free_clean = repaired = fell_back = 0
    n = len(test_set)

    for i, (query, expected, group) in enumerate(test_set, 1):
        t = time.perf_counter()
        d = decide(query)
        dt = time.perf_counter() - t
        latencies.append(dt)

        if d.repaired:
            repaired += 1
        elif d.parsed:
            free_clean += 1
        else:
            fell_back += 1

        if expected == CONSISTENCY:
            outcome = "consist"
        else:
            graded += 1
            hit = (d.route == expected)
            correct += hit
            outcome = "ok" if hit else "WRONG"
        row = (outcome, group, expected, d.route, d.repaired, dt, query)
        rows.append(row)
        if progress:
            print(f"[{i:2}/{n}]{_format_row(row)}", flush=True)

    consistency = []
    if not quick:
        if progress:
            print(f"\n[consistency: {len(consistency_set)} queries × "
                  f"{consistency_runs} runs]", flush=True)
        for query in consistency_set:
            routes = [decide(query).route for _ in range(consistency_runs)]
            counts = Counter(routes)
            top = counts.most_common(1)[0][1]
            consistency.append((query, top, consistency_runs, dict(counts)))
            if progress:
                stable = "STABLE" if top == consistency_runs else "VARIES"
                print(f"  [{stable}] {top}/{consistency_runs}  {dict(counts)}  "
                      f"| {query[:38]!r}", flush=True)

    return {
        "rows": rows,
        "graded": graded, "correct": correct,
        "free_clean": free_clean, "repaired": repaired, "fell_back": fell_back,
        "n": len(test_set), "latencies": latencies,
        "consistency": consistency, "consistency_runs": consistency_runs,
    }


def print_report(res, show_rows=True):
    if show_rows:
        print("\n── per-query ──")
        for row in res["rows"]:
            print(_format_row(row))

    n = res["n"]; lat = res["latencies"]
    acc = res["correct"] / res["graded"] * 100 if res["graded"] else 0
    print("\n── verdict ──")
    print(f"  routing correctness : {res['correct']}/{res['graded']}  ({acc:.0f}%)"
          f"   [{n - res['graded']} consistency-only, excluded]")
    print(f"  free-pass clean     : {res['free_clean']}/{n}")
    print(f"  needed repair (T2)  : {res['repaired']}/{n}")
    print(f"  fell back to direct : {res['fell_back']}/{n}")
    print(f"  decision latency    : avg {sum(lat)/len(lat):.2f}s "
          f"(min {min(lat):.2f} / max {max(lat):.2f})")
    if show_rows and res["consistency"]:
        print(f"\n  consistency ({res['consistency_runs']} runs each):")
        for query, top, k, counts in res["consistency"]:
            stable = "STABLE" if top == k else "VARIES"
            print(f"    [{stable}] {top}/{k}  {counts}  | {query[:38]!r}")

    fails = [r for r in res["rows"] if r[0] == "WRONG"]
    unstable = [c for c in res["consistency"] if c[1] != c[2]]
    ok = not fails and not unstable
    print("\n" + "=" * 60)
    print("ALL GREEN — baseline held" if ok
          else f"ATTENTION — {len(fails)} misroute(s), {len(unstable)} unstable")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--consistency-runs", type=int, default=5)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    env = REPO_ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    if args.model:
        os.environ["MODEL"] = args.model

    from core.inference.config import load_config
    from core.inference.llama_cpp_backend import LlamaCppBackend
    from core.inference.router import route

    cfg = load_config()
    if not Path(cfg.model_path).exists():
        print(f"model not found: {cfg.model_path}")
        return 2

    backend = LlamaCppBackend(cfg)
    backend.load()
    print(f"model: {Path(cfg.model_path).name}  "
          f"emits_think={backend.emits_think_reasoning}")
    backend.warmup()

    res = run_eval(lambda q: route(backend, q),
                   consistency_runs=args.consistency_runs, quick=args.quick,
                   progress=True)
    rc = print_report(res, show_rows=False)
    backend.unload()
    return rc


if __name__ == "__main__":
    sys.exit(main())
