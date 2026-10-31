# F4 — Parity Acceptance Checklist

**Purpose.** The go/no-go gate for cutting `app.py` over from v1
(`core/llm_inference.py`) to v2 (`core/inference/engine.ask_stream`). It maps every
v1 capability to its v2 proof, logs the deltas we've accepted, and — most
importantly — lists what is **not yet proven**, so the cutover is a conscious
sign-off rather than a leap.

**Legend:** ✅ proven (evidence named) · ⚠️ accepted delta · ⬜ gap (not yet proven)

---

## 1. Core request flow

- ✅ **Routing** — `router_eval` on three real models (Gemma 4 12B 94%, Gemma 4 31B
  97%, Qwen 3.5 90%; free-pass parse 100%; consistency stable) and `parity_check`
  (both models, 0 pipeline errors).
- ✅ **Skill dispatch** — `smoke_dispatch` (12 cases: every per-skill kwarg
  convention, errors captured, drift detection) + `parity_check`.
- ✅ **Streaming, reasoning-strip, truncation leak-guard** — `smoke_streaming` +
  `smoke_stream_ui` on hardware (real Qwen truncation suppressed, no CoT leak).
- ✅ **SSE event contract** — `smoke_pipeline_exec` (status/token/ctx/done/file/
  error/title, all branches).

## 2. Skills (10) — execute end-to-end on the v2 core

All via `engine.ask_stream` in `parity_check`, Gemma 4 12B **and** Qwen 3.5,
**0 pipeline errors**. API-dependent skills confirmed live (keys/network working).

- ✅ calculator *(see delta D1)* · ✅ currency · ✅ weather · ✅ datetime ·
  ✅ news · ✅ directions · ✅ wikipedia · ✅ web_search · ✅ memory
- ⬜ **image** — not exercised on hardware (needs an uploaded image). Gap **G2**.

## 3. Special behaviours

- ✅ **memory FINAL_TEXT** (exact reply, no model reword) — `smoke_pipeline_exec` +
  `parity_check`.
- ✅ **web_search enrich-then-answer** — `parity_check`.
- ✅ **file ingestion consumption** (`store_file_context → get_file_contexts →
  attach to prompt`) — `verify_periphery`.
- ✅ **[FILE:] builder** (disk write + `message_files` linkage) — `verify_periphery`.
- ⬜ **memory two-turn clear confirmation** — pre-route unit-tested
  (`smoke_pipeline_exec`); not run end-to-end on hardware. Gap **G1**.
- ⬜ **image SKILL_DELIVER** file delivery — parsing unit-tested; not on hardware.
  Gap **G2**.
- ⬜ **wikipedia → web_search fallback** — unit-tested (`smoke_pipeline`); not on
  hardware. Gap **G3**.
- ⬜ **[FILE:] end-to-end** (model emits a block → file event → downloadable) — the
  builder is proven, but a live model-emitted block through `ask_stream` is not.
  Gap **G4**.

## 4. Periphery (reused, verified under the v2 core)

- ✅ db under new-core traffic; `init_db`/migrations/**fresh-DB bootstrap** —
  `verify_periphery`.
- ✅ UUID identity + conversation/message isolation — `verify_periphery`.
- ✅ file-gen disk + message linkage — `verify_periphery`.
- ✅ age-band personalisation, 4 bands, MyAI identity, no Gemma/cutoff leakage —
  `verify_periphery` + `smoke_prompts`.
- ✅ system-role handling across families — `smoke_sysrole`,
  `smoke_cross_family_templating`, hardware.
- ⬜ **UI rendering** (templates/static, smooth token stream, download cards,
  age-band styling) — browser-only. Gap **G5** (deferred to cutover by nature).

## 5. Accepted deltas (v2 intentionally differs from v1)

- **D1 — calculator self-serve.** Smaller models answer simple arithmetic/
  conversions directly instead of routing to `calculator` ("15% of 240",
  "5 miles to km" on the 12B). Confirmed **3×** (router_eval + both parity runs);
  the 31B routes them correctly. Model-capability, not a pipeline fault.
  **Accepted.** Optional prompt-tune noted and deferred (risks pulling
  definitional conversions into calculator).
- **D2 — borderline volatility.** "population of France" → direct on the 12B
  (→ web_search on 31B/Qwen). **Accepted.**
- **D3 — Qwen routing latency** ~26 s/decision (verbose reasoning). Qwen is
  correct-but-too-slow to daily-drive as the router; **Gemma is the router.**
  **Accepted / documented.**
- **D4 — `[SEARCH:]` retired** → `web_search` route (ADR-0002). Behaviourally
  equivalent, proven. **Intentional.**

## 6. Removed at cutover (ADR-0002)

- ✅ replacement proven → safe to delete: `[SEARCH:]` tag protocol (regexes +
  prompt instruction) → `web_search` route; two-pass classifier + Arm A prefix
  cache → the router.
- ⏸ **`[FILE:]` tag protocol — KEPT** this cutover (the `generate_file` tool is
  deferred). No `[FILE:]` removal now; it retires when the tool lands.

## 7. Gaps — resolve before sign-off

| Gap | What's unproven | Recommendation |
|-----|-----------------|----------------|
| **G1** | memory two-turn clear on hardware | **Prove** — cheap, and it's a destructive op (wiping memories) worth confirming live. |
| **G4** | `[FILE:]` end-to-end generation on hardware | **Prove** — a headline family feature; confirm a live block becomes a downloadable file. |
| **G2** | image SKILL_DELIVER on hardware | **Prove or accept** — needs a real image upload; image skill is unchanged v1 code + parsing unit-tested, so lower risk. |
| **G3** | wikipedia → web fallback on hardware | **Accept or quick-check** — unit-tested, low risk. |
| **G5** | UI rendering in browser | **Must do at cutover** — blocker for family *release*, not for the cutover-to-staging step. |

## 8. Go / No-Go

**GO to cut over when:** every ✅ holds, the accepted deltas (D1–D4) are
acknowledged, and **G1 + G4 are proven** (G2/G3 proven or explicitly accepted by
you). **G5 must be signed off before the family release** that follows cutover.

**Rollback.** v1 stays recoverable: the cutover is a one-line import swap in
`app.py`, reversible instantly; ADR-0002 deletions land only after the manual UI
pass. Restart runbook: `lsof -ti:8080 | xargs kill -9`.

**Current standing:** 4 of 6 F4 stories done; this gate (Story 5) closes once
G1/G4 are green; Story 6 (cutover + release) follows.
