# ADR-0002 — Retiring the tag protocols and the classifier

**Status:** Accepted as a spec (MyAI v2, F3 S3). Physical removal is executed at
F4 (parity cutover), not now — see "Why this is a spec, not a change."
**Related:** ADR-0001 (inference-backend seam); `core/inference/router.py` (the
structured-selection replacement, built and validated in F3 S1/S2).

---

## Context

v1 routes and requests side effects through **model-emitted text tags**:

- `[SEARCH: query]` — the model, mid-answer, emits a tag to request a web search.
- `[FILE:name]…[/FILE]` — the model emits a block to request file generation.

Both are brittle: the model mis-forms them under load (the search regex literally
matches `SEARSEARCH` typos, `SEARCH_TAG_RE`), and the Gemma 3 file-gen bug is the
proof point. v1 also routes skills through a **grammar-constrained two-pass
classifier** that forces a category *before* the model can reason — the misrouting
Arm B fixed.

F3 built the replacement: the single-pass reason-then-emit **router**, which emits
a structured `{"route": …}` selection and is validated at ~90–97% correctness /
stable consistency across Gemma 4 12B, Gemma 4 31B, and Qwen 3.5. The router's
`web_search` route *is* the structured replacement for `[SEARCH:]`.

## Why this is a spec, not a change (yet)

The tags and the classifier live in the **live v1 pipeline** (`_ask_stream_impl`
in `core/llm_inference.py`) and in **shared** `core/prompts.py`. The router is
built but **not yet wired into any running request flow** — that wiring (router
decision → skill dispatch → answer pass) is F4.

Removing the tags now would strand the running system: delete `SEARCH_TAG_RE` or
the `[SEARCH:]` instruction from `prompts.py` and v1's live search breaks, with no
v2 pipeline to take over. So retirement must coincide with the router-driven
pipeline going live — i.e. the F4 cutover. This ADR is the executable checklist so
that cutover removes everything in one clean move, leaving no dead tag-handling
behind. This is the same discipline as F6's deferred seams: spec precisely, build
at the right moment.

## Decision — what retires, and what replaces it

### 1. `[SEARCH:]` → the router's `web_search` route

**Delete (at F4):**
- `core/llm_inference.py`: `SEARCH_TAG_RE` (~L116); `_SEARCH_INSTRUCTION_BLOCK_RE`
  (~L718); `_strip_search_instructions()` (~L729); `_strip_search_tags_from_answer()`
  (~L739); the Pass-1 `search_match = SEARCH_TAG_RE.search(...)` branch and its
  `[SEARCH:]`-triggered search flow; the belt-and-braces scrub (~L1148).
- `core/prompts.py`: the `[SEARCH:]` instruction inside `_PROMPT_RULES` (the "How to
  handle knowledge & uncertainty" block that tells the model to emit
  `[SEARCH: query]`). The volatility policy it encoded now lives in the **router
  prompt**, so the chat/answer system prompt must no longer mention search tags.

**Replaced by:** `router.route()` deciding `web_search` up front. The web-search
skill runs *before* the answer and its results are fed to the answer pass — which
is exactly what the classifier's existing `web_search` path already does via
`_enrich_answer_message()`. So the **enrich-then-answer mechanism is reused**; only
the tag-based *triggering* is deleted. Net effect: the model never emits or sees a
search tag again.

### 2. `[FILE:]` → a `generate_file` tool (tools schema)

**Delete / replace (at F4 or later — larger build):**
- `core/llm_inference.py`: `FILE_GENERATION_SYSTEM_SNIPPET`; `_FILE_REQUEST_PATTERNS`;
  `_NO_FILE_RE`; `_detect_requested_extension()` (~L464); `_wrap_as_file_block()`
  (~L542); `_correct_file_extensions()` (~L546); all `has_file_block` /
  `strip_file_blocks` call sites (~L1147, 1196, 1214, 1228, 1280, 1298, 1312).
- `core/file_generation.py`: the `[FILE:]` **parsing** protocol — `extract_and_generate`,
  `has_file_block`, `strip_file_blocks`. The actual file-*building* logic
  (content → docx/pdf/…) is **kept and reused**; only the tag protocol goes.

**Replaced by:** a `generate_file` tool exposed through the OpenAI **tools schema**
— the passthrough seam already built and tested in F2 S3 T2 (`tools` forwarded
unmodified, `tool_calls` returned unmodified). The model calls
`generate_file(filename, content)`; the pipeline runs the retained builder on the
returned `tool_calls`. This is a genuinely larger change than `[SEARCH:]` (file
requests are interleaved with answer prose today), so it is **design-now,
build-at-F4-or-later**, not a simple deletion.

### 3. The two-pass classifier + Arm A prefix cache → the router (companion deletion)

Not strictly "tags," but the same cutover retires the classifier the router
replaces:

**Delete (at F4):**
- `_CLASSIFIER_GBNF` (~L214); `_CLASSIFIER_GRAMMAR` (~L219); `_build_classifier_body()`
  (~L240); `_classify_chat()` (~L280); `_classify_intent()` (~L322);
  `_CATEGORY_TO_SKILL` (~L189).
- Arm A prefix caching: `_CLASSIFIER_PREFIX_STATE`, `_CLASSIFIER_PREFIX_TRIED`,
  `_CLASSIFIER_PREFIX_QUERY`, `_ensure_classifier_prefix_state()`,
  `_restore_classifier_prefix()`, and their `save_state`/`load_state` calls
  (already excluded from the v2 backend seam per ADR-0001).

**Replaced by:** `router.route()` + a route→skill dispatch map (F4 S1). The router's
`ROUTES` names already match skill names, so the dispatch map replaces
`_CATEGORY_TO_SKILL` directly.

## Consequences

- F4 executes this checklist as part of standing up the router-driven `ask_stream`,
  in one move — no interim state where half the tags are gone.
- `[SEARCH:]`'s replacement is proven today; `[FILE:]`'s is designed against a seam
  that exists but is unbuilt for this use.
- Retained and reused, untouched: the web-search enrich-then-answer flow, the file
  *builder*, memory/image direct-delivery, and the skill contract.
- Deleted for good: every model-emitted control tag, the typo-tolerant tag regexes,
  and the grammar-first classifier that forced a label before reasoning.

## F4 execution checklist

1. Stand up the router-driven answer flow (router → dispatch → answer), replacing
   `_ask_stream_impl`'s Pass 0 + Pass 1.
2. Delete the `[SEARCH:]` surfaces (§1) and drop the `[SEARCH:]` block from
   `_PROMPT_RULES`.
3. Delete the classifier + Arm A surfaces (§3); wire `route→skill` dispatch.
4. Keep `[FILE:]` working through the old path *until* the `generate_file` tool
   (§2) lands, then delete the `[FILE:]` surfaces. (Or land the tool in the same
   cutover if F4 scope allows.)
5. Verify against the parity checklist (F4) that search, file-gen, and every skill
   still work end-to-end on the new core before removing the last v1 fallback.
