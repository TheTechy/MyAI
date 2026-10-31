# ADR-0001 — The inference-backend seam

**Status:** Accepted (MyAI v2, F1 Story 1)
**Files:** `core/inference/base.py` (the seam), `core/inference/config.py` (selection & config)
**Supersedes:** the module-global `Llama()` and hand-built Gemma markers in v1 `core/llm_inference.py`

---

## Context

v2 replaces the Gemma-shackled model core so MyAI can swap models — and eventually
whole backends — freely, without a rewrite each time. The v1.1 spike already proved
the mechanism: `llm_inference.py`'s `_run_chat_silent` and `_stream_chat` call
`create_chat_completion(messages=...)` and let each model's own embedded chat
template build the prompt. F1 does not invent a new completion path; it extracts a
formal seam around the shape that already works, so a second backend can slot in
behind it later.

Two product tiers set the requirement. MyAI-Family ships a zero-dependency,
in-process backend and wants no configuration beyond pointing at a model.
MyAI-SME later gains an optional backend that speaks the same contract to a local
server or cloud endpoint over real HTTP (the hybrid offload path). That second
backend is **not built in milestone 1** — but the seam must exist now so it drops
in without disturbing the core.

## Decision

The contract is the OpenAI chat-completions **data shape**, implemented in-process.
To be unambiguous: this means the message/response *data format* only — **not** the
OpenAI company, their servers, HTTP, or an API key. The default backend runs
entirely on the user's machine; nothing leaves it.

Every model call in v2 goes through one abstract class, `InferenceBackend`. Given
OpenAI-shaped `messages`, a backend returns an OpenAI-shaped completion (or a stream
of OpenAI-shaped chunks) by applying the model's own chat template. The required
surface is `load`, `unload`, `create_chat_completion`, `emits_think_reasoning`,
`context_window`, and `context_used`; `warmup` is optional with a no-op default.

The individual choices, with rationale:

**ABC, not `typing.Protocol`.** Python refuses to instantiate an incomplete
subclass, giving a future backend author an immediate, precise failure instead of a
type-checker warning, and the Story 3 registry wants concrete classes registered by
name. The cost — a backend must inherit `InferenceBackend` — is accepted.

**A backend-neutral `OutputConstraint`, not `grammar=LlamaGrammar`.**
`LlamaGrammar` is a llama-cpp type an HTTP backend cannot accept, so it must not
appear in the interface. Constraints are expressed as an `OutputConstraint` carrying
an optional GBNF string (honoured by the in-process backend) and an optional JSON
schema (honoured by a future structured-outputs backend). Constraints are applied
**only to the final routing-selection line** (v2 routing decision), never to the
reasoning that precedes it. A backend handed a constraint it cannot honour **must
raise, not silently emit unconstrained text** — a silent fallback would let a
malformed routing line through, which is exactly the brittleness v2 removes.

**`save_state`/`load_state` is deliberately excluded from the seam.** It existed
only to make v1's standalone classifier prefix cheap (the "Arm A" KV snapshot), and
that classifier is deleted in v2. Forcing a KV-snapshot method onto every backend
would burden the HTTP backend with a llama-cpp-only trick it cannot perform.

**Backend-specific config sits in its own nested block.** `BackendConfig` holds the
common fields (`kind`, `model_path`, `warmup`) plus a nested `llama_cpp` block for
the in-process tuning knobs. A later HTTP backend adds an `http` block and touches
nothing above it. `load_config` stays deliberately ignorant of which `kind` values
are valid — it just reads the string; the registry (Story 3) is the single place
that rejects an unknown kind.

Two behaviour changes from v1, both intentional, both one-line reversions if
unwanted: `MODEL` is **required** and raises an actionable error when unset (Family
must still say which model to run); `CTX_SIZE` is now **optional**, defaulting to
8192 with a log line, so a minimal `.env` with only `MODEL` boots.

## Contract for a future HTTP-backend author

To add the SME/HTTP backend later, subclass `InferenceBackend` and honour these
exactly — the seam guarantees nothing else in the core needs to change:

- **Return the OpenAI shape at the exact access paths the pipeline reads.**
  Non-stream text lives at `choices[0]["message"]["content"]`. Stream text lives at
  `choices[0]["delta"]["content"]`, and a final `choices[0]["finish_reason"] ==
  "length"` signals truncation. These two paths differ deliberately; conflating them
  is the classic porting bug. The minimal TypedDicts in `base.py` document the fields
  actually consumed.
- **Apply the model's own chat template.** Assemble the prompt from `messages`;
  never hand-build turn markers.
- **Honour `OutputConstraint` or raise.** Translate `json_schema` to your structured
  -output mechanism. If you cannot constrain output, raise — do not emit free text.
- **Report what you can; return `None` for what you can't.** `context_used()` may
  return `None` where a remote endpoint can't report resident tokens; callers already
  degrade the `ctx` status readout gracefully. `context_window()` must return a real
  maximum.
- **Decide `emits_think_reasoning` from the model, once.** It gates the reasoning
  strip; a wrong answer here either leaks `<think>` text or no-ops incorrectly.
- **Add a config block, don't mutate the shared one.** Introduce `kind="http"`, a
  nested `http` params block, and register the class in the registry. Leave
  `llama_cpp` and the common fields untouched.

## Consequences

The in-process backend (Story 2) becomes a thin wrapper: it forwards to
`llama-cpp-python`'s `create_chat_completion`, translates `OutputConstraint.gbnf`
to a `LlamaGrammar`, and sniffs the chat template once for `emits_think_reasoning`.
Model swapping is now explicit (`unload` old, `load` new) rather than an
import-time global. The HTTP backend is unblocked but unbuilt: the seam is present,
the contract is written, and no core code assumes llama-cpp specifics. The one
standing risk is `OutputConstraint` drift — if the in-process and HTTP backends ever
interpret the same constraint differently, routing consistency diverges; the
"must raise, never silently fall back" rule is the guard against that.

`messages` content is a plain string today. The content-part array shape (text +
image) needed for multimodal is reserved, not designed out — the interface accepts
it without a signature change when multimodal lands (F2 / F6).
