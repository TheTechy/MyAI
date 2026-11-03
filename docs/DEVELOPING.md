# Developing MyAI

How MyAI is put together, how to run its tests, and the full settings reference.
For installing and using MyAI, see the [README](../README.md).

## Architecture

MyAI v2 uses an OpenAI-shaped message format **in-process**: there are no API
keys and no HTTP calls to any model provider. A request flows like this:

```
user turn → router (reason-then-emit) → dispatch to a skill (or answer directly)
          → stream the answer → Server-Sent Events → browser
```

- **Core engine:** [`core/inference/`](../core/inference/): backend, router, dispatch,
  pipeline, streaming.
- **Skills:** [`myai_skills/`](../myai_skills/). Each tool (weather, calculator,
  directions, …) is a module, collected in a registry.
- **Web app:** `app.py` (Flask, served by waitress) plus `templates/` and `static/`.
- **Storage:** a single SQLite database (`instance/myai.db` by default) via
  [`core/db.py`](../core/db.py).

Key design decisions are recorded as ADRs in this folder:

- [ADR-0001](ADR-0001-inference-backend-seam.md): the pluggable inference-backend
  seam (in-process by default).
- [ADR-0002](ADR-0002-retire-tag-protocols.md): retiring the old text-tag routing
  in favour of structured selection.

## Project structure

```
install.sh             # launcher: finds Python 3.13+, makes .venv, hands over to…
setup_myai.py          # …the setup app (screens only)
installer/             # setup logic (system.py) and the stdlib-only terminal UI (tui.py)
start.sh               # start MyAI (what users run)
debug.sh               # clear Python caches, then start (for development)
app.py                 # Flask web app + SSE endpoint
core/
  db.py, prompts.py, file_generation.py, file_ingestion.py
  inference/           # the v2 engine
myai_skills/           # built-in skills + registry
data/                  # age-band personalities
static/ templates/     # web UI
test_suite/            # smoke tests and pytest suite
scripts/               # evaluation and maintenance tools
docs/                  # this file and the ADRs
```

## The AI engine (llama-cpp-python)

MyAI pins `llama-cpp-python` (see `LLAMA_VERSION` in
[`installer/system.py`](../installer/system.py)) and **builds it from source**
with Metal enabled. It's deliberately left out of `requirements.txt` so it's
only rebuilt when the pinned version changes.

The prebuilt Metal wheels for 0.3.33–0.3.36 on the llama-cpp-python wheel index
are corrupt upstream (their checksums match the published ones, but the archives
are broken), so don't switch back to them without checking.

To build by hand, inside the venv:

```bash
CMAKE_ARGS="-DGGML_METAL=on" .venv/bin/pip install --no-cache-dir llama-cpp-python==0.3.35
```

This needs Apple's Command Line Tools (`xcode-select --install`). pip fetches
cmake itself.

## Manual setup

`./install.sh` is the supported route. By hand, it's:

```bash
python3.13 -m venv .venv
CMAKE_ARGS="-DGGML_METAL=on" .venv/bin/pip install llama-cpp-python==0.3.35
.venv/bin/pip install -r requirements.txt
cp .env.example .env      # then set MODEL, SECRET_KEY and the rest (below)
./start.sh
```

Generate a `SECRET_KEY` with `python3 -c "import secrets; print(secrets.token_hex(32))"`.
On an empty database MyAI creates a temporary **Setup** account (PIN `0000`). It
removes itself once you create a real account, as long as it hasn't been used.

## Settings (`.env`)

The setup app writes `.env` for you. For reference:

| Variable | Required | Description |
|---|---|---|
| `MODEL` | ✅ | Path to the `.gguf` model file |
| `SECRET_KEY` | ✅ | Flask session secret; setup generates one and keeps it on re-runs |
| `PORT` | ✅ | Web server port (setup defaults to `8080`) |
| `MYAIDB` | – | Path to the SQLite database file (default `./instance/myai.db`) |
| `CTX_SIZE` | – | Context window in tokens (`0` = the model's trained maximum) |
| `WARMUP` | – | `1` to warm the model up at start-up (faster first reply) |
| `FILE_OUTPUT_DIR` | – | Where generated files are saved |
| `OWM_API_KEY` | – | OpenWeatherMap key; turns on the weather skill |
| `OWM_UNITS` | – | `metric`, `imperial` or `standard` |
| `OWM_HOME_CITY` | – | Default place for "will it rain today?" |
| `SEARCH_REGION` | – | Web-search region, e.g. `uk-en`, `wt-wt` |
| `WHISPER_MODEL` | – | Whisper model for voice input, e.g. `base.en` (blank = off) |
| `WHISPER_DEVICE` | – | `cpu` (setup's choice on a Mac) |
| `WHISPER_COMPUTE` | – | Compute precision, e.g. `int8` |

## Skills

| Skill | Example | Uses |
|---|---|---|
| Calculator | "what's 15% of 240", "convert 100 miles to km" | local |
| Date & time | "what time is it in Tokyo?" | local |
| Image editing | *(attach)* "resize this to 400px wide" | local |
| Memory | "remember I drink tea" | local |
| Weather | "what's the weather in Rochdale?" | OpenWeatherMap (key) |
| Web search | "who won the match last night?" | Public search engines (`ddgs`) |
| News | "what's in the news today?" | public RSS feeds |
| Directions | "driving directions from Manchester to Leeds" | Nominatim + OSRM |
| Currency | "how much is 50 USD in GBP?" | Frankfurter |
| Wikipedia | "look up the Eiffel Tower" | Wikipedia API |

Anything the model already knows is answered directly, without a tool.

## Running in development

```bash
./debug.sh
```

This clears `__pycache__` and stray `.pyc` files (not inside `.venv`), then runs
`./start.sh`.

## Tests

**Smoke tests** are plain scripts that use fakes, so they need no model or real
database. Each prints `ALL GREEN` when it passes:

```bash
PYTHONPATH=. .venv/bin/python test_suite/smoke_router.py

# all of them:
for f in test_suite/smoke_*.py; do PYTHONPATH=. .venv/bin/python "$f" >/dev/null 2>&1 || echo "FAIL $f"; done
```

They cover the engine, router, pipeline and skills, plus the installer
(`smoke_install_sh`, `smoke_tui`, `smoke_setup_system`, `smoke_setup_flow`) and
the start scripts (`smoke_start_sh`).

**pytest** covers the database and the web API, against a temporary database and
the fake engine in `test_suite/fake_llama_cpp.py`. pytest isn't in
`requirements.txt`, so install it into the venv first:

```bash
.venv/bin/pip install pytest
.venv/bin/python -m pytest test_suite
```

**With a real model**, to evaluate routing:

```bash
.venv/bin/python scripts/router_eval.py --model ./models/<your-model>.gguf
```

## Roadmap

- **Windows and Linux.** The engine is OS-agnostic; the installer is Mac-first,
  with the platform-specific parts kept in `installer/system.py`.
- **Proper account security.** Hashed PINs, and only adults able to manage accounts.
- **User-created skills.** A guided, safe way for people to build simple tools of
  their own.
