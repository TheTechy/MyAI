# MyAI
```
A privacy-first local LLM chat application with an intelligent skill routing system.
```
MyAI runs entirely on your own hardware. Your conversations, files and data never leave your machine.

Built with Python, Flask and llama-cpp-python, MyAI wraps a local GGUF model in a polished web interface with streaming responses, file handling and a plugin skill system that routes queries to the right tool automatically.

---

![MyAI](/static/images/myai.png)

---

## Features

- **100% local inference** — GPU-accelerated via llama-cpp-python (Metal, CUDA, ROCm or CPU)
- **LLM intent classifier** — a silent low-temperature call routes each query to the right skill in milliseconds
- **Plugin skill system** — 10 built-in skills, each self-contained and extensible
- **File generation** — request a Word doc, Excel spreadsheet, Python script or any common file type and download it instantly
- **File understanding** — upload PDF, DOCX, XLSX, CSV or code files and use their content as conversation context
- **Intelligent web search** — DuckDuckGo + BeautifulSoup content extraction for live answers when training data isn't enough
- **Live news headlines** — BBC, Sky News, The Guardian, Daily Mail, Independent, Express and Metro via RSS — no API key needed
- **Driving directions** — turn-by-turn routing with an interactive Leaflet map, powered by OpenStreetMap and OSRM — no API key needed
- **Wikipedia lookup** — instant encyclopaedia summaries via the Wikipedia REST API — no API key needed
- **Persistent conversations** — SQLite-backed multi-turn history with automatic titling and sidebar search
- **Memory / user notes** — tell MyAI to *remember* facts about you and they're injected into every conversation; browse and delete them on a dedicated **Memories** page
- **Multi-user** — PIN authentication with per-user personalised system prompts
- **Streaming responses** — Server-Sent Events with live status updates, thinking bubble and context window indicator
- **Voice input** — click-to-speak mic button transcribed locally via faster-whisper (no cloud, no API key)

---

## Skills

Each skill is a self-contained module in `myai_skills/`. The LLM classifier routes queries automatically — no keywords or commands needed.

| Skill | Trigger examples | API / dependency |
|-------|-----------------|-----------------|
| 🧮 **Calculator** | `What is 15% of £340?`, `Convert 100 miles to km`, `Square root of 1764` | None — pure Python |
| 💱 **Currency** | `Convert £500 to USD`, `GBP to EUR rate` | [Frankfurter](https://www.frankfurter.app) — free, no key |
| 🌤 **Weather** | `Will it rain in Leeds today?`, `5-day forecast for Tokyo` | [OpenWeatherMap](https://openweathermap.org) — free tier |
| 📅 **Date & Time** | `What time is it in New York?`, `How many days until Christmas?`, `What day was D-Day?` | None — pure Python |
| 🔍 **Web Search** | `Who won the last F1 race?`, `Current Bitcoin price` | DuckDuckGo + BeautifulSoup |
| 🖼️ **Image** | `Resize this image to 800x600`, `Convert this PNG to JPEG`, `Make this image greyscale` | Pillow |
| 🗺️ **Directions** | `Give me directions from Liverpool to Birmingham`, `Give me the fastest route from Glasgow to London` | Nominatim + OSRM — free, no key |
| 📖 **Wikipedia** | `Who was Isaac Newton?`, `Explain the Magna Carta`, `What is quantum entanglement?` | Wikipedia REST API — free, no key |
| 📰 **News** | `Latest BBC headlines`, `What's in the news today?`, `Latest tech news` | RSS feeds — free, no key | 14 RSS feeds and routes automatically by topic or source mentioned
| 🧠 **Memory** | `Remember I'm allergic to peanuts`, `What do you remember about me?`, `Forget that I'm allergic to peanuts` | None — SQLite, per-user |

### Adding a new skill

1. Create `myai_skills/my_skill.py` inheriting from `BaseSkill`
2. Set `name`, `description` and implement `execute(query) -> str`
3. Register it in `myai_skills/__init__.py`
4. Add `"news"` (or your skill name) to `_VALID_CATEGORIES`, `_CATEGORY_TO_SKILL`, the classifier prompt string, and `format_skill_prompt`'s label map in `core/llm_inference.py`
5. Add classification examples to `data/classifier_examples.json`
6. Restart the app

```python
from .base import BaseSkill

class MySkill(BaseSkill):
    name        = "my_skill"
    description = "Does something useful"

    def execute(self, query: str) -> str:
        return f"Result for: {query}"
```

---

## Requirements

- Python 3.11+
- A GGUF model file (MyAI was tested with [Gemma 4 E4B UD-Q8_K_XL from Unsloth](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF) - Serch for: `UD-Q8_K_XL` on this page)
- GPU recommended (Apple Silicon Metal, NVIDIA CUDA, or AMD ROCm)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for voice input (optional)

---

## Installation

### Quick Start ✨ (recommended)

Clone the repo and run the setup wizard — it handles everything:

```bash
git clone https://github.com/TheTechy/MyAI.git
cd MyAI
python3 setup_myai.py
```

The wizard will:
- ✅ Check your Python version
- ✅ Detect your GPU and install `llama-cpp-python` with the correct build flags (Metal, CUDA, ROCm or CPU)
- ✅ Install all required and optional packages
- ✅ Walk you through configuration and write your `.env` file
- ✅ Create directories and initialise the database
- ✅ Set up user accounts and PINs
- ✅ Generate personalised system prompts per user

Then drop your `.gguf` model file into the `models/` folder and start the app:

```bash
python3 app.py
```

Open your browser at `http://localhost:8080` (or the port you chose) and you're good to go.

> **Note:** New OpenWeatherMap API keys can take up to 2 hours to activate after registration. If the weather skill returns an error immediately after setup, simply wait and try again later.

---

### Manual Installation

For those who prefer full control over each step.

#### 1. Clone the repository

```bash
git clone https://github.com/TheTechy/MyAI.git
cd MyAI
```

#### 2. Install llama-cpp-python for your hardware

**Apple Silicon (Metal):**
```bash
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

**NVIDIA CUDA:**
```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

**CPU only:**
```bash
pip install llama-cpp-python
```

#### 3. Install dependencies

```bash
pip install flask waitress python-dotenv pypdf python-docx openpyxl pillow duckduckgo-search beautifulsoup4 requests faster-whisper
```

#### 4. Configure your environment

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

```dotenv
MODEL=./models/your-model.gguf
CTX_SIZE=12288
PORT=8080
MYAIDB=./instance/myai.db
USERS=alice,bob
SECRET_KEY=your-secret-key-here
FILE_OUTPUT_DIR=generated_files
OWM_API_KEY=your_openweathermap_key      # optional — weather skill only
OWM_UNITS=metric                          # metric | imperial | standard
OWM_HOME_CITY=London, GB                  # optional — default location for weather
SEARCH_REGION=uk-en                       # wt-wt | uk-en | us-en | de-de etc.

# Voice input — faster-whisper (optional, needed for mic button)
WHISPER_MODEL=base.en                     # tiny.en | base.en | small.en | medium.en
WHISPER_DEVICE=cpu                        # cpu | cuda  (cuda requires NVIDIA GPU + CUDA drivers)
WHISPER_COMPUTE=int8                      # int8 (cpu/cuda) | float16 (cuda only)
```

Generate a secret key:
```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

#### 5. Initialise the database

```bash
python3 -c "from core.db import init_db; init_db()"
```

#### 6. Run

```bash
python3 app.py
```

Open your browser at `http://localhost:8080`

---

## File Generation

Ask MyAI to create a file and it will generate and deliver it as a download card:

```
Write a Python class that implements a thread-safe LRU cache and save it as lru_cache.py
Create an Excel spreadsheet comparing the planets of the Solar System and save it as planets.xlsx
Write a Word document outlining our C# coding standards and save it as standards.docx
```

Supported formats: `.txt` `.md` `.py` `.js` `.ts` `.jsx` `.tsx` `.html` `.css` `.json` `.yaml` `.sh` `.sql` `.xml` `.csv` `.toml` `.docx` `.xlsx`

When the file type is specified in the request, MyAI generates the file immediately without asking for confirmation. To analyse an uploaded file **without** generating a new one:

```
You are a data analyst. I have uploaded a CSV. Provide your analysis inline, DO NOT generate a file.
```

---

## File Upload

Attach files to give the model context. Supported types:

| Type | Extensions |
|------|-----------|
| Documents | `.pdf` `.docx` |
| Spreadsheets | `.xlsx` `.csv` |
| Code | `.py` `.js` `.ts` `.cs` `.java` `.cpp` `.html` `.css` `.sql` `.sh` |
| Text | `.txt` `.md` `.json` `.yaml` `.xml` `.toml` `.env` |
| Images | `.jpg` `.jpeg` `.png` `.webp` `.gif` |

Files are extracted to text and stored in the conversation context. Nothing is written to disk.

---

## Driving Directions

Ask for directions and MyAI returns turn-by-turn steps plus an interactive HTML map file as a download card. Open it in any browser — it works fully offline once loaded.

```
Directions from Manchester Picadilly to London Euston Station
How do I get from Leeds to Sheffield?
Route from London to Edinburgh by car
```

Powered by [Nominatim](https://nominatim.openstreetmap.org) (geocoding) and [OSRM](https://router.project-osrm.org) (routing) — both free, no API key required. Map tiles served by OpenStreetMap.

---

## Wikipedia

Ask about any person, place, concept or event and MyAI fetches the Wikipedia summary directly, giving accurate, sourced answers rather than relying on the model's training data.

```
Who was Alan Turing?
Explain the Magna Carta
What is the Large Hadron Collider?
Tell me about the Battle of Hastings
```

Uses the [Wikipedia REST API](https://en.wikipedia.org/api/rest_v1/) — free, no key required.

---

## Memory

Tell MyAI to remember facts about you and it keeps them in a per-user store, injecting them into the system prompt of every conversation so the model can use them naturally. Memory is **explicit only** — MyAI never infers or saves anything unless you ask it to.

```
Remember I'm allergic to peanuts
Remember my home address is 1 Acacia Avenue, London. WA1 1AW
What do you remember about me?
Forget that I'm allergic to peanuts
Forget everything
```

Operations are recognised automatically:

| You say | What happens |
|---------|--------------|
| `Remember …` / `Note that …` | Saves the fact |
| `What do you remember about me?` / `List my memories` | Lists everything saved |
| `Forget that …` | Removes any memory matching what you describe |
| `Forget everything` | Asks you to confirm, then clears all your memories |

`Forget everything` is a **two-step confirmation** — MyAI asks first and only wipes everything once you reply `yes`. Memories are private to each signed-in user; you only ever see your own.

Other skills can use your memories too. For example, once a home or work address is saved, the **Directions** skill resolves `directions from home to …` to that address instead of geocoding the literal word.

### Memories page

Visit **Memories** from the sidebar (or `/memories`) to browse everything MyAI remembers, delete individual entries, or clear the lot. Memories are stored in the `user_memories` table in your local SQLite database — nothing leaves your machine.

---

## Voice Input

Click the 🎙 mic button in the chat input bar to dictate instead of type. Transcription runs entirely on your machine via [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — no audio is ever sent to the cloud.

### How it works

1. Click the mic icon — your browser asks for microphone permission once
2. Speak your message
3. Click the mic icon again to stop
4. Transcribed text appears in the input box — edit if needed, then send as normal

A status pill shows each phase: **Recording → Sending audio → Transcribing**.

### Configuration

| Variable | Default | Notes |
|----------|---------|-------|
| `WHISPER_MODEL` | `base.en` | See model sizes below |
| `WHISPER_DEVICE` | `cpu` | `cpu` or `cuda` (NVIDIA only) |
| `WHISPER_COMPUTE` | `int8` | `int8` for CPU; `int8` or `float16` for CUDA |

### Model sizes

| Model | Size | Speed (CPU) | Accuracy |
|-------|------|-------------|----------|
| `tiny.en` | ~39 MB | Fastest | Good |
| `base.en` | ~145 MB | Fast | Better — recommended default |
| `small.en` | ~244 MB | Moderate | Great |
| `medium.en` | ~769 MB | Slow | Excellent |

> ⚠️ `float16` is only valid when `WHISPER_DEVICE=cuda`. Using it on CPU will crash the transcription endpoint.

---

## Intent Classifier

The classifier uses few-shot examples loaded from `data/classifier_examples.json`. Edit this file to improve routing accuracy — no Python knowledge required. The file currently ships with ~115 examples across all 10 skill categories.

```json
{
  "examples": [
    {"query": "What is 15% of £340?",             "category": "calculator"},
    {"query": "Give me the latest BBC headlines",  "category": "news"},
    {"query": "Directions from Leeds to York",     "category": "directions"},
    {"query": "Who was Marie Curie?",              "category": "wikipedia"}
  ]
}
```

Valid categories: `calculator` `currency` `weather` `datetime` `web_search` `image` `directions` `wikipedia` `news` `memory` `general`

Restart the app after editing. On startup you'll see:
```
[classifier] loaded 114 examples from classifier_examples.json
```

---

## Testing

MyAI ships with a [pytest](https://docs.pytest.org) suite in `test_suite/`. The tests are fully self-contained: `test_suite/conftest.py` mocks `llama_cpp` and `faster_whisper` and sets the required environment variables (model path, database, port, etc.) before any app code is imported, so you don't need a real GGUF model, a configured `.env`, or a populated database to run them. Install the test dependency with `pip install pytest` and run the whole suite from the project root:

```bash
pytest                                  # run everything
pytest test_suite/test_calculator.py    # one file
pytest -k "directions"                  # tests matching a keyword
pytest -x                               # stop on first failure
```

Configuration lives in `pytest.ini` (test path, `pythonpath` and verbose output), so no extra flags are needed. The suite is split by concern: `test_db.py` (SQLite helpers), `test_api.py` (Flask endpoints and the SSE prompt stream), `test_routing_logic.py` (filename and question-detection helpers), and one file per skill (e.g. `test_calculator.py`, `test_datetime_skill.py`).

### Adding tests for a new skill

Most skills are pure Python with no network or LLM dependency, so the fastest path is to mirror `test_suite/test_datetime_skill.py`:

1. Create `test_suite/test_my_skill.py`.
2. Import your skill and expose it via a fixture, then assert on the output of `execute()`:

   ```python
   import pytest
   from myai_skills.my_skill import MySkill

   @pytest.fixture(scope="module")
   def skill():
       return MySkill()

   class TestMySkill:
       def test_basic_query(self, skill):
           result = skill.execute("some query")
           assert "expected" in result.lower(), f"got: {result}"
   ```

3. Prefer **deterministic inputs** (fixed historical dates, known constants) for exact assertions; for anything time- or location-sensitive, assert on *structure* (e.g. a regex for a `HH:MM` time or a number) rather than an exact value, as the datetime tests do.

For a skill that calls an external API, monkeypatch the network call (`monkeypatch.setattr(...)`) so tests stay offline and fast. To exercise a skill end-to-end through the Flask app instead, use the `auth_client` fixture from `conftest.py` and the SSE helpers in `test_suite/helpers.py` (`parse_sse`, `event_names`, `token_text`) — see `test_api.py::TestPromptEndpoint` for the pattern. The default `llm_mock` returns a fixed "general" response; reconfigure it via the `llm_mock` fixture if your test needs a specific model output.

---

## Project Structure

```
MyAI/
├── app.py                     # Flask routes and file upload handling
├── setup_myai.py              # Interactive setup wizard
├── core/
│   ├── llm_inference.py       # LLM classifier, skill router, streaming
│   ├── file_generation.py     # [FILE:] tag parsing, DOCX/XLSX builders
│   ├── file_ingestion.py      # Text extraction from uploaded files
│   ├── db.py                  # SQLite schema and helpers
│   └── prompts.py             # Per-user system prompts
├── myai_skills/
│   ├── __init__.py            # Skill registry
│   ├── base.py                # BaseSkill abstract class
│   ├── calculator.py          # Maths, conversions, percentages
│   ├── currency.py            # Live exchange rates (Frankfurter API)
│   ├── datetime_skill.py      # Time, timezones, date arithmetic
│   ├── weather.py             # Current weather + forecast (OWM API)
│   ├── web_search.py          # DuckDuckGo + BeautifulSoup scraping
│   ├── image_skill.py         # Image manipulation (Pillow)
│   ├── directions.py          # Driving directions + Leaflet map (OSM/OSRM)
│   ├── wikipedia_skill.py     # Encyclopaedia lookups (Wikipedia REST API)
│   ├── news_skill.py          # Live headlines from 14 RSS feeds
│   └── memory_skill.py        # Per-user remembered facts (SQLite)
├── data/
│   └── classifier_examples.json  # Few-shot examples for the intent classifier
├── templates/
│   ├── chat.html              # Main chat interface
│   ├── memories.html          # Memories browse / delete page
│   └── index.html             # Login page
├── static/
│   ├── style/chat.css
│   └── scripts/chat.js
├── instance/                  # Created by setup — SQLite database
│   └── myai.db
├── models/                    # Drop your .gguf model file here
├── generated_files/           # File generation output
└── scripts/                   # Utility shell scripts
```

---

## Per-User Prompts

Each user can have a personalised system prompt defined in `core/prompts.py`. This lets you tailor the assistant's tone, knowledge depth and content guidelines per person — useful for family setups where users have very different needs.

```python
USER_PROMPTS = {
    "alice": "You are a helpful assistant...",
    "bob":   "You are a patient tutor...",
}
```

Users not in `USER_PROMPTS` fall back to the default prompt.

---

## Roadmap

- ✅ **Voice input** — faster-whisper transcription via mic button
- ✅ **Interactive setup script** — guided per-user prompt generation
- ✅ **Wikipedia skill** — encyclopaedia lookups via REST API
- ✅ **News headlines skill** — 14 RSS feeds, topic and source routing, inline links
- ✅ **Directions skill** — turn-by-turn routing with interactive Leaflet map
- ✅ **Image skill** — resize, crop, convert, rotate via Pillow
- ✅ **Inline markdown links** — clickable URLs rendered in chat bubbles
- ✅ **Live status updates** — thinking bubble updates as skills run
- ✅ **Memory / user notes** — persistent per-user facts injected into context, with a Memories management page
- ✅ **Test suite** — pytest coverage for the database, API, routing logic and skills
- 🔜 **Local RAG** — embed your own documents for retrieval-augmented answers
- 🔜 **Dynamic dashboard** — morning briefing combining weather, news and reminders etc with dynamic web components the LLM generates
- 🔜 **MTP** — Implmentation of multi-token prediction.
- 🔜 **Voice output (TTS)** — Piper TTS for full two-way voice

---

## Q&A
Q: Why did you build MyAI?<br>
A: The last 6-9 months has seen a dramatic improvement in local LLMs and I wanted to take advantage of these improvements. I also wanted to keep my data local and not be reliant on the pricing fluctuations of OpenAI, Anthropic or other LLM providers. Beyond the practical side, there's something genuinely satisfying about owning the whole stack, the model, the conversations, the skills, the data. Nothing leaves my machine, nothing gets logged for training, nothing disappears when a service changes its terms.

Q: Is MyAI as "good" as something like ChatGPT or Claude?<br>
A: In a word, no and I'm not going to pretend otherwise. The frontier models from OpenAI, Anthropic, and Google are genuinely better at deep reasoning, complex coding tasks, long-context analysis, and nuanced creative writing. They're trained on far more compute and have hundreds of billions of parameters. MyAI runs a ~4 billion parameter model on consumer hardware. That said, for the vast majority of everyday queries "what's the weather", "convert this to PDF", "explain quantum entanglement", "directions to Manchester", "give me the latest BBC headlines", "help me write a Python function", "what year was the Battle of Hastings", MyAI is comfortably 90-95% as good as a paid service, and arrives at the answer in seconds without sending your query anywhere. For the kind of questions most people ask most of the time, it's more than enough.

Q: How private is it, really?<br>
A: Privacy is the foundation of the architecture, not an afterthought. Your conversations are stored in a SQLite database on your own machine. The LLM inference runs locally via llama-cpp-python. Voice transcription runs locally via faster-whisper, audio is never sent to the cloud. The only outbound network calls happen when a skill genuinely needs external data (weather forecasts, exchange rates, news headlines, web search), and even then the request goes directly from your machine to the data source, there's no MyAI server in the middle observing or logging anything.

Q: What about the API calls the skills make — aren't those tracking me?<br>
A: Some, technically, yes — but it's far less invasive than using a hosted LLM service. When the weather skill calls OpenWeatherMap, it sends a city name. When the directions skill calls OSRM, it sends two coordinates. None of these services see your full conversation, your other queries, your name, or anything about who you are beyond the single API call. Compare this to a hosted LLM where every word you type is processed on someone else's servers, potentially logged, potentially used to train the next model.

Q: Will I get charged for using MyAI?<br>
A: No. MyAI itself is free and open source. Of the external APIs it uses:
- Frankfurter (currency), Nominatim (geocoding), OSRM (routing), Wikipedia REST API, and the RSS feeds for news are all completely free with no key required.
- OpenWeatherMap has a generous free tier (1,000 calls/day) that's more than enough for personal use.
- Voice transcription runs locally — no cloud cost.

The only "cost" is the hardware you already own and a bit of electricity.

Q: What hardware do I need?<br>
A: Anything reasonably modern will work. MyAI was developed and tested on Apple Silicon (M-series Macs use the Metal backend), but it also runs on NVIDIA GPUs via CUDA, AMD GPUs via ROCm, or pure CPU on any machine. CPU-only is slower but perfectly usable for a single user. A modest GPU makes it feel snappy. You don't need a £3,000 workstation — a five-year-old laptop with 16GB of RAM will run it comfortably.

Q: Why does it sometimes feel slower than ChatGPT?<br>
A: Because ChatGPT is running on a data centre full of H100 GPUs costing millions of pounds and MyAI is running on consumer grade hardware. That's the trade-off. The first token might take a second or two on a smaller machine; subsequent tokens stream as fast as your hardware allows.

Q: Can multiple people in my house use it?<br>
A: Yes — MyAI is built for multi-user from the ground up. Each user gets their own PIN, their own conversation history, and their own personalised system prompt (so the assistant can be tuned differently for different people in the household). One person can be running their work queries while another asks for help with homework.

Q: How extensible is it?<br>
A: Very. The skill system is deliberately simple — a new skill is one Python file inheriting from BaseSkill, with three things to define: name, description, and an execute() method. The intent classifier learns to route to it via examples in a JSON file. I've added nine skills covering everything from currency conversion to live news headlines to driving directions with interactive maps, and the same pattern would work for whatever you want to add next — home automation, stock prices, sports scores, anything.

Q: What can MyAI not do?<br>
A: Honest list:
- Long, complex multi-step reasoning — frontier models handle this better
- Generating production-quality code for unfamiliar frameworks — fine for snippets and common languages, but ask for a 500-line React app and the results will be patchy.
- Image generation — there's no image-generation model bundled in. (Image processing is supported via the image skill.)
- Anything involving live real-time data not covered by an existing skill — but skills are easy to add.
- Vision/multimodal input — currently text-only, no image understanding.

Q: Can MyAI be used as an agent?<br>
A: Yes, kind of (but not yet). On the roadmap there are plans to have MyAI act as a host with open endpoints so 3rd party applications such as Slack, Smart home integration and then be able to intagrate with 3rd party APIs, for example gmail could... "Send an email to jane.smith@email.com, subject: Meeting, body: Apologies, I will be 10 minutes late to our meeting later today."

Q: Will it get better over time?<br>
A: Yes, on two fronts. New local models keep getting released and getting better — when a stronger GGUF comes out (and they're coming out monthly now), you can swap your .gguf file and instantly get the improvement. And the skill system means you can keep adding new capabilities without waiting on anyone else.

---

## Contributing

Pull requests are welcome. For significant changes, please open an issue first.
When adding a skill, follow the pattern in `myai_skills/web_search.py` and add classification examples to `data/classifier_examples.json`.

---

## Licence
MIT