# MyAI

> A privacy-first local LLM chat application with an intelligent skill routing system.

MyAI runs entirely on your own hardware. Your conversations, files and data never leave your machine.

Built with Python, Flask and [llama-cpp-python](https://github.com/abetlen/llama-cpp-python), MyAI wraps a local GGUF model in a polished web interface with streaming responses, file handling and a plugin skill system that routes queries to the right tool automatically.

---

## Features

- 🧠 **100% local inference** — GPU-accelerated via llama-cpp-python (Metal, CUDA, ROCm or CPU)
- 🎯 **LLM intent classifier** — a silent low-temperature call routes each query to the right skill in milliseconds
- 🔧 **Plugin skill system** — weather, currency, calculator, date/time and web search, each self-contained and extensible
- 📄 **File generation** — request a Word doc, Excel spreadsheet, Python script or any common file type and download it instantly
- 📁 **File understanding** — upload PDF, DOCX, XLSX, CSV or code files and use their content as conversation context
- 🌐 **Intelligent web search** — DuckDuckGo + BeautifulSoup content extraction for live answers when training data isn't enough
- 💬 **Persistent conversations** — SQLite-backed multi-turn history with automatic titling
- 👥 **Multi-user** — PIN authentication with per-user personalised system prompts
- ⚡ **Streaming responses** — Server-Sent Events with a thinking bubble and context window indicator

---

## Skills

Each skill is a self-contained module in `myai_skills/`. The LLM classifier routes queries automatically — no keywords or commands needed.

| Skill | Trigger examples | API / dependency |
|-------|-----------------|-----------------|
| 🧮 **Calculator** | `What is 15% of £340?`, `Convert 100 miles to km`, `Square root of 1764` | None — pure Python |
| 💱 **Currency** | `Convert £500 to USD`, `GBP to EUR rate`, `What was the rate on 1st Jan 2020?` | [Frankfurter](https://www.frankfurter.app) — free, no key |
| 🌤 **Weather** | `Will it rain in Leeds today?`, `5-day forecast for Tokyo` | [OpenWeatherMap](https://openweathermap.org) — free tier |
| 📅 **Date & Time** | `What time is it in New York?`, `How many days until Christmas?`, `What day was D-Day?` | None — pure Python |
| 🔍 **Web Search** | `Who won the last F1 race?`, `Latest AI news today`, `Current Bitcoin price` | DuckDuckGo + BeautifulSoup |

### Adding a new skill

1. Create `myai_skills/my_skill.py` inheriting from `BaseSkill`
2. Set `name`, `description` and implement `execute(query) -> str`
3. Register it in `myai_skills/__init__.py`
4. Add classification examples to `classifier_examples.json`
5. Restart the app

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
- A GGUF model file (MyAI was tested with [Gemma 4 E4B UD-Q8_K_XL from Unsloth] ((https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF))
- GPU recommended (Apple Silicon Metal, NVIDIA CUDA, or AMD ROCm)

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

>**Note:** New OpenWeatherMap API keys can take up to 2 hours to activate after registration. If the weather skill returns an error immediately after setup, simply wait and try again later.


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
pip install flask waitress python-dotenv pypdf python-docx openpyxl duckduckgo-search beautifulsoup4 requests
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
MYAIDB=./DB/myai.db
USERS=alice,bob
SECRET_KEY=your-secret-key-here
FILE_OUTPUT_DIR=generated_files
OWM_API_KEY=your_openweathermap_key      # optional — weather skill only
OWM_UNITS=metric                          # metric | imperial | standard
OWM_HOME_CITY=London, GB                  # optional — default location for weather
SEARCH_REGION=uk-en                       # wt-wt | uk-en | us-en | de-de etc.
```

Generate a secret key:
```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

#### 5. Initialise the database

```bash
python3 -c "from db import init_db; init_db()"
```

#### 6. Run

```bash
python3 app.py
```

Open your browser at `http://localhost:8080`

---

## File Generation

Ask MyAI to create a file and it will generate and deliver it as a download:

```
Write a Python class that implements a thread-safe LRU cache and save it as lru_cache.py
Create an Excel spreadsheet comparing the planets of the Solar System and save it as planets.xlsx
Write a Word document outlining our C# coding standards and save it as standards.docx
```

Supported formats: `.txt` `.md` `.py` `.js` `.ts` `.jsx` `.tsx` `.html` `.css` `.json` `.yaml` `.sh` `.sql` `.xml` `.csv` `.toml` `.docx` `.xlsx`

To analyse an uploaded file **without** generating a new one:
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

Files are extracted to text and stored in the conversation context. Nothing is written to disk.

---

## Classifier Tuning

The intent classifier uses few-shot examples loaded from `classifier_examples.json`. Edit this file to improve routing accuracy for your use cases — no Python knowledge required.

```json
{
  "examples": [
    {"query": "What is 15% of £340?",        "category": "calculator"},
    {"query": "Current price of Bitcoin?",    "category": "web_search"},
    {"query": "Will it snow in Edinburgh?",   "category": "weather"}
  ]
}
```

Valid categories: `calculator` `currency` `weather` `datetime` `web_search` `general`

Restart the app after editing. On startup you'll see:
```
[classifier] loaded 27 examples from classifier_examples.json
```

---

## Project Structure

```
MyAI/
├── app.py                     # Flask routes and file upload handling
├── llm_inference.py           # LLM classifier, skill router, streaming
├── file_generation.py         # [FILE:] tag parsing, DOCX/XLSX builders
├── file_ingestion.py          # Text extraction from uploaded files
├── db.py                      # SQLite schema and helpers
├── prompts.py                 # Per-user system prompts
├── classifier_examples.json   # Few-shot examples for the intent classifier
├── myai_skills/
│   ├── __init__.py            # Skill registry
│   ├── base.py                # BaseSkill abstract class
│   ├── calculator.py          # Maths, conversions, percentages
│   ├── currency.py            # Live exchange rates (Frankfurter API)
│   ├── datetime_skill.py      # Time, timezones, date arithmetic
│   ├── weather.py             # Current weather + forecast (OWM API)
│   └── web_search.py          # DuckDuckGo + BeautifulSoup scraping
├── Templates/
│   ├── chat.html              # Main chat interface
│   └── index.html             # Login page
└── static/
    ├── style/chat.css
    └── scripts/chat.js
```

---

## Per-User Prompts

Each user can have a personalised system prompt defined in `prompts.py`. This lets you tailor the assistant's tone, knowledge depth and content guidelines per person — useful for family setups where users have very different needs.

```python
USER_PROMPTS = {
    "alice": "You are a helpful assistant...",
    "bob":   "You are a patient tutor...",
}
```

Users not in `USER_PROMPTS` fall back to the default prompt.

---

## Roadmap

- ⚙️ Interactive setup script with guided per-user prompt generation
- 🧪 Test suite
- 📖 Wikipedia skill
- 📰 News headlines skill
- 🗺️ Location / Maps skill
- 🖼️ Image / OCR support
- 🕷️ Crawl4AI integration for richer web scraping

---

## Contributing

Pull requests are welcome. For significant changes, please open an issue first.

When adding a skill, follow the pattern in `myai_skills/web_search.py` and add classification examples to `classifier_examples.json`.

---

## Licence

MIT