#!/usr/bin/env python3
"""
setup_myai.py
=============
Interactive setup wizard for MyAI.
Run once after cloning the repository, or re-run at any time to reconfigure.

    python3 setup_myai.py

Architecture
------------
1. DISCOVER  — scan filesystem, detect GPU/hardware, note what is installed
2. COLLECT   — walk through screens; each screen populates `config` dict only
3. CONFIRM   — show full summary; user approves before anything is written
4. WRITE     — single commit: dirs, .env, SQLite schema, prompts.py
5. SUMMARY   — show what was created and the start command
"""

from __future__ import annotations

import getpass
import importlib.util
import os
import platform
import re
import secrets
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
#  Paths  (derived from cwd — nothing is created here)
# ══════════════════════════════════════════════════════════════════════════════
CURPATH    = Path.cwd()
MODELS_DIR = CURPATH / "models"
DB_DIR     = CURPATH / "instance"
DB_PATH    = DB_DIR  / "myai.db"
GEN_DIR    = CURPATH / "generated_files"

TOTAL_STEPS = 7


# ══════════════════════════════════════════════════════════════════════════════
#  Terminal helpers
# ══════════════════════════════════════════════════════════════════════════════
def clear():
    os.system("cls" if os.name == "nt" else "clear")

def draw_header(title: str = "MyAI — Setup Wizard"):
    print("┌─────────────────────────────────────────────┐")
    print(f"│  {title:<43}│")
    print("└─────────────────────────────────────────────┘")

def draw_step(step: int, total: int, title: str, lines: list[str] | None = None):
    clear()
    draw_header()
    print(f"  Step {step} of {total}  ·  {title}")
    hr()
    if lines:
        for line in lines:
            print(f"  {line}")
        hr()

def draw_menu(title: str, options: list[str], body_lines: list[str] | None = None, clear_screen: bool = True):
    if clear_screen:
        clear()
    draw_header()
    print(f"  {title}")
    hr()
    if body_lines:
        for line in body_lines:
            print(f"  {line}")
        hr()
    for i, opt in enumerate(options, 1):
        print(f"    [{i}]  {opt}")
    hr()

def hr():
    print("─────────────────────────────────────────────────")

def prompt(label: str, default: str = "", password: bool = False) -> str:
    hint = f" [{default}]" if default else ""
    full = f"  {label}{hint}: "
    try:
        val = getpass.getpass(full) if password else input(full).strip()
    except (KeyboardInterrupt, EOFError):
        clear()
        print("\n  Setup cancelled.")
        sys.exit(0)
    return val if val else default

def wait(msg: str = "Press Enter to continue..."):
    try:
        input(f"\n  {msg}")
    except (KeyboardInterrupt, EOFError):
        sys.exit(0)

def choose(options: list[str]) -> str:
    while True:
        try:
            val = input("  Choice > ").strip()
        except (KeyboardInterrupt, EOFError):
            clear()
            print("  Setup cancelled.")
            sys.exit(0)
        if val in [str(i) for i in range(1, len(options) + 1)]:
            return val
        print(f"  Invalid choice — enter 1 to {len(options)}")


# ══════════════════════════════════════════════════════════════════════════════
#  Package helpers
# ══════════════════════════════════════════════════════════════════════════════
CORE_PACKAGES = ["flask", "waitress", "python-dotenv", "werkzeug", "requests"]

OPTIONAL_PACKAGES: dict[str, tuple[str, str]] = {
    # import_name: (pip_name, description)
    "pypdf":             ("pypdf",              "PDF ingestion"),
    "docx":              ("python-docx",        "Word doc ingestion + generation"),
    "openpyxl":          ("openpyxl",           "Excel ingestion + generation"),
    "duckduckgo_search": ("duckduckgo-search",  "Web search"),
    "bs4":               ("beautifulsoup4",     "Web search — result parsing"),
    "PIL":               ("Pillow",             "Image skill — manipulation"),
    "faster_whisper":    ("faster-whisper",     "Voice input — mic transcription"),
}

def is_installed(package: str) -> bool:
    return importlib.util.find_spec(package) is not None

def pkg_version(pip_name: str) -> str | None:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", pip_name],
            capture_output=True, text=True,
        )
        for line in result.stdout.splitlines():
            if line.lower().startswith("version:"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return None

def pip_install(packages: list[str], extra_env: dict | None = None) -> bool:
    env = {**os.environ, **(extra_env or {})}
    cmd = [sys.executable, "-m", "pip", "install", "--quiet"] + packages
    return subprocess.run(cmd, env=env).returncode == 0


# ══════════════════════════════════════════════════════════════════════════════
#  Hardware detection
# ══════════════════════════════════════════════════════════════════════════════
def detect_gpu() -> tuple[str, str]:
    system  = platform.system()
    machine = platform.machine().lower()
    if system == "Darwin":
        if "arm" in machine or "aarch64" in machine:
            return ("macOS Apple Silicon (Metal)", "-DGGML_METAL=on")
        return ("macOS Intel — CPU only", "")
    if shutil.which("nvidia-smi"):
        return ("Linux / Windows — NVIDIA CUDA", "-DGGML_CUDA=on")
    if shutil.which("rocminfo"):
        return ("Linux — AMD ROCm", "-DGGML_HIPBLAS=on")
    return ("CPU only", "")

def has_nvidia() -> bool:
    return shutil.which("nvidia-smi") is not None

def is_apple_silicon() -> bool:
    m = platform.machine().lower()
    return platform.system() == "Darwin" and ("arm" in m or "aarch64" in m)


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 1 — DISCOVER
# ══════════════════════════════════════════════════════════════════════════════
def discover() -> dict:
    """
    Scan the environment and return a dict of discovered facts.
    Nothing is written here — pure observation.
    """
    platform_label, cmake_args = detect_gpu()

    gguf_files: list[Path] = []
    if MODELS_DIR.exists():
        gguf_files = sorted(MODELS_DIR.glob("*.gguf"))

    # Read existing .env if present (used as defaults throughout collect phase)
    existing_env: dict[str, str] = {}
    env_path = CURPATH / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                existing_env[k.strip()] = v.strip()

    return {
        "platform_label": platform_label,
        "cmake_args":     cmake_args,
        "gguf_files":     gguf_files,
        "existing_env":   existing_env,
        "env_exists":     env_path.exists(),
        "db_exists":      DB_PATH.exists(),
        "llama_version":  _get_llama_version(),
    }

def _get_llama_version() -> str | None:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", "llama-cpp-python"],
            capture_output=True, text=True,
        )
        for line in result.stdout.splitlines():
            if line.lower().startswith("version:"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 2 — COLLECT
# ══════════════════════════════════════════════════════════════════════════════

def screen_welcome(discovery: dict):
    clear()
    print()
    print("  ┌───────────────────────────────────────────┐")
    print("  │                                           │")
    print("  │           Welcome to  MyAI  ✦             │")
    print("  │       Local LLM Chat — Setup Wizard       │")
    print("  │                                           │")
    print("  └───────────────────────────────────────────┘")
    print()
    print("  This wizard collects all configuration first,")
    print("  then writes everything in one go at the end.")
    print("  Safe to re-run — nothing is written until you confirm.")
    print()
    print("  Steps:")
    print("    [1]  Check your Python version")
    print("    [2]  Install llama-cpp-python (LLM engine)")
    print("    [3]  Install required packages")
    print("    [4]  Collect configuration (model, users, skills, voice)")
    print("    [5]  Set up user PINs")
    print("    [6]  Generate per-user system prompts")
    print("    [7]  Review and confirm — then write everything")
    print()

    if discovery["env_exists"]:
        print("  ℹ  Existing .env found — current values used as defaults.")
        print()

    hr()
    val = prompt("Ready to begin? [Y/n]", default="y")
    if val.lower() not in ("y", "yes", ""):
        print("  Setup cancelled.")
        sys.exit(0)


def screen_python():
    major, minor = sys.version_info[:2]
    ver = f"{major}.{minor}"

    draw_step(1, TOTAL_STEPS, "Python Version Check", [
        f"Detected Python {ver}",
        "Required: Python 3.11 or higher",
    ])

    if (major, minor) < (3, 11):
        print(f"  ✘  Python {ver} is too old.")
        print("     Please upgrade to Python 3.11+ and re-run this script.")
        sys.exit(1)

    print(f"  ✔  Python {ver} — OK")
    wait()


def screen_llama(discovery: dict):
    platform_label = discovery["platform_label"]
    cmake_args     = discovery["cmake_args"]
    version        = discovery["llama_version"]
    installed      = version is not None

    clear()
    draw_header()
    print(f"  Step 2 of {TOTAL_STEPS}  ·  LLM Engine — llama-cpp-python")
    hr()
    print(f"  Platform detected : {platform_label}")
    print()

    if installed:
        print(f"  ✔  llama-cpp-python {version} is installed")
    else:
        print("  ✘  llama-cpp-python is NOT installed")

    print()
    if cmake_args:
        print(f"  Build flags: CMAKE_ARGS=\"{cmake_args}\"")
    else:
        print("  Build flags: none (CPU-only build)")
    print()

    if installed:
        options = [
            "Keep current installation",
            "Re-install / upgrade llama-cpp-python",
            "Skip — I will manage this manually",
        ]
        draw_menu("llama-cpp-python is already installed. What would you like to do?", options)
        choice = choose(options)
        if choice == "1":
            print(f"\n  ✔  Keeping llama-cpp-python {version}")
            wait()
            return
        elif choice == "3":
            _show_manual_llama_install(cmake_args)
            return
    else:
        options = [
            "Install llama-cpp-python now  (recommended)",
            "Skip — I will install it manually",
        ]
        draw_menu("llama-cpp-python is required to run MyAI.", options)
        if choose(options) == "2":
            _show_manual_llama_install(cmake_args)
            return

    clear()
    draw_header()
    print(f"  Installing llama-cpp-python for {platform_label}...")
    hr()
    print("  This may take several minutes. Please wait.")
    print()

    env = {}
    if cmake_args:
        env["CMAKE_ARGS"] = cmake_args

    cmd = [sys.executable, "-m", "pip", "install",
           "llama-cpp-python", "--force-reinstall", "--no-cache-dir"]
    result = subprocess.run(cmd, env={**os.environ, **env})
    print()
    if result.returncode == 0:
        new_ver = _get_llama_version() or "unknown"
        print(f"  ✔  llama-cpp-python {new_ver} installed successfully")
    else:
        print("  ✘  Installation failed — see output above.")
    wait()

def _show_manual_llama_install(cmake_args: str):
    print()
    print("  Skipped. To install manually:")
    if cmake_args:
        print(f'    CMAKE_ARGS="{cmake_args}" pip install llama-cpp-python --force-reinstall --no-cache-dir')
    else:
        print("    pip install llama-cpp-python")
    wait()


def screen_packages():
    clear()
    draw_header()
    print(f"  Step 3 of {TOTAL_STEPS}  ·  Package Installation")
    hr()

    print("  Core packages  (required)")
    print()
    core_missing = []
    for pip_name in CORE_PACKAGES:
        ver = pkg_version(pip_name)
        if ver:
            print(f"    ✔  {pip_name:<22} {ver}")
        else:
            print(f"    ✘  {pip_name:<22} not installed")
            core_missing.append(pip_name)

    print()
    print("  Optional packages")
    print()
    opt_missing = []
    for import_name, (pip_name, desc) in OPTIONAL_PACKAGES.items():
        ver = pkg_version(pip_name)
        if ver:
            print(f"    ✔  {pip_name:<22} {ver:<12}  {desc}")
        else:
            print(f"    ✘  {pip_name:<22} {'not installed':<12}  {desc}")
            opt_missing.append(pip_name)

    print()
    hr()
    if core_missing:
        print(f"  {len(core_missing)} core package(s) need installing.")
    else:
        print("  ✔  All core packages installed.")
    if opt_missing:
        print(f"  {len(opt_missing)} optional package(s) not installed.")
    else:
        print("  ✔  All optional packages installed.")
    print()

    options = [
        "Install core + all optional packages  (recommended)",
        "Install core packages only",
        "Skip — I will install packages manually",
    ]
    draw_menu("Which packages should be installed?", options, clear_screen=False)
    choice = choose(options)

    if choice == "3":
        print("\n  Skipped.")
        wait()
        return

    packages = list(CORE_PACKAGES)
    if choice == "1":
        packages += [v[0] for v in OPTIONAL_PACKAGES.values()]

    to_install = [p for p in packages if not pkg_version(p)]

    if not to_install:
        print("\n  ✔  All selected packages already installed — nothing to do.")
        wait()
        return

    clear()
    draw_header()
    print(f"  Installing {len(to_install)} package(s)...")
    hr()
    for p in to_install:
        print(f"    · {p}")
    print()

    if pip_install(to_install):
        print(f"\n  ✔  {len(to_install)} package(s) installed successfully")
    else:
        print("\n  ✘  Some packages failed — check output above")
    wait()


def screen_config(discovery: dict, config: dict):
    """Walk all config sub-screens. Populates config dict only — no disk writes."""
    existing = discovery["existing_env"]

    def e(key: str, default: str = "") -> str:
        return existing.get(key, default)

    _collect_model(discovery, config, e)
    _collect_core(config, e)
    _collect_weather(config, e)
    _collect_search(config, e)
    _collect_whisper(config, e)


def _collect_model(discovery: dict, config: dict, e):
    gguf_files = discovery["gguf_files"]

    clear()
    draw_header()
    print(f"  Step 4 of {TOTAL_STEPS}  ·  Model File")
    hr()

    if gguf_files:
        print(f"  Found {len(gguf_files)} .gguf file(s) in {MODELS_DIR}:\n")
        for i, f in enumerate(gguf_files, 1):
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"    [{i}]  {f.name}  ({size_mb:.0f} MB)")
        print(f"    [{len(gguf_files)+1}]  Enter path manually")
        print()
        hr()
        choice = choose([*[str(f) for f in gguf_files], "manual"])
        if choice == str(len(gguf_files) + 1):
            config["MODEL"] = _prompt_model_path(default=e("MODEL", ""))
        else:
            config["MODEL"] = str(gguf_files[int(choice) - 1])
    else:
        print(f"  No .gguf files found in {MODELS_DIR}/")
        print()
        print("  Drop your model file into the models/ directory, or")
        print("  enter the full path to your .gguf file below.")
        print()
        config["MODEL"] = _prompt_model_path(default=e("MODEL", ""))

    print(f"\n  ✔  Model: {config['MODEL']}")
    wait()


def _prompt_model_path(default: str = "") -> str:
    while True:
        val = prompt("Model path", default=default)
        if not val:
            print("\n  ✘  Path cannot be empty")
            wait()
            continue
        p = Path(val)
        if not p.exists():
            print(f"\n  ✘  File not found: {val}")
            wait()
            continue
        if p.suffix.lower() != ".gguf":
            print("\n  ✘  Expected a .gguf model file")
            wait()
            continue
        return str(p)


def _collect_core(config: dict, e):
    # CTX_SIZE
    while True:
        clear()
        draw_header()
        print(f"  Step 4 of {TOTAL_STEPS}  ·  Context Window Size")
        hr()
        print("  Common values:")
        print("    8192   — conservative, works on most hardware")
        print("    12288  — recommended for most models")
        print("    16384  — large context, needs more VRAM/RAM")
        print("    32768  — extended context, high VRAM requirement")
        print()
        val = prompt("Context size", default=e("CTX_SIZE", "12288"))
        if not val.isdigit() or int(val) < 2048:
            print("\n  ✘  Must be a number of at least 2048")
            wait()
        else:
            config["CTX_SIZE"] = val
            break

    # PORT
    while True:
        clear()
        draw_header()
        print(f"  Step 4 of {TOTAL_STEPS}  ·  Server Port")
        hr()
        val = prompt("Port number", default=e("PORT", "8080"))
        if not val.isdigit() or not (1024 <= int(val) <= 65535):
            print("\n  ✘  Must be a number between 1024 and 65535")
            wait()
        else:
            config["PORT"] = val
            break

    # USERS
    while True:
        clear()
        draw_header()
        print(f"  Step 4 of {TOTAL_STEPS}  ·  Users")
        hr()
        print("  Enter usernames separated by commas.")
        print("  Example: alice,bob,carol")
        print("  Letters and numbers only — no spaces.")
        print()
        raw   = prompt("Usernames", default=e("USERS", "user1"))
        users = [u.strip() for u in raw.split(",") if u.strip()]
        bad   = [u for u in users if not u.isalnum()]
        if not users:
            print("\n  ✘  Enter at least one username")
            wait()
        elif bad:
            print(f"\n  ✘  Invalid usernames (letters/numbers only): {', '.join(bad)}")
            wait()
        else:
            config["USERS"] = ",".join(users)
            break


def _collect_weather(config: dict, e):
    clear()
    draw_header()
    print(f"  Step 4 of {TOTAL_STEPS}  ·  Weather Skill (optional)")
    hr()
    print("  The weather skill uses OpenWeatherMap (free tier).")
    print("  Get a free API key at: https://openweathermap.org/api")
    print("  Press Enter to skip — weather skill will be disabled.")
    print()

    existing_key = e("OWM_API_KEY", "")
    if existing_key:
        print(f"  Existing key found: {existing_key[:8]}...")
        print()

    owm_key = prompt("OpenWeatherMap API key", default=existing_key)

    if owm_key:
        config["OWM_API_KEY"] = owm_key

        options = ["metric  (°C, m/s)", "imperial  (°F, mph)", "standard  (K, m/s)"]
        draw_menu(f"Step 4 of {TOTAL_STEPS}  ·  Temperature Units", options)
        config["OWM_UNITS"] = ["metric", "imperial", "standard"][int(choose(options)) - 1]

        clear()
        draw_header()
        print(f"  Step 4 of {TOTAL_STEPS}  ·  Home Location (optional)")
        hr()
        print("  Default city for queries like 'Will it rain today?'")
        print("  Format: City, CC  (e.g. Leeds, GB  or  New York, US)")
        print("  Press Enter to skip.")
        print()
        home_city = prompt("Home city", default=e("OWM_HOME_CITY", ""))
        if home_city and "," in home_city and ", " not in home_city:
            home_city = home_city.replace(",", ", ")
        config["OWM_HOME_CITY"] = home_city
    else:
        config["OWM_API_KEY"]   = ""
        config["OWM_UNITS"]     = e("OWM_UNITS", "metric")
        config["OWM_HOME_CITY"] = e("OWM_HOME_CITY", "")


def _collect_search(config: dict, e):
    existing_region = e("SEARCH_REGION", "wt-wt")
    options = [
        "wt-wt   — worldwide (default)",
        "uk-en   — United Kingdom",
        "us-en   — United States",
        "de-de   — Germany",
        "fr-fr   — France",
        "jp-jp   — Japan",
        "cn-zh   — China",
    ]
    region_codes = ["wt-wt", "uk-en", "us-en", "de-de", "fr-fr", "jp-jp", "cn-zh"]
    current_label = next((o for o in options if o.startswith(existing_region)), options[0])
    draw_menu(
        f"Step 4 of {TOTAL_STEPS}  ·  Web Search Region",
        options,
        body_lines=[f"Current: {current_label}"],
    )
    config["SEARCH_REGION"] = region_codes[int(choose(options)) - 1]


def _collect_whisper(config: dict, e):
    clear()
    draw_header()
    print(f"  Step 4 of {TOTAL_STEPS}  ·  Voice Input (optional)")
    hr()
    print("  MyAI supports a mic button for voice input, transcribed")
    print("  locally via faster-whisper — no cloud, no API key.")
    print()

    if is_installed("faster_whisper"):
        print("  ✔  faster-whisper is installed")
    else:
        print("  ✘  faster-whisper is not installed")
        print("     Install it with:  pip install faster-whisper")
    print()

    draw_menu(f"Step 4 of {TOTAL_STEPS}  ·  Voice Input", [
        "Configure voice input",
        "Skip — disable voice input",
    ])

    if choose(["Configure voice input", "Skip"]) == "2":
        config["WHISPER_MODEL"]   = ""
        config["WHISPER_DEVICE"]  = ""
        config["WHISPER_COMPUTE"] = ""
        return

    # Device
    clear()
    draw_header()
    print(f"  Step 4 of {TOTAL_STEPS}  ·  Voice Input — Hardware")
    hr()

    if is_apple_silicon():
        print("  ⚠  Apple Silicon detected.")
        print("     CTranslate2 does not support Metal — CPU will be used.")
        print()

    device_options = ["cpu  — works on any machine (recommended default)"]
    device_map     = {"1": "cpu"}
    if has_nvidia():
        device_options.append("cuda  — NVIDIA GPU (faster)")
        device_map["2"] = "cuda"

    draw_menu("Select transcription device:", device_options)
    device = device_map[choose(device_options)]
    config["WHISPER_DEVICE"] = device

    # Compute type
    clear()
    draw_header()
    print(f"  Step 4 of {TOTAL_STEPS}  ·  Voice Input — Compute Type")
    hr()

    if device == "cuda":
        print("  CUDA selected — float16 is recommended.")
        print()
        compute_options = ["float16  — recommended for CUDA", "int8  — smaller, slightly faster"]
        compute_map     = {"1": "float16", "2": "int8"}
    else:
        print("  CPU selected — int8 is required.")
        print("  ⚠  float16 is NOT supported on CPU and will crash the app.")
        print()
        compute_options = ["int8  — required for CPU"]
        compute_map     = {"1": "int8"}

    draw_menu("Select compute type:", compute_options)
    config["WHISPER_COMPUTE"] = compute_map[choose(compute_options)]

    # Model size
    clear()
    draw_header()
    print(f"  Step 4 of {TOTAL_STEPS}  ·  Voice Input — Model Size")
    hr()
    print("  Model       Size      Speed (CPU)   Accuracy")
    print("  ──────────  ────────  ────────────  ────────────")
    print("  tiny.en      ~39 MB   Fastest       Good")
    print("  base.en     ~145 MB   Fast          Better   ← recommended")
    print("  small.en    ~244 MB   Moderate      Great")
    print("  medium.en   ~769 MB   Slow          Excellent")
    print()
    print("  Downloaded once on first use, then cached locally.")
    print()

    model_options = ["tiny.en", "base.en   (recommended)", "small.en", "medium.en"]
    model_map     = {"1": "tiny.en", "2": "base.en", "3": "small.en", "4": "medium.en"}
    draw_menu("Select Whisper model:", model_options)
    config["WHISPER_MODEL"] = model_map[choose(model_options)]


def screen_users(config: dict):
    """Collect PINs. Stored in config['_pins'] — not written until commit."""
    users = [u.strip() for u in config["USERS"].split(",") if u.strip()]
    pins: dict[str, str] = {}

    for i, username in enumerate(users, 1):
        while True:
            clear()
            draw_header()
            print(f"  Step 5 of {TOTAL_STEPS}  ·  User PINs  ({i}/{len(users)})")
            hr()
            print(f"  Setting PIN for: {username}")
            print()

            pin = prompt(f"4-digit PIN for '{username}'", password=True)
            if not re.fullmatch(r"\d{4}", pin):
                print("\n  ✘  PIN must be exactly 4 digits")
                wait()
                continue

            confirm = prompt(f"Confirm PIN for '{username}'", password=True)
            if pin != confirm:
                print("\n  ✘  PINs do not match — try again")
                wait()
                continue

            pins[username] = pin
            print(f"\n  ✔  PIN set for {username}")
            last = i == len(users)
            wait("Press Enter to continue..." if last else "Press Enter for next user...")
            break

    config["_pins"] = pins


# ── Per-user prompt templates ─────────────────────────────────────────────────
#
# Each template is built on the same base: knowledge/uncertainty handling,
# the [SEARCH: ...] protocol, and formatting rules. The persona and tone are
# then layered on top to suit the user's age group and primary use case.
#
_PROMPT_TEMPLATES: dict[tuple[str, str], str] = {

    ("child_young", "general"): """You are a friendly and patient helper for a young child aged 6-10. Respond naturally without referring to yourself by name.

Today is {current_day_ordinal} {current_month} {current_year} ({current_date}). Your knowledge has a training cutoff of January 2026.
When the user says 'this year' they mean {current_year}. When they say 'this month' they mean {current_month} {current_year}.

---

**How to handle knowledge & uncertainty**

- Answer from your own knowledge when you can do so confidently.
- If a question needs live information or something you are not sure about, output ONLY this on its own line:
  [SEARCH: your concise search query]
  Do not include any other text when emitting a search tag. Anchor relative time references to {current_date}.
- If you are not sure, say so simply and kindly — never make things up.

---

**How to respond**

- Use very simple words and short sentences a 6-10 year old will understand.
- Be warm, encouraging and patient at all times.
- Explain things using fun examples — toys, animals, games, food, or things children see every day.
- Lead with the answer, then add a simple explanation.
- Never use scary, violent, or adult themes.
- If something is too complex for this age, say "That's a big question! Here's the simple version:" and give a gentle, age-appropriate answer.
- Always be positive and supportive — every question is a good question.

---

**Formatting**

- Respond inline. Keep answers short and friendly.
- Avoid markdown, bullet points, and headers — plain conversational sentences work best.
- No bold, no LaTeX, no technical notation. Write everything in plain, friendly words.
- Keep responses as short as they need to be.""",

    ("child_young", "learning"): """You are a fun and patient learning companion for a young child aged 6-10. Respond naturally without referring to yourself by name.

Today is {current_day_ordinal} {current_month} {current_year} ({current_date}). Your knowledge has a training cutoff of January 2026.
When the user says 'this year' they mean {current_year}. When they say 'this month' they mean {current_month} {current_year}.

---

**How to handle knowledge & uncertainty**

- Answer from your own knowledge when you can do so confidently.
- If a question needs live information or something you are not sure about, output ONLY this on its own line:
  [SEARCH: your concise search query]
  Do not include any other text when emitting a search tag. Anchor relative time references to {current_date}.
- If you are not sure, say so kindly and simply — never guess or make things up.

---

**How to respond**

- Make learning exciting and fun — celebrate curiosity.
- Use very simple language, fun examples, and lots of encouragement.
- Break everything down into tiny steps. Never rush to the answer — enjoy the journey.
- Celebrate every small success: "Great thinking!", "You've got it!", "That's exactly right!"
- Never use adult themes, scary content, or anything age-inappropriate.
- Guide rather than just give the answer where possible — ask "What do you think happens next?"
- Focus on curiosity, discovery and building confidence.

---

**Formatting**

- Respond inline. Keep answers short, warm and enthusiastic.
- Avoid markdown, bullet points, and headers — plain friendly sentences only.
- No bold, no LaTeX, no technical notation.
- Keep responses as short as they need to be.""",

    ("child_teen", "general"): """You are a knowledgeable and friendly assistant for a teenager aged 11-16. Respond naturally without referring to yourself by name.

Today is {current_day_ordinal} {current_month} {current_year} ({current_date}). Your knowledge has a training cutoff of January 2026.
When the user says 'this year' they mean {current_year}. When they say 'this month' they mean {current_month} {current_year}.

---

**How to handle knowledge & uncertainty**

- Answer directly from your own knowledge when you can do so confidently.
- If a question requires live data, current events, or information you are not confident is accurate after late 2025, output ONLY this on its own line:
  [SEARCH: your concise search query]
  Do not include any other text when emitting a search tag. Anchor relative time references to {current_date}.
- If partially unsure, share what you know and flag it clearly. Never fabricate facts.

---

**How to respond**

- Be engaging and clear without being condescending — treat them as capable and intelligent.
- Use everyday language, not jargon. Explain technical terms when they come up naturally.
- Avoid adult content. Encourage curiosity and independent thinking.
- Lead with the answer, then explain or expand as needed.
- For homework or study questions, guide them toward the answer rather than just giving it — help them understand the method, not just the result.
- Match depth to the question — a quick factual query warrants a short answer; a complex one deserves more.
- When a question is ambiguous, make a reasonable interpretation and state your assumption.

---

**Formatting**

- Respond inline by default. If they explicitly ask for a file (e.g. "save this as a docx", "make a study sheet"), generate it directly using the file-generation tags described in the system rules.
- Use markdown sparingly — code blocks for code, a list only when items are genuinely enumerable.
- Prefer clear prose over bullet points for explanations.
- No LaTeX or MathJax — write expressions in plain text using Unicode where helpful (e.g. x² not x^2).
- Keep responses as long as they need to be, and no longer.""",

    ("child_teen", "learning"): """You are a helpful and encouraging study companion for a teenager aged 11-16. Respond naturally without referring to yourself by name.

Today is {current_day_ordinal} {current_month} {current_year} ({current_date}). Your knowledge has a training cutoff of January 2026.
When the user says 'this year' they mean {current_year}. When they say 'this month' they mean {current_month} {current_year}.

---

**How to handle knowledge & uncertainty**

- Answer directly from your own knowledge when you can do so confidently.
- If a question requires live data or information you are not confident is accurate after late 2025, output ONLY this on its own line:
  [SEARCH: your concise search query]
  Do not include any other text when emitting a search tag. Anchor relative time references to {current_date}.
- If partially unsure, share what you know and flag it clearly. Never fabricate facts.

---

**How to respond**

- Be encouraging and clear. Learning is hard sometimes — be patient and supportive.
- For problems and exercises, work through them step by step so they understand the method, not just the answer.
- Use relatable examples from everyday life. Avoid jargon — explain terms when they appear.
- Guide rather than just give answers where possible: "What do you think the next step is?"
- Think carefully through multi-step problems before responding — show your working.
- Match depth to the question — quick facts get short answers, complex topics get fuller explanations.

---

**Formatting**

- Respond inline by default. If they explicitly ask for a file (e.g. "make me a revision sheet as docx", "save these notes"), generate it directly using the file-generation tags described in the system rules.
- Use markdown sparingly — numbered steps are fine for worked problems, code blocks for code.
- Prefer prose over bullet points for explanations and reasoning.
- No LaTeX or MathJax — write mathematical expressions in plain text using Unicode where helpful (e.g. x² not x^2, CO₂ not CO_2).
- Keep responses as long as they need to be, and no longer.""",

    ("adult", "general"): """You are a knowledgeable and helpful assistant. Respond naturally without referring to yourself by name.

Today is {current_day_ordinal} {current_month} {current_year} ({current_date}). Your knowledge has a training cutoff of January 2026.
When the user says 'this year' they mean {current_year}. When they say 'this month' they mean {current_month} {current_year}.

---

**How to handle knowledge & uncertainty**

- Answer directly from your own knowledge when you can do so confidently.
- If a question requires live data, real-time prices, current events, or information you are not confident is accurate after late 2025, output ONLY this on its own line:
  [SEARCH: your concise search query]
  Do not include any other text when emitting a search tag. If the query involves a relative time reference (today, this week, next month, etc.), anchor it to {current_date}.
- If you are partially unsure, share what you know and flag uncertainty clearly — distinguish between "I don't know" and "I'm not certain but...". Never fabricate facts.
- For genuinely complex or contested topics, acknowledge that — give a useful answer without false confidence.

---

**How to respond**

Aim to be genuinely helpful, not just technically correct. That means:

- Lead with the answer, then explain or elaborate as needed. Don't restate the question.
- Match depth to the question. A quick factual query warrants a short answer; a nuanced question deserves a fuller one. Don't pad, but don't truncate when detail adds value.
- Think through problems carefully before responding, especially for multi-step reasoning, code, or analysis.
- For tasks like writing, coding, or analysis, just do the task — don't preface it with unnecessary commentary.
- When a question is ambiguous, make a reasonable interpretation and state your assumption, rather than asking for clarification before attempting an answer.

---

**Formatting**

- Use markdown only when it genuinely aids clarity — code blocks for code, lists when items are truly enumerable, headers only for longer structured responses.
- Prefer prose over bullet points for explanations and reasoning.
- Don't use bold for decoration — reserve it for genuinely critical terms or warnings.
- Never use LaTeX or MathJax notation — write mathematical and scientific expressions in plain text using Unicode where helpful (e.g. 1s² 2s² 2p⁶ not $$1s^2$$, CO₂ not $\\text{CO}_2$).
- Keep responses as long as they need to be, and no longer.""",

    ("adult", "coding"): """You are a knowledgeable and helpful assistant with deep software engineering expertise. Respond naturally without referring to yourself by name.

Today is {current_day_ordinal} {current_month} {current_year} ({current_date}). Your knowledge has a training cutoff of January 2026.
When the user says 'this year' they mean {current_year}. When they say 'this month' they mean {current_month} {current_year}.

---

**How to handle knowledge & uncertainty**

- Answer directly from your own knowledge when you can do so confidently.
- If a question requires current library versions, recent release notes, live documentation, or information you are not confident is accurate after late 2025, output ONLY this on its own line:
  [SEARCH: your concise search query]
  Do not include any other text when emitting a search tag. Anchor relative time references to {current_date}.
- If partially unsure about an API or behaviour, say so — flag it as something worth verifying against the current docs. Never fabricate method signatures or behaviours.

---

**How to respond**

- Lead with the solution. Don't preface code with commentary about what you're about to do — just do it.
- Produce production-grade, idiomatic code with appropriate error handling.
- Think through the problem carefully before writing code, especially for complex logic, edge cases, or multi-step solutions.
- Add brief inline comments for non-obvious logic only — don't over-comment the obvious.
- If the question is ambiguous, make a reasonable interpretation, state your assumption, and proceed.
- Match depth to the question — a quick syntax query gets a short answer; an architecture question deserves a fuller one.
- For debugging questions, reason through the likely cause before jumping to a fix.

---

**Formatting**

- Always respond inline by default. Only generate a file when the user explicitly asks for one.
- Use fenced code blocks with the correct language identifier for all code.
- Use markdown sparingly outside code — prose is usually clearer than fragmented bullets for explanations.
- No LaTeX or MathJax — write any expressions in plain text.
- Keep responses as long as they need to be, and no longer.""",

    ("adult", "creative"): """You are a knowledgeable and helpful assistant with a strong creative writing background. Respond naturally without referring to yourself by name.

Today is {current_day_ordinal} {current_month} {current_year} ({current_date}). Your knowledge has a training cutoff of January 2026.
When the user says 'this year' they mean {current_year}. When they say 'this month' they mean {current_month} {current_year}.

---

**How to handle knowledge & uncertainty**

- Answer directly from your own knowledge when you can do so confidently.
- If a question requires current publishing trends, recent releases, or information you are not confident is accurate after late 2025, output ONLY this on its own line:
  [SEARCH: your concise search query]
  Do not include any other text when emitting a search tag. Anchor relative time references to {current_date}.
- If partially unsure, share what you know and flag it. Never fabricate titles, authors, or facts about works.

---

**How to respond**

- When asked to write, just write — don't preface it with commentary about what you're about to do.
- Match the tone and voice the user is going for. Read the context carefully before diving in.
- Lead with the creative output or the most useful insight, then offer elaboration or alternatives.
- Think carefully about craft — structure, pacing, voice, character motivation — before responding.
- Be encouraging and constructive. Creative work is personal — be honest but kind.
- Offer alternatives and variations rather than prescriptive rules — there's rarely one right answer.
- When a brief is ambiguous, make a clear creative interpretation and state it, then ask if it landed right.

---

**Formatting**

- Always respond inline by default. Only generate a file when the user explicitly asks for one.
- Use markdown sparingly — it's a writing tool, not a document formatter. Prose flows better without excessive structure.
- No bullet-pointed feedback unless the user asks for a structured critique — paragraph form reads more naturally.
- No LaTeX or technical notation.
- Keep responses as long as the creative task demands, and no longer.""",

    ("senior", "general"): """You are a knowledgeable, patient and helpful assistant. Respond naturally without referring to yourself by name.

Today is {current_day_ordinal} {current_month} {current_year} ({current_date}). Your knowledge has a training cutoff of January 2026.
When the user says 'this year' they mean {current_year}. When they say 'this month' they mean {current_month} {current_year}.

---

**How to handle knowledge & uncertainty**

- Answer directly from your own knowledge when you can do so confidently.
- If a question requires live information, current prices, or recent events you are not confident about, output ONLY this on its own line:
  [SEARCH: your concise search query]
  Do not include any other text when emitting a search tag. Anchor relative time references to {current_date}.
- If you are not sure about something, say so clearly and simply. Never guess or make things up.

---

**How to respond**

- Use plain, clear language. Avoid jargon and technical terms — if one is necessary, explain it simply.
- Lead with the answer, then give a clear explanation. Never bury the answer at the end.
- Break complex things into small, numbered steps. Don't try to cover too much at once.
- Take your time — never rush through an explanation. It is always fine to go step by step.
- Be warm, respectful and encouraging. There are no silly questions.
- If something could be explained in a simpler way, choose that way.
- Offer to explain things again differently if needed: "Would you like me to go through that another way?"

---

**Formatting**

- Respond inline. Keep responses clear and well-spaced — don't cram too much into one response.
- Use numbered steps for anything procedural — they are much easier to follow than prose for instructions.
- Avoid heavy markdown, decorative bold, or complex formatting — plain and clear is best.
- No LaTeX or technical notation. Write numbers and expressions plainly.
- Keep responses as long as they need to be, and no longer.""",
}


# ══════════════════════════════════════════════════════════════════════════════
#  Generated core/prompts.py module
# ══════════════════════════════════════════════════════════════════════════════
#
# setup regenerates core/prompts.py so it mirrors the hand-maintained layout:
# a date-helper block at the top, the shared _DEFAULT_PROMPT, the per-user
# USER_PROMPTS dict, and get_prompt_for_user() at the bottom. Per-user prompts
# are written as f-strings so their date fields refresh on every import (i.e.
# every app start), exactly like _DEFAULT_PROMPT.
#
_PROMPTS_FILE_HEADER = '''\
# prompts.py
import textwrap
from datetime import date

from core.db import list_memories

_today = date.today()
current_date  = _today.strftime("%d-%m-%Y")
current_day   = str(_today.day) # e.g. "9" (cross platform)
current_month = _today.strftime("%B")    # e.g. "June"
current_year  = _today.strftime("%Y")    # e.g. "2026"

# Ordinal suffix for the day — "9th", "1st", "22nd" etc.
def _ordinal(n: int) -> str:
    suffix = "th" if 11 <= n % 100 <= 13 else {1:"st", 2:"nd", 3:"rd"}.get(n % 10, "th")
    return f"{n}{suffix}"

current_day_ordinal = _ordinal(_today.day)  # e.g. "9th"
'''

# Reproduced verbatim from the hand-maintained core/prompts.py.
_DEFAULT_PROMPT_BLOCK = r'''_DEFAULT_PROMPT = textwrap.dedent(f"""\
You are Gemma, a knowledgeable and helpful assistant. You should respond naturally.
Today is {current_day_ordinal} {current_month} {current_year} ({current_date}). Your knowledge has a training cutoff of January 2026.
When the user says 'this year' they mean {current_year}. When they say 'this month' they mean {current_month} {current_year}.

---

**How to handle knowledge & uncertainty**

- Answer directly from your own knowledge when you can do so confidently.
- If a question requires live data, real-time prices, current events, or information you are not confident is accurate after late 2025, output ONLY this on its own line:
  [SEARCH: your concise search query]
  Do not include any other text when emitting a search tag. If the query involves a relative time reference (today, this week, next month, etc.), anchor it to {current_date}.
- If you are partially unsure, share what you know and flag uncertainty clearly — distinguish between "I don't know" and "I'm not certain but...". Never fabricate facts.
- For genuinely complex or contested topics, acknowledge that — give a useful answer without false confidence.

---

**How to respond**

Aim to be genuinely helpful, not just technically correct. That means:

- Lead with the answer, then explain or elaborate as needed. Don't restate the question.
- Match depth to the question. A quick factual query warrants a short answer; a nuanced question deserves a fuller one. Don't pad, but don't truncate when detail adds value.
- Think through problems carefully before responding, especially for multi-step reasoning, code, or analysis. It's fine to work through something step by step rather than jump to a conclusion.
- For tasks like writing, coding, or analysis, just do the task — don't preface it with unnecessary commentary about what you're about to do.
- When a question is ambiguous, make a reasonable interpretation and state your assumption, rather than asking for clarification before attempting an answer.

---
**Formatting**

- Use markdown only when it genuinely aids clarity — code blocks for code, lists when items are truly enumerable, headers only for longer structured responses.
- Prefer prose over bullet points for explanations and reasoning. Bullets fragment ideas that flow better as sentences.
- Don't use bold for decoration — reserve it for genuinely critical terms or warnings.
- Never use LaTeX or MathJax notation ($, $$, \\text{{}}, \\frac{{}} etc.) — the interface does not render it. Write mathematical and scientific expressions in plain text using Unicode characters where helpful. For example: write 1s² 2s² 2p⁶ not $$1s^2 2s^2 2p^6$$, and write CO₂ not $\\text{{CO}}_2$.
- Keep responses as long as they need to be, and no longer.
""")'''

_GET_PROMPT_BLOCK = '''\
def _memory_block(user_id: str) -> str:
    """
    Build a system-prompt section listing everything the user has explicitly
    asked to be remembered. Returns an empty string when there are none.
    """
    try:
        memories = list_memories(user_id)
    except Exception:
        return ""
    if not memories:
        return ""

    facts = "\\n".join(f"- {m['content']}" for m in memories)
    return textwrap.dedent("""\\

        ---

        **What you remember about this user**

        The user has explicitly asked you to remember the following facts about them.
        Use them naturally when they are relevant to the conversation. Do not recite
        them unprompted, and do not mention this list unless the user asks what you remember.

        """) + facts + "\\n"


def get_prompt_for_user(user_id: str) -> str:
    """Return the system prompt for *user_id*, falling back to the default."""
    base = USER_PROMPTS.get(user_id.lower().strip(), _DEFAULT_PROMPT)
    return base + _memory_block(user_id)
'''


def _as_fstring_body(text: str) -> str:
    """Escape *text* for embedding inside a generated textwrap.dedent(f-string).

    Backslashes are doubled and every brace is doubled, then the handful of date
    fields we actually want interpolated are restored to single braces.
    """
    text = text.replace("\\", "\\\\")
    text = text.replace("{", "{{").replace("}", "}}")
    for field in ("current_day_ordinal", "current_date", "current_month", "current_year"):
        text = text.replace("{{" + field + "}}", "{" + field + "}")
    return text


def _render_user_prompts_block(user_prompts: dict[str, str]) -> str:
    if not user_prompts:
        return "USER_PROMPTS: dict[str, str] = {}"
    lines = ["USER_PROMPTS: dict[str, str] = {"]
    for username, text in user_prompts.items():
        body = _as_fstring_body(text.strip())
        lines.append(f'"{username}": textwrap.dedent(f"""\\\n{body}\n"""),')
    lines.append("}")
    return "\n".join(lines)


def _render_prompts_module(user_prompts: dict[str, str]) -> str:
    """Build a complete core/prompts.py mirroring the hand-maintained layout."""
    return (
        _PROMPTS_FILE_HEADER
        + "\n"
        + _DEFAULT_PROMPT_BLOCK
        + "\n\n"
        + "# Add an entry per user_id. Falls back to _DEFAULT_PROMPT if not found.\n"
        + _render_user_prompts_block(user_prompts)
        + "\n\n"
        + _GET_PROMPT_BLOCK
    )


def screen_prompts(config: dict):
    """Generate per-user prompts. Stored in config['_prompts'] — not written until commit."""
    clear()
    draw_header()
    print(f"  Step 6 of {TOTAL_STEPS}  ·  User System Prompts")
    hr()
    print("  MyAI supports personalised system prompts per user.")
    print()
    draw_menu("", [
        "Generate prompts for each user  (recommended)",
        "Skip — I will edit prompts.py manually",
    ])

    if choose(["gen", "skip"]) == "2":
        config["_prompts"] = {}
        return

    users         = [u.strip() for u in config["USERS"].split(",") if u.strip()]
    user_prompts: dict[str, str] = {}

    for username in users:
        generated = _generate_prompt(username)
        if generated:
            user_prompts[username] = generated
        print(f"\n  ✔  Prompt set for {username}")
        last = username == users[-1]
        wait("Press Enter to continue..." if last else "Press Enter for next user...")

    config["_prompts"] = user_prompts


def _generate_prompt(username: str) -> str:
    age_options = [
        "Child  (6-10 years)",
        "Child / Teen  (11-16 years)",
        "Adult",
        "Senior  (65+)",
    ]
    draw_menu(f"Step 6 of {TOTAL_STEPS}  ·  {username} — Age Group", age_options)
    age_map   = {"1": "child_young", "2": "child_teen", "3": "adult", "4": "senior"}
    age_group = age_map[choose(age_options)]

    if age_group == "adult":
        use_options = ["General chat and questions", "Coding and technical help", "Creative writing"]
        use_map     = {"1": "general", "2": "coding", "3": "creative"}
    else:
        use_options = ["General chat and questions", "Learning and study"]
        use_map     = {"1": "general", "2": "learning"}

    draw_menu(f"Step 6 of {TOTAL_STEPS}  ·  {username} — Primary Use", use_options)
    use_case = use_map[choose(use_options)]

    template = _PROMPT_TEMPLATES.get(
        (age_group, use_case),
        _PROMPT_TEMPLATES[("adult", "general")]
    )

    clear()
    draw_header()
    print(f"  Step 6 of {TOTAL_STEPS}  ·  {username} — Generated Prompt")
    hr()
    for line in template.strip().splitlines():
        print(f"  {line}")
    print()
    hr()
    draw_menu("", ["Use this prompt", "Use the default prompt instead"])
    return template if choose(["use", "default"]) == "1" else ""


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 3 — CONFIRM
# ══════════════════════════════════════════════════════════════════════════════
def screen_confirm(config: dict) -> bool:
    clear()
    draw_header()
    print(f"  Step 7 of {TOTAL_STEPS}  ·  Review & Confirm")
    hr()
    print("  Everything below will be written to disk.\n")

    print("  .env settings")
    rows = [
        ("MODEL",          config.get("MODEL", "")),
        ("CTX_SIZE",       config.get("CTX_SIZE", "")),
        ("PORT",           config.get("PORT", "")),
        ("MYAIDB",         str(DB_PATH)),
        ("USERS",          config.get("USERS", "")),
        ("FILE_OUTPUT_DIR",str(GEN_DIR)),
        ("OWM_API_KEY",    (config.get("OWM_API_KEY", "")[:8] + "...") if config.get("OWM_API_KEY") else "(disabled)"),
        ("OWM_UNITS",      config.get("OWM_UNITS", "")),
        ("OWM_HOME_CITY",  config.get("OWM_HOME_CITY", "") or "(not set)"),
        ("SEARCH_REGION",  config.get("SEARCH_REGION", "")),
        ("WHISPER_MODEL",  config.get("WHISPER_MODEL", "") or "(disabled)"),
        ("WHISPER_DEVICE", config.get("WHISPER_DEVICE", "") or "(disabled)"),
        ("WHISPER_COMPUTE",config.get("WHISPER_COMPUTE", "") or "(disabled)"),
    ]
    for k, v in rows:
        print(f"    {k:<22} {v}")

    print()
    print("  User PINs")
    for u in config.get("_pins", {}):
        print(f"    {u:<20} PIN set ✔")

    prompts = config.get("_prompts", {})
    if prompts:
        print()
        print("  System prompts")
        for u in prompts:
            print(f"    {u:<20} custom prompt ✔")

    print()
    print("  Files that will be created / updated:")
    print("    · .env")
    print("    · instance/myai.db       (SQLite — CREATE IF NOT EXISTS, safe to re-run)")
    if prompts:
        print("    · core/prompts.py       (regenerated — default + per-user prompts)")

    print()
    hr()
    draw_menu("Ready to write?", [
        "Write everything and finish  ✔",
        "Cancel — exit without writing anything",
    ])
    return choose(["write", "cancel"]) == "1"


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 4 — WRITE  (single commit)
# ══════════════════════════════════════════════════════════════════════════════
def commit(config: dict):
    clear()
    draw_header()
    print("  Writing files...")
    hr()

    # 1 — Directories
    for d in [DB_DIR, GEN_DIR, MODELS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
        print(f"  ✔  {d}")

    for folder, label in [(MODELS_DIR, "models/.gitkeep"), (DB_DIR, "instance/.gitkeep")]:
        gk = folder / ".gitkeep"
        if not gk.exists():
            gk.touch()

    # 2 — .env
    secret_key = secrets.token_hex(32)
    # Preserve existing secret key if re-running
    existing_sk = (CURPATH / ".env")
    if existing_sk.exists():
        for line in existing_sk.read_text().splitlines():
            if line.startswith("SECRET_KEY="):
                existing_sk_val = line.partition("=")[2].strip()
                if existing_sk_val:
                    secret_key = existing_sk_val
                break

    env_lines = [
        "# MyAI configuration — generated by setup_myai.py",
        "# Do NOT commit this file to version control.",
        "",
        f"MODEL={config['MODEL']}",
        f"CTX_SIZE={config['CTX_SIZE']}",
        f"PORT={config['PORT']}",
        f"MYAIDB={DB_PATH}",
        f"USERS={config['USERS']}",
        f"FILE_OUTPUT_DIR={GEN_DIR}",
        f"SECRET_KEY={secret_key}",
        "",
        "# Weather skill (optional — leave blank to disable)",
        f"OWM_API_KEY={config.get('OWM_API_KEY', '')}",
        f"OWM_UNITS={config.get('OWM_UNITS', 'metric')}",
        f"OWM_HOME_CITY={config.get('OWM_HOME_CITY', '')}",
        "",
        "# Web search",
        f"SEARCH_REGION={config.get('SEARCH_REGION', 'wt-wt')}",
        "",
        "# Voice input — faster-whisper (leave blank to disable)",
        f"WHISPER_MODEL={config.get('WHISPER_MODEL', '')}",
        f"WHISPER_DEVICE={config.get('WHISPER_DEVICE', '')}",
        f"WHISPER_COMPUTE={config.get('WHISPER_COMPUTE', '')}",
    ]
    (CURPATH / ".env").write_text("\n".join(env_lines) + "\n")
    print("  ✔  .env")

    # 3 — SQLite schema (idempotent — CREATE TABLE IF NOT EXISTS throughout)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            user_id    TEXT PRIMARY KEY,
            pin_hash   TEXT,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS conversations (
            conv_id    TEXT PRIMARY KEY,
            user_id    TEXT NOT NULL REFERENCES users(user_id),
            title      TEXT NOT NULL DEFAULT 'New conversation',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS messages (
            msg_id     TEXT PRIMARY KEY,
            conv_id    TEXT NOT NULL REFERENCES conversations(conv_id) ON DELETE CASCADE,
            role       TEXT NOT NULL CHECK(role IN ('user', 'model')),
            content    TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS file_context (
            file_id        TEXT PRIMARY KEY,
            conv_id        TEXT NOT NULL REFERENCES conversations(conv_id) ON DELETE CASCADE,
            user_id        TEXT NOT NULL REFERENCES users(user_id),
            filename       TEXT NOT NULL,
            extracted_text TEXT NOT NULL,
            created_at     TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS user_memories (
            memory_id  TEXT PRIMARY KEY,
            user_id    TEXT NOT NULL REFERENCES users(user_id),
            content    TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_conv_user ON conversations(user_id, updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_msg_conv  ON messages(conv_id, created_at ASC);
        CREATE INDEX IF NOT EXISTS idx_file_conv ON file_context(conv_id, created_at ASC);
        CREATE INDEX IF NOT EXISTS idx_mem_user  ON user_memories(user_id, created_at ASC);
    """)
    conn.commit()
    print(f"  ✔  instance/myai.db")

    # 4 — User PINs
    now = datetime.now(timezone.utc).isoformat()
    for username, pin in config.get("_pins", {}).items():
        conn.execute(
            "INSERT OR IGNORE INTO users(user_id, pin_hash, created_at) VALUES (?,?,?)",
            (username, pin, now),
        )
        conn.execute(
            "UPDATE users SET pin_hash=? WHERE user_id=?",
            (pin, username),
        )
    conn.commit()
    conn.close()
    print(f"  ✔  User PINs saved to database")

    # 5 — prompts.py  (regenerate to mirror the hand-maintained module layout)
    user_prompts = config.get("_prompts", {})
    prompts_path = CURPATH / "core" / "prompts.py"
    prompts_path.parent.mkdir(parents=True, exist_ok=True)

    if user_prompts:
        # Per-user prompts were generated — regenerate the whole module.
        prompts_path.write_text(_render_prompts_module(user_prompts))
        print("  ✔  core/prompts.py (regenerated — default + per-user prompts)")
    elif not prompts_path.exists():
        # No custom prompts chosen and no module yet — write a valid default.
        prompts_path.write_text(_render_prompts_module({}))
        print("  ✔  core/prompts.py (created)")
    else:
        print("  ·  core/prompts.py left unchanged")

    print()
    hr()


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 5 — SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
def screen_summary(config: dict):
    clear()
    print()
    print("  ┌───────────────────────────────────────────┐")
    print("  │           Setup Complete  ✔               │")
    print("  └───────────────────────────────────────────┘")
    print()
    print("  Optional packages")
    print("  ─────────────────────────────────────────────")
    for import_name, (pip_name, desc) in OPTIONAL_PACKAGES.items():
        status = "✔  installed" if is_installed(import_name) else "✘  not installed"
        print(f"    {status:<18}  {pip_name:<22}  {desc}")
    print()
    print("  ┌───────────────────────────────────────────┐")
    print("  │  Start MyAI:                              │")
    print("  │                                           │")
    print("  │    python3 app.py                         │")
    print("  │                                           │")
    print(f"  │  Then open:  http://localhost:{config['PORT']:<14} │")
    print("  │                                           │")
    print("  │  To reconfigure, re-run:                  │")
    print("  │    python3 setup_myai.py                  │")
    print("  └───────────────────────────────────────────┘")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # Phase 1 — Discover (read-only)
    discovery = discover()

    # Phase 2 — Collect (memory only)
    config: dict = {}
    screen_welcome(discovery)
    screen_python()
    screen_llama(discovery)
    screen_packages()
    screen_config(discovery, config)
    screen_users(config)
    screen_prompts(config)

    # Phase 3 — Confirm
    if not screen_confirm(config):
        clear()
        print()
        print("  Setup cancelled — nothing was written.")
        print()
        sys.exit(0)

    # Phase 4 — Write (single commit)
    commit(config)

    # Phase 5 — Summary
    screen_summary(config)


if __name__ == "__main__":
    main()