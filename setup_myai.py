#!/usr/bin/env python3
"""
setup_myai.py
=============
Interactive setup wizard for MyAI.
Run once after cloning the repository, or re-run at any time to reconfigure.

    python3 setup_myai.py

Architecture
------------
1. DISCOVER	— scan filesystem, detect GPU/hardware, note what is installed
2. COLLECT 	— walk through screens; each screen populates `config` dict only
3. CONFIRM 	— show full summary; user approves before anything is written
4. WRITE   	— single commit: dirs, .env, and first user (via core.db)
5. SUMMARY 	— show what was created and the start command

The wizard no longer generates prompts.py or manages a users list — MyAI
uses core.db.init_db() to create the schema and manages users through the
Users page in the UI. This script's job is to get you as far as a working
.env and one usable login account.
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

TOTAL_STEPS = 6


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


def screen_first_user(config: dict):
    """
    Collect one first user (display name, PIN, age band) — if and only if
    the DB has no users yet. Stored in config['_first_user'] and not written
    until commit.

    Skipped entirely if the DB already contains users, matching the "DB is
    the source of truth" principle. On existing installs, the wizard just
    updates .env and leaves users alone.
    """
    # Skip if the DB already has users
    if _db_has_users():
        config["_first_user"] = None
        clear()
        draw_header()
        print(f"  Step 5 of {TOTAL_STEPS}  ·  Users")
        hr()
        print("  Existing users detected in the database — skipping first-user setup.")
        print("  Manage users via the Users page in the app after login.")
        print()
        wait()
        return

    # Collect display name
    while True:
        clear()
        draw_header()
        print(f"  Step 5 of {TOTAL_STEPS}  ·  Create Your First User")
        hr()
        print("  Set up the first user account so you can sign in after install.")
        print("  Add more users later via the Users page in the app.")
        print()
        display_id = prompt("Display name (e.g. 'George')").strip()
        if not display_id:
            print("\n  ✘  A display name is required")
            wait()
            continue
        if len(display_id) > 40:
            print("\n  ✘  Display name too long (max 40 characters)")
            wait()
            continue
        break

    # Collect PIN + confirmation
    while True:
        clear()
        draw_header()
        print(f"  Step 5 of {TOTAL_STEPS}  ·  User PIN")
        hr()
        print(f"  Setting a 4-digit PIN for '{display_id}'.")
        print()

        pin = prompt("4-digit PIN", password=True)
        if not re.fullmatch(r"\d{4}", pin):
            print("\n  ✘  PIN must be exactly 4 digits")
            wait()
            continue

        confirm = prompt("Confirm PIN", password=True)
        if pin != confirm:
            print("\n  ✘  PINs do not match — try again")
            wait()
            continue
        break

    # Choose age band with a nice menu
    age_options = [
        ("adult",       "Adult"),
        ("child_young", "Child (6-10)"),
        ("child_teen",  "Teen (11-16)"),
        ("senior",      "Senior"),
    ]
    while True:
        clear()
        draw_header()
        print(f"  Step 5 of {TOTAL_STEPS}  ·  Age Band")
        hr()
        print(f"  Choose an age band for '{display_id}'.")
        print("  This picks an age-appropriate system prompt style. You can change")
        print("  it later via the Users page.")
        print()
        for i, (_, label) in enumerate(age_options, 1):
            print(f"    {i}) {label}")
        print()
        pick = prompt("Enter a number (1-4)", default="1").strip()
        if not pick.isdigit() or not (1 <= int(pick) <= len(age_options)):
            print(f"\n  ✘  Enter a number between 1 and {len(age_options)}")
            wait()
            continue
        age_band, _ = age_options[int(pick) - 1]
        break

    config["_first_user"] = {
        "display_id": display_id,
        "pin":        pin,
        "age_band":   age_band,
    }


def _db_has_users() -> bool:
    """
    True if the SQLite DB exists AND already has at least one row in `users`.
    Used to skip first-user creation on re-runs of the wizard.
    """
    if not DB_PATH.exists():
        return False
    try:
        with sqlite3.connect(DB_PATH) as conn:
            # Users table might not exist yet on a partially-initialised DB
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='users'"
            ).fetchone()
            if not row:
                return False
            return conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] > 0
    except sqlite3.DatabaseError:
        return False


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 3 — CONFIRM
# ══════════════════════════════════════════════════════════════════════════════
def screen_confirm(config: dict) -> bool:
    clear()
    draw_header()
    print(f"  Step 6 of {TOTAL_STEPS}  ·  Review & Confirm")
    hr()
    print("  Everything below will be written to disk.\n")

    print("  .env settings")
    rows = [
        ("MODEL",          config.get("MODEL", "")),
        ("CTX_SIZE",       config.get("CTX_SIZE", "")),
        ("PORT",           config.get("PORT", "")),
        ("MYAIDB",         str(DB_PATH)),
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

    first_user = config.get("_first_user")
    print()
    if first_user:
        print("  First user")
        print(f"    Display name:   {first_user['display_id']}")
        print(f"    PIN:            (set ✔)")
        print(f"    Age band:       {first_user['age_band']}")
    else:
        print("  First user")
        print("    (skipped — existing users already in the database)")

    print()
    print("  Files that will be created / updated:")
    print("    · .env")
    print("    · instance/myai.db       (schema created via core.db.init_db — safe to re-run)")

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
        f"FILE_OUTPUT_DIR={GEN_DIR}",
        f"SECRET_KEY={secret_key}",
        f"WARMUP=1",
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

    # 3 — SQLite schema
    # We delegate to core.db.init_db() so the schema, migrations, and any
    # future changes live in exactly one place. MYAIDB must point at the
    # right location — set it explicitly before importing.
    os.environ["MYAIDB"] = str(DB_PATH)

    # Ensure the project root is importable even when the wizard is run
    # from a subdirectory or via an absolute path.
    if str(CURPATH) not in sys.path:
        sys.path.insert(0, str(CURPATH))

    try:
        from core.db import init_db, create_user, list_users
    except ImportError as e:
        print(f"  ✘  Could not import core.db: {e}")
        print("     Make sure you're running this from the MyAI project root.")
        sys.exit(1)

    init_db()
    print(f"  ✔  instance/myai.db (schema ready)")

    # 4 — First user (only if the wizard collected one)
    first_user = config.get("_first_user")
    if first_user:
        # Guard: db.init_db() may have created a bootstrap "Setup" user on a
        # totally fresh DB. If we now go and create ANOTHER user, we'd end up
        # with two on install. Remove the bootstrap if it exists AND has the
        # default PIN AND we're about to create a real user.
        existing = list_users(include_inactive=True)
        bootstrap = next(
            (u for u in existing if u["display_id"] == "Setup" and u["has_pin"]),
            None,
        )
        if bootstrap:
            # Directly remove — no memories or conversations exist yet
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute("DELETE FROM users WHERE user_id = ?", (bootstrap["user_id"],))

        try:
            create_user(
                first_user["display_id"],
                first_user["pin"],
                first_user["age_band"],
            )
            print(f"  ✔  Created user '{first_user['display_id']}' ({first_user['age_band']})")
        except ValueError as e:
            print(f"  ✘  Could not create user: {e}")
            print("     You can create users via the Users page after signing in.")
    else:
        # Existing DB with existing users — leave them alone
        print(f"  ·  Users left unchanged")

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
    screen_first_user(config)

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