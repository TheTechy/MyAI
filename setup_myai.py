#!/usr/bin/env python3
"""
setup_myai.py
=============
Interactive setup wizard for MyAI.
Run once after cloning the repository.

    python3 setup_myai.py
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

# ── Runtime constants ──────────────────────────────────────────────────────────
CURPATH = os.getcwd()
MODELS_DIR = os.path.join(CURPATH, "models")
DB_PATH = os.path.join(DB_DIR, "myai.db")
GEN_DIR = os.path.join(CURPATH, "generated_files")


# ── Terminal helpers ───────────────────────────────────────────────────────────
def clear():
    os.system("cls" if os.name == "nt" else "clear")

def draw_header(title: str = "MyAI — Setup Wizard"):
    print("┌─────────────────────────────────────────────┐")
    print(f"│  {title:<43}│")
    print("└─────────────────────────────────────────────┘")

def draw_footer(hint: str = ""):
    print("─────────────────────────────────────────────────")
    if hint:
        print(f"  {hint}")

def draw_step(step: int, total: int, title: str, lines: list[str]):
    clear()
    draw_header()
    print(f"  Step {step} of {total}  ·  {title}")
    print("─────────────────────────────────────────────────")
    for line in lines:
        print(f"  {line}")
    print("─────────────────────────────────────────────────")

def draw_menu(title: str, options: list[str], body_lines: list[str] | None = None):
    clear()
    draw_header()
    print(f"  {title}")
    print("─────────────────────────────────────────────────")
    if body_lines:
        for line in body_lines:
            print(f"  {line}")
        print("─────────────────────────────────────────────────")
    for i, opt in enumerate(options, 1):
        print(f"    [{i}]  {opt}")
    print("─────────────────────────────────────────────────")

def draw_result(lines: list[str], success: bool = True):
    icon = "✔" if success else "✘"
    for line in lines:
        print(f"  {icon}  {line}")
    print()

def prompt(label: str, default: str = "", password: bool = False) -> str:
    hint = f" [{default}]" if default else ""
    full = f"  {label}{hint}: "
    try:
        val = getpass.getpass(full) if password else input(full).strip()
    except (KeyboardInterrupt, EOFError):
        clear()
        print("  Setup cancelled.")
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


# ── Dependency helpers ─────────────────────────────────────────────────────────
def is_installed(package: str) -> bool:
    return importlib.util.find_spec(package) is not None

def pip_install(packages: list[str], extra_env: dict | None = None) -> bool:
    env = {**os.environ, **(extra_env or {})}
    cmd = [sys.executable, "-m", "pip", "install", "--quiet"] + packages
    return subprocess.run(cmd, env=env).returncode == 0

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


# ══════════════════════════════════════════════════════════════════════════════
#  Screen 0 — Welcome
# ══════════════════════════════════════════════════════════════════════════════
def screen_welcome():
    clear()
    print()
    print("  ┌───────────────────────────────────────────┐")
    print("  │                                           │")
    print("  │           Welcome to  MyAI  ✦             │")
    print("  │       Local LLM Chat — Setup Wizard       │")
    print("  │                                           │")
    print("  └───────────────────────────────────────────┘")
    print()
    print("  This wizard will:")
    print("    [1]  Check your Python version")
    print("    [2]  Install llama-cpp-python (LLM engine)")
    print("    [3]  Install required packages")
    print("    [4]  Configure your .env file")
    print("    [5]  Create directories and database")
    print("    [6]  Set up user accounts and PINs")
    print()
    print("─────────────────────────────────────────────────")
    val = prompt("Ready to begin? [Y/n]", default="y")
    if val.lower() not in ("y", "yes", ""):
        print("  Setup cancelled.")
        sys.exit(0)


# ══════════════════════════════════════════════════════════════════════════════
#  Screen 1 — Python version
# ══════════════════════════════════════════════════════════════════════════════
def screen_python():
    major, minor = sys.version_info[:2]
    ver = f"{major}.{minor}"

    draw_step(1, 7, "Python Version Check", [
        f"Detected Python {ver}",
        f"Required: Python 3.11 or higher",
    ])

    if (major, minor) < (3, 11):
        print(f"  ✘  Python {ver} is too old.")
        print("     Please upgrade to Python 3.11+ and re-run this script.")
        sys.exit(1)

    print(f"  ✔  Python {ver} — OK")
    wait()


# ══════════════════════════════════════════════════════════════════════════════
#  Screen 2 — llama-cpp-python
# ══════════════════════════════════════════════════════════════════════════════
def _get_llama_version() -> str | None:
    """Return the installed llama-cpp-python version string, or None."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", "llama-cpp-python"],
            capture_output=True, text=True
        )
        for line in result.stdout.splitlines():
            if line.lower().startswith("version:"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return None


def screen_llama():
    platform_label, cmake_args = detect_gpu()
    version = _get_llama_version()
    installed = version is not None

    clear()
    draw_header()
    print("  Step 2 of 7  ·  LLM Engine — llama-cpp-python")
    print("─────────────────────────────────────────────────")
    print(f"  Platform detected : {platform_label}")
    print()

    if installed:
        print(f"  ✔  llama-cpp-python is installed")
        print(f"     Version : {version}")
    else:
        print(f"  ✘  llama-cpp-python is NOT installed")

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
            _show_manual_install(cmake_args)
            return
        # choice == "2" falls through to install
    else:
        options = [
            "Install llama-cpp-python now  (recommended)",
            "Skip — I will install it manually",
        ]
        draw_menu("llama-cpp-python is required to run MyAI.", options)
        choice = choose(options)

        if choice == "2":
            _show_manual_install(cmake_args)
            return
        # choice == "1" falls through to install

    # ── Install / re-install ───────────────────────────────────────────────────
    clear()
    draw_header()
    print(f"  Installing llama-cpp-python for {platform_label}...")
    print("─────────────────────────────────────────────────")
    print("  This may take several minutes. Please wait.")
    print()

    env = {}
    if cmake_args:
        env["CMAKE_ARGS"] = cmake_args

    cmd = [
        sys.executable, "-m", "pip", "install",
        "llama-cpp-python", "--force-reinstall", "--no-cache-dir",
    ]
    result = subprocess.run(cmd, env={**os.environ, **env})
    print()
    if result.returncode == 0:
        new_version = _get_llama_version() or "unknown"
        print(f"  ✔  llama-cpp-python {new_version} installed successfully")
    else:
        print("  ✘  Installation failed.")
        print("     Please install manually — see README for commands.")
    wait()


def _show_manual_install(cmake_args: str):
    print()
    print("  Skipped. To install manually:")
    if cmake_args:
        print(f'    CMAKE_ARGS="{cmake_args}" pip install llama-cpp-python --force-reinstall --no-cache-dir')
    else:
        print("    pip install llama-cpp-python")
    wait()
    return


# ══════════════════════════════════════════════════════════════════════════════
#  Screen 3 — Packages
# ══════════════════════════════════════════════════════════════════════════════
CORE_PACKAGES = ["flask", "waitress", "python-dotenv", "werkzeug"]

OPTIONAL_PACKAGES = {
    "pypdf":              ("pypdf",               "PDF ingestion"),
    "docx":               ("python-docx",         "Word doc ingestion + generation"),
    "openpyxl":           ("openpyxl",            "Excel ingestion + generation"),
    "duckduckgo_search":  ("duckduckgo-search",   "Web search"),
}

def _pkg_version(pip_name: str) -> str | None:
    """Return installed version of a pip package, or None."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", pip_name],
            capture_output=True, text=True
        )
        for line in result.stdout.splitlines():
            if line.lower().startswith("version:"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return None


def screen_packages():
    clear()
    draw_header()
    print("  Step 3 of 7  ·  Package Installation")
    print("─────────────────────────────────────────────────")

    # ── Core packages ──────────────────────────────────────────────────────────
    print("  Core packages  (required)")
    print()
    core_missing = []
    for pip_name in CORE_PACKAGES:
        ver = _pkg_version(pip_name)
        if ver:
            print(f"    ✔  {pip_name:<22} {ver}")
        else:
            print(f"    ✘  {pip_name:<22} not installed")
            core_missing.append(pip_name)

    # ── Optional packages ──────────────────────────────────────────────────────
    print()
    print("  Optional packages")
    print()
    opt_missing = []
    for import_name, (pip_name, desc) in OPTIONAL_PACKAGES.items():
        ver = _pkg_version(pip_name)
        if ver:
            print(f"    ✔  {pip_name:<22} {ver:<12}  {desc}")
        else:
            print(f"    ✘  {pip_name:<22} {'not installed':<12}  {desc}")
            opt_missing.append(pip_name)

    # ── Summary ────────────────────────────────────────────────────────────────
    print()
    print("─────────────────────────────────────────────────")
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
    draw_menu("Which packages should be installed?", options)
    choice = choose(options)

    if choice == "3":
        print("\n  Skipped.")
        wait()
        return

    packages = list(CORE_PACKAGES)
    if choice == "1":
        packages += [v[0] for v in OPTIONAL_PACKAGES.values()]

    # Only install what is actually missing
    to_install = [p for p in packages if not _pkg_version(p)]

    if not to_install:
        print("\n  ✔  All selected packages are already installed — nothing to do.")
        wait()
        return

    clear()
    draw_header()
    print(f"  Installing {len(to_install)} package(s)...")
    print("─────────────────────────────────────────────────")
    for p in to_install:
        print(f"    · {p}")
    print()

    if pip_install(to_install):
        print(f"\n  ✔  {len(to_install)} package(s) installed successfully")
    else:
        print("\n  ✘  Some packages failed — check output above")
    wait()


# ══════════════════════════════════════════════════════════════════════════════
#  Screen 4 — .env configuration
# ══════════════════════════════════════════════════════════════════════════════
def _validate_model(path: str) -> str | None:
    if not path:
        return "Path cannot be empty"
    p = Path(path)
    if not p.exists():
        return f"File not found: {path}"
    if p.suffix.lower() != ".gguf":
        return "Expected a .gguf model file"
    return None

def _validate_port(val: str) -> str | None:
    if not val.isdigit():
        return "Port must be a number"
    if not (1024 <= int(val) <= 65535):
        return "Port must be between 1024 and 65535"
    return None

def _validate_ctx_size(val: str) -> str | None:
    if not val.isdigit():
        return "Context size must be a number"
    if int(val) < 2048:
        return "Context size should be at least 2048"
    return None

def screen_env() -> dict:
    draw_step(4, 7, "Environment Configuration", [
        "Configure your .env file.",
        "These values will be saved and used when MyAI starts.",
        "",
        f"DB path        : {DB_PATH}  (auto)",
        f"Generated files: {GEN_DIR}  (auto)",
        f"Secret key     : auto-generated",
    ])
    wait("Press Enter to start configuration...")

    config = {
        "MYAIDB":          DB_PATH,
        "FILE_OUTPUT_DIR": GEN_DIR,
        "SECRET_KEY":      secrets.token_hex(32),
    }

    # MODEL
    while True:
        clear()
        draw_header()
        print("  Step 4 of 7  ·  Model Path")
        print("─────────────────────────────────────────────────")
        print("  Enter the full path to your .gguf model file.")
        print("  Example: ./models/gemma-4-it.gguf")
        print()
        val = prompt("Model path")
        err = _validate_model(val)
        if err:
            print(f"\n  ✘  {err}")
            wait()
        else:
            config["MODEL"] = val
            break

    # CTX_SIZE
    while True:
        clear()
        draw_header()
        print("  Step 4 of 7  ·  Context Window Size")
        print("─────────────────────────────────────────────────")
        print("  The context window size in tokens.")
        print("  Check your model card for the supported maximum.")
        print()
        print("  Common values:")
        print("    8192   — conservative, works on most hardware")
        print("    12288  — recommended for most models")
        print("    32768  — large context, needs more VRAM/RAM")
        print()
        val = prompt("Context size", default="12288")
        err = _validate_ctx_size(val)
        if err:
            print(f"\n  ✘  {err}")
            wait()
        else:
            config["CTX_SIZE"] = val
            break

    # PORT
    while True:
        clear()
        draw_header()
        print("  Step 4 of 7  ·  Server Port")
        print("─────────────────────────────────────────────────")
        val = prompt("Port number", default="8080")
        err = _validate_port(val)
        if err:
            print(f"\n  ✘  {err}")
            wait()
        else:
            config["PORT"] = val
            break

    # USERS
    while True:
        clear()
        draw_header()
        print("  Step 4 of 7  ·  Users")
        print("─────────────────────────────────────────────────")
        print("  Enter usernames separated by commas.")
        print("  Example: alice,bob,carol")
        print()
        raw   = prompt("Usernames", default="user1")
        users = [u.strip() for u in raw.split(",") if u.strip()]
        bad   = [u for u in users if " " in u or not u.isalnum()]
        if not users:
            print("\n  ✘  Enter at least one username")
            wait()
        elif bad:
            print(f"\n  ✘  Invalid usernames (letters and numbers only): {', '.join(bad)}")
            wait()
        else:
            config["USERS"] = ",".join(users)
            break

    # OWM (optional)
    clear()
    draw_header()
    print("  Step 4 of 7  ·  Weather Skill (optional)")
    print("─────────────────────────────────────────────────")
    print("  The weather skill uses OpenWeatherMap (free tier).")
    print("  Get a free API key at: https://openweathermap.org/api")
    print("  Press Enter to skip — weather skill will be disabled.")
    print()
    owm_key = prompt("OpenWeatherMap API key", default="")
    if owm_key:
        config["OWM_API_KEY"] = owm_key

        # OWM_UNITS
        options = ["metric  (°C, m/s)", "imperial  (°F, mph)", "standard  (K, m/s)"]
        draw_menu("Temperature units:", options)
        units_choice = choose(options)
        config["OWM_UNITS"] = ["metric", "imperial", "standard"][int(units_choice) - 1]

        # OWM_HOME_CITY
        clear()
        draw_header()
        print("  Step 4 of 7  ·  Home Location (optional)")
        print("─────────────────────────────────────────────────")
        print("  Set a default city for queries like 'Will it rain today?'")
        print("  Format: City, CC  (e.g. Leeds, GB  or  New York, US)")
        print("  Press Enter to skip.")
        print()
        home_city = prompt("Home city", default="")
        if home_city:
            # Ensure space after comma
            if "," in home_city and ", " not in home_city:
                home_city = home_city.replace(",", ", ")
            config["OWM_HOME_CITY"] = home_city
    else:
        config["OWM_API_KEY"]  = ""
        config["OWM_UNITS"]    = "metric"
        config["OWM_HOME_CITY"] = ""

    # SEARCH_REGION
    clear()
    draw_header()
    print("  Step 4 of 7  ·  Web Search Region")
    print("─────────────────────────────────────────────────")
    print("  Sets the regional focus for DuckDuckGo searches.")
    print()
    options = [
        "wt-wt   — worldwide (default)",
        "uk-en   — United Kingdom",
        "us-en   — United States",
        "de-de   — Germany",
        "fr-fr   — France",
        "jp-jp   — Japan",
        "cn-zh   — China",
    ]
    draw_menu("Select your search region:", options)
    region_codes = ["wt-wt", "uk-en", "us-en", "de-de", "fr-fr", "jp-jp", "cn-zh"]
    config["SEARCH_REGION"] = region_codes[int(choose(options)) - 1]

    return config


def write_env(config: dict):
    env_path = Path(".env")
    exists   = env_path.exists()

    if exists:
        clear()
        draw_header()
        print("  Step 4 of 7  ·  Write .env")
        print("─────────────────────────────────────────────────")
        print("  A .env file already exists.")
        options = ["Overwrite it with new configuration", "Keep existing .env file"]
        draw_menu("What would you like to do?", options)
        if choose(options) == "2":
            print("\n  Kept existing .env")
            wait()
            return

    lines = [
        "# MyAI configuration — generated by setup_myai.py",
        "# Do NOT commit this file to version control.",
        "",
        f"MODEL={config['MODEL']}",
        f"CTX_SIZE={config['CTX_SIZE']}",
        f"PORT={config['PORT']}",
        f"MYAIDB={config['MYAIDB']}",
        f"USERS={config['USERS']}",
        f"FILE_OUTPUT_DIR={config['FILE_OUTPUT_DIR']}",
        f"SECRET_KEY={config['SECRET_KEY']}",
        "",
        "# Weather skill (optional)",
        f"OWM_API_KEY={config.get('OWM_API_KEY', '')}",
        f"OWM_UNITS={config.get('OWM_UNITS', 'metric')}",
        f"OWM_HOME_CITY={config.get('OWM_HOME_CITY', '')}",
        "",
        "# Web search",
        f"SEARCH_REGION={config.get('SEARCH_REGION', 'wt-wt')}",
    ]
    env_path.write_text("\n".join(lines) + "\n")
    print(f"\n  ✔  .env written")
    wait()


# ══════════════════════════════════════════════════════════════════════════════
#  Screen 5 — Directories + database
# ══════════════════════════════════════════════════════════════════════════════
def screen_dirs_and_db():
    draw_step(5, 7, "Directories + Database", [
        "Creating required directories and initialising database.",
        "",
        f"  DB/              → {DB_DIR}",
        f"  generated_files/ → {GEN_DIR}",
        f"  models/          → {MODELS_DIR}",
    ])

    # Directories
    for d in [DB_DIR, GEN_DIR, MODELS_DIR]:
        Path(d).mkdir(parents=True, exist_ok=True)
        print(f"  ✔  Created {d}")

    # Add .gitkeep files so git tracks the empty folders
    for folder, label in [(MODELS_DIR, "models/.gitkeep"), (DB_DIR, "DB/.gitkeep")]:
        gitkeep = Path(folder) / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()
            print(f"  ✔  Added {label}")

    # SQLite schema
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
        CREATE INDEX IF NOT EXISTS idx_conv_user ON conversations(user_id, updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_msg_conv  ON messages(conv_id, created_at ASC);
        CREATE INDEX IF NOT EXISTS idx_file_conv ON file_context(conv_id, created_at ASC);
    """)
    conn.commit()
    print(f"  ✔  Database initialised at {DB_PATH}")
    wait()
    return conn


# ══════════════════════════════════════════════════════════════════════════════
#  Screen 6 — User PINs
# ══════════════════════════════════════════════════════════════════════════════
def screen_users(config: dict, conn: sqlite3.Connection):
    users = [u.strip() for u in config["USERS"].split(",") if u.strip()]
    now   = datetime.now(timezone.utc).isoformat()

    for i, username in enumerate(users, 1):
        while True:
            clear()
            draw_header()
            print(f"  Step 6 of 7  ·  User Setup  ({i}/{len(users)})")
            print("─────────────────────────────────────────────────")
            print(f"  Setting up user: {username}")
            print()

            existing = conn.execute(
                "SELECT pin_hash FROM users WHERE user_id=?", (username,)
            ).fetchone()

            if existing and existing[0]:
                print(f"  User '{username}' already has a PIN.")
                options = ["Keep existing PIN", "Set a new PIN"]
                draw_menu("What would you like to do?", options)
                if choose(options) == "1":
                    print(f"\n  ✔  Kept existing PIN for {username}")
                    wait()
                    break

            pin = prompt(f"  4-digit PIN for '{username}'", password=True)
            if not re.fullmatch(r"\d{4}", pin):
                print("\n  ✘  PIN must be exactly 4 digits")
                wait()
                continue

            confirm = prompt(f"  Confirm PIN for '{username}'", password=True)
            if pin != confirm:
                print("\n  ✘  PINs do not match — try again")
                wait()
                continue

            conn.execute(
                "INSERT OR IGNORE INTO users(user_id, pin_hash, created_at) VALUES (?,?,?)",
                (username, pin, now),
            )
            conn.execute(
                "UPDATE users SET pin_hash=? WHERE user_id=?",
                (pin, username),
            )
            conn.commit()
            print(f"\n  ✔  {username} configured")
            wait()
            break

    conn.close()


# ══════════════════════════════════════════════════════════════════════════════
#  Screen 6b — Per-user prompt generator
# ══════════════════════════════════════════════════════════════════════════════

_PROMPT_TEMPLATES = {
    # (age_group, use_case) → system prompt
    ("child_young", "general"): """You are a friendly and patient assistant for a young child aged 6-10.
Use very simple words and short sentences. Be encouraging and kind.
Avoid any adult themes, violence, or complex concepts.
When explaining things, use fun examples like toys, animals, or games.
Always be positive and supportive.""",

    ("child_young", "learning"): """You are a fun and patient learning companion for a child aged 6-10.
Make learning exciting! Use simple language, fun examples and lots of encouragement.
Break things down into tiny steps. Celebrate every small success.
Avoid adult themes. Focus on curiosity, discovery and confidence.""",

    ("child_teen", "general"): """You are a knowledgeable and friendly assistant for a teenager aged 11-16.
Be engaging and clear without being condescending. Use everyday language.
Avoid adult content. Encourage curiosity and independent thinking.
For homework or study topics, guide them to the answer rather than just giving it.""",

    ("child_teen", "learning"): """You are a helpful study companion for a teenager aged 11-16.
Be encouraging and clear. For questions, work through the problem step by step
so they understand the method, not just the answer.
Use relatable examples. Avoid jargon. Keep it engaging.""",

    ("adult", "general"): """You are a knowledgeable and concise assistant.
Answer directly and clearly. Match the depth of your response to the question.
For factual questions, lead with the answer then explain.
Use markdown only when it genuinely aids clarity.""",

    ("adult", "coding"): """You are an expert software engineer and concise technical assistant.
Produce production-grade, idiomatic code with appropriate error handling.
Lead with the solution. Add brief inline comments for complex logic only.
Use fenced code blocks with the correct language identifier.""",

    ("adult", "creative"): """You are a creative writing assistant with a strong narrative voice.
Help with story ideas, character development, dialogue and prose style.
Be encouraging and constructive. Match the tone the user is going for.
Offer alternatives and suggestions rather than prescriptive rules.""",

    ("senior", "general"): """You are a patient, clear and helpful assistant.
Use plain language and avoid jargon. Keep explanations simple and direct.
If something is complex, break it into small clear steps.
Never rush. Offer to explain things again in a different way if needed.
Be warm, respectful and encouraging.""",
}

def _generate_prompt(username: str) -> str:
    """Interactive per-user prompt generator. Returns a system prompt string."""
    clear()
    draw_header()
    print(f"  Step 6 of 7  ·  User Prompt — {username}")
    print("─────────────────────────────────────────────────")
    print(f"  Generating a personalised system prompt for {username}.")
    print()

    # Age group
    age_options = [
        "Child  (6-10 years)",
        "Child / Teen  (11-16 years)",
        "Adult",
        "Senior  (65+)",
    ]
    draw_menu("What is this user's age group?", age_options)
    age_choice = choose(age_options)
    age_map = {"1": "child_young", "2": "child_teen", "3": "adult", "4": "senior"}
    age_group = age_map[age_choice]

    # Use case
    if age_group == "adult":
        use_options = [
            "General chat and questions",
            "Coding and technical help",
            "Creative writing",
        ]
        use_map = {"1": "general", "2": "coding", "3": "creative"}
    else:
        use_options = [
            "General chat and questions",
            "Learning and study",
        ]
        use_map = {"1": "general", "2": "learning"}

    clear()
    draw_header()
    print(f"  Step 6 of 7  ·  User Prompt — {username}")
    print("─────────────────────────────────────────────────")
    draw_menu("What is this user's primary use?", use_options)
    use_choice = choose(use_options)
    use_case = use_map[use_choice]

    # Get template
    template = _PROMPT_TEMPLATES.get(
        (age_group, use_case),
        _PROMPT_TEMPLATES[("adult", "general")]
    )

    # Show preview
    clear()
    draw_header()
    print(f"  Step 6 of 7  ·  User Prompt — {username}")
    print("─────────────────────────────────────────────────")
    print(f"  Generated prompt for {username}:")
    print()
    for line in template.strip().splitlines():
        print(f"  {line}")
    print()
    print("─────────────────────────────────────────────────")

    options = ["Use this prompt", "Use the default prompt instead"]
    draw_menu("", options)
    if choose(options) == "2":
        return ""  # empty = use default

    return template


def screen_prompts(config: dict):
    """Generate per-user system prompts and write them into prompts.py."""
    users = [u.strip() for u in config["USERS"].split(",") if u.strip()]

    # Check if prompts.py exists
    prompts_path = Path("prompts.py")
    if not prompts_path.exists():
        return  # nothing to do

    clear()
    draw_header()
    print("  Step 6 of 7  ·  User System Prompts")
    print("─────────────────────────────────────────────────")
    print("  MyAI supports personalised prompts per user.")
    print("  This step will help generate a prompt for each user.")
    print()
    options = [
        "Generate prompts for each user  (recommended)",
        "Skip — I will edit prompts.py manually",
    ]
    draw_menu("", options)
    if choose(options) == "2":
        print("\n  Skipped — edit prompts.py to customise user prompts.")
        wait()
        return

    user_prompts = {}
    for username in users:
        generated = _generate_prompt(username)
        if generated:
            user_prompts[username] = generated
        print(f"\n  ✔  Prompt set for {username}")
        wait("Press Enter for next user...")

    if not user_prompts:
        return

    # Write into prompts.py USER_PROMPTS dict
    with open(prompts_path, 'r') as f:
        prompts_src = f.read()

    entries = []
    for username, prompt_text in user_prompts.items():
        escaped = prompt_text.replace('\\', '\\\\').replace('"', '\\"')
        lines   = escaped.splitlines()
        body    = '\n'.join(f'            {l}' for l in lines)
        entries.append(f'    "{username}": textwrap.dedent("""\n{body}\n    """),' )

    if 'USER_PROMPTS: dict[str, str] = {' in prompts_src:
        new_block = 'USER_PROMPTS: dict[str, str] = {\n' + '\n'.join(entries) + '\n}'
        import re as _re
        prompts_src = _re.sub(
            r'USER_PROMPTS: dict\[str, str\] = \{.*?\}',
            new_block,
            prompts_src,
            flags=_re.DOTALL
        )
        with open(prompts_path, 'w') as f:
            f.write(prompts_src)
        print("\n  ✔  prompts.py updated with user prompts")
    else:
        print("\n  ⚠  Could not find USER_PROMPTS in prompts.py — edit manually")
    wait()


# ══════════════════════════════════════════════════════════════════════════════
#  Screen 7 — Summary
# ══════════════════════════════════════════════════════════════════════════════
def screen_summary(config: dict):
    clear()
    print()
    print("  ┌───────────────────────────────────────────┐")
    print("  │           Setup Complete  ✔               │")
    print("  └───────────────────────────────────────────┘")
    print()
    print("  Configuration")
    print("  ─────────────────────────────────────────────")
    safe = {**config, "SECRET_KEY": config["SECRET_KEY"][:8] + "..."}
    for k, v in safe.items():
        print(f"    {k:<20} {v}")

    print()
    print("  Optional packages")
    print("  ─────────────────────────────────────────────")
    for import_name, (pip_name, desc) in OPTIONAL_PACKAGES.items():
        status = "✔  installed" if is_installed(import_name) else "✘  not installed"
        print(f"    {status:<18}  {pip_name}")

    print()
    print("  ┌───────────────────────────────────────────┐")
    print("  │  Start MyAI:                              │")
    print("  │                                           │")
    print(f"  │    python3 app.py                         │")
    print("  │                                           │")
    print(f"  │  Then open:  http://localhost:{config['PORT']:<14} │")
    print("  │                                           │")
    print("  │  To add more users later, re-run:         │")
    print("  │    python3 setup_myai.py                  │")
    print("  └───────────────────────────────────────────┘")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    screen_welcome()
    screen_python()
    screen_llama()
    screen_packages()
    config = screen_env()
    write_env(config)
    conn   = screen_dirs_and_db()
    screen_users(config, conn)
    screen_prompts(config)
    screen_summary(config)


if __name__ == "__main__":
    main()