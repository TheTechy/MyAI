"""
installer/system.py — what the setup app knows and does, without any screens.

Facts about the machine (build tools, memory, installed versions), the .env
file, the pinned engine build and the database. Kept apart from the screens so
it can be tested headlessly. Standard library only: this runs before MyAI's own
requirements are installed.
"""

from __future__ import annotations

import json
import os
import platform
import re
import secrets
import sqlite3
import subprocess
import sys
import tempfile
from importlib import metadata
from pathlib import Path
from typing import Mapping, Sequence

from core.db import BOOTSTRAP_DISPLAY_ID as BOOTSTRAP_USER   # core.db is stdlib-only

# ── Pins and choices (the single place to change them) ────────────────────────
MIN_PYTHON = (3, 13)
LLAMA_PACKAGE = "llama-cpp-python"
LLAMA_VERSION = "0.3.35"
# Prebuilt Metal wheels for 0.3.33–0.3.36 are corrupt upstream, so the engine is
# built from source; on Apple Silicon these flags give it GPU (Metal) support.
APPLE_SILICON_CMAKE_ARGS = "-DGGML_METAL=on"

DEFAULT_PORT = "8080"

CTX_CHOICES = [
    ("8192", "Short chats — lightest on memory"),
    ("16384", "Everyday use — a good balance"),
    ("32768", "Long chats and big documents — needs plenty of memory"),
]
AGE_BANDS = [
    ("adult", "Adult"),
    ("child_teen", "Teen (11–16)"),
    ("child_young", "Child (6–10)"),
    ("senior", "Senior"),
]
OWM_UNITS = [("metric", "°C, km/h"), ("imperial", "°F, mph"), ("standard", "Kelvin")]
SEARCH_REGIONS = [
    ("wt-wt", "Worldwide"), ("uk-en", "United Kingdom"), ("us-en", "United States"),
    ("de-de", "Germany"), ("fr-fr", "France"), ("jp-jp", "Japan"),
]
WHISPER_MODELS = [
    ("tiny.en", "Fastest, least accurate (~40 MB)"),
    ("base.en", "Quick and accurate — recommended (~145 MB)"),
    ("small.en", "More accurate, slower (~245 MB)"),
]

GB = 1024 ** 3


# ── The interpreter we're running in ─────────────────────────────────────────

def in_venv() -> bool:
    return sys.prefix != sys.base_prefix


def python_ok(version: tuple = sys.version_info[:2]) -> bool:
    return tuple(version) >= MIN_PYTHON


def installed_version(dist: str) -> str | None:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


# ── The machine ───────────────────────────────────────────────────────────────

def is_apple_silicon(system: str | None = None, machine: str | None = None) -> bool:
    system = system or platform.system()
    machine = (machine or platform.machine()).lower()
    return system == "Darwin" and machine in ("arm64", "aarch64")


def build_env(base: Mapping[str, str], apple_silicon: bool) -> dict[str, str]:
    """Environment for building the engine: GPU flags on Apple Silicon, CPU elsewhere."""
    env = dict(base)
    if apple_silicon:
        env["CMAKE_ARGS"] = APPLE_SILICON_CMAKE_ARGS
    else:
        env.pop("CMAKE_ARGS", None)
    return env


def has_build_tools() -> bool:
    """Apple's Command Line Tools (the C/C++ compiler the engine build needs).
    `clang` alone proves nothing: /usr/bin/clang is a stub that offers to install them."""
    try:
        return subprocess.run(["xcode-select", "-p"], capture_output=True).returncode == 0
    except FileNotFoundError:
        return False


def request_build_tools() -> None:
    """Open Apple's own installer dialog for the Command Line Tools (no sudo needed)."""
    subprocess.run(["xcode-select", "--install"], capture_output=True)


def total_ram_bytes() -> int | None:
    try:
        out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
        return int(out.stdout.strip())
    except (FileNotFoundError, ValueError):
        return None


def gpu_budget(ram: int) -> int:
    """How much memory macOS lets the GPU use by default: about two-thirds on
    smaller Macs, three-quarters above 36 GB."""
    return ram * 3 // 4 if ram > 36 * GB else ram * 2 // 3


def model_fit(model_bytes: int, ram: int | None) -> str | None:
    """'fits', 'tight' or 'too_big' for this machine (None if RAM is unknown).
    The weights must sit inside the GPU budget with room left for the chat itself."""
    if not ram:
        return None
    budget = gpu_budget(ram)
    if model_bytes <= budget * 0.8:
        return "fits"
    if model_bytes <= budget:
        return "tight"
    return "too_big"


def recommended_ctx(ram: int | None) -> str:
    return "32768" if ram and ram >= 32 * GB else "16384"


def gb(n: int) -> str:
    # Binary GB, as macOS reports memory: an "18 GB" Mac is 18 × 1024³ bytes.
    return f"{n // GB} GB" if n % GB == 0 else f"{n / GB:.1f} GB"


# ── .env ──────────────────────────────────────────────────────────────────────

_INLINE_COMMENT = re.compile(r"\s+#.*$")


def read_env(path: Path) -> dict[str, str]:
    """KEY=value pairs from an existing .env (comments and quotes stripped)."""
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if value[:1] in ("'", '"') and value[-1:] == value[:1] and len(value) > 1:
            value = value[1:-1]
        else:
            value = _INLINE_COMMENT.sub("", value)
        env[key.strip()] = value
    return env


def render_env(cfg: Mapping[str, str]) -> str:
    """The .env file for `cfg`. Same layout the app has always used."""
    g = lambda k, d="": cfg.get(k, d)
    return "\n".join([
        "# MyAI configuration — written by the MyAI setup app (./install.sh).",
        "# Do NOT commit this file to version control.",
        "",
        f"MODEL={g('MODEL')}",
        f"CTX_SIZE={g('CTX_SIZE', '16384')}",
        f"PORT={g('PORT', DEFAULT_PORT)}",
        f"MYAIDB={g('MYAIDB')}",
        f"FILE_OUTPUT_DIR={g('FILE_OUTPUT_DIR')}",
        f"SECRET_KEY={g('SECRET_KEY')}",
        f"WARMUP={g('WARMUP', '1')}",
        "",
        "# Weather skill (optional — leave blank to disable)",
        f"OWM_API_KEY={g('OWM_API_KEY')}",
        f"OWM_UNITS={g('OWM_UNITS', 'metric')}",
        f"OWM_HOME_CITY={g('OWM_HOME_CITY')}",
        "",
        "# Web search",
        f"SEARCH_REGION={g('SEARCH_REGION', 'wt-wt')}",
        "",
        "# Voice input — faster-whisper (leave blank to disable)",
        f"WHISPER_MODEL={g('WHISPER_MODEL')}",
        f"WHISPER_DEVICE={g('WHISPER_DEVICE')}",
        f"WHISPER_COMPUTE={g('WHISPER_COMPUTE')}",
        "",
    ])


def write_env(path: Path, cfg: Mapping[str, str]) -> bool:
    """Write .env, keeping any existing SECRET_KEY so logged-in sessions survive
    a re-run. Returns True if a new key was generated."""
    cfg = dict(cfg)
    existing = read_env(path).get("SECRET_KEY", "")
    cfg["SECRET_KEY"] = existing or secrets.token_hex(32)
    path.write_text(render_env(cfg), encoding="utf-8")
    return not existing


# ── Installing ────────────────────────────────────────────────────────────────

def llama_install_cmd(python: str = sys.executable) -> list[str]:
    # --no-cache-dir: never reuse a wheel from an earlier (possibly broken) attempt.
    return [python, "-m", "pip", "install", "--no-cache-dir",
            "--disable-pip-version-check", f"{LLAMA_PACKAGE}=={LLAMA_VERSION}"]


def requirements_without_llama(text: str) -> str:
    """requirements.txt minus any llama-cpp-python line: the engine is pinned and
    built separately, and a second pin there would fight it."""
    keep = [l for l in text.splitlines()
            if not re.match(r"\s*llama[_-]cpp[_-]python\b", l, re.I)]
    return "\n".join(keep) + "\n"


def requirements_file(repo: Path) -> Path:
    """A temp copy of requirements.txt without the engine line. Caller deletes it."""
    text = (repo / "requirements.txt").read_text(encoding="utf-8")
    fd, name = tempfile.mkstemp(prefix="myai-req-", suffix=".txt")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(requirements_without_llama(text))
    return Path(name)


def requirements_cmd(req: Path, python: str = sys.executable) -> list[str]:
    return [python, "-m", "pip", "install", "--disable-pip-version-check", "-r", str(req)]


def engine_check_cmd(python: str = sys.executable) -> list[str]:
    """Imports the engine and reports whether it can use the GPU."""
    return [python, "-c",
            "import llama_cpp; print('gpu' if llama_cpp.llama_supports_gpu_offload() else 'cpu')"]


# ── Database ──────────────────────────────────────────────────────────────────

def has_real_users(db_path: Path) -> bool:
    """True if the DB holds accounts other than core.db's 'Setup' placeholder."""
    if not db_path.exists():
        return False
    try:
        with sqlite3.connect(db_path) as conn:
            if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' "
                                "AND name='users'").fetchone():
                return False
            n = conn.execute("SELECT COUNT(*) FROM users WHERE display_id != ?",
                             (BOOTSTRAP_USER,)).fetchone()[0]
            return n > 0
    except sqlite3.DatabaseError:
        return False


# Database work runs in a child process so core.db reads MYAIDB fresh (it caches
# the path at import). Requests arrive on stdin, keeping the PIN out of `ps`.
_DB_SETUP = """
import json, sys
from core.db import init_db, create_user, remove_untouched_bootstrap
init_db()
req = json.loads(sys.stdin.read())
if req["user"]:
    u = req["user"]
    create_user(u["name"], u["pin"], u["age_band"])   # also retires the placeholder
if req["remove_placeholder"]:
    remove_untouched_bootstrap()
"""
_DB_HAS_PLACEHOLDER = """
from core.db import untouched_bootstrap_id
print("yes" if untouched_bootstrap_id() else "no")
"""


def _db_run(repo: Path, db_path: Path, script: str, stdin: str = "",
            python: str = sys.executable) -> subprocess.CompletedProcess:
    return subprocess.run(
        [python, "-c", script], input=stdin, text=True, capture_output=True,
        cwd=repo, env={**os.environ, "MYAIDB": str(db_path)},
    )


def setup_database(repo: Path, db_path: Path, user: Mapping[str, str] | None,
                   remove_placeholder: bool = False,
                   python: str = sys.executable) -> tuple[bool, str]:
    """Create/upgrade the schema via core.db, add the first user if given and,
    if asked, remove the unused 'Setup' placeholder. Returns (ok, error text)."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    proc = _db_run(repo, db_path, _DB_SETUP, python=python,
                   stdin=json.dumps({"user": user, "remove_placeholder": remove_placeholder}))
    return proc.returncode == 0, (proc.stderr.strip().splitlines() or [""])[-1]


def has_unused_placeholder(repo: Path, db_path: Path, python: str = sys.executable) -> bool:
    """True if an existing DB still has the untouched 'Setup' / 0000 account
    alongside real ones (left over from before the app removed it itself)."""
    if not db_path.exists():
        return False
    proc = _db_run(repo, db_path, _DB_HAS_PLACEHOLDER, python=python)
    # An unreadable or very old DB just means "nothing to offer".
    return proc.returncode == 0 and proc.stdout.strip() == "yes"


# ── Answer checks (return an error message, or None if fine) ─────────────────

def check_name(v: str) -> str | None:
    if not v:
        return "Please enter a name."
    if len(v) > 40:
        return "That's a bit long — 40 characters at most."
    if v == BOOTSTRAP_USER:
        return f"'{BOOTSTRAP_USER}' is reserved — please pick another name."
    return None


def check_pin(v: str) -> str | None:
    return None if re.fullmatch(r"\d{4}", v) else "The PIN must be exactly 4 digits."


def check_port(v: str) -> str | None:
    return None if v.isdigit() and 1024 <= int(v) <= 65535 else \
        "Please enter a number between 1024 and 65535."


def check_model_path(v: str) -> str | None:
    p = Path(v).expanduser()
    if not v:
        return "Please enter the path to a .gguf file."
    if not p.is_file():
        return f"There's no file at {p}."
    if p.suffix.lower() != ".gguf":
        return "That isn't a .gguf model file."
    return None


def choice_index(choices: Sequence[tuple[str, str]], value: str, fallback: int = 0) -> int:
    return next((i for i, (v, _) in enumerate(choices) if v == value), fallback)
