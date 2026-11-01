#!/usr/bin/env python3
"""
setup_myai.py — the MyAI setup app.

Started by ./install.sh, which finds a suitable Python and creates .venv first;
this runs inside that venv. Re-run it any time to change settings.

It asks everything up front, shows a review screen, and only then writes the
settings and does the slow installs. Settings are saved *before* the long engine
build, so if a build fails, re-running ./install.sh starts from the same answers.

The look lives in installer/tui.py; the machine facts and actions live in
installer/system.py. This file is just the screens, in order.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from installer import system as S
from installer.tui import TUI, SHOW_CURSOR

REPO = Path(__file__).resolve().parent
ENV_PATH = REPO / ".env"
MODELS_DIR = REPO / "models"
DB_PATH = REPO / "instance" / "myai.db"
GEN_DIR = REPO / "generated_files"
LOG_PATH = REPO / "install.log"

STEPS = 6       # check · model · extras · account · review · install


def main() -> int:
    if not S.in_venv() or not S.python_ok():
        print("\n  Please start MyAI's setup with ./install.sh — it gets Python ready first.\n")
        return 1

    ui = TUI(total_steps=STEPS)
    existing = S.read_env(ENV_PATH)
    saved = False
    try:
        welcome(ui, existing)
        ram = check_mac(ui)
        cfg: dict[str, str] = {}
        choose_model(ui, cfg, existing, ram)
        choose_extras(ui, cfg, existing)
        user, remove_placeholder = choose_account(ui)
        if not review(ui, cfg, user, remove_placeholder, existing):
            ui.show(["Setup cancelled. Nothing was changed."], wait=False)
            return 0
        saved = True
        return install(ui, cfg, user, remove_placeholder)
    except KeyboardInterrupt:
        msg = ("Setup stopped. Your settings are saved — run ./install.sh again to finish."
               if saved else "Setup cancelled. Nothing was changed.")
        sys.stdout.write(f"{SHOW_CURSOR}\n\n  {msg}\n\n")
        return 130


# ── 0 · Welcome ───────────────────────────────────────────────────────────────

def welcome(ui: TUI, existing: dict) -> None:
    ui.step(0, "Welcome")
    body = [
        f"Welcome to {ui.bold('MyAI')} — a private AI assistant that runs entirely on "
        "this Mac.",
        "Nothing you type ever leaves it.",
        "",
        "This setup will:",
        "  1. Check this Mac is ready",
        "  2. Help you choose an AI model",
        "  3. Set up optional extras (weather, voice)",
        "  4. Create your account",
        "  5. Install everything — about 5–10 minutes",
        "",
        "Nothing is changed until you've reviewed your choices.",
    ]
    if existing:
        body += ["", ui.dim("Found your existing settings — they'll be the starting answers.")]
    ui.show(body)


# ── 1 · Check this Mac ────────────────────────────────────────────────────────

def check_mac(ui: TUI) -> int | None:
    ui.step(1, "Check this Mac")
    ram = S.total_ram_bytes()
    while True:
        v = sys.version_info
        lines = [ui.ok(f"Python {v.major}.{v.minor}.{v.micro}")]
        lines.append(ui.ok("Apple Silicon — MyAI can use the GPU") if S.is_apple_silicon()
                     else ui.warn("Not Apple Silicon — MyAI will run on the CPU (slowly)"))
        if ram:
            lines.append(ui.ok(f"{S.gb(ram)} of memory"))
        if S.has_build_tools():
            lines.append(ui.ok("Apple's developer tools are installed"))
            ui.show(lines)
            return ram

        lines.append(ui.err("Apple's developer tools are missing"))
        choice = ui.menu(
            "MyAI needs Apple's free developer tools to build its AI engine.",
            ["Install Apple's developer tools", "I've installed them — check again",
             "Quit setup"],
            intro=lines)
        if choice == 0:
            S.request_build_tools()
            ui.show(["An Apple window should have opened.", "",
                     "Click Install, accept the licence and wait for it to finish — "
                     "this can take 5–15 minutes.",
                     "", "Then come back here and press Enter."])
        elif choice == 2:
            raise KeyboardInterrupt


# ── 2 · Choose a model ────────────────────────────────────────────────────────

FIT_LABEL = {"fits": ("fits", "green"), "tight": ("tight fit", "yellow"),
             "too_big": ("too big", "red")}


def choose_model(ui: TUI, cfg: dict, existing: dict, ram: int | None) -> None:
    ui.step(2, "Choose a model")
    models = sorted(MODELS_DIR.glob("*.gguf"))
    current = Path(existing.get("MODEL", "")).expanduser()
    if current.is_file() and current.resolve() not in [m.resolve() for m in models]:
        models.append(current)

    fits = [S.model_fit(m.stat().st_size, ram) for m in models]
    labels = []
    for m, fit in zip(models, fits):
        tag = ui.paint(*FIT_LABEL[fit]) if fit else ""
        labels.append(f"{m.name[:36]:<36} {S.gb(m.stat().st_size):>8}  {tag}")
    other, later = len(models), len(models) + 1
    labels += ["Use a model stored somewhere else…", "Skip — I'll add a model later"]

    if models:
        intro = ["MyAI found these models. "
                 + (f"This Mac has {S.gb(ram)} of memory; models marked 'fits' "
                    "will run comfortably." if ram else "")]
    else:
        intro = ["No models found yet. A model is the AI's 'brain' — a single .gguf "
                 "file, usually several GB.",
                 "", "Download one (Gemma 4 12B at Q5 is a good start for most Macs) "
                 "and put it in:", f"  {MODELS_DIR}"]

    if current.is_file():
        default = next(i for i, m in enumerate(models) if m.resolve() == current.resolve())
    else:
        default = fits.index("fits") if "fits" in fits else 0

    while True:
        pick = ui.menu("Which model should MyAI use?", labels, default=default, intro=intro)
        if pick == later:
            cfg["MODEL"] = ""
            break
        if pick == other:
            path = ui.ask("Where is the model file?", default=existing.get("MODEL", ""),
                          intro=["Type or paste the full path to a .gguf file.",
                                 ui.dim("Tip: drag the file onto this window to paste its path.")],
                          validate=lambda v: S.check_model_path(_unquote(v)))
            chosen = Path(_unquote(path)).expanduser()
            fit = S.model_fit(chosen.stat().st_size, ram)
        else:
            chosen, fit = models[pick], fits[pick]
        if fit == "too_big" and not ui.confirm(
                "Use it anyway?", default=False,
                intro=[ui.warn(f"{chosen.name} is bigger than this Mac's GPU can hold."),
                       "It will probably fail to load, or run very slowly."]):
            continue
        cfg["MODEL"] = str(chosen.resolve())
        break

    rec = S.recommended_ctx(ram)
    i = ui.menu("How much should MyAI remember in a conversation?", context_options(ui, rec),
                default=S.choice_index(S.CTX_CHOICES, existing.get("CTX_SIZE", rec),
                                       S.choice_index(S.CTX_CHOICES, rec)),
                intro=["More memory means longer conversations, but uses more of this "
                       "Mac's RAM. You can change this later."])
    cfg["CTX_SIZE"] = S.CTX_CHOICES[i][0]


def context_options(ui: TUI, recommended: str) -> list[str]:
    return [f"{n:>6} tokens — {desc}" + (ui.dim("  (recommended)") if n == recommended else "")
            for n, desc in S.CTX_CHOICES]


def _unquote(v: str) -> str:
    # Dragging a file into Terminal pastes it quoted or with escaped spaces.
    v = v.strip()
    if len(v) > 1 and v[0] == v[-1] and v[0] in "'\"":
        v = v[1:-1]
    return v.replace("\\ ", " ")


# ── 3 · Optional extras ───────────────────────────────────────────────────────

def choose_extras(ui: TUI, cfg: dict, existing: dict) -> None:
    ui.step(3, "Optional extras")
    e = existing.get

    if ui.confirm("Would you like weather forecasts?", default=bool(e("OWM_API_KEY")),
                  intro=["MyAI can answer questions like 'Will it rain today?' using "
                         "OpenWeatherMap.",
                         "It needs a free API key from openweathermap.org/api — you "
                         "can add one later by re-running setup."]):
        cfg["OWM_API_KEY"] = ui.ask("Your OpenWeatherMap API key", default=e("OWM_API_KEY", ""),
                                    validate=lambda v: None if v else "Please paste your key.")
        i = ui.menu("Which units?", [f"{u:<9} {d}" for u, d in S.OWM_UNITS],
                    default=S.choice_index(S.OWM_UNITS, e("OWM_UNITS", "metric")))
        cfg["OWM_UNITS"] = S.OWM_UNITS[i][0]
        city = ui.ask("Your home town (optional)", default=e("OWM_HOME_CITY", ""),
                      intro=["Used when you ask about the weather without saying where.",
                             "Format: City, country code — e.g. Leeds, GB or Austin, US"])
        cfg["OWM_HOME_CITY"] = ", ".join(p.strip() for p in city.split(",")) if city else ""
    else:
        cfg.update(OWM_API_KEY="", OWM_UNITS=e("OWM_UNITS", "metric"), OWM_HOME_CITY="")

    i = ui.menu("Where should web searches be focused?",
                [label for _, label in S.SEARCH_REGIONS],
                default=S.choice_index(S.SEARCH_REGIONS, e("SEARCH_REGION", "wt-wt")))
    cfg["SEARCH_REGION"] = S.SEARCH_REGIONS[i][0]

    if ui.confirm("Would you like to talk to MyAI using your microphone?",
                  default=bool(e("WHISPER_MODEL", "base.en")),
                  intro=["Speech is turned into text on this Mac — nothing is sent anywhere."]):
        i = ui.menu("Which speech model?", [f"{m:<9} {d}" for m, d in S.WHISPER_MODELS],
                    default=S.choice_index(S.WHISPER_MODELS, e("WHISPER_MODEL", "base.en"), 1))
        # faster-whisper has no Metal support; on a Mac, CPU + int8 is the combination that works.
        cfg.update(WHISPER_MODEL=S.WHISPER_MODELS[i][0], WHISPER_DEVICE="cpu",
                   WHISPER_COMPUTE="int8")
    else:
        cfg.update(WHISPER_MODEL="", WHISPER_DEVICE="", WHISPER_COMPUTE="")


# ── 4 · Your account ──────────────────────────────────────────────────────────

def choose_account(ui: TUI) -> tuple[dict | None, bool]:
    """Returns (first user to create or None, remove the unused 'Setup' account?)."""
    ui.step(4, "Your account")
    if S.has_real_users(DB_PATH):
        intro = ["MyAI already has accounts set up.",
                 "You can add or change accounts on the Users page once you've signed in."]
        if S.has_unused_placeholder(REPO, DB_PATH):
            remove = ui.confirm(
                "Remove the unused 'Setup' account (PIN 0000)?", default=True,
                intro=intro + ["", ui.warn("There's also a temporary 'Setup' account left "
                                           "over from when MyAI was first set up."),
                               "It isn't needed any more, and anyone who knows its PIN "
                               "(0000) can sign in with it."])
            return None, remove
        ui.show(intro)
        return None, False

    name = ui.ask("What's your name?", validate=S.check_name,
                  intro=["This is the first account. You'll sign in with your name and a "
                         "4-digit PIN.",
                         "You can add family members later from the Users page."])
    problem = []
    while True:
        pin = ui.ask("Choose a 4-digit PIN", secret=True, validate=S.check_pin, intro=problem)
        if ui.ask("Type the PIN again", secret=True) == pin:
            break
        problem = [ui.err("Those PINs didn't match — let's try again.")]
    i = ui.menu(f"Which age group is {name} in?", [label for _, label in S.AGE_BANDS],
                intro=["MyAI adjusts its tone to suit. You can change this later."])
    return {"name": name, "pin": pin, "age_band": S.AGE_BANDS[i][0]}, False


# ── 5 · Review ────────────────────────────────────────────────────────────────

def review(ui: TUI, cfg: dict, user: dict | None, remove_placeholder: bool,
           existing: dict) -> bool:
    ui.step(5, "Review")
    cfg.update(PORT=existing.get("PORT", S.DEFAULT_PORT), MYAIDB=str(DB_PATH),
               FILE_OUTPUT_DIR=str(GEN_DIR), WARMUP=existing.get("WARMUP", "1"))
    label = lambda choices, v: dict(choices).get(v, v)
    while True:
        rows = [
            ("Model", Path(cfg["MODEL"]).name if cfg["MODEL"] else ui.warn("none yet — add one later")),
            ("Conversation memory", f"{cfg['CTX_SIZE']} tokens"),
            ("Weather", f"on — {cfg['OWM_HOME_CITY'] or 'no home town'}" if cfg["OWM_API_KEY"] else "off"),
            ("Web search", label(S.SEARCH_REGIONS, cfg["SEARCH_REGION"])),
            ("Voice input", f"on — {cfg['WHISPER_MODEL']}" if cfg["WHISPER_MODEL"] else "off"),
            ("Account", f"{user['name']} ({label(S.AGE_BANDS, user['age_band'])})" if user
             else "keeping your existing accounts"
             + (", removing 'Setup'" if remove_placeholder else "")),
            ("Address", f"http://localhost:{cfg['PORT']}"),
        ]
        body = [f"{ui.dim(f'{k}:'.ljust(21))} {v}" for k, v in rows]
        body += ["", "Setup will now save these settings and install MyAI's components.",
                 "Building the AI engine is the slow part — about 5–10 minutes."]
        choice = ui.menu("Ready?", ["Install MyAI",
                                    f"Change the port (advanced — now {cfg['PORT']})",
                                    "Cancel — leave everything as it was"], intro=body)
        if choice == 0:
            return True
        if choice == 2:
            return False
        cfg["PORT"] = ui.ask("Which port should MyAI use?", default=cfg["PORT"],
                             validate=S.check_port,
                             intro=["Only change this if something else on this Mac already "
                                    "uses port 8080."])


# ── 6 · Install ───────────────────────────────────────────────────────────────

def install(ui: TUI, cfg: dict, user: dict | None, remove_placeholder: bool) -> int:
    ui.step(6, "Installing")
    done: list[str] = []

    for d in (DB_PATH.parent, GEN_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    S.write_env(ENV_PATH, cfg)
    done.append(ui.ok("Settings saved"))

    ok, error = S.setup_database(REPO, DB_PATH, user, remove_placeholder)
    if not ok:
        return _failed(ui, done, "Couldn't set up the database.", [error])
    done.append(ui.ok(f"Account '{user['name']}' created" if user else "Database ready"))
    if remove_placeholder:
        done.append(ui.ok("Unused 'Setup' account removed"))

    have = S.installed_version(S.LLAMA_PACKAGE)
    if have == S.LLAMA_VERSION:
        done.append(ui.ok("AI engine already installed"))
    else:
        r = ui.run(done + ["", "Building the AI engine for this Mac.",
                           "This is the slow part — usually 5–10 minutes. "
                           "Time for a cuppa.", "",
                           ui.dim(f"Details are written to {LOG_PATH.name}")],
                   "Building the AI engine", S.llama_install_cmd(), LOG_PATH,
                   env=S.build_env(os.environ, S.is_apple_silicon()))
        if not r.ok:
            return _failed(ui, done, "The AI engine didn't build.", r.tail)
        done.append(ui.ok("AI engine built"))

    req = S.requirements_file(REPO)
    try:
        r = ui.run(done + ["", "Installing the rest of MyAI's components…"],
                   "Installing components", S.requirements_cmd(req), LOG_PATH)
    finally:
        req.unlink(missing_ok=True)
    if not r.ok:
        return _failed(ui, done, "Some components didn't install.", r.tail)
    done.append(ui.ok("Components installed"))

    r = ui.run(done + ["", "Checking everything works…"], "Checking the AI engine",
               S.engine_check_cmd(), LOG_PATH)
    if not r.ok:
        return _failed(ui, done, "The AI engine is installed but won't start.", r.tail)
    gpu = "gpu" in r.tail
    done.append(ui.ok("AI engine uses the GPU") if gpu
                else ui.warn("AI engine will run on the CPU only (slower)"))

    ui.step(STEPS, "All done")
    body = done + ["", ui.bold("MyAI is ready."), ""]
    if not cfg["MODEL"]:
        body += [ui.warn("One thing left: add a model."),
                 f"Put a .gguf file in {MODELS_DIR.name}/ and run ./install.sh again "
                 "to choose it.", ""]
    body += ["To start MyAI, run this in the MyAI folder:", "",
             "    " + ui.paint("./start.sh", "cyan"), "",
             f"Then open http://localhost:{cfg['PORT']} in your browser"
             + (f" and sign in as {user['name']}." if user else ".")]
    ui.show(body, wait=False)
    return 0


def _failed(ui: TUI, done: list[str], what: str, tail: list[str]) -> int:
    body = done + ["", ui.err(what), ""]
    if any(tail):
        body += [ui.dim("Last lines of the log:")] + [ui.dim("  " + l) for l in tail] + [""]
    body += [f"Full details: {LOG_PATH}",
             "Your settings are saved. Once the problem is fixed, run ./install.sh "
             "again — it picks up where it left off."]
    ui.show(body, wait=False)
    return 1


if __name__ == "__main__":
    sys.exit(main())
