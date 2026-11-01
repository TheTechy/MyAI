"""End-to-end run of setup_myai.py with scripted answers.

The screens fall back to numbered prompts (stdout isn't a terminal), answers come
from a list, and the slow/system-changing parts — pip, the engine build, Apple's
tools — are replaced by stand-ins. Everything else is real: the .env written, and
the database created through core.db. Paths point at a temp folder.
"""
import builtins, contextlib, io, sqlite3, sys, tempfile
from pathlib import Path

import installer.tui as T
from installer import system as S
import setup_myai as M

PY = sys.executable
_real_input = builtins.input
calls: list[str] = []

def script(answers):
    it = iter(answers)
    def fake(prompt=""):
        try:
            return next(it)
        except StopIteration:
            raise EOFError          # ran out of answers → treated as Ctrl-C
    builtins.input = fake
    T._getpass = fake
    return it

def fake_cmd(name, output="", rc=0):
    def make(*a, **k):
        calls.append(name)
        return [PY, "-c", f"print({output!r}); raise SystemExit({rc})"]
    return make

def run(answers, *, engine=None):
    S.installed_version = lambda dist: engine
    it = script(answers)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = M.main()
    left = list(it)
    return rc, out.getvalue(), left

S.has_build_tools = lambda: True
S.total_ram_bytes = lambda: 18 * S.GB
S.llama_install_cmd = fake_cmd("build", "built")
S.requirements_cmd = fake_cmd("reqs", "reqs ok")
S.engine_check_cmd = fake_cmd("check", "gpu")

try:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        M.ENV_PATH, M.MODELS_DIR = tmp / ".env", tmp / "models"
        M.DB_PATH, M.GEN_DIR, M.LOG_PATH = tmp / "instance" / "myai.db", tmp / "gen", tmp / "install.log"
        M.MODELS_DIR.mkdir()
        model = M.MODELS_DIR / "tiny-test-model.gguf"
        model.write_bytes(b"\0" * 1024)

        # 1) Cancel at the review screen → nothing written.
        rc, out, left = run([
            "",            # welcome
            "",            # Mac check OK
            "1", "",       # model 1, default context
            "2",           # weather: No
            "1",           # search: worldwide
            "2",           # voice: No
            "George", "1234", "1234", "",   # name, PIN twice, age band default (adult)
            "3",           # review: Cancel
        ])
        assert rc == 0 and not left, (rc, left)
        assert not M.ENV_PATH.exists() and not M.DB_PATH.exists() and calls == []
        assert "Nothing was changed" in out
        print("ok  cancel at review → nothing written, nothing installed")

        # 2) Full first install: answers → .env + DB user → build → components → check.
        rc, out, left = run([
            "", "",
            "1", "3",                  # model 1, context 32768
            "1", "my-owm-key", "1", "leeds,gb",   # weather on, metric, home town
            "2",                       # search: UK
            "1", "",                   # voice on, default (base.en)
            "George", "1234", "4321",  # PINs don't match…
            "1234", "1234", "4",       # …so asked again; age band: senior
            "2", "8888",               # review: change port
            "1",                       # review: install
        ])
        assert rc == 0 and not left, (rc, left, out[-2000:])
        env = S.read_env(M.ENV_PATH)
        assert env["MODEL"] == str(model.resolve()) and env["CTX_SIZE"] == "32768"
        assert env["OWM_API_KEY"] == "my-owm-key" and env["OWM_HOME_CITY"] == "leeds, gb"
        assert env["SEARCH_REGION"] == "uk-en" and env["PORT"] == "8888"
        assert (env["WHISPER_MODEL"], env["WHISPER_DEVICE"], env["WHISPER_COMPUTE"]) == \
               ("base.en", "cpu", "int8")
        assert env["MYAIDB"] == str(M.DB_PATH) and len(env["SECRET_KEY"]) == 64
        with sqlite3.connect(M.DB_PATH) as c:
            assert c.execute("SELECT display_id, pin, age_band FROM users").fetchall() == \
                   [("George", "1234", "senior")]
        assert calls == ["build", "reqs", "check"], calls
        assert "built" in M.LOG_PATH.read_text() and "MyAI is ready" in out
        assert "didn't match" in out and "http://localhost:8888" in out and "./start.sh" in out
        print("ok  first install: .env, DB user, build → components → check, PIN re-ask")

        # 3) Re-run: existing answers are the defaults (Enter all the way), the
        #    account step is skipped, the engine isn't rebuilt, SECRET_KEY survives.
        calls.clear()
        rc, out, left = run([
            "", "",
            "", "",          # model + context: keep
            "", "", "", "",  # weather: keep on, key, units, home town
            "",              # search: keep
            "", "",          # voice: keep
            "",              # account: already set up
            "1",             # install
        ], engine=S.LLAMA_VERSION)
        assert rc == 0 and not left, (rc, left, out[-2000:])
        again = S.read_env(M.ENV_PATH)
        assert again == env, {k: (env.get(k), again.get(k)) for k in env if env.get(k) != again.get(k)}
        assert calls == ["reqs", "check"], calls
        assert "already has accounts" in out and "already installed" in out
        print("ok  re-run keeps answers + SECRET_KEY, skips account and engine build")

        # 3b) A leftover 'Setup' / 0000 account next to real ones (an install from
        #     before the app cleaned it up) → offered for removal, Yes by default.
        with sqlite3.connect(M.DB_PATH) as c:
            c.execute("INSERT INTO users(user_id, display_id, pin, age_band, active, created_at) "
                      "VALUES ('s', 'Setup', '0000', 'adult', 1, '2026-01-01')")
        calls.clear()
        rc, out, left = run(["", "", "", "", "", "", "", "", "", "", "",
                             "",      # account: "Remove the unused 'Setup' account?" → Yes
                             "1"], engine=S.LLAMA_VERSION)
        assert rc == 0 and not left, (rc, left, out[-2000:])
        assert "Remove the unused 'Setup' account" in out and "removing 'Setup'" in out
        assert "Unused 'Setup' account removed" in out
        with sqlite3.connect(M.DB_PATH) as c:
            assert [r[0] for r in c.execute("SELECT display_id FROM users")] == ["George"]
        print("ok  leftover 'Setup' account → offered, removed (default Yes)")

        # 4) A failed build stops the install, shows the log tail, and says how to recover.
        S.llama_install_cmd = fake_cmd("build", "error: compiler exploded", rc=1)
        calls.clear()
        rc, out, left = run(["", "", "", "", "", "", "", "", "", "", "", "", "1"])
        assert rc == 1 and calls == ["build"], (rc, calls)
        assert "didn't build" in out and "compiler exploded" in out and "./install.sh" in out
        print("ok  failed build → stops, shows log tail + how to recover")

        # 5) Ctrl-C (here: answers running out) mid-questions → clean exit, nothing changed.
        M.ENV_PATH.unlink()
        rc, out, left = run(["", ""])
        assert rc == 130 and "Nothing was changed" in out and not M.ENV_PATH.exists()
        print("ok  Ctrl-C mid-way → exit 130, nothing changed")

    # 6) Started outside a venv → refuses, points at ./install.sh.
    S_in_venv = S.in_venv
    S.in_venv = lambda: False
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert M.main() == 1
    assert "./install.sh" in out.getvalue()
    S.in_venv = S_in_venv
    print("ok  outside a venv → refuses, points at ./install.sh")
finally:
    builtins.input = _real_input

# The context screen's options fit the widest panel without being cut off with
# '…', whichever one carries "(recommended)" and the "▶ " selection marker.
inner = T.MAX_WIDTH - 4
for rec, _ in S.CTX_CHOICES:
    for option in M.context_options(T.TUI(), rec):
        assert T.vis_len("▶ " + option) <= inner, (rec, option, T.vis_len("▶ " + option))
print("ok  context options fit the panel")

print("\nALL GREEN")
