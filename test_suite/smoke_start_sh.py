"""start.sh / debug.sh, run from copies in a temp folder with a stand-in
.venv/bin/python, so the real app never starts."""
import os, shutil, subprocess, tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent          # test_suite/ -> repo root
START, DEBUG = _REPO / "start.sh", _REPO / "debug.sh"

# Syntax check under the bash macOS actually ships (3.2).
for s in (START, DEBUG):
    subprocess.run(["bash", "-n", str(s)], check=True)
print("ok  bash -n (start.sh, debug.sh)")


def port_from(env_text: str) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / ".env").write_text(env_text)
        r = subprocess.run(["bash", "-c", f'source "{START}"; port_from_env'], cwd=tmp,
                           capture_output=True, text=True,
                           env={**os.environ, "MYAI_START_NO_MAIN": "1"})
        return r.stdout.strip()

# 1) PORT is read the way the setup app writes it, and as people hand-edit it.
assert port_from("MODEL=/m.gguf\nPORT=8888\n") == "8888"
assert port_from('PORT="9000"\t\t# web port\n') == "9000"
assert port_from("PORT=1\nPORT=8081\n") == "8081"            # last one wins, like .env
assert port_from("# PORT=1234\nMODEL=x\n") == ""
print("ok  port_from_env (plain, quoted + comment, last wins, commented out)")


def folder(tmp: Path, *, app_exit=None, env=True) -> Path:
    """A copy of the scripts, plus a fake venv whose python exits with app_exit."""
    for s in (START, DEBUG):
        shutil.copy(s, tmp / s.name)
    if env:
        (tmp / ".env").write_text("PORT=8888\n")
    if app_exit is not None:
        py = tmp / ".venv" / "bin" / "python"
        py.parent.mkdir(parents=True)
        py.write_text(f'#!/bin/sh\necho "app ran: $1" > ran.txt\nprintf "serving"\nexit {app_exit}\n')
        py.chmod(0o755)
    return tmp


def run(script: Path) -> subprocess.CompletedProcess:
    # Run from elsewhere: the scripts must find their own folder.
    return subprocess.run(["bash", str(script)], cwd="/", capture_output=True, text=True,
                          env={**os.environ, "TERM": "dumb"})

with tempfile.TemporaryDirectory() as tmp:
    # 2) Not installed yet → points at ./install.sh, doesn't try to start.
    d = folder(Path(tmp))
    r = run(d / "start.sh")
    assert r.returncode == 1 and "./install.sh" in r.stdout, r
print("ok  not set up → points at ./install.sh")

with tempfile.TemporaryDirectory() as tmp:
    # 3) Normal start/stop: runs app.py from the MyAI folder, shows the
    #    address, and ends the app's unfinished line before saying it stopped.
    d = folder(Path(tmp), app_exit=0)
    r = run(d / "start.sh")
    assert r.returncode == 0, r
    assert (d / "ran.txt").read_text().strip() == "app ran: app.py"
    assert "http://localhost:8888" in r.stdout and "Ctrl-C" in r.stdout
    assert "serving\n  MyAI has stopped." in r.stdout, r.stdout
print("ok  starts app.py in its own folder, shows the address, clean stop")

with tempfile.TemporaryDirectory() as tmp:
    # 4) Ctrl-C (the app exits 130) is a normal stop, not an error.
    d = folder(Path(tmp), app_exit=130)
    r = run(d / "start.sh")
    assert r.returncode == 0 and "MyAI has stopped." in r.stdout, r
print("ok  exit 130 (Ctrl-C) → normal stop")

with tempfile.TemporaryDirectory() as tmp:
    # 5) A crash is reported and its exit code passed on.
    d = folder(Path(tmp), app_exit=3)
    r = run(d / "start.sh")
    assert r.returncode == 3 and "stopped with an error (code 3)" in r.stdout, r
print("ok  crash → reported, exit code passed on")

with tempfile.TemporaryDirectory() as tmp:
    # 6) debug.sh clears the project's caches but not the venv's, then starts.
    d = folder(Path(tmp), app_exit=0)
    for p in ("core/__pycache__/db.cpython-313.pyc", "stray.pyc",
              ".venv/lib/__pycache__/keep.pyc"):
        (d / p).parent.mkdir(parents=True, exist_ok=True)
        (d / p).write_bytes(b"")
    r = run(d / "debug.sh")
    assert r.returncode == 0 and "MyAI has stopped." in r.stdout, r
    assert not (d / "core" / "__pycache__").exists() and not (d / "stray.pyc").exists()
    assert (d / ".venv/lib/__pycache__/keep.pyc").exists()
    assert (d / "ran.txt").exists()
print("ok  debug.sh clears project caches (not .venv), then starts")

print("\nALL GREEN")
