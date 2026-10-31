import os, stat, subprocess, tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent          # test_suite/ -> repo root
SCRIPT = _REPO / "install.sh"

# Syntax check under the bash macOS actually ships (3.2) — no bash-4-isms.
subprocess.run(["bash", "-n", str(SCRIPT)], check=True); print("ok  bash -n")


def sh(snippet: str, **env) -> subprocess.CompletedProcess:
    """Source install.sh without running main, then run `snippet`."""
    return subprocess.run(
        ["bash", "-c", f'source "{SCRIPT}"; {snippet}'],
        capture_output=True, text=True,
        env={**os.environ, "MYAI_INSTALL_NO_MAIN": "1", "NO_COLOR": "1", **env},
    )


def fake_python(d: Path, name: str, version: str) -> str:
    """A stand-in interpreter that reports `version` (as 'MAJOR.MINOR')."""
    p = d / name
    p.write_text(f'#!/bin/sh\necho {version}\n')
    p.chmod(p.stat().st_mode | stat.S_IXUSR)
    return str(p)


# 1) version_ok honours MIN_MAJOR.MIN_MINOR (default 3.13).
for v, want in [("3.9", 1), ("3.12", 1), ("3.13", 0), ("3.14", 0), ("4.0", 0)]:
    assert sh(f"version_ok {v}").returncode == want, v
assert sh("version_ok 3.13", MIN_MINOR="14").returncode == 1
print("ok  version_ok gates at 3.13 (and respects MIN_MINOR)")

with tempfile.TemporaryDirectory() as tmp:
    d = Path(tmp)
    old   = fake_python(d, "apple_py", "3.9")
    mid   = fake_python(d, "py312", "3.12")
    good  = fake_python(d, "py313", "3.13")
    newer = fake_python(d, "py314", "3.14")
    missing = str(d / "nope")

    # 2) Skips too-old and missing interpreters; takes the first that qualifies.
    r = sh(f'pick_python "{missing}" "{old}" "{good}" "{newer}"')
    assert r.returncode == 0 and r.stdout == good, (r.returncode, r.stdout)
    print("ok  pick_python skips missing/too-old, picks first suitable")

    # 3) Nothing suitable → fails and reports the newest version it saw,
    #    whatever order the candidates came in.
    r = sh(f'pick_python "{mid}" "{old}" "{missing}"')
    assert r.returncode == 1 and r.stdout == "3.12", (r.returncode, r.stdout)
    print("ok  pick_python reports newest too-old version on failure")

    # 4) Nothing at all → fails with empty output.
    r = sh(f'pick_python "{missing}"')
    assert r.returncode == 1 and r.stdout == "", (r.returncode, r.stdout)
    print("ok  pick_python with no interpreters → empty, rc=1")

# 5) Candidate list never comes back empty (an empty array would trip `set -u`
#    in bash 3.2) and never contains an unexpanded glob.
r = sh("python_candidates")
lines = r.stdout.split()
assert lines and "/usr/bin/python3" in lines and not any("*" in l for l in lines), lines
print("ok  python_candidates non-empty, no stray globs")

# 6) No suitable Python + no keyboard (stdin closed) → explains, exits 1, and
#    does NOT open the browser. `open` is shadowed so a regression can't
#    actually launch anything.
r = subprocess.run(
    ["bash", "-c", f'source "{SCRIPT}"; open(){{ echo OPENED; }}; ensure_python'],
    capture_output=True, text=True, stdin=subprocess.DEVNULL,
    env={**os.environ, "MYAI_INSTALL_NO_MAIN": "1", "NO_COLOR": "1", "MIN_MINOR": "99"},
)
assert r.returncode == 1, r.returncode
assert "python.org" in r.stdout and "OPENED" not in r.stdout, r.stdout
print("ok  no suitable Python + closed stdin → exit 1, browser not opened")

print("\nALL GREEN")
