import os, sqlite3, sys, tempfile
from pathlib import Path

from installer import system as S

_REPO = Path(__file__).resolve().parent.parent
GB = S.GB

# 1) read_env copes with what people actually have: comments, the tab-aligned
#    inline comments in .env.example, quotes, blanks, values with spaces.
with tempfile.TemporaryDirectory() as tmp:
    p = Path(tmp) / ".env"
    p.write_text("# header\n\nMODEL=/m/x.gguf\t\t\t# Path to your model\n"
                 "OWM_HOME_CITY=Leeds, GB\nOWM_API_KEY=\nSECRET_KEY=\"abc#123\"\n"
                 "not a setting\n", encoding="utf-8")
    env = S.read_env(p)
    assert env == {"MODEL": "/m/x.gguf", "OWM_HOME_CITY": "Leeds, GB",
                   "OWM_API_KEY": "", "SECRET_KEY": "abc#123"}, env
    assert S.read_env(Path(tmp) / "missing") == {}
print("ok  read_env (inline comments, quotes, blanks, spaces)")

# 2) render → read round-trips; write_env keeps an existing SECRET_KEY and only
#    generates one when there isn't one.
cfg = {"MODEL": "/m/x.gguf", "CTX_SIZE": "8192", "PORT": "8888", "MYAIDB": "/db",
       "FILE_OUTPUT_DIR": "/gen", "OWM_API_KEY": "k", "OWM_UNITS": "imperial",
       "OWM_HOME_CITY": "Austin, US", "SEARCH_REGION": "us-en", "WHISPER_MODEL": "base.en",
       "WHISPER_DEVICE": "cpu", "WHISPER_COMPUTE": "int8", "WARMUP": "1"}
with tempfile.TemporaryDirectory() as tmp:
    p = Path(tmp) / ".env"
    assert S.write_env(p, cfg) is True                       # new key generated
    first = S.read_env(p)
    assert {k: first[k] for k in cfg} == cfg and len(first["SECRET_KEY"]) == 64
    assert S.write_env(p, {**cfg, "PORT": "9000"}) is False  # kept
    second = S.read_env(p)
    assert second["SECRET_KEY"] == first["SECRET_KEY"] and second["PORT"] == "9000"
print("ok  render/write_env round-trip, SECRET_KEY preserved")

# 3) The engine line is stripped from requirements, whatever its spelling.
req = "Flask==3.1.3\nllama_cpp_python==0.3.33\nLlama-CPP-Python>=0.3\nnumpy==2.5.1\n"
assert S.requirements_without_llama(req) == "Flask==3.1.3\nnumpy==2.5.1\n"
real = (_REPO / "requirements.txt").read_text()
kept = S.requirements_without_llama(real)
assert "flask" in kept.lower() and "llama" not in kept.lower()
print("ok  requirements_without_llama")

# 4) Memory guidance. 18 GB Mac → ~12 GB for the GPU; 64 GB Mac → 48 GB.
assert S.gpu_budget(18 * GB) == 12 * GB and S.gpu_budget(64 * GB) == 48 * GB
g = lambda x: int(x * GB)
assert S.model_fit(g(8.4), 18 * GB) == "fits"       # 12B Q5 on the MacBook
assert S.model_fit(g(11), 18 * GB) == "tight"
assert S.model_fit(g(20.2), 18 * GB) == "too_big"   # the 27B that wouldn't load
assert S.model_fit(g(20.2), 64 * GB) == "fits"      # …but fine on the Mini
assert S.model_fit(g(8), None) is None
assert S.recommended_ctx(18 * GB) == "16384" and S.recommended_ctx(64 * GB) == "32768"
print("ok  gpu_budget / model_fit / recommended_ctx")

# 5) Build flags: Metal on Apple Silicon only; a stray CMAKE_ARGS never leaks into a CPU build.
assert S.is_apple_silicon("Darwin", "arm64") and not S.is_apple_silicon("Darwin", "x86_64")
assert not S.is_apple_silicon("Linux", "aarch64")
assert S.build_env({"PATH": "/x"}, True) == {"PATH": "/x", "CMAKE_ARGS": "-DGGML_METAL=on"}
assert S.build_env({"PATH": "/x", "CMAKE_ARGS": "junk"}, False) == {"PATH": "/x"}
assert S.llama_install_cmd("py")[-1] == f"llama-cpp-python=={S.LLAMA_VERSION}"
print("ok  build_env / llama_install_cmd pinned")

# 6) Answer checks.
assert S.check_pin("1234") is None and S.check_pin("123") and S.check_pin("12a4")
assert S.check_port("8080") is None and S.check_port("80") and S.check_port("x")
assert S.check_name("George") is None and S.check_name("") and S.check_name("x" * 41)
assert S.check_name("Setup")                          # reserved for the placeholder
with tempfile.TemporaryDirectory() as tmp:
    m = Path(tmp) / "m.gguf"; m.write_bytes(b"x")
    t = Path(tmp) / "m.txt"; t.write_bytes(b"x")
    assert S.check_model_path(str(m)) is None
    assert S.check_model_path(str(t)) and S.check_model_path(str(Path(tmp) / "no.gguf"))
assert S.choice_index(S.CTX_CHOICES, "32768") == 2 and S.choice_index(S.CTX_CHOICES, "?", 1) == 1
print("ok  check_pin / check_port / check_name / check_model_path / choice_index")

# 7) Database, against the real core.db: an empty DB gets core.db's 'Setup'
#    placeholder, which doesn't count as a real user; a first user replaces it;
#    a re-run without a user leaves accounts alone.
assert S.in_venv(), "run this with the project venv"
with tempfile.TemporaryDirectory() as tmp:
    db = Path(tmp) / "instance" / "myai.db"
    ok, err = S.setup_database(_REPO, db, None)
    assert ok, err
    assert not S.has_real_users(db)
    ok, err = S.setup_database(_REPO, db, {"name": "George", "pin": "1234", "age_band": "adult"})
    assert ok, err
    with sqlite3.connect(db) as c:
        rows = c.execute("SELECT display_id, pin, age_band FROM users").fetchall()
    assert rows == [("George", "1234", "adult")], rows
    assert S.has_real_users(db)
    ok, err = S.setup_database(_REPO, db, None)
    with sqlite3.connect(db) as c:
        assert c.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 1
    # A clash is reported, not swallowed.
    ok, err = S.setup_database(_REPO, db, {"name": "George", "pin": "9999", "age_band": "adult"})
    assert not ok and "already in use" in err, err
    assert not S.has_real_users(Path(tmp) / "nope.db")
print("ok  setup_database / has_real_users (real core.db)")

# 8) A leftover placeholder (from before the app removed it itself) is spotted
#    and removed only when asked; a fresh DB's lone placeholder never is.
with tempfile.TemporaryDirectory() as tmp:
    db = Path(tmp) / "myai.db"
    assert not S.has_unused_placeholder(_REPO, db)                 # no DB yet
    S.setup_database(_REPO, db, None)
    assert not S.has_unused_placeholder(_REPO, db)                 # it's the only way in
    with sqlite3.connect(db) as c:                                 # an old-style install
        c.execute("INSERT INTO users(user_id, display_id, pin, age_band, active, created_at) "
                  "VALUES ('g', 'George', '1234', 'adult', 1, '2026-01-01')")
    assert S.has_unused_placeholder(_REPO, db)
    ok, err = S.setup_database(_REPO, db, None)                    # not asked → kept
    assert ok and S.has_unused_placeholder(_REPO, db), err
    ok, err = S.setup_database(_REPO, db, None, remove_placeholder=True)
    assert ok, err
    with sqlite3.connect(db) as c:
        assert [r[0] for r in c.execute("SELECT display_id FROM users")] == ["George"]
    assert not S.has_unused_placeholder(_REPO, db)
print("ok  has_unused_placeholder / remove on request only")

print("\nALL GREEN")
