"""
test_suite/conftest.py
======================
Session-level setup for the MyAI test suite.

Responsibilities
----------------
- Set required environment variables *before* any app module is imported
  (DB path, model path, port, etc.)
- Replace llama_cpp with test_suite/fake_llama_cpp so no real model file is needed
- Stub faster-whisper for machines where it may not be installed
- Expose session/function-scoped pytest fixtures used across all test files
"""
from __future__ import annotations

import os
import sys
import tempfile
from unittest.mock import MagicMock

import pytest


# ── Environment variables ──────────────────────────────────────────────────────
# These must be set at module import time, before any app code is loaded.

_tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_tmp_db.close()

os.environ.setdefault("MODEL",      "/tmp/fake.gguf")
os.environ.setdefault("CTX_SIZE",   "4096")
os.environ.setdefault("PORT",       "8080")
os.environ.setdefault("SECRET_KEY", "test-secret-key-pytest-only")
os.environ.setdefault("MYAIDB",     _tmp_db.name)
os.environ.setdefault("USERS",      "testuser")


# ── llama_cpp stub ─────────────────────────────────────────────────────────────
# The real Llama constructor loads a multi-GB model file; we short-circuit it
# with the same fake the smoke tests use. A bare MagicMock isn't enough: the
# engine reads string metadata (the chat template) and the chat-completion API.
# The fake answers "hello" to every call, so classification falls back to
# "general" and the SSE pipeline takes the happy path.
import fake_llama_cpp  # noqa: E402 — test_suite/ is on sys.path via conftest

sys.modules.setdefault("llama_cpp", fake_llama_cpp)

# Stub faster_whisper — may not be installed on all machines
sys.modules.setdefault("faster_whisper", MagicMock(name="faster_whisper"))


# ── Fixtures ───────────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def llm_mock():
    """
    Exposes the fake Llama class. The model loads lazily, so the instance the
    app is using is Llama.instances[-1] once a request has run.
    """
    return fake_llama_cpp.Llama


@pytest.fixture(scope="session", autouse=True)
def test_user_row():
    """
    Make the user the tests sign in as a real users row. Conversations, files
    and memories reference users(user_id) with foreign keys on, so without it
    every write for that user fails.
    """
    from core.db import get_conn, init_db  # noqa: PLC0415 — after MYAIDB is set
    from helpers import TEST_USER_ID  # noqa: PLC0415

    init_db()
    with get_conn() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO users (user_id, display_id, pin, created_at) "
            "VALUES (?, 'testuser', '1234', datetime('now'))",
            (TEST_USER_ID,),
        )


@pytest.fixture(scope="session")
def flask_app():
    """
    Create the Flask application once for the whole test session.
    App is imported here so the stubs above are in place first.
    """
    from app import app  # noqa: PLC0415 — deferred import is intentional

    app.config["TESTING"] = True
    app.config["WTF_CSRF_ENABLED"] = False
    return app


@pytest.fixture
def client(flask_app):
    """Unauthenticated Flask test client."""
    with flask_app.test_client() as c:
        yield c


@pytest.fixture
def auth_client(flask_app):
    """
    Authenticated Flask test client.
    Signs in as the seeded test user so every request passes is_authenticated().
    """
    from helpers import TEST_USER_ID  # noqa: PLC0415

    with flask_app.test_client() as c:
        with c.session_transaction() as sess:
            sess["user_id"] = TEST_USER_ID
        yield c
