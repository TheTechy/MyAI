"""
test_suite/conftest.py
======================
Session-level setup for the MyAI test suite.

Responsibilities
----------------
- Set required environment variables *before* any app module is imported
  (DB path, model path, port, etc.)
- Replace llama_cpp with a MagicMock so no real model file is needed
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
# The real Llama constructor loads a multi-GB model file; we short-circuit it.
# The stub is configured to return a plausible "general" response for all calls
# so the SSE pipeline exercises the happy path without requiring any model.

_llm_mock = MagicMock(name="llm_instance")
_llm_mock.n_ctx.return_value = 4096
_llm_mock.n_tokens = 0
# Default return value for both classifier and routing calls (non-streaming):
# "Paris is the capital of France." is not a valid category, so the classifier
# falls back to "general", and the routing pass returns it as the answer.
_llm_mock.return_value = {
    "choices": [{"text": "Paris is the capital of France.", "finish_reason": "stop"}]
}

_llama_cpp_stub = MagicMock(name="llama_cpp_module")
_llama_cpp_stub.Llama = MagicMock(return_value=_llm_mock)
sys.modules.setdefault("llama_cpp", _llama_cpp_stub)

# Stub faster_whisper — may not be installed on all machines
sys.modules.setdefault("faster_whisper", MagicMock(name="faster_whisper"))


# ── Fixtures ───────────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def llm_mock() -> MagicMock:
    """Exposes the mock Llama instance so tests can inspect or reconfigure it."""
    return _llm_mock


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
    Injects 'testuser' into the session so every request passes is_authenticated().
    """
    with flask_app.test_client() as c:
        with c.session_transaction() as sess:
            sess["user_id"] = "testuser"
        yield c
