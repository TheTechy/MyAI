"""
test_suite/test_api.py
======================
Integration tests for the Flask HTTP API.

Coverage
--------
  /auth                    — authentication endpoint
  /conversations           — create / list / delete / search
  /conversations/<id>/messages — message retrieval
  /upload                  — file upload validation
  /prompt                  — SSE streaming endpoint (mocked LLM)
  /logout                  — session teardown

The LLM is replaced by the MagicMock defined in conftest.py.  Endpoint
tests that touch /prompt configure the mock to return a short, plain-text
answer so the full SSE pipeline can be exercised without a real model.
"""
from __future__ import annotations

import io
import json

import pytest

from helpers import event_names, parse_sse, token_text


# ── /auth ──────────────────────────────────────────────────────────────────────
class TestAuth:

    def test_missing_body_returns_400(self, client):
        r = client.post("/auth", data="not-json", content_type="application/json")
        assert r.status_code in (400, 415)

    def test_missing_pin_field_returns_400(self, client):
        r = client.post("/auth", json={"user_id": "testuser"})
        assert r.status_code == 400

    def test_unknown_user_returns_401(self, client):
        r = client.post("/auth", json={"user_id": "ghost", "pin": "0000"})
        assert r.status_code == 401

    def test_wrong_pin_returns_401(self, client):
        r = client.post("/auth", json={"user_id": "testuser", "pin": "9999"})
        assert r.status_code == 401


# ── Unauthenticated access ─────────────────────────────────────────────────────
class TestUnauthenticated:

    def test_conversations_list_requires_auth(self, client):
        r = client.get("/conversations/testuser")
        assert r.status_code == 401

    def test_new_conversation_requires_auth(self, client):
        r = client.post("/conversations", json={"user_id": "testuser"})
        assert r.status_code == 401

    def test_delete_conversation_requires_auth(self, client):
        r = client.delete("/conversations/some-uuid")
        assert r.status_code == 401

    def test_prompt_requires_auth(self, client):
        r = client.post("/prompt", json={"prompt": "hello", "name": "testuser"})
        assert r.status_code == 401

    def test_upload_requires_auth(self, client):
        r = client.post("/upload", data={})
        assert r.status_code == 401

    def test_transcribe_requires_auth(self, client):
        r = client.post("/transcribe", data={})
        assert r.status_code == 401


# ── /logout ────────────────────────────────────────────────────────────────────
class TestLogout:

    def test_logout_redirects(self, auth_client):
        r = auth_client.get("/logout")
        assert r.status_code in (301, 302)

    def test_logout_clears_session(self, flask_app):
        with flask_app.test_client() as c:
            with c.session_transaction() as sess:
                sess["user_id"] = "testuser"
            c.get("/logout")
            # After logout, /conversations should require auth again
            r = c.get("/conversations/testuser")
            assert r.status_code == 401


# ── /conversations ─────────────────────────────────────────────────────────────
class TestConversations:

    def test_create_conversation_returns_conv_id(self, auth_client):
        r = auth_client.post("/conversations", json={"user_id": "testuser"})
        assert r.status_code == 200
        body = json.loads(r.data)
        assert "conv_id" in body
        assert len(body["conv_id"]) == 36  # UUID

    def test_create_conversation_missing_user_id(self, auth_client):
        r = auth_client.post("/conversations", json={})
        assert r.status_code == 400

    def test_list_conversations_returns_list(self, auth_client):
        auth_client.post("/conversations", json={"user_id": "testuser"})
        r = auth_client.get("/conversations/testuser")
        assert r.status_code == 200
        assert isinstance(json.loads(r.data), list)

    def test_created_conversation_appears_in_list(self, auth_client):
        r = auth_client.post("/conversations", json={"user_id": "testuser"})
        conv_id = json.loads(r.data)["conv_id"]
        r2 = auth_client.get("/conversations/testuser")
        ids = [c["conv_id"] for c in json.loads(r2.data)]
        assert conv_id in ids

    def test_delete_conversation(self, auth_client):
        r = auth_client.post("/conversations", json={"user_id": "testuser"})
        conv_id = json.loads(r.data)["conv_id"]
        rd = auth_client.delete(f"/conversations/{conv_id}")
        assert rd.status_code == 200
        assert json.loads(rd.data)["deleted"] == conv_id
        ids = [c["conv_id"] for c in json.loads(auth_client.get("/conversations/testuser").data)]
        assert conv_id not in ids

    def test_get_messages_returns_list(self, auth_client):
        r = auth_client.post("/conversations", json={"user_id": "testuser"})
        conv_id = json.loads(r.data)["conv_id"]
        r2 = auth_client.get(f"/conversations/{conv_id}/messages")
        assert r2.status_code == 200
        assert isinstance(json.loads(r2.data), list)

    def test_search_conversations(self, auth_client):
        r = auth_client.get("/conversations/testuser/search?q=anything")
        assert r.status_code == 200
        assert isinstance(json.loads(r.data), list)

    def test_search_empty_query_returns_all(self, auth_client):
        r = auth_client.get("/conversations/testuser/search?q=")
        assert r.status_code == 200


# ── /upload ────────────────────────────────────────────────────────────────────
class TestUpload:

    def _new_conv(self, auth_client) -> str:
        r = auth_client.post("/conversations", json={"user_id": "testuser"})
        return json.loads(r.data)["conv_id"]

    def test_no_file_part_returns_400(self, auth_client):
        conv_id = self._new_conv(auth_client)
        r = auth_client.post("/upload", data={"conv_id": conv_id})
        assert r.status_code == 400

    def test_empty_filename_returns_400(self, auth_client):
        conv_id = self._new_conv(auth_client)
        r = auth_client.post(
            "/upload",
            data={"file": (io.BytesIO(b""), ""), "conv_id": conv_id},
            content_type="multipart/form-data",
        )
        assert r.status_code == 400

    def test_disallowed_extension_returns_415(self, auth_client):
        conv_id = self._new_conv(auth_client)
        r = auth_client.post(
            "/upload",
            data={"file": (io.BytesIO(b"bad"), "virus.exe"), "conv_id": conv_id},
            content_type="multipart/form-data",
        )
        assert r.status_code == 415

    def test_missing_conv_id_returns_400(self, auth_client):
        r = auth_client.post(
            "/upload",
            data={"file": (io.BytesIO(b"hello"), "sample.txt")},
            content_type="multipart/form-data",
        )
        assert r.status_code == 400

    def test_txt_upload_succeeds(self, auth_client):
        conv_id = self._new_conv(auth_client)
        r = auth_client.post(
            "/upload",
            data={"file": (io.BytesIO(b"Hello world"), "sample.txt"), "conv_id": conv_id},
            content_type="multipart/form-data",
        )
        assert r.status_code == 200
        body = json.loads(r.data)
        assert "file_id" in body
        assert body["filename"] == "sample.txt"
        assert body["chars"] > 0

    def test_delete_uploaded_file(self, auth_client):
        conv_id = self._new_conv(auth_client)
        r = auth_client.post(
            "/upload",
            data={"file": (io.BytesIO(b"To delete"), "delete_me.txt"), "conv_id": conv_id},
            content_type="multipart/form-data",
        )
        file_id = json.loads(r.data)["file_id"]
        rd = auth_client.delete(f"/upload/{file_id}")
        assert rd.status_code == 200
        assert json.loads(rd.data)["deleted"] == file_id


# ── /prompt (SSE) ──────────────────────────────────────────────────────────────
class TestPromptEndpoint:
    """
    Smoke tests for the /prompt SSE endpoint with the mocked LLM.

    The mock (configured in conftest.py) returns a plain-text answer for all
    calls, so the general/no-search code path is exercised.
    """

    def _new_conv(self, auth_client) -> str:
        r = auth_client.post("/conversations", json={"user_id": "testuser"})
        return json.loads(r.data)["conv_id"]

    def test_empty_prompt_returns_400(self, auth_client):
        r = auth_client.post("/prompt", json={"prompt": "   ", "name": "testuser"})
        assert r.status_code == 400

    def test_missing_prompt_returns_400(self, auth_client):
        r = auth_client.post("/prompt", json={"name": "testuser"})
        assert r.status_code == 400

    def test_response_is_event_stream(self, auth_client, llm_mock):
        conv_id = self._new_conv(auth_client)
        r = auth_client.post(
            "/prompt",
            json={"prompt": "What is the capital of France?", "name": "testuser", "conv_id": conv_id},
        )
        assert r.status_code == 200
        assert "text/event-stream" in r.content_type

    def test_done_event_is_present(self, auth_client, llm_mock):
        conv_id = self._new_conv(auth_client)
        r = auth_client.post(
            "/prompt",
            json={"prompt": "What is the capital of France?", "name": "testuser", "conv_id": conv_id},
        )
        events = parse_sse(r.data)
        assert "done" in event_names(events), f"No 'done' event in: {event_names(events)}"

    def test_status_event_is_present(self, auth_client, llm_mock):
        conv_id = self._new_conv(auth_client)
        r = auth_client.post(
            "/prompt",
            json={"prompt": "What is the capital of France?", "name": "testuser", "conv_id": conv_id},
        )
        events = parse_sse(r.data)
        assert "status" in event_names(events), f"No 'status' event in: {event_names(events)}"

    def test_token_or_response_contains_text(self, auth_client, llm_mock):
        conv_id = self._new_conv(auth_client)
        r = auth_client.post(
            "/prompt",
            json={"prompt": "What is the capital of France?", "name": "testuser", "conv_id": conv_id},
        )
        events = parse_sse(r.data)
        text = token_text(events)
        assert text, f"No token text in response events: {events}"

    def test_autocreated_conv_id_event(self, auth_client, llm_mock):
        # When no conv_id is supplied, the server creates one and emits it
        r = auth_client.post(
            "/prompt",
            json={"prompt": "Hello!", "name": "testuser"},
        )
        events = parse_sse(r.data)
        conv_events = [e for e in events if e.get("event") == "conv_id"]
        assert conv_events, "Expected a conv_id event when no conv_id was provided"

    def test_message_stored_after_prompt(self, auth_client, llm_mock):
        conv_id = self._new_conv(auth_client)
        resp = auth_client.post(
            "/prompt",
            json={"prompt": "What year did WW2 end?", "name": "testuser", "conv_id": conv_id},
        )
        # Drain the SSE stream — with stream_with_context, the generator
        # (and therefore the message-storage code at the end of ask_stream)
        # only runs as the response body is consumed.
        _ = resp.data
        r = auth_client.get(f"/conversations/{conv_id}/messages")
        msgs = json.loads(r.data)
        assert len(msgs) >= 2  # at least one user + one model message
        roles = [m["role"] for m in msgs]
        assert "user" in roles
        assert "model" in roles