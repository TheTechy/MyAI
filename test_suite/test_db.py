"""
test_suite/test_db.py
=====================
Unit tests for core/db.py — CRUD operations against a real (temp) SQLite file.

The database path is set in conftest.py via MYAIDB.  Each test gets a
freshly initialised schema so data doesn't bleed between tests.
"""
import pytest

from core.db import (
    append_message,
    auto_title_from_first_message,
    create_conversation,
    delete_conversation,
    delete_file_context,
    ensure_user,
    get_conversation_title,
    get_file_contexts,
    get_messages,
    init_db,
    list_conversations,
    search_conversations,
    store_file_context,
)


@pytest.fixture(autouse=True)
def fresh_schema():
    """Ensure tables exist before every test (idempotent — safe to call repeatedly)."""
    init_db()


# ── Conversations ──────────────────────────────────────────────────────────────
class TestConversations:

    def test_create_returns_uuid(self):
        conv_id = create_conversation("testuser")
        assert conv_id
        assert len(conv_id) == 36  # standard UUID string

    def test_created_conversation_appears_in_list(self):
        conv_id = create_conversation("testuser")
        ids = [c["conv_id"] for c in list_conversations("testuser")]
        assert conv_id in ids

    def test_list_returns_newest_first(self):
        id1 = create_conversation("testuser")
        id2 = create_conversation("testuser")
        convs = list_conversations("testuser")
        ids = [c["conv_id"] for c in convs]
        # id2 was created last — should appear before id1
        assert ids.index(id2) < ids.index(id1)

    def test_list_empty_for_unknown_user(self):
        assert list_conversations("unknown_user_xyz") == []

    def test_delete_removes_conversation(self):
        conv_id = create_conversation("testuser")
        delete_conversation(conv_id)
        ids = [c["conv_id"] for c in list_conversations("testuser")]
        assert conv_id not in ids

    def test_default_title_is_new_conversation(self):
        conv_id = create_conversation("testuser")
        title = get_conversation_title(conv_id)
        assert title == "New conversation"

    def test_get_title_for_missing_conv_returns_none(self):
        assert get_conversation_title("nonexistent-uuid") is None


# ── Messages ───────────────────────────────────────────────────────────────────
class TestMessages:

    def test_append_and_retrieve(self):
        conv_id = create_conversation("testuser")
        append_message(conv_id, "user", "Hello!")
        append_message(conv_id, "model", "Hi there!")
        msgs = get_messages(conv_id, max_turns=10)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "Hello!"
        assert msgs[1]["role"] == "model"
        assert msgs[1]["content"] == "Hi there!"

    def test_empty_conversation_returns_empty_list(self):
        conv_id = create_conversation("testuser")
        assert get_messages(conv_id, max_turns=10) == []

    def test_max_turns_limits_results(self):
        conv_id = create_conversation("testuser")
        for i in range(10):
            append_message(conv_id, "user", f"Question {i}")
            append_message(conv_id, "model", f"Answer {i}")
        msgs = get_messages(conv_id, max_turns=3)
        # max_turns=3 → last 3 pairs → at most 6 messages
        assert len(msgs) <= 6

    def test_messages_are_ordered_asc(self):
        conv_id = create_conversation("testuser")
        for i in range(4):
            append_message(conv_id, "user", str(i))
        msgs = get_messages(conv_id, max_turns=10)
        contents = [m["content"] for m in msgs]
        assert contents == sorted(contents, key=lambda x: int(x))


# ── Auto-titling ───────────────────────────────────────────────────────────────
class TestAutoTitle:

    def test_title_set_from_first_message(self):
        conv_id = create_conversation("testuser")
        auto_title_from_first_message(conv_id, "Tell me about the Battle of Britain")
        assert get_conversation_title(conv_id) == "Tell me about the Battle of Britain"

    def test_title_truncated_at_60_chars(self):
        conv_id = create_conversation("testuser")
        long_msg = "A" * 100
        auto_title_from_first_message(conv_id, long_msg)
        title = get_conversation_title(conv_id)
        assert len(title) <= 60

    def test_title_not_overwritten_if_already_set(self):
        conv_id = create_conversation("testuser")
        auto_title_from_first_message(conv_id, "First message")
        auto_title_from_first_message(conv_id, "Second message — should be ignored")
        assert get_conversation_title(conv_id) == "First message"

    def test_short_message_becomes_title(self):
        conv_id = create_conversation("testuser")
        auto_title_from_first_message(conv_id, "Hi")
        assert get_conversation_title(conv_id) == "Hi"


# ── Search ─────────────────────────────────────────────────────────────────────
class TestSearch:

    def test_search_by_title(self):
        conv_id = create_conversation("testuser")
        auto_title_from_first_message(conv_id, "Battle of Britain research")
        results = search_conversations("testuser", "Britain")
        ids = [r["conv_id"] for r in results]
        assert conv_id in ids

    def test_search_by_message_content(self):
        conv_id = create_conversation("testuser")
        append_message(conv_id, "user", "quantum entanglement explanation")
        results = search_conversations("testuser", "quantum")
        ids = [r["conv_id"] for r in results]
        assert conv_id in ids

    def test_search_no_match_returns_empty(self):
        results = search_conversations("testuser", "zzznomatch_xyz")
        assert results == []


# ── File contexts ──────────────────────────────────────────────────────────────
class TestFileContexts:

    def test_store_and_retrieve(self):
        conv_id = create_conversation("testuser")
        file_id = store_file_context(conv_id, "testuser", "report.txt", "Some content here")
        contexts = get_file_contexts(conv_id)
        assert any(fc["file_id"] == file_id for fc in contexts)
        assert any(fc["filename"] == "report.txt" for fc in contexts)
        assert any(fc["extracted_text"] == "Some content here" for fc in contexts)

    def test_delete_file_context(self):
        conv_id = create_conversation("testuser")
        file_id = store_file_context(conv_id, "testuser", "delete_me.txt", "temp")
        delete_file_context(file_id)
        contexts = get_file_contexts(conv_id)
        assert not any(fc["file_id"] == file_id for fc in contexts)

    def test_multiple_files_per_conversation(self):
        conv_id = create_conversation("testuser")
        ids = [
            store_file_context(conv_id, "testuser", f"file{i}.txt", f"content {i}")
            for i in range(3)
        ]
        contexts = get_file_contexts(conv_id)
        stored_ids = [fc["file_id"] for fc in contexts]
        for fid in ids:
            assert fid in stored_ids

    def test_cascading_delete_removes_files(self):
        conv_id = create_conversation("testuser")
        store_file_context(conv_id, "testuser", "cascade.txt", "will be deleted")
        delete_conversation(conv_id)
        # File context should be gone because of ON DELETE CASCADE
        contexts = get_file_contexts(conv_id)
        assert contexts == []
