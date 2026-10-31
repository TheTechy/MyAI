"""
test_suite/test_bootstrap_user.py
=================================
The 'Setup' / PIN 0000 placeholder that init_db() creates on an empty DB must
go as soon as a real account exists — otherwise anyone can sign in with 0000 —
but never when it holds data or is the only way in.

Each test gets its own temp DB (core.db reads DB_PATH at call time).
"""
from __future__ import annotations

import pytest

import core.db as db


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "myai.db"))
    db.init_db()
    return db


def placeholder():
    return db.get_user_by_display_id(db.BOOTSTRAP_DISPLAY_ID)


# ── core.db ────────────────────────────────────────────────────────────────────
class TestPlaceholder:

    def test_empty_db_gets_placeholder_and_keeps_it(self, fresh_db):
        assert placeholder() is not None
        assert db.untouched_bootstrap_id() is None      # it's the only way in
        assert db.remove_untouched_bootstrap() is None
        assert placeholder() is not None

    def test_first_real_user_removes_it(self, fresh_db):
        db.create_user("George", "1234", "adult")
        assert placeholder() is None
        assert [u["display_id"] for u in db.list_users()] == ["George"]

    def test_kept_if_pin_changed(self, fresh_db):
        db.set_pin(placeholder()["user_id"], "1111")
        db.create_user("George", "1234", "adult")
        assert placeholder() is not None

    def test_kept_if_it_has_conversations(self, fresh_db):
        db.create_conversation(placeholder()["user_id"])
        db.create_user("George", "1234", "adult")
        assert placeholder() is not None

    def test_kept_if_it_has_memories(self, fresh_db):
        db.add_memory(placeholder()["user_id"], "likes tea")
        db.create_user("George", "1234", "adult")
        assert placeholder() is not None

    def test_kept_if_no_other_active_user(self, fresh_db):
        setup_id = placeholder()["user_id"]
        db.set_pin(setup_id, "1111")                     # keep it through the create…
        george = db.create_user("George", "1234", "adult")
        db.update_user(george, active=False)             # …then leave it the only active user
        db.set_pin(setup_id, db.BOOTSTRAP_PIN)
        assert db.untouched_bootstrap_id() is None

    def test_leftover_placeholder_can_be_removed(self, fresh_db):
        # An install from before this fix: real users created while 'Setup' survived.
        setup_id = placeholder()["user_id"]
        db.set_pin(setup_id, "1111")
        db.create_user("George", "1234", "adult")
        db.set_pin(setup_id, db.BOOTSTRAP_PIN)
        assert db.untouched_bootstrap_id() == setup_id
        assert db.remove_untouched_bootstrap() == setup_id
        assert placeholder() is None


# ── /user/create ───────────────────────────────────────────────────────────────
class TestUserCreateRoute:

    def test_signed_in_as_placeholder_is_signed_out(self, flask_app, fresh_db):
        with flask_app.test_client() as c:
            with c.session_transaction() as sess:
                sess["user_id"] = placeholder()["user_id"]
            r = c.post("/user/create", json={"display_id": "George", "pin": "1234"})
            assert r.status_code == 200
            body = r.get_json(force=True)
            assert body["ok"] and body["signed_out"] and body["display_id"] == "George"
            with c.session_transaction() as sess:
                assert "user_id" not in sess
        assert placeholder() is None

    def test_signed_in_real_user_stays_signed_in(self, flask_app, fresh_db):
        ann = db.create_user("Ann", "1234", "adult")
        with flask_app.test_client() as c:
            with c.session_transaction() as sess:
                sess["user_id"] = ann
            r = c.post("/user/create", json={"display_id": "Bob", "pin": "5678"})
            body = r.get_json(force=True)
            assert r.status_code == 200 and body["ok"] and "signed_out" not in body
            with c.session_transaction() as sess:
                assert sess["user_id"] == ann
