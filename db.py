"""
db.py
==================
"""

import os
import sqlite3
import uuid
from datetime import datetime, timezone
from contextlib import contextmanager
# from werkzeug.security import generate_password_hash, check_password_hash

# DB_PATH = "chat_history.db"
DB_PATH = os.getenv("MYAIDB", "./DB/myai.db")

# ── Connection ─────────────────────────────────────────────────────────────────
@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # safer concurrent reads
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Schema ─────────────────────────────────────────────────────────────────────
def init_db():
    """Create tables if they don't exist. Call once at startup."""
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                user_id    TEXT PRIMARY KEY,
                pin_hash   TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS conversations (
                conv_id    TEXT PRIMARY KEY,
                user_id    TEXT NOT NULL REFERENCES users(user_id),
                title      TEXT NOT NULL DEFAULT 'New conversation',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                msg_id     TEXT PRIMARY KEY,
                conv_id    TEXT NOT NULL REFERENCES conversations(conv_id) ON DELETE CASCADE,
                role       TEXT NOT NULL CHECK(role IN ('user', 'model')),
                content    TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS file_context (
                file_id        TEXT PRIMARY KEY,
                conv_id        TEXT NOT NULL REFERENCES conversations(conv_id) ON DELETE CASCADE,
                user_id        TEXT NOT NULL REFERENCES users(user_id),
                filename       TEXT NOT NULL,
                extracted_text TEXT NOT NULL,
                created_at     TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_conv_user   ON conversations(user_id, updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_msg_conv    ON messages(conv_id, created_at ASC);
            CREATE INDEX IF NOT EXISTS idx_file_conv   ON file_context(conv_id, created_at ASC);
        """)

    # Migration: add pin_hash to existing databases that predate this column
    with get_conn() as conn:
        existing = {
            row[1]
            for row in conn.execute("PRAGMA table_info(users)").fetchall()
        }
        if "pin_hash" not in existing:
            conn.execute("ALTER TABLE users ADD COLUMN pin_hash TEXT")


# ── Users ──────────────────────────────────────────────────────────────────────
def ensure_user(user_id: str):
    """Create user row if it doesn't already exist."""
    with get_conn() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO users(user_id, created_at) VALUES (?, ?)",
            (user_id, _now()),
        )


# ── PIN management ─────────────────────────────────────────────────────────────
def set_pin(user_id: str, pin: str):
    """
    Hash and store a 4-digit PIN for a user.
    Creates the user row first if it doesn't exist.
    Called by seed_pins.py — not exposed via the API.
    """
    ensure_user(user_id)
    pin_hash = generate_password_hash(pin)
    with get_conn() as conn:
        conn.execute(
            "UPDATE users SET pin_hash = ? WHERE user_id = ?",
            (pin_hash, user_id),
        )


def verify_pin(user_id: str, pin: str) -> bool:
    """
    Return True if *pin* matches the stored hash for *user_id*.
    Returns False for unknown users or users with no PIN set.
    """
    with get_conn() as conn:
        row = conn.execute(
            "SELECT pin_hash FROM users WHERE user_id = ?", (user_id,)
        ).fetchone()

    if not row or not row["pin_hash"]:
        return False

    if row["pin_hash"] == pin:
        return True

# ── Conversations ──────────────────────────────────────────────────────────────
def create_conversation(user_id: str, title: str = "New conversation") -> str:
    """Create a new conversation and return its conv_id."""
    ensure_user(user_id)
    conv_id = str(uuid.uuid4())
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO conversations(conv_id, user_id, title, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?)""",
            (conv_id, user_id, title, _now(), _now()),
        )
    return conv_id


def list_conversations(user_id: str) -> list[dict]:
    """Return all conversations for a user, newest first."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT conv_id, title, created_at, updated_at
               FROM conversations
               WHERE user_id = ?
               ORDER BY updated_at DESC""",
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_conversation_title(conv_id: str) -> str | None:
    """Return the current title for a conversation, or None if not found."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT title FROM conversations WHERE conv_id = ?", (conv_id,)
        ).fetchone()
    return row["title"] if row else None


def rename_conversation(conv_id: str, new_title: str):
    with get_conn() as conn:
        conn.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE conv_id = ?",
            (new_title, _now(), conv_id),
        )


def delete_conversation(conv_id: str):
    """Delete a conversation and all its messages (CASCADE handles messages)."""
    with get_conn() as conn:
        conn.execute("DELETE FROM conversations WHERE conv_id = ?", (conv_id,))


# ── Messages ───────────────────────────────────────────────────────────────────
def append_message(conv_id: str, role: str, content: str):
    """Append one message and bump the conversation's updated_at timestamp."""
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO messages(msg_id, conv_id, role, content, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (str(uuid.uuid4()), conv_id, role, content, _now()),
        )
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE conv_id = ?",
            (_now(), conv_id),
        )


def get_messages(conv_id: str, max_turns: int = 10) -> list[dict]:
    """
    Return the last max_turns pairs of messages as a list of
    {"role": ..., "content": ...} dicts — ready to drop into your prompt builder.
    """
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT role, content FROM messages
               WHERE conv_id = ?
               ORDER BY created_at ASC""",
            (conv_id,),
        ).fetchall()

    pairs = [dict(r) for r in rows]

    # Trim to last N turns (each turn = user + model = 2 rows)
    max_entries = max_turns * 2
    return pairs[-max_entries:] if len(pairs) > max_entries else pairs


def auto_title_from_first_message(conv_id: str, user_message: str, max_len: int = 60):
    """Set the conversation title from the first user message if still default."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT title FROM conversations WHERE conv_id = ?", (conv_id,)
        ).fetchone()
        if row and row["title"] == "New conversation":
            title = user_message[:max_len].strip()
            conn.execute(
                "UPDATE conversations SET title = ? WHERE conv_id = ?",
                (title, conv_id),
            )


# ── File context ───────────────────────────────────────────────────────────────
def store_file_context(conv_id: str, user_id: str, filename: str, extracted_text: str) -> str:
    """
    Persist extracted file text for a conversation and return the new file_id.
    """
    file_id = str(uuid.uuid4())
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO file_context(file_id, conv_id, user_id, filename, extracted_text, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (file_id, conv_id, user_id, filename, extracted_text, _now()),
        )
    return file_id


def get_file_contexts(conv_id: str) -> list[dict]:
    """
    Return all file contexts attached to a conversation, oldest first.
    Each dict has keys: file_id, filename, extracted_text.
    """
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT file_id, filename, extracted_text
               FROM file_context
               WHERE conv_id = ?
               ORDER BY created_at ASC""",
            (conv_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def delete_file_context(file_id: str):
    """Remove a single file context entry."""
    with get_conn() as conn:
        conn.execute("DELETE FROM file_context WHERE file_id = ?", (file_id,))


# ── Helpers ────────────────────────────────────────────────────────────────────
def _now() -> str:
    return datetime.now(timezone.utc).isoformat()