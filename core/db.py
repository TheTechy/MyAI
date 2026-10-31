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
DB_PATH = os.getenv("MYAIDB", "./instance/myai.db")

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
                display_id TEXT NOT NULL,
                pin        TEXT,
                age_band   TEXT NOT NULL DEFAULT 'adult'
                              CHECK(age_band IN ('child_young','child_teen','adult','senior')),
                active     INTEGER NOT NULL DEFAULT 1 CHECK(active IN (0,1)),
                created_at TEXT NOT NULL,
                last_login TEXT
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

            CREATE TABLE IF NOT EXISTS user_memories (
                memory_id  TEXT PRIMARY KEY,
                user_id    TEXT NOT NULL REFERENCES users(user_id),
                content    TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS message_files (
                file_id    TEXT PRIMARY KEY,
                msg_id     TEXT NOT NULL REFERENCES messages(msg_id) ON DELETE CASCADE,
                filename   TEXT NOT NULL,
                mime_type  TEXT NOT NULL,
                size       INTEGER NOT NULL,
                disk_path  TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_conv_user      ON conversations(user_id, updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_msg_conv       ON messages(conv_id, created_at ASC);
            CREATE INDEX IF NOT EXISTS idx_file_conv      ON file_context(conv_id, created_at ASC);
            CREATE INDEX IF NOT EXISTS idx_mem_user       ON user_memories(user_id, created_at ASC);
            CREATE INDEX IF NOT EXISTS idx_msgfile_msg    ON message_files(msg_id, created_at ASC);
        """)

    # ── Schema migrations for existing databases ──────────────────────────────
    # These add columns to pre-existing users tables. Idempotent: only runs
    # when the column is missing.
    with get_conn() as conn:
        existing = {
            row[1]
            for row in conn.execute("PRAGMA table_info(users)").fetchall()
        }
        # From very first DB shape — normalise the PIN column name to 'pin'.
        # Older schemas used 'pin_hash' (which stored plaintext anyway); we
        # add 'pin' if missing and copy any legacy values across during the
        # data migration below.
        if "pin" not in existing:
            conn.execute("ALTER TABLE users ADD COLUMN pin TEXT")
        # New columns for user management
        if "display_id" not in existing:
            conn.execute("ALTER TABLE users ADD COLUMN display_id TEXT")
        if "age_band" not in existing:
            conn.execute("ALTER TABLE users ADD COLUMN age_band TEXT DEFAULT 'adult'")
        if "active" not in existing:
            conn.execute("ALTER TABLE users ADD COLUMN active INTEGER DEFAULT 1")
        if "last_login" not in existing:
            conn.execute("ALTER TABLE users ADD COLUMN last_login TEXT")

    # Indexes that depend on new columns — created after the ALTERs above
    with get_conn() as conn:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_users_display ON users(LOWER(display_id))")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_users_active  ON users(active)")

    # ── Data migration ────────────────────────────────────────────────────────
    # If any row still has user_id that doesn't look like a UUID (i.e. it's a
    # display name like "Tata"), promote it: mint a UUID as the new primary
    # key and lift the old name into display_id. All FKs follow.
    #
    # If the DB is empty after migration, mint a single bootstrap user so the
    # system is usable. Beyond that, all user creation goes through the UI —
    # `.env USERS=` is no longer read.
    _migrate_users_to_uuid()


def _looks_like_uuid(value: str) -> bool:
    """True for a well-formed UUID4 string."""
    if not value or len(value) != 36:
        return False
    try:
        uuid.UUID(value)
        return True
    except (ValueError, AttributeError):
        return False


def _migrate_users_to_uuid():
    """
    One-time migration: convert plain-name user_ids into UUIDs.

    Steps:
      1. Find rows where user_id is NOT a UUID (legacy shape).
      2. For each, generate a UUID, lift the old name into display_id, backfill
         pin from pin_hash if that column exists.
      3. Update every FK in child tables to point at the new UUID.
      4. Delete the old row (was preserved during update since PK changed).
      5. If the DB ends up empty (very first run), create a bootstrap user
         so someone can sign in and manage users through the UI.

    Idempotent: no-op if every existing row already has a UUID primary key
    and at least one user exists.
    """
    with get_conn() as conn:
        # Check if pin_hash column exists (very old schema)
        columns = {r[1] for r in conn.execute("PRAGMA table_info(users)").fetchall()}
        has_pin_hash = "pin_hash" in columns

        # Copy any pin_hash values into pin before we touch anything else.
        # Old schema stored plaintext in pin_hash despite the name; this is
        # a straight column rename.
        if has_pin_hash:
            conn.execute(
                "UPDATE users SET pin = pin_hash WHERE pin IS NULL AND pin_hash IS NOT NULL"
            )

        rows = conn.execute(
            "SELECT user_id, display_id, pin FROM users"
        ).fetchall()

        legacy_rows = [r for r in rows if not _looks_like_uuid(r["user_id"])]

        if legacy_rows:
            print(f"[db] Migrating {len(legacy_rows)} legacy user(s) to UUID primary keys…")
            conn.execute("PRAGMA foreign_keys = OFF")
            try:
                for row in legacy_rows:
                    old_id      = row["user_id"]
                    new_id      = str(uuid.uuid4())
                    display_id  = row["display_id"] or old_id  # preserve original casing
                    pin_value   = row["pin"]

                    # Insert the new UUID row
                    conn.execute(
                        """INSERT INTO users(user_id, display_id, pin, age_band, active, created_at)
                           VALUES (?, ?, ?, 'adult', 1, ?)""",
                        (new_id, display_id, pin_value, _now()),
                    )

                    # Update FKs on child tables (parent still exists briefly)
                    conn.execute("UPDATE conversations  SET user_id = ? WHERE user_id = ?", (new_id, old_id))
                    conn.execute("UPDATE file_context   SET user_id = ? WHERE user_id = ?", (new_id, old_id))
                    conn.execute("UPDATE user_memories  SET user_id = ? WHERE user_id = ?", (new_id, old_id))

                    # Now drop the legacy parent row
                    conn.execute("DELETE FROM users WHERE user_id = ?", (old_id,))
            finally:
                conn.execute("PRAGMA foreign_keys = ON")
            print(f"[db] Legacy user migration complete.")

    # First-run bootstrap. If the DB has no users AT ALL after migration,
    # mint one to make the system usable — otherwise there's nobody to sign
    # in as. `.env USERS=` is intentionally ignored: the DB is the source
    # of truth. First thing the setup user should do after signing in is
    # rename themself, change the PIN, and add real users via the UI.
    with get_conn() as conn:
        user_count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        if user_count == 0:
            new_id = str(uuid.uuid4())
            conn.execute(
                """INSERT INTO users(user_id, display_id, pin, age_band, active, created_at)
                   VALUES (?, ?, ?, 'adult', 1, ?)""",
                (new_id, BOOTSTRAP_DISPLAY_ID, BOOTSTRAP_PIN, _now()),
            )
            print("[db] No users found — created bootstrap user 'Setup' with PIN 0000.")
            print("[db] Sign in, then rename this user and change the PIN via the Users page.")





# ── Users ──────────────────────────────────────────────────────────────────────
# Backwards-compat shim: some legacy callers still pass in what used to be a
# name-style user_id (e.g. from seed scripts). With the new schema we only ever
# have UUIDs in that column, so this becomes a no-op — user creation flows
# exclusively through create_user() below.
def ensure_user(user_id: str):
    """Deprecated. No-op retained so legacy call sites don't crash."""
    return


_VALID_AGE_BANDS = ("child_young", "child_teen", "adult", "senior")


def _row_to_user_dict(row: sqlite3.Row) -> dict:
    """Format a users row as an API-friendly dict."""
    return {
        "user_id":    row["user_id"],
        "display_id": row["display_id"],
        "age_band":   row["age_band"],
        "active":     bool(row["active"]),
        "has_pin":    bool(row["pin"]),
        "created_at": row["created_at"],
        "last_login": row["last_login"],
    }


def list_users(include_inactive: bool = False) -> list[dict]:
    """
    Return all users as dicts. By default excludes inactive (soft-deleted).
    Ordered by created_at ascending so display order is stable.
    """
    query = "SELECT * FROM users"
    if not include_inactive:
        query += " WHERE active = 1"
    query += " ORDER BY created_at ASC"

    with get_conn() as conn:
        rows = conn.execute(query).fetchall()
    return [_row_to_user_dict(r) for r in rows]


def get_user_by_id(user_id: str) -> dict | None:
    """Look up a user by UUID. Returns None if not found."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE user_id = ?", (user_id,)
        ).fetchone()
    return _row_to_user_dict(row) if row else None


def get_user_by_display_id(display_id: str) -> dict | None:
    """Look up a user by display_id (case-insensitive). Returns None if not found."""
    if not display_id:
        return None
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE LOWER(display_id) = LOWER(?)",
            (display_id.strip(),),
        ).fetchone()
    return _row_to_user_dict(row) if row else None


def is_display_id_available(display_id: str) -> bool:
    """
    True if the display_id is not already taken (case-insensitive).
    Used for the live 'name available?' check in the UI.
    """
    return get_user_by_display_id(display_id) is None


def count_active_users() -> int:
    """How many users have active = 1. Used to block deactivating the last one."""
    with get_conn() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM users WHERE active = 1"
        ).fetchone()[0]


# The placeholder init_db() creates on an empty DB. It's "untouched" while it's
# still called Setup, still has PIN 0000 and owns no data — and it should go as
# soon as a real account exists, or anyone can sign in with 0000.
BOOTSTRAP_DISPLAY_ID = "Setup"
BOOTSTRAP_PIN = "0000"


def untouched_bootstrap_id() -> str | None:
    """
    The placeholder's user_id if it's untouched AND another active user exists
    (so removing it can never leave nobody able to sign in); otherwise None.
    """
    with get_conn() as conn:
        row = conn.execute(
            """SELECT u.user_id FROM users u
               WHERE u.display_id = ? AND u.pin = ?
                 AND NOT EXISTS (SELECT 1 FROM conversations c WHERE c.user_id = u.user_id)
                 AND NOT EXISTS (SELECT 1 FROM user_memories m WHERE m.user_id = u.user_id)
                 AND NOT EXISTS (SELECT 1 FROM file_context  f WHERE f.user_id = u.user_id)
                 AND EXISTS (SELECT 1 FROM users o
                             WHERE o.user_id != u.user_id AND o.active = 1)""",
            (BOOTSTRAP_DISPLAY_ID, BOOTSTRAP_PIN),
        ).fetchone()
    return row["user_id"] if row else None


def remove_untouched_bootstrap() -> str | None:
    """Delete the untouched placeholder, if there is one. Returns its user_id."""
    user_id = untouched_bootstrap_id()
    if user_id:
        with get_conn() as conn:
            conn.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
    return user_id


def create_user(display_id: str, pin: str, age_band: str = "adult") -> str:
    """
    Create a new user. Returns the new user_id (UUID).
    Raises ValueError if display_id is already taken or age_band is invalid.

    Creating the first real account also removes the untouched 'Setup'
    placeholder (see remove_untouched_bootstrap). Callers signed in as that
    placeholder must check their session user still exists afterwards.
    """
    display_id = (display_id or "").strip()
    if not display_id:
        raise ValueError("display_id is required")
    if age_band not in _VALID_AGE_BANDS:
        raise ValueError(f"age_band must be one of {_VALID_AGE_BANDS}")
    if not is_display_id_available(display_id):
        raise ValueError(f"display_id '{display_id}' is already in use")

    new_id = str(uuid.uuid4())
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO users(user_id, display_id, pin, age_band, active, created_at)
               VALUES (?, ?, ?, ?, 1, ?)""",
            (new_id, display_id, pin, age_band, _now()),
        )
    remove_untouched_bootstrap()
    return new_id


def update_user(
    user_id: str,
    display_id: str | None = None,
    pin: str | None = None,
    age_band: str | None = None,
    active: bool | None = None,
) -> bool:
    """
    Partial update — only fields explicitly passed get changed.
    Returns True if a row was updated, False if the user doesn't exist.

    Enforces:
      * display_id must remain unique (case-insensitive) if changed
      * age_band must be valid
      * cannot deactivate the last active user
    """
    existing = get_user_by_id(user_id)
    if not existing:
        return False

    fields = []
    values = []

    if display_id is not None:
        display_id = display_id.strip()
        if not display_id:
            raise ValueError("display_id cannot be blank")
        # Only enforce uniqueness if the name is actually changing
        if display_id.lower() != (existing["display_id"] or "").lower():
            if not is_display_id_available(display_id):
                raise ValueError(f"display_id '{display_id}' is already in use")
        fields.append("display_id = ?")
        values.append(display_id)

    if pin is not None:
        fields.append("pin = ?")
        values.append(pin)

    if age_band is not None:
        if age_band not in _VALID_AGE_BANDS:
            raise ValueError(f"age_band must be one of {_VALID_AGE_BANDS}")
        fields.append("age_band = ?")
        values.append(age_band)

    if active is not None:
        active_int = 1 if active else 0
        # Refuse to deactivate the last active user
        if active_int == 0 and existing["active"] and count_active_users() <= 1:
            raise ValueError("Cannot deactivate the last active user")
        fields.append("active = ?")
        values.append(active_int)

    if not fields:
        return True  # nothing to do, but the user exists

    values.append(user_id)
    with get_conn() as conn:
        conn.execute(
            f"UPDATE users SET {', '.join(fields)} WHERE user_id = ?",
            values,
        )
    return True


def touch_last_login(user_id: str):
    """Update last_login to now. Called from /auth on successful sign-in."""
    with get_conn() as conn:
        conn.execute(
            "UPDATE users SET last_login = ? WHERE user_id = ?",
            (_now(), user_id),
        )


# ── PIN management ─────────────────────────────────────────────────────────────
def set_pin(user_id: str, pin: str):
    """
    Store a 4-digit PIN for a user (plaintext — matches current implementation).
    user_id must be an existing UUID.
    """
    with get_conn() as conn:
        conn.execute(
            "UPDATE users SET pin = ? WHERE user_id = ?",
            (pin, user_id),
        )


def verify_pin(user_id: str, pin: str) -> bool:
    """
    Verify PIN against a stored user_id (UUID).
    Returns False if the user doesn't exist, has no PIN, is inactive, or the
    PIN doesn't match.
    """
    with get_conn() as conn:
        row = conn.execute(
            "SELECT pin, active FROM users WHERE user_id = ?", (user_id,)
        ).fetchone()

    if not row or not row["pin"] or not row["active"]:
        return False
    return row["pin"] == pin


def verify_pin_by_display_id(display_id: str, pin: str) -> str | None:
    """
    Convenience: look up a user by display_id, verify PIN, return their
    user_id (UUID) on success or None on failure.
    Used by the /auth route so the UI never has to touch UUIDs.
    """
    user = get_user_by_display_id(display_id)
    if not user or not user["active"] or not user["has_pin"]:
        return None
    if verify_pin(user["user_id"], pin):
        return user["user_id"]
    return None


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


def search_conversations(user_id: str, query: str) -> list[dict]:
    """
    Search conversations by title or message content for a user.
    Returns matches newest-first, deduplicated by conv_id.
    """
    pattern = f"%{query}%"
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT DISTINCT c.conv_id, c.title, c.updated_at
               FROM conversations c
               LEFT JOIN messages m ON c.conv_id = m.conv_id
               WHERE c.user_id = ?
               AND (c.title LIKE ? OR m.content LIKE ?)
               ORDER BY c.updated_at DESC""",
            (user_id, pattern, pattern),
        ).fetchall()
    return [dict(r) for r in rows]


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
def append_message(conv_id: str, role: str, content: str) -> str:
    """
    Append one message and bump the conversation's updated_at timestamp.
    Returns the new msg_id so callers can link related rows (e.g. message_files).
    """
    msg_id = str(uuid.uuid4())
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO messages(msg_id, conv_id, role, content, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (msg_id, conv_id, role, content, _now()),
        )
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE conv_id = ?",
            (_now(), conv_id),
        )
    return msg_id


def get_messages(conv_id: str, max_turns: int = 10) -> list[dict]:
    """
    Return the last max_turns pairs of messages as a list of
    {"msg_id": ..., "role": ..., "content": ..., "files": [...]} dicts —
    ready to drop into your prompt builder or render in the UI.

    The "files" list contains any download cards attached to the message:
        [{ "file_id", "filename", "mime_type", "size" }, ...]
    Empty list when no files are attached.
    """
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT msg_id, role, content FROM messages
               WHERE conv_id = ?
               ORDER BY created_at ASC""",
            (conv_id,),
        ).fetchall()

        pairs = [dict(r) for r in rows]

        # Trim to last N turns (each turn = user + model = 2 rows)
        max_entries = max_turns * 2
        if len(pairs) > max_entries:
            pairs = pairs[-max_entries:]

        # Attach any message_files in a single query, then group by msg_id
        if pairs:
            msg_ids = [p["msg_id"] for p in pairs]
            placeholders = ",".join("?" * len(msg_ids))
            file_rows = conn.execute(
                f"""SELECT file_id, msg_id, filename, mime_type, size
                    FROM message_files
                    WHERE msg_id IN ({placeholders})
                    ORDER BY created_at ASC""",
                msg_ids,
            ).fetchall()

            files_by_msg: dict[str, list[dict]] = {}
            for fr in file_rows:
                files_by_msg.setdefault(fr["msg_id"], []).append({
                    "file_id":   fr["file_id"],
                    "filename":  fr["filename"],
                    "mime_type": fr["mime_type"],
                    "size":      fr["size"],
                })

            for p in pairs:
                p["files"] = files_by_msg.get(p["msg_id"], [])
        return pairs


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


# ── Message files (generated download cards) ──────────────────────────────────
# Persistent record of files emitted by the LLM as [FILE:] blocks. The bytes
# themselves live in generated_files/; this table just remembers which message
# they belong to so the download card can be restored when the conversation is
# reloaded.
def store_message_file(
    msg_id: str,
    filename: str,
    mime_type: str,
    size: int,
    disk_path: str,
) -> str:
    """Persist one generated file linked to *msg_id*. Returns the new file_id."""
    file_id = str(uuid.uuid4())
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO message_files(file_id, msg_id, filename, mime_type, size, disk_path, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (file_id, msg_id, filename, mime_type, size, disk_path, _now()),
        )
    return file_id


def get_message_file(file_id: str) -> dict | None:
    """
    Return the row for a single message_file, including the on-disk path
    and the conv_id / user_id (via JOIN) so the download route can verify
    ownership before serving the bytes.
    """
    with get_conn() as conn:
        row = conn.execute(
            """SELECT mf.file_id, mf.msg_id, mf.filename, mf.mime_type,
                      mf.size, mf.disk_path, mf.created_at,
                      m.conv_id, c.user_id
               FROM message_files mf
               JOIN messages       m ON mf.msg_id = m.msg_id
               JOIN conversations  c ON m.conv_id = c.conv_id
               WHERE mf.file_id = ?""",
            (file_id,),
        ).fetchone()
    return dict(row) if row else None


# ── User memories ──────────────────────────────────────────────────────────────
# Explicit, user-stated facts ("remember I'm allergic to peanuts"). Per-user
# isolated — every helper is scoped by user_id so one user never sees another's.
def add_memory(user_id: str, content: str) -> str:
    """Store one memory for a user and return its memory_id."""
    ensure_user(user_id)
    memory_id = str(uuid.uuid4())
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO user_memories(memory_id, user_id, content, created_at)
               VALUES (?, ?, ?, ?)""",
            (memory_id, user_id, content, _now()),
        )
    return memory_id


def list_memories(user_id: str) -> list[dict]:
    """Return all memories for a user, oldest first."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT memory_id, content, created_at
               FROM user_memories
               WHERE user_id = ?
               ORDER BY created_at ASC""",
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def delete_memory(user_id: str, memory_id: str) -> bool:
    """Delete a single memory by id, scoped to the owning user. Returns True if a row was removed."""
    with get_conn() as conn:
        cur = conn.execute(
            "DELETE FROM user_memories WHERE memory_id = ? AND user_id = ?",
            (memory_id, user_id),
        )
        return cur.rowcount > 0


def delete_memories_matching(user_id: str, term: str) -> list[dict]:
    """
    Delete memories whose content contains *term* (case-insensitive substring),
    scoped to the user. Returns the list of deleted {memory_id, content} dicts.
    """
    term = term.strip()
    if not term:
        return []
    pattern = f"%{term}%"
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT memory_id, content FROM user_memories
               WHERE user_id = ? AND content LIKE ? COLLATE NOCASE""",
            (user_id, pattern),
        ).fetchall()
        matched = [dict(r) for r in rows]
        if matched:
            conn.executemany(
                "DELETE FROM user_memories WHERE memory_id = ? AND user_id = ?",
                [(r["memory_id"], user_id) for r in matched],
            )
    return matched


def clear_memories(user_id: str) -> int:
    """Delete all memories for a user. Returns the number removed."""
    with get_conn() as conn:
        cur = conn.execute(
            "DELETE FROM user_memories WHERE user_id = ?", (user_id,)
        )
        return cur.rowcount


# ── Helpers ────────────────────────────────────────────────────────────────────
def _now() -> str:
    return datetime.now(timezone.utc).isoformat()