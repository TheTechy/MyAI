import os
import re
import json
import uuid
from pathlib import Path
from flask import Flask, request, render_template, Response, stream_with_context, session, redirect, url_for, send_file, abort
from werkzeug.utils import secure_filename
#from core.llm_inference import ask_stream, warmup
from core.inference.engine import ask_stream, warmup
from core.file_ingestion import extract_text, ExtractionError, SUPPORTED_EXTENSIONS, IMAGE_EXTENSIONS
from waitress import serve
from dotenv import load_dotenv
from core.db import (
	create_conversation,
	list_conversations,
	search_conversations,
	delete_conversation,
	get_messages,
	verify_pin,
	verify_pin_by_display_id,
	list_users,
	touch_last_login,
	create_user,
	get_user_by_id,
	update_user,
	is_display_id_available,
	init_db,
	store_file_context,
	delete_file_context,
	list_memories,
	delete_memory,
	clear_memories,
	get_message_file,
)

load_dotenv()  # loads from .env in the current directory

# ── Version ────────────────────────────────────────────────────────────────────
VERSION = "0.1.0"


# ── Pretty model name ──────────────────────────────────────────────────────────
def model_display_name(model_path: str) -> str:
	"""
	Turn a MODEL env var like './models/gemma-4-E4B-it-UD-Q8_K_XL.gguf' into a
	short human-readable label like 'Gemma 4 E4B' for the header badge and
	startup banner. Strips path, extension, quant suffixes, and role tags.
	"""
	if not model_path:
		return "Unknown"

	name = Path(model_path).stem

	# Strip everything from the first quant marker to the end of the name.
	name = re.sub(r'[-_.](?:UD[-_]?)?Q\d.*$', '', name, flags=re.IGNORECASE)
	name = re.sub(r'[-_.](?:F16|FP16|BF16|F32|GGUF)(?:[-_.].*)?$', '', name, flags=re.IGNORECASE)
	# Role tags as trailing tokens only
	name = re.sub(r'[-_.](?:it|instruct|chat|base)$', '', name, flags=re.IGNORECASE)

	# Convert separators to spaces, but ONLY for hyphens and underscores —
	# keep dots so version strings like '3.5' or '4.1' stay intact.
	name = re.sub(r'[-_]+', ' ', name).strip()

	parts = []
	for token in name.split():
		# Pure-version tokens (digits and dots): leave intact (e.g. '3.5', '4.1')
		if re.match(r'^[\d.]+$', token):
			parts.append(token)
		# Mixed alphanumeric tokens (E4B, R1, 14B, 9B, 8b): uppercase
		elif re.match(r'^[A-Za-z]*\d+[A-Za-z]*$', token):
			parts.append(token.upper())
		# Regular word
		else:
			parts.append(token.capitalize())

	return ' '.join(parts) or "Unknown"


# ── Load / instantiate variables ───────────────────────────────────────────────
ALLOWED_EXTENSIONS = {ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS}
MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5 MB
# NOTE: `.env USERS=` is now legacy — the DB is the source of truth for users.
# Kept only so the startup banner can still show a rough "N configured" count
# without hitting the DB before init_db() has been called.
users = os.getenv("USERS", "").split(",")
port = os.getenv("PORT")
if not port:
    raise ValueError("PORT environment variable not set")

app = Flask(__name__, template_folder='templates')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# Required for sessions. Set SECRET_KEY in your .env file.
# Generate a strong one with: python -c "import secrets; print(secrets.token_hex(32))"
app.secret_key = os.getenv("SECRET_KEY", "CHANGE-THIS-WHEN-DEPLOYED")

# —— Initialise app ───────────────────────────────────────────────────────────—
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Directory for uploaded images (raw bytes, not stored in DB)
IMAGE_UPLOAD_DIR = Path(os.getenv("FILE_OUTPUT_DIR", "generated_files")) / "uploads"
IMAGE_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


init_db()

# Pre-load Whisper model at startup so the first transcription isn't slow
def _preload_whisper():
    try:
        from faster_whisper import WhisperModel as _W
        global _whisper_model
        model_size   = os.getenv("WHISPER_MODEL",   "base.en")
        device       = os.getenv("WHISPER_DEVICE",  "cpu")
        compute_type = os.getenv("WHISPER_COMPUTE", "int8")
        _whisper_model = _W(model_size, device=device, compute_type=compute_type)
        print(f"Whisper model '{model_size}' loaded.")
    except Exception as e:
        print(f"Whisper pre-load skipped: {e}")
        return ""

# Module-level cache so the model is only loaded once per server process.
_whisper_model = None
_preload_whisper()

@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, private, max-age=0"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    return r


# ── Auth helper ────────────────────────────────────────────────────────────────
def is_authenticated() -> bool:
    return "user_id" in session


# ── Page routes ────────────────────────────────────────────────────────────────
@app.route('/', methods=['GET'])
def index():
    # Login dropdown shows ACTIVE users only. Soft-deleted (inactive) users
    # remain in the DB for audit / undo but aren't sign-in-able.
    active_users = list_users(include_inactive=False)
    display_names = [u["display_id"] for u in active_users]
    return render_template('index.html', users=display_names)


@app.route('/chat', methods=['GET'])
def chat():
    if not is_authenticated():
        return redirect(url_for('index'))
    return render_template(
        'chat.html',
        model_name=model_display_name(os.getenv("MODEL", "")),
    )


@app.route('/logout', methods=['GET'])
def logout():
    session.clear()
    return redirect(url_for('index'))


@app.route('/memories', methods=['GET'])
def memories_page():
    if not is_authenticated():
        return redirect(url_for('index'))
    return render_template('memories.html')


@app.route('/users', methods=['GET'])
def users_page():
    if not is_authenticated():
        return redirect(url_for('index'))
    return render_template('users.html')


# ── Authentication route ───────────────────────────────────────────────────────
@app.route('/auth', methods=['POST'])
def auth():
    """
    Verify a user's 4-digit PIN.

    Request JSON : { "display_id": "Bob", "pin": "1234" }
                   (also accepts "user_id" for backwards compat with older clients)
    Response JSON: { "ok": true }  or  { "error": "..." }  (HTTP 401)

    On success:
      * session["user_id"] is set to the user's UUID (opaque to the client)
      * last_login is refreshed
    """
    data = request.get_json()
    if not data:
        return json.dumps({"error": "display_id and pin required"}), 400

    # Accept either 'display_id' (new) or 'user_id' (legacy) as the name field.
    # The value here is a display name like "Bob", not the underlying UUID.
    display_name = (data.get("display_id") or data.get("user_id") or "").strip()
    pin          = str(data.get("pin", "")).strip()

    if not display_name or not pin:
        return json.dumps({"error": "display_id and pin required"}), 400

    # Look up by display_id (case-insensitive), verify PIN, get back the UUID.
    # Same generic error whether the user is unknown, inactive, or the PIN is
    # wrong — no leaking of which usernames exist.
    user_uuid = verify_pin_by_display_id(display_name, pin)
    if not user_uuid:
        return json.dumps({"error": "Invalid PIN"}), 401

    session["user_id"] = user_uuid
    touch_last_login(user_uuid)
    return json.dumps({"ok": True})


# ── Conversation management routes ─────────────────────────────────────────────
@app.route('/conversations', methods=['POST'])
def new_conversation():
    """
    Create a new conversation for the signed-in user.

    Response JSON: { "conv_id": "<uuid>" }
    """
    if not is_authenticated():
        return json.dumps({"error": "Unauthorized"}), 401

    # No client-supplied user_id — the session is the source of truth so users
    # can only ever create conversations belonging to themselves.
    conv_id = create_conversation(session["user_id"])
    return json.dumps({"conv_id": conv_id})


@app.route('/conversations/<user_id>', methods=['GET'])
def get_conversations(user_id: str):
    """
    List all conversations for the signed-in user, newest first.
    (The URL parameter is legacy and ignored — session is authoritative.)

    Response JSON: [{ "conv_id": ..., "title": ..., "updated_at": ... }, ...]
    """
    if not is_authenticated():
        return json.dumps({"error": "Unauthorized"}), 401

    return json.dumps(list_conversations(session["user_id"]))


@app.route('/conversations/<user_id>/search', methods=['GET'])
def search_conversations_route(user_id: str):
    """
    Search conversations by title or message content.
    (The URL parameter is legacy and ignored — session is authoritative.)

    Query param: ?q=<search term>
    Response JSON: [{ "conv_id": ..., "title": ..., "updated_at": ... }, ...]
    """
    if not is_authenticated():
        return json.dumps({"error": "Unauthorized"}), 401

    query = request.args.get('q', '').strip()
    if not query:
        return json.dumps(list_conversations(session["user_id"]))

    return json.dumps(search_conversations(session["user_id"], query))


@app.route('/conversations/<conv_id>/messages', methods=['GET'])
def get_conversation_messages(conv_id: str):
    """
    Return all messages for a conversation (used to render past chats in the UI).

    Response JSON: [{ "msg_id": ..., "role": "user"|"model", "content": "...",
                      "files": [{ "file_id", "filename", "mime_type", "size" }] }, ...]
    """
    if not is_authenticated():
        return json.dumps({"error": "Unauthorized"}), 401

    messages = get_messages(conv_id, max_turns=9999)
    return json.dumps(messages)


@app.route('/files/<file_id>/download', methods=['GET'])
def download_message_file(file_id: str):
    """
    Stream a previously-generated file back to the client.
    Only the owning user (the one who originally generated it) may download.
    """
    if not is_authenticated():
        return json.dumps({"error": "Unauthorized"}), 401

    row = get_message_file(file_id)
    if not row:
        abort(404)

    # Ownership check — the file's conversation must belong to the session user
    if row["user_id"] != session.get("user_id"):
        abort(403)

    disk_path = Path(row["disk_path"])
    if not disk_path.is_file():
        abort(404)

    return send_file(
        disk_path,
        mimetype=row["mime_type"],
        as_attachment=True,
        download_name=row["filename"],
    )

@app.route('/conversations/<conv_id>', methods=['DELETE'])
def remove_conversation(conv_id: str):
    """
    Delete a conversation and all its messages.
    """
    if not is_authenticated():
        return json.dumps({"error": "Unauthorized"}), 401

    delete_conversation(conv_id)
    return json.dumps({"deleted": conv_id})


# ── Memory management routes ───────────────────────────────────────────────────
# All scoped to the signed-in user via session["user_id"] — never trust a
# client-supplied id, so users can only ever see or delete their own memories.
@app.route('/api/memories', methods=['GET'])
def api_list_memories():
    """Return the signed-in user's memories, oldest first."""
    if not is_authenticated():
        return json.dumps({"error": "Unauthorized"}), 401

    return json.dumps(list_memories(session["user_id"]))


@app.route('/api/memories/<memory_id>', methods=['DELETE'])
def api_delete_memory(memory_id: str):
    """Delete a single memory belonging to the signed-in user."""
    if not is_authenticated():
        return json.dumps({"error": "Unauthorized"}), 401

    removed = delete_memory(session["user_id"], memory_id)
    if not removed:
        return json.dumps({"error": "Not found"}), 404
    return json.dumps({"deleted": memory_id})


@app.route('/api/memories', methods=['DELETE'])
def api_clear_memories():
    """Delete all memories belonging to the signed-in user."""
    if not is_authenticated():
        return json.dumps({"error": "Unauthorized"}), 401

    removed = clear_memories(session["user_id"])
    return json.dumps({"cleared": removed})


# ── User management (CRUD) API ─────────────────────────────────────────────────
# Any signed-in user can manage users. Kept intentionally simple for family scale
# — if you later want an admin role or PIN re-confirmation on destructive
# actions, add it here.
#
# Never returns the raw pin field — only has_pin (boolean) — so a compromised
# session can't leak PINs even in transit.

def _validate_pin_or_none(pin) -> str | None:
    """Return a normalised 4-digit PIN, or raise ValueError if malformed."""
    if pin is None or pin == "":
        return None
    pin = str(pin).strip()
    if not (pin.isdigit() and len(pin) == 4):
        raise ValueError("PIN must be exactly 4 digits")
    return pin


@app.route('/user/create', methods=['POST'])
def user_create():
    """
    Create a new user.

    Body : { "display_id": "Bob", "pin": "1234", "age_band": "adult" }
    Returns: { "ok": true, "user_id": "<uuid>", "display_id": "Bob", ... }
    """
    if not is_authenticated():
        return json.dumps({"error": "Unauthorized"}), 401

    data = request.get_json(silent=True) or {}
    display_id = (data.get("display_id") or "").strip()
    age_band   = (data.get("age_band") or "adult").strip()

    try:
        pin = _validate_pin_or_none(data.get("pin"))
    except ValueError as e:
        return json.dumps({"error": str(e)}), 400

    if not display_id:
        return json.dumps({"error": "display_id is required"}), 400
    if not pin:
        return json.dumps({"error": "pin is required"}), 400

    signed_in_user_existed = get_user_by_id(session["user_id"]) is not None
    try:
        new_id = create_user(display_id, pin, age_band)
    except ValueError as e:
        return json.dumps({"error": str(e)}), 400

    user = get_user_by_id(new_id)
    # Creating the first real account removes the 'Setup' placeholder. If that's
    # who is signed in, end the session now — it points at a user that's gone.
    if signed_in_user_existed and get_user_by_id(session["user_id"]) is None:
        session.clear()
        return json.dumps({"ok": True, "signed_out": True, **user})
    return json.dumps({"ok": True, **user})


@app.route('/user/read', methods=['GET'])
def user_read_all():
    """
    List all users, active and inactive.
    Returns: [ { user_id, display_id, age_band, active, has_pin, created_at, last_login }, ... ]
    """
    if not is_authenticated():
        return json.dumps({"error": "Unauthorized"}), 401

    return json.dumps(list_users(include_inactive=True))


@app.route('/user/read/<user_id>', methods=['GET'])
def user_read_one(user_id: str):
    """Return a single user by UUID, or 404 if not found."""
    if not is_authenticated():
        return json.dumps({"error": "Unauthorized"}), 401

    user = get_user_by_id(user_id)
    if not user:
        return json.dumps({"error": "User not found"}), 404
    return json.dumps(user)


@app.route('/user/update/<user_id>', methods=['POST'])
def user_update(user_id: str):
    """
    Partial update. Only fields present in the body are changed.

    Body : any subset of { display_id, pin, age_band, active }
    Returns: { "ok": true, ...updated user... }
    """
    if not is_authenticated():
        return json.dumps({"error": "Unauthorized"}), 401

    existing = get_user_by_id(user_id)
    if not existing:
        return json.dumps({"error": "User not found"}), 404

    data = request.get_json(silent=True) or {}

    kwargs: dict = {}
    if "display_id" in data:
        kwargs["display_id"] = (data["display_id"] or "").strip()
    if "age_band" in data:
        kwargs["age_band"] = data["age_band"]
    if "active" in data:
        kwargs["active"] = bool(data["active"])
    if "pin" in data:
        try:
            new_pin = _validate_pin_or_none(data.get("pin"))
        except ValueError as e:
            return json.dumps({"error": str(e)}), 400
        if new_pin is None:
            return json.dumps({"error": "pin cannot be empty"}), 400
        kwargs["pin"] = new_pin

    try:
        update_user(user_id, **kwargs)
    except ValueError as e:
        return json.dumps({"error": str(e)}), 400

    return json.dumps({"ok": True, **get_user_by_id(user_id)})


@app.route('/user/check-display-id', methods=['GET'])
def user_check_display_id():
    """
    Real-time availability check for the create-user UI.

    Query param: ?name=<candidate>
    Returns    : { "available": true|false }

    Case-insensitive. Available means: no existing user (active or inactive)
    has this display_id.
    """
    if not is_authenticated():
        return json.dumps({"error": "Unauthorized"}), 401

    name = (request.args.get("name") or "").strip()
    if not name:
        return json.dumps({"available": False, "reason": "name required"}), 400

    return json.dumps({"available": is_display_id_available(name)})


# ── Prompt / inference route ───────────────────────────────────────────────────
@app.route('/prompt', methods=['POST'])
def execute_prompt():
    """
    Stream a response to a user message.

    Request JSON: { "prompt": "...", "name": "Bob", "conv_id": "<uuid>" }
    """
    if not is_authenticated():
        return json.dumps({"error": "Unauthorized"}), 401

    data = request.get_json()

    if not data or "prompt" not in data:
        return json.dumps({"error": "Invalid request data"}), 400

    prompt = data["prompt"].strip()
    if not prompt:
        return json.dumps({"error": "Empty prompt"}), 400

    user_id = session["user_id"]  # UUID — authoritative, never trust client
    conv_id = data.get("conv_id")

    # Auto-create a conversation if the client didn't supply one
    new_conv = conv_id is None
    if new_conv:
        conv_id = create_conversation(user_id)

    def generate():
        # Tell the client the conv_id so it can attach future messages to it
        if new_conv:
            yield f"event: conv_id\ndata: {json.dumps({'conv_id': conv_id})}\n\n"
        yield from ask_stream(prompt, user_id, conv_id)

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'X-Accel-Buffering': 'no',
            'Cache-Control': 'no-cache',
        }
    )


# ── File upload route ──────────────────────────────────────────────────────────
@app.route("/upload", methods=["POST"])
def upload_file():
    """
    Upload a file, extract its text, and store the context in the DB.

    Request : multipart/form-data with fields:
                "file"    — the file itself
                "conv_id" — conversation to attach it to
    Response: { "file_id": "...", "filename": "...", "chars": 1234 }
    """
    if not is_authenticated():
        return json.dumps({"error": "Unauthorized"}), 401

    if "file" not in request.files:
        return json.dumps({"error": "No file part in request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return json.dumps({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return json.dumps({"error": "File type not allowed"}), 415

    conv_id = request.form.get("conv_id")
    if not conv_id:
        return json.dumps({"error": "conv_id required"}), 400

    user_id  = session["user_id"]
    filename = secure_filename(file.filename)
    ext      = "." + filename.rsplit(".", 1)[1].lower()

    file_bytes = file.read()

    # ── Images: save to disk, store only metadata in DB ───────────────────────
    if ext in IMAGE_EXTENSIONS:
        try:
            from PIL import Image as PilImage
            import io as _io
            img      = PilImage.open(_io.BytesIO(file_bytes))
            img.load()
            fmt      = img.format or ext.lstrip(".").upper()
            dims     = f"{img.width}x{img.height}"
            mode     = img.mode
        except Exception:
            fmt, dims, mode = ext.lstrip(".").upper(), "unknown", "unknown"

        size_kb  = len(file_bytes) / 1024
        size_str = f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb/1024:.1f} MB"

        # Save raw bytes to disk with unique name
        unique_name = f"{uuid.uuid4().hex}_{filename}"
        save_path   = IMAGE_UPLOAD_DIR / unique_name
        save_path.write_bytes(file_bytes)

        # Store only metadata in DB — no base64, no context blowout
        extracted = (
            f"[IMAGE]\n"
            f"Filename: {filename}\n"
            f"Path: {save_path}\n"
            f"Format: {fmt}\n"
            f"Dimensions: {dims}\n"
            f"Mode: {mode}\n"
            f"File size: {size_str}\n"
            f"[/IMAGE]"
        )

        file_id = store_file_context(conv_id, user_id, filename, extracted)
        return json.dumps({
            "file_id":  file_id,
            "filename": filename,
            "chars":    len(extracted),
        })

    # ── Documents / text: extract and store content in DB ────────────────────
    try:
        extracted = extract_text(file_bytes, ext)
    except ExtractionError as e:
        return json.dumps({"error": str(e)}), 422

    file_id = store_file_context(conv_id, user_id, filename, extracted)

    return json.dumps({
        "file_id":  file_id,
        "filename": filename,
        "chars":    len(extracted),
    })


@app.route("/upload/<file_id>", methods=["DELETE"])
def remove_uploaded_file(file_id: str):
    """Remove a previously uploaded file context from the DB."""
    if not is_authenticated():
        return json.dumps({"error": "Unauthorized"}), 401

    delete_file_context(file_id)
    return json.dumps({"deleted": file_id})

# ── Voice transcription route ──────────────────────────────────────────────────
@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    """
    Transcribe an audio clip using faster-whisper (runs entirely on-device).

    Request : multipart/form-data with field "audio" — any format MediaRecorder
              produces (webm/ogg/opus on Chrome/Firefox, mp4 on Safari).
    Response: { "text": "transcribed words..." }
              { "error": "..." }  on failure (HTTP 4xx/5xx)

    faster-whisper is loaded once at import time and cached in the module.
    Model choice follows the WHISPER_MODEL env var (default: "base.en").
    Set WHISPER_DEVICE=cuda and WHISPER_COMPUTE=float16 for GPU boxes.
    """
    if not is_authenticated():
        return json.dumps({"error": "Unauthorized"}), 401

    if "audio" not in request.files:
        return json.dumps({"error": "No audio field in request"}), 400

    audio_file = request.files["audio"]
    if audio_file.filename == "":
        return json.dumps({"error": "Empty audio file"}), 400

    # ── Lazy-load the Whisper model (cached after first call) ──────────────────
    try:
        from faster_whisper import WhisperModel as _WhisperModel
    except ImportError:
        return json.dumps({"error": "faster-whisper is not installed"}), 500

    global _whisper_model
    if _whisper_model is None:
        model_size   = os.getenv("WHISPER_MODEL",   "base.en")
        device       = os.getenv("WHISPER_DEVICE",  "cpu")
        compute_type = os.getenv("WHISPER_COMPUTE", "int8")
        _whisper_model = _WhisperModel(model_size, device=device, compute_type=compute_type)

    # ── Save upload to a temp file (faster-whisper needs a file path) ──────────
    import tempfile
    suffix = Path(audio_file.filename).suffix or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        audio_file.save(tmp_path)

    try:
        segments, _ = _whisper_model.transcribe(tmp_path, language="en")
        text = " ".join(seg.text.strip() for seg in segments).strip()
    except Exception as exc:
        return json.dumps({"error": f"Transcription failed: {exc}"}), 500
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return json.dumps({"text": text})

# ── Startup banner ─────────────────────────────────────────────────────────────
def _print_banner():
    """Print the MyAI ASCII banner and runtime info on startup."""
    # ANSI colours — work in all modern terminals; gracefully ignored if not.
    C  = "\033[38;5;43m"   # accent teal
    D  = "\033[38;5;245m"  # dim grey
    Y  = "\033[38;5;221m"  # warming yellow
    B  = "\033[1m"         # bold
    R  = "\033[0m"         # reset

    model_path = os.getenv("MODEL", "unknown")
    model_name = model_display_name(model_path) if model_path != "unknown" else "unknown"
    ctx_size   = os.getenv("CTX_SIZE", "?")
    # User count now comes from the DB (source of truth). Fall back to the
    # legacy env var if the DB read fails for any reason so the banner still
    # renders during early bring-up.
    try:
        user_count = len(list_users(include_inactive=False))
    except Exception:
        user_count = len([u for u in users if u.strip()])
    warmup_on  = os.getenv("WARMUP", "1") == "1"

    banner = f"""{C}
    ███╗   ███╗██╗   ██╗ █████╗ ██╗
    ████╗ ████║╚██╗ ██╔╝██╔══██╗██║
    ██╔████╔██║ ╚████╔╝ ███████║██║
    ██║╚██╔╝██║  ╚██╔╝  ██╔══██║██║
    ██║ ╚═╝ ██║   ██║   ██║  ██║██║
    ╚═╝     ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝{R}
    {D}Private. Local. Yours.{R}  {D}· v{VERSION}{R}

    {B}Model{R}   {model_name}
    {B}Context{R} {ctx_size} tokens
    {B}Users{R}   {user_count} configured
    {B}URL{R}     http://localhost:{port}
"""
    print(banner)

    # Status line — show "warming…" while the warmup call runs, then flip to
    # "ready" by overwriting the same line. The trailing \r returns the cursor
    # to the start of the line so the next print clears the previous status.
    if warmup_on:
        print(f"    {Y}● warming…{R}   {D}— priming model weights{R}", end="\r", flush=True)
        warmed = warmup()
        # Clear the warming line, then print the ready status
        print(" " * 70, end="\r")  # blank the line
        ready_note = "" if warmed else f"   {D}(warmup skipped){R}"
        print(f"    {C}● ready{R}{ready_note}   {D}— press Ctrl+C to stop{R}\n")
    else:
        # Warmup disabled — print ready immediately with a hint
        print(f"    {C}● ready{R}   {D}— warmup off; first prompt may be slow{R}\n")


if __name__ == "__main__":
    _print_banner()
    serve(app, host="0.0.0.0", port=port)
