import os
import json
import uuid
from pathlib import Path
from flask import Flask, request, render_template, Response, stream_with_context, session, redirect, url_for, send_file, abort
from werkzeug.utils import secure_filename
from core.llm_inference import ask_stream
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

# ── Load / instantiate variables ───────────────────────────────────────────────
ALLOWED_EXTENSIONS = {ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS}
MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5 MB
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
app.secret_key = os.getenv("SECRET_KEY", "e0048a069f6a595970fe8e230f2c016cb38f6d51ed38c61c0c886164c7cbf088")

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
    return render_template('index.html', users=users)


@app.route('/chat', methods=['GET'])
def chat():
    if not is_authenticated():
        return redirect(url_for('index'))
    return render_template('chat.html')


@app.route('/logout', methods=['GET'])
def logout():
    session.clear()
    return redirect(url_for('index'))


@app.route('/memories', methods=['GET'])
def memories_page():
    if not is_authenticated():
        return redirect(url_for('index'))
    return render_template('memories.html')


# ── Authentication route ───────────────────────────────────────────────────────
@app.route('/auth', methods=['POST'])
def auth():
    """
    Verify a user's 4-digit PIN.

    Request JSON : { "user_id": "Bob", "pin": "1234" }
    Response JSON: { "ok": true }  or  { "error": "..." }  (HTTP 401)
    """
    data = request.get_json()
    if not data or "user_id" not in data or "pin" not in data:
        return json.dumps({"error": "user_id and pin required"}), 400

    user_id = data["user_id"].strip()
    pin     = str(data["pin"]).strip()

    if user_id not in users:
        # Avoid leaking which users exist — same response as wrong PIN
        return json.dumps({"error": "Invalid PIN"}), 401

    if not verify_pin(user_id, pin):
        return json.dumps({"error": "Invalid PIN"}), 401

    session["user_id"] = user_id
    return json.dumps({"ok": True})


# ── Conversation management routes ─────────────────────────────────────────────
@app.route('/conversations', methods=['POST'])
def new_conversation():
    """
    Create a new conversation for a user.

    Request JSON : { "user_id": "Bob" }
    Response JSON: { "conv_id": "<uuid>" }
    """
    if not is_authenticated():
        return json.dumps({"error": "Unauthorized"}), 401

    data = request.get_json()
    if not data or "user_id" not in data:
        return json.dumps({"error": "user_id required"}), 400

    conv_id = create_conversation(data["user_id"])
    return json.dumps({"conv_id": conv_id})


@app.route('/conversations/<user_id>', methods=['GET'])
def get_conversations(user_id: str):
    """
    List all conversations for a user, newest first.

    Response JSON: [{ "conv_id": ..., "title": ..., "updated_at": ... }, ...]
    """
    if not is_authenticated():
        return json.dumps({"error": "Unauthorized"}), 401

    return json.dumps(list_conversations(user_id))


@app.route('/conversations/<user_id>/search', methods=['GET'])
def search_conversations_route(user_id: str):
    """
    Search conversations by title or message content.

    Query param: ?q=<search term>
    Response JSON: [{ "conv_id": ..., "title": ..., "updated_at": ... }, ...]
    """
    if not is_authenticated():
        return json.dumps({"error": "Unauthorized"}), 401

    query = request.args.get('q', '').strip()
    if not query:
        return json.dumps(list_conversations(user_id))

    return json.dumps(search_conversations(user_id, query))


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

    user_id = data.get("name", "default")
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
    B  = "\033[1m"         # bold
    R  = "\033[0m"         # reset

    model_path = os.getenv("MODEL", "unknown")
    model_name = Path(model_path).name if model_path != "unknown" else "unknown"
    ctx_size   = os.getenv("CTX_SIZE", "?")
    user_count = len([u for u in users if u.strip()])

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

    {C}● ready{R}   {D}— press Ctrl+C to stop{R}
"""
    print(banner)


if __name__ == "__main__":
    _print_banner()
    serve(app, host="0.0.0.0", port=port)