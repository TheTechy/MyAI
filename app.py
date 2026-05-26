import os
import json
from flask import Flask, request, render_template, Response, stream_with_context, session, redirect, url_for
from werkzeug.utils import secure_filename
from llm_inference import ask_stream
from file_ingestion import extract_text, ExtractionError, SUPPORTED_EXTENSIONS
from waitress import serve
from dotenv import load_dotenv
from db import (
	create_conversation,
	list_conversations,
	delete_conversation,
	get_messages,
	verify_pin,
	init_db,
	store_file_context,
	delete_file_context,
)

load_dotenv()  # loads from .env in the current directory

# ── Load / instantiate variables ───────────────────────────────────────────────
ALLOWED_EXTENSIONS = {ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS}
MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5 MB
users = os.getenv("USERS", "").split(",")
port = os.getenv("PORT")
if not port:
    raise ValueError("PORT environment variable not set")

app = Flask(__name__, template_folder='Templates')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# Required for sessions. Set SECRET_KEY in your .env file.
# Generate a strong one with: python -c "import secrets; print(secrets.token_hex(32))"
app.secret_key = os.getenv("SECRET_KEY", "change-me-in-production")

# —— Initialise app ───────────────────────────────────────────────────────────—
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

init_db()

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


@app.route('/conversations/<conv_id>/messages', methods=['GET'])
def get_conversation_messages(conv_id: str):
    """
    Return all messages for a conversation (used to render past chats in the UI).

    Response JSON: [{ "role": "user"|"model", "content": "..." }, ...]
    """
    if not is_authenticated():
        return json.dumps({"error": "Unauthorized"}), 401

    messages = get_messages(conv_id, max_turns=9999)
    return json.dumps(messages)


@app.route('/conversations/<conv_id>', methods=['DELETE'])
def remove_conversation(conv_id: str):
    """
    Delete a conversation and all its messages.
    """
    if not is_authenticated():
        return json.dumps({"error": "Unauthorized"}), 401

    delete_conversation(conv_id)
    return json.dumps({"deleted": conv_id})


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

if __name__ == "__main__":
    print(f"Starting Waitress server on http://0.0.0.0:{port}")
    serve(app, host="0.0.0.0", port=port)