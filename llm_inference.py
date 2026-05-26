import os
import re
import json
from pathlib import Path
from llama_cpp import Llama
from dotenv import load_dotenv

from db import (
    init_db,
    get_messages,
    append_message,
    auto_title_from_first_message,
    get_conversation_title,
    get_file_contexts,
)
from prompts import get_prompt_for_user
from myai_skills import REGISTERED_SKILLS
from file_generation import (
    has_file_block,
    strip_file_blocks,
    extract_and_generate,
    FILE_GENERATION_SYSTEM_SNIPPET,
    FILE_BLOCK_RE,
)


# ── Load / instantiate variables ───────────────────────────────────────────────
load_dotenv()
model    = os.getenv("MODEL")
ctx_size = os.getenv("CTX_SIZE")

RESPONSE_TOKENS = [
    "<end_of_turn>", "</end_of_turn>", "[/end_of_turn]", "<eos>",
    "<start_of_turn>", "</start_of_turn>", "[/start_of_turn]", "<bos>",
    "<channel|>","<channel>",
    "<pad>",
]

# Regex to strip stray model control tokens that leak into output
_CONTROL_TOKEN_RE = re.compile(r'<(eos|end_of_turn|start_of_turn|/?turn)>', re.IGNORECASE)

def _clean(text: str) -> str:
    """Strip any stray control tokens from model output."""
    text = _CONTROL_TOKEN_RE.sub("", text)
    return text.strip()

# ── Initialise DB ──────────────────────────────────────────────────────────────
init_db()

# ── Model ──────────────────────────────────────────────────────────────────────
llm = Llama(
    model_path=model,
    n_ctx=int(ctx_size),
    n_gpu_layers=-1,
    verbose=False,
    n_threads=6,
    n_threads_batch=12,
    n_batch=512,
    n_ubatch=512,
    flash_attn=True,
    use_mmap=True,
    type_k=8,
    type_v=8
)

MAX_HISTORY_TURNS = 10

SEARCH_TAG_RE = re.compile(r'\[SEAR(?:C?SEARCH|CH):\s*(.+?)\]', re.IGNORECASE)

# ── Classifier examples ────────────────────────────────────────────────────────
def _load_classifier_examples() -> str:
    """
    Load few-shot examples from classifier_examples.json.
    Returns a formatted string ready to inject into the classifier prompt.
    Falls back to an empty string if the file is missing or malformed —
    the classifier still works, just with less accuracy.
    """
    path = Path(__file__).parent / "classifier_examples.json"
    try:
        data    = json.loads(path.read_text(encoding="utf-8"))
        lines   = [
            f"Query: {ex['query']:<55} → {ex['category']}"
            for ex in data.get("examples", [])
            if "query" in ex and "category" in ex
        ]
        if lines:
            print(f"[classifier] loaded {len(lines)} examples from {path.name}")
        return "\n".join(lines)
    except FileNotFoundError:
        print(f"[classifier] warning: {path.name} not found — using built-in examples")
        return ""
    except Exception as e:
        print(f"[classifier] warning: could not load examples: {e}")
        return ""

_CLASSIFIER_EXAMPLES = _load_classifier_examples()

# ── Intent classifier ──────────────────────────────────────────────────────────
_VALID_CATEGORIES = {"calculator", "currency", "weather", "datetime", "web_search", "general"}

_CATEGORY_TO_SKILL = {
    "calculator": "calculator",
    "currency":   "currency",
    "weather":    "weather",
    "datetime":   "datetime",
    "web_search": "web_search",
    "general":    None,
}


def _classify_intent(user_input: str):
    """
    Silent low-temperature LLM call to classify the user query into a skill.
    Returns the matched skill instance or None for general queries.
    """
    prompt = (
        "<start_of_turn>user\n"
        "Classify the following query into exactly ONE category. "
        "Reply with only the category name, nothing else.\n\n"
        "Valid categories: calculator, currency, weather, datetime, web_search, general\n\n"
        f"Examples:\n{_CLASSIFIER_EXAMPLES}\n\n"
        f"Query: {user_input}\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )

    try:
        output = llm(
            prompt,
            max_tokens=8,
            stop=["<end_of_turn>", "\n"],
            echo=False,
            stream=False,
            temperature=0.1,
        )
        category = output["choices"][0]["text"].strip().lower()
        category = category.replace("→", "").strip()

        if category not in _VALID_CATEGORIES:
            for valid in _VALID_CATEGORIES:
                if valid in category:
                    category = valid
                    break
            else:
                category = "general"

    except Exception as e:
        print(f"[classifier] error: {e}")
        category = "general"

    print(f"[classifier] '{user_input[:60]}' → {category}")

    if category == "general":
        return None

    skill_name = _CATEGORY_TO_SKILL.get(category)
    if not skill_name:
        return None

    return next((s for s in REGISTERED_SKILLS if s.name == skill_name), None)


# ── File request detection & post-processing ───────────────────────────────────
_FILE_REQUEST_PATTERNS = [
    (re.compile(r'\b(word\s+doc(?:ument)?|\.docx?)\b',        re.IGNORECASE), '.docx'),
    (re.compile(r'\b(excel|spreadsheet|\.xlsx?)\b',            re.IGNORECASE), '.xlsx'),
    (re.compile(r'\b(markdown|\.md)\b',                        re.IGNORECASE), '.md'),
    (re.compile(r'\b(python\s+(?:script|file)|\.py)\b',        re.IGNORECASE), '.py'),
    (re.compile(r'\b(javascript|\.js)\b',                      re.IGNORECASE), '.js'),
    (re.compile(r'\b(typescript|\.ts)\b',                      re.IGNORECASE), '.ts'),
    (re.compile(r'\b(html\s+(?:file|page)|\.html?)\b',         re.IGNORECASE), '.html'),
    (re.compile(r'\b(css\s+(?:file|stylesheet)|\.css)\b',      re.IGNORECASE), '.css'),
    (re.compile(r'\b(csv\s+(?:file|data)|\.csv)\b',            re.IGNORECASE), '.csv'),
    (re.compile(r'\b(json\s+file|\.json)\b',                   re.IGNORECASE), '.json'),
    (re.compile(r'\b(sql\s+(?:file|script)|\.sql)\b',          re.IGNORECASE), '.sql'),
    (re.compile(r'\b(shell\s+script|bash\s+script|\.sh)\b',    re.IGNORECASE), '.sh'),
    (re.compile(r'\b(text\s+file|\.txt)\b',                    re.IGNORECASE), '.txt'),
]

_FILENAME_STOPWORDS = {
    'a', 'an', 'the', 'generate', 'create', 'make', 'write', 'give', 'produce',
    'me', 'us', 'my', 'please', 'page', 'two', 'three', 'file', 'document',
    'word', 'excel', 'that', 'with', 'and', 'or', 'for', 'about', 'on', 'in',
    'of', 'to', 'is', 'it', 'be', 'as', 'at', 'by', 'we', 'do', 'can',
    'format', 'formatted', 'professional', 'detailed', 'comprehensive', 'full',
}

# Explicit no-file instructions — these override any file type mention
_NO_FILE_RE = re.compile(
    r'\b(do\s+not|don\'t|dont|no|without|not)\s+(generate|create|make|produce|save|output|write)\s+a?\s*(file|document|spreadsheet|doc|xlsx?|docx?|csv)\b|'
    r'\b(inline|in.?line|in\s+the\s+chat|as\s+text|as\s+prose|in\s+your\s+response)\b|'
    r'\bdo\s+not\s+generate\s+a\s+file\b',
    re.IGNORECASE
)

# Creation-intent keywords that must accompany a file type mention
_CREATE_INTENT_RE = re.compile(
    r'\b(create|generate|make|write|build|produce|give me|output)\b', re.IGNORECASE
)

def _detect_requested_extension(user_input: str) -> str | None:
    """
    Return the file extension the user requested, or None.
    Requires both a file-type signal AND a creation-intent verb.
    Returns None immediately if the user explicitly said NOT to generate a file.
    """
    # Explicit no-file instruction takes priority over everything
    if _NO_FILE_RE.search(user_input):
        return None
    if not _CREATE_INTENT_RE.search(user_input):
        return None
    for pattern, ext in _FILE_REQUEST_PATTERNS:
        if pattern.search(user_input):
            return ext
    return None


def _make_filename(user_input: str, ext: str) -> str:
    """Derive a reasonable filename from the user's request."""
    words    = re.findall(r'[a-zA-Z]+', user_input.lower())
    keywords = [w for w in words if w not in _FILENAME_STOPWORDS and len(w) > 2][:5]
    stem     = '_'.join(keywords) if keywords else 'output'
    return f"{stem}{ext}"


def _wrap_as_file_block(content: str, filename: str) -> str:
    """Wrap prose into a [FILE:…][/FILE] block for extract_and_generate."""
    return f"[FILE:{filename}]\n{content.strip()}\n[/FILE]"


def _correct_file_extensions(raw_answer: str, user_input: str) -> str:
    """
    If the model used [FILE:] tags but chose the wrong extension,
    rewrite the filename to match what the user actually requested.
    """
    requested_ext = _detect_requested_extension(user_input)
    if not requested_ext:
        return raw_answer

    from pathlib import Path as _Path

    def _fix(match):
        raw_name    = match.group(1).strip()
        content     = match.group(2)
        current_ext = _Path(raw_name).suffix.lower()
        if current_ext != requested_ext:
            stem     = _Path(raw_name).stem or 'output'
            new_name = f"{stem}{requested_ext}"
            return f"[FILE:{new_name}]\n{content}[/FILE]"
        return match.group(0)

    return FILE_BLOCK_RE.sub(_fix, raw_answer)


def _suppress_prose_if_file_delivered(answer: str, files_delivered: int) -> str:
    """
    When files were successfully delivered, suppress long boilerplate prose.
    Keep only short follow-ups (≤2 lines, ≤200 chars).
    """
    if files_delivered == 0:
        return answer
    lines = [l for l in answer.splitlines() if l.strip()]
    if len(lines) <= 2 and len(answer) <= 200:
        return answer
    return ""


def _file_description(files: list[dict]) -> str:
    """Generate a short default description when the model doesn't provide one."""
    if not files:
        return ""
    if len(files) == 1:
        fname = files[0]["filename"]
        ext   = fname.rsplit(".", 1)[-1].upper() if "." in fname else ""
        label = {
            "DOCX": "Word document", "XLSX": "Excel spreadsheet",
            "PDF":  "PDF document",  "MD":   "Markdown file",
            "PY":   "Python script", "JS":   "JavaScript file",
            "CSV":  "CSV file",      "TXT":  "text file",
        }.get(ext, "file")
        return f"Your {label} **{fname}** is ready to download."
    return f"{len(files)} files are ready to download."


# ── Prompt builders ────────────────────────────────────────────────────────────
def _build_prompt_with_history(
    history: list[dict],
    system_instruction: str,
    new_user_message: str,
) -> str:
    parts = []

    for i, turn in enumerate(history):
        role    = turn["role"]
        content = turn["content"]
        if role == "user":
            if i == 0 and system_instruction:
                content = f"{system_instruction}\n\nUser question: {content}"
            parts.append(f"<start_of_turn>user\n{content}<end_of_turn>")
        elif role == "model":
            parts.append(f"<start_of_turn>model\n{content}<end_of_turn>")

    if not history and system_instruction:
        new_user_message = f"{system_instruction}\n\nUser question: {new_user_message}"
    parts.append(f"<start_of_turn>user\n{new_user_message}<end_of_turn>")
    parts.append("<start_of_turn>model\n")

    return "\n".join(parts)


def format_routing_prompt(user_message: str, history: list[dict], system_prompt: str) -> str:
    return _build_prompt_with_history(history, system_prompt, user_message)


def format_answer_prompt(user_message: str, search_context: str, history: list[dict]) -> str:
    enriched_message = (
        f"Use the web search results below to answer the user's question. "
        f"Do not emit any [SEARCH:] tags.\n\n"
        f"Web search results:\n{search_context}\n\n"
        f"User question: {user_message}"
    )
    return _build_prompt_with_history(history, "", enriched_message)


def format_skill_prompt(user_message: str, skill_name: str, context: str, history: list[dict]) -> str:
    """Prompt builder for non-search skills — avoids 'web search results' framing."""
    label_map = {
        "calculator": "calculation result",
        "weather":    "weather data",
        "datetime":   "date and time data",
        "currency":   "currency conversion data",
    }
    label = label_map.get(skill_name, "data")
    enriched_message = (
        f"Use the {label} below to answer the user's question directly and concisely.\n\n"
        f"{label.capitalize()}:\n{context}\n\n"
        f"User question: {user_message}"
    )
    return _build_prompt_with_history(history, "", enriched_message)


# ── LLM helpers ────────────────────────────────────────────────────────────────
def _run_llm_silent(prompt: str, temperature: float | None = None) -> str:
    """Non-streaming LLM pass — used for Pass 1 routing decision."""
    kwargs = {
        "max_tokens": None,
        "stop":       RESPONSE_TOKENS,
        "echo":       False,
        "stream":     False,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    output = llm(prompt, **kwargs)
    return output["choices"][0]["text"]


def _sse(event: str, data: dict) -> str:
    """Format a Server-Sent Event string."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _title_sse_if_updated(conv_id: str) -> str | None:
    """Return a 'title' SSE if the conversation title has been set."""
    title = get_conversation_title(conv_id)
    if title and title != "New conversation":
        return _sse("title", {"conv_id": conv_id, "title": title})
    return None


# ── Main streaming entry point ─────────────────────────────────────────────────
def ask_stream(user_input: str, user_id: str, conv_id: str):
    """
    Generator that yields SSE-formatted strings.
    Consumers must set mimetype='text/event-stream'.

    Events emitted
    --------------
    status  – { "message": str }
    ctx     – { used, max, remaining, pct }
    token   – { "text": str }
    done    – { "truncated": bool }
    title   – { "conv_id": str, "title": str }
    error   – { "message": str }
    file    – { filename, mime_type, encoding, data, size }
    """
    history       = get_messages(conv_id, max_turns=MAX_HISTORY_TURNS)
    system_prompt = get_prompt_for_user(user_id)
    system_prompt = f"{system_prompt}\n\n{FILE_GENERATION_SYSTEM_SNIPPET}"

    # ── File context injection ─────────────────────────────────────────────────
    file_contexts  = get_file_contexts(conv_id)
    uploaded_names = {fc['filename'] for fc in file_contexts}
    if file_contexts:
        blocks = "\n\n".join(
            f"[Attached file: {fc['filename']}]\n{fc['extracted_text']}\n[/Attached file]"
            for fc in file_contexts
        )
        prompt_input = f"{blocks}\n\nUser question: {user_input}"
    else:
        prompt_input = user_input

    # ── Pass 0: intent classifier ──────────────────────────────────────────────
    # File generation bypasses the classifier — handled by the LLM in Pass 1
    # with FILE_GENERATION_SYSTEM_SNIPPET in the system prompt.
    if _detect_requested_extension(user_input):
        matched_skill = None
    else:
        matched_skill = _classify_intent(user_input)

    if matched_skill:
        yield _sse("status", {"message": "Searching…"})
        try:
            context = matched_skill.execute(user_input)
        except Exception as e:
            yield _sse("error", {"message": f"Skill error: {e}"})
            return

        yield _sse("status", {"message": "Generating answer…"})
        try:
            prompt_fn = (
                format_answer_prompt(prompt_input, search_context=context, history=history)
                if matched_skill.name == "web_search"
                else format_skill_prompt(prompt_input, matched_skill.name, context, history)
            )
            output = llm(
                prompt_fn,
                max_tokens=None,
                stop=RESPONSE_TOKENS,
                echo=False,
                stream=True,
            )
            truncated        = False
            collected_tokens = []
            for chunk in output:
                choice     = chunk["choices"][0]
                token_text = choice.get("text", "")
                if choice.get("finish_reason") == "length":
                    truncated = True
                if token_text:
                    collected_tokens.append(token_text)
                    yield _sse("token", {"text": token_text})

            full_answer = _clean("".join(collected_tokens))
            full_answer = strip_file_blocks(full_answer)

            append_message(conv_id, "user", user_input)
            append_message(conv_id, "model", full_answer)
            auto_title_from_first_message(conv_id, user_input)

            yield _sse("done", {"truncated": truncated})
            title_event = _title_sse_if_updated(conv_id)
            if title_event:
                yield title_event
        except Exception as e:
            yield _sse("error", {"message": str(e)})
        return

    # ── Pass 1: LLM routing ────────────────────────────────────────────────────
    yield _sse("status", {"message": "Thinking…"})

    try:
        first_response = _run_llm_silent(
            format_routing_prompt(prompt_input, history, system_prompt)
        )

        max_ctx   = llm.n_ctx()
        used_ctx  = llm.n_tokens
        remaining = max_ctx - used_ctx
        pct       = used_ctx / max_ctx * 100

        yield _sse("ctx", {
            "used":      used_ctx,
            "max":       max_ctx,
            "remaining": remaining,
            "pct":       round(pct, 1),
        })
    except Exception as e:
        yield _sse("error", {"message": str(e)})
        return

    search_match = SEARCH_TAG_RE.search(first_response)

    # ── No search needed: direct answer (+ optional file generation) ───────────
    if not search_match:
        raw_answer = _clean(first_response)

        if has_file_block(raw_answer):
            raw_answer = _correct_file_extensions(raw_answer, user_input)
        else:
            requested_ext = _detect_requested_extension(user_input)
            if requested_ext:
                filename   = _make_filename(user_input, requested_ext)
                raw_answer = _wrap_as_file_block(raw_answer, filename)

        files_delivered = 0
        files = []
        if has_file_block(raw_answer):
            yield _sse("status", {"message": "Generating file(s)…"})
            try:
                files = [
                    f for f in extract_and_generate(raw_answer)
                    if f['filename'] not in uploaded_names
                ]
                for f in files:
                    yield _sse("file", f)
                files_delivered = len(files)
            except Exception as e:
                yield _sse("error", {"message": f"File generation failed: {e}"})

        answer = _suppress_prose_if_file_delivered(
            strip_file_blocks(raw_answer), files_delivered
        )
        if files_delivered > 0 and not answer.strip():
            answer = _file_description(files)

        append_message(conv_id, "user", user_input)
        append_message(conv_id, "model", answer)
        auto_title_from_first_message(conv_id, user_input)

        if answer:
            yield _sse("token", {"text": answer})
        yield _sse("done", {"truncated": False})

        title_event = _title_sse_if_updated(conv_id)
        if title_event:
            yield title_event
        return

    # ── Pass 2: search context → streamed answer ───────────────────────────────
    query     = search_match.group(1).strip()
    web_skill = next((s for s in REGISTERED_SKILLS if s.name == "web_search"), None)
    if not web_skill:
        yield _sse("error", {"message": "Web search skill not available."})
        return

    yield _sse("status", {"message": f"Searching: '{query}'…"})
    try:
        context = web_skill.execute(query)
    except Exception as e:
        yield _sse("error", {"message": f"Search failed: {e}"})
        return

    yield _sse("status", {"message": "Search complete — generating answer…"})

    try:
        output = llm(
            format_answer_prompt(prompt_input, search_context=context, history=history),
            max_tokens=None,
            stop=RESPONSE_TOKENS,
            echo=False,
            stream=True,
        )
        truncated        = False
        collected_tokens = []

        for chunk in output:
            choice     = chunk["choices"][0]
            token_text = choice.get("text", "")
            if choice.get("finish_reason") == "length":
                truncated = True
            if token_text:
                collected_tokens.append(token_text)
                yield _sse("token", {"text": token_text})

        full_answer = _clean("".join(collected_tokens))

        if has_file_block(full_answer):
            full_answer = _correct_file_extensions(full_answer, user_input)
        else:
            requested_ext = _detect_requested_extension(user_input)
            if requested_ext:
                filename    = _make_filename(user_input, requested_ext)
                full_answer = _wrap_as_file_block(full_answer, filename)

        files_delivered = 0
        files = []
        if has_file_block(full_answer):
            yield _sse("status", {"message": "Generating file(s)…"})
            try:
                files = [
                    f for f in extract_and_generate(full_answer)
                    if f['filename'] not in uploaded_names
                ]
                for f in files:
                    yield _sse("file", f)
                files_delivered = len(files)
            except Exception as e:
                yield _sse("error", {"message": f"File generation failed: {e}"})

        full_answer = _suppress_prose_if_file_delivered(
            strip_file_blocks(full_answer), files_delivered
        )
        if files_delivered > 0 and not full_answer.strip():
            full_answer = _file_description(files)

        append_message(conv_id, "user", user_input)
        append_message(conv_id, "model", full_answer)
        auto_title_from_first_message(conv_id, user_input)

        yield _sse("done", {"truncated": truncated})
        title_event = _title_sse_if_updated(conv_id)
        if title_event:
            yield title_event

    except Exception as e:
        yield _sse("error", {"message": str(e)})