"""
core/inference/filegen.py — v1 [FILE:] file-request handling, ported for parity
(MyAI v2, F4 Story 5, Option A)

A VERBATIM port of the file-request helpers from v1's llm_inference.py, so the v2
direct-answer path reaches exact parity: file requests that the model answers as
prose still get wrapped into a file, extensions are corrected, duplicate prose is
suppressed when a file is delivered, uploaded files aren't re-emitted, and the raw
[FILE:] block never leaks into the stream.

INTENTIONALLY ISOLATED. ADR-0002 retires the [FILE:] tag protocol in favour of a
generate_file tool. This whole module + its call sites in pipeline.py are the
end-of-project cleanup's single deletion target once that tool lands. Nothing new
is invented here — it mirrors v1 so behaviour is identical.

The orchestrator prepare_files() takes the file-generation functions via `deps`
(has_file_block / extract_and_generate / strip_file_blocks) so this module stays
import-light and testable without Pillow/docx/openpyxl.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, List, Optional, Tuple

# Mirrors core.file_generation.FILE_BLOCK_RE. Duplicated deliberately to keep this
# leaf module dependency-free; the cleanup task removes both together.
FILE_BLOCK_RE = re.compile(r'\[FILE:([^\]\n]+)\]\n?(.*?)\[/FILE\]', re.DOTALL | re.IGNORECASE)

# ── file-request patterns (verbatim from v1) ────────────────────────────────────
_FILE_REQUEST_PATTERNS = [
    (re.compile(r'\b(html\s+(?:file|page|document)|web\s*page|\.html?)\b', re.IGNORECASE), '.html'),
    (re.compile(r'\b(word\s+doc(?:ument)?|\.docx?)\b', re.IGNORECASE), '.docx'),
    (re.compile(r'\b(excel|spreadsheet|\.xlsx?)\b', re.IGNORECASE), '.xlsx'),
    (re.compile(r'\b(markdown|\.md)\b', re.IGNORECASE), '.md'),
    (re.compile(r'\b(css\s+(?:file|stylesheet)|\.css)\b', re.IGNORECASE), '.css'),
    (re.compile(r'\b(csv\s+(?:file|data)?|\.csv)\b', re.IGNORECASE), '.csv'),
    (re.compile(r'\b(json\s+file|\.json)\b', re.IGNORECASE), '.json'),
    (re.compile(r'\b(sql\s+(?:file|script)|\.sql)\b', re.IGNORECASE), '.sql'),
    (re.compile(r'\b(shell\s+script|bash\s+script|\.sh)\b', re.IGNORECASE), '.sh'),
    (re.compile(r'\b(text\s+file|\.txt)\b', re.IGNORECASE), '.txt'),
    (re.compile(r'\b(python\s+(?:script|file)|\.py)\b', re.IGNORECASE), '.py'),
    (re.compile(r'\b(javascript\s+(?:file|script)|\.js)\b', re.IGNORECASE), '.js'),
    (re.compile(r'\b(typescript\s+(?:file|script)|\.ts)\b', re.IGNORECASE), '.ts'),
]

_LANG_NEGATION_RE = re.compile(
    r'\b(no|without|not|never|skip|avoid|exclude|don\'?t\s+use|no\s+need\s+for)\s+'
    r'(?:any\s+)?'
    r'(javascript|js|typescript|ts|python|css|html|sql|bash|shell)\b',
    re.IGNORECASE,
)

_FILENAME_STOPWORDS = {
    'a', 'an', 'the', 'me', 'us', 'my', 'our', 'your', 'that', 'this',
    'with', 'and', 'or', 'for', 'about', 'on', 'in', 'of', 'to', 'is', 'it',
    'be', 'as', 'at', 'by', 'we', 'do', 'can', 'should', 'must', 'will',
    'has', 'have', 'are', 'was', 'were', 'so', 'if', 'but', 'than', 'then',
    'one', 'two', 'three', 'four', 'five', 'some', 'any', 'all', 'each',
    'only', 'just', 'very', 'also', 'such', 'more', 'most', 'less', 'few',
    'generate', 'create', 'make', 'write', 'give', 'produce', 'build',
    'craft', 'design', 'develop', 'output', 'save', 'export',
    'file', 'document', 'page', 'script', 'sheet', 'spreadsheet',
    'word', 'excel', 'html', 'css', 'javascript', 'python', 'markdown',
    'json', 'csv', 'sql', 'bash', 'shell', 'text', 'pdf', 'doc', 'web',
    'language', 'extension', 'format', 'syntax',
    'please', 'kindly', 'formatted', 'professional', 'detailed',
    'comprehensive', 'full', 'simple', 'basic', 'modern', 'responsive',
    'polished', 'clean', 'nice', 'good', 'great', 'visually', 'centered',
    'beautiful', 'styled', 'styling', 'using', 'use', 'single', 'includes',
    'including', 'similar', 'shown', 'show', 'look', 'looks', 'like',
}

_NO_FILE_RE = re.compile(
    r'\b(do\s+not|don\'t|dont|no|without|not)\s+(generate|create|make|produce|save|output|write)\s+a?\s*(file|document|spreadsheet|doc|xlsx?|docx?|csv)\b|'
    r'\b(inline|in.?line|in\s+the\s+chat|as\s+text|as\s+prose|in\s+your\s+response)\b|'
    r'\bdo\s+not\s+generate\s+a\s+file\b',
    re.IGNORECASE,
)

_CREATE_INTENT_RE = re.compile(
    r'\b(create|generate|make|write|build|produce|give\s+me|save\s+(?:it\s+)?as|output\s+(?:a\s+)?file)\b',
    re.IGNORECASE,
)


# ── helpers (verbatim from v1) ──────────────────────────────────────────────────
def detect_requested_extension(user_input: str) -> Optional[str]:
    if _NO_FILE_RE.search(user_input):
        return None
    if not _CREATE_INTENT_RE.search(user_input):
        return None
    cleaned = _LANG_NEGATION_RE.sub('', user_input)
    for pattern, ext in _FILE_REQUEST_PATTERNS:
        if pattern.search(cleaned):
            return ext
    return None


def make_filename(user_input: str, ext: str) -> str:
    text = _LANG_NEGATION_RE.sub('', user_input)
    cue_match = None
    for pattern, _ in _FILE_REQUEST_PATTERNS:
        m = pattern.search(text)
        if m and (cue_match is None or m.start() < cue_match.start()):
            cue_match = m
    words = re.findall(r'[a-zA-Z]+', text.lower())
    candidates = [w for w in words if w not in _FILENAME_STOPWORDS and len(w) > 2]
    if not candidates:
        return f"output{ext}"
    if cue_match:
        all_words = re.findall(r'[a-zA-Z]+', text)
        try:
            cue_word = cue_match.group(0).split()[0].lower()
            cue_idx = next(i for i, w in enumerate(all_words) if w.lower() == cue_word)
            nearby = [w.lower() for w in all_words[max(0, cue_idx - 5):cue_idx + 6]]
            near = [w for w in nearby if w in candidates]
            if near:
                seen, ordered = set(), []
                for w in near:
                    if w not in seen:
                        seen.add(w)
                        ordered.append(w)
                if ordered:
                    return f"{'_'.join(ordered[:4])}{ext}"
        except (StopIteration, IndexError):
            pass
    return f"{'_'.join(candidates[:3])}{ext}"


def wrap_as_file_block(content: str, filename: str) -> str:
    return f"[FILE:{filename}]\n{content.strip()}\n[/FILE]"


def correct_file_extensions(raw_answer: str, user_input: str) -> str:
    requested_ext = detect_requested_extension(user_input)
    if not requested_ext:
        return raw_answer

    def _fix(match):
        raw_name = match.group(1).strip()
        content = match.group(2)
        current_ext = Path(raw_name).suffix.lower()
        if current_ext != requested_ext:
            stem = Path(raw_name).stem or 'output'
            return f"[FILE:{stem}{requested_ext}]\n{content}[/FILE]"
        return match.group(0)

    return FILE_BLOCK_RE.sub(_fix, raw_answer)


def suppress_prose_if_file_delivered(answer: str, files_delivered: int) -> str:
    if files_delivered == 0:
        return answer
    lines = [l for l in answer.splitlines() if l.strip()]
    if len(lines) <= 2 and len(answer) <= 200:
        return answer
    return ""


def file_description(files: List[dict]) -> str:
    if not files:
        return ""
    if len(files) == 1:
        fname = files[0]["filename"]
        ext = fname.rsplit(".", 1)[-1].upper() if "." in fname else ""
        label = {
            "DOCX": "Word document", "XLSX": "Excel spreadsheet",
            "PDF": "PDF document", "MD": "Markdown file",
            "PY": "Python script", "JS": "JavaScript file",
            "CSV": "CSV file", "TXT": "text file",
        }.get(ext, "file")
        return f"Your {label} **{fname}** is ready to download."
    return f"{len(files)} files are ready to download."


def looks_like_question(text: str) -> bool:
    stripped = text.strip()
    if '?' in stripped:
        return True
    question_starters = re.compile(
        r'^(could you|can you|what |which |how |do you|would you|are you|is it|should i|'
        r'before (i|we) (begin|start|proceed)|to (clarify|confirm)|'
        r'i (need|have) (a few|some) (question|clarif))',
        re.IGNORECASE,
    )
    if question_starters.match(stripped):
        return True
    clarify_phrases = re.compile(
        r'\b(clarif|confirm|before (i|we) (begin|start|proceed)|'
        r'need (more|further|some) (detail|info|clarif|context)|'
        r'a few (question|thing)s?|could you (clarify|confirm|tell|let me know)|'
        r'please (clarify|confirm|let me know))\b',
        re.IGNORECASE,
    )
    return bool(clarify_phrases.search(stripped))


def find_historical_file_request(history: Optional[List[dict]]) -> Optional[str]:
    if not history:
        return None
    last_model_turn = next(
        (t for t in reversed(history) if t.get("role") == "model"), None)
    if not last_model_turn:
        return None
    if re.search(r'ready to download', last_model_turn.get("content", ""), re.IGNORECASE):
        return None
    if not looks_like_question(last_model_turn.get("content", "")):
        return None
    for turn in reversed(history):
        if turn.get("role") == "user":
            ext = detect_requested_extension(turn["content"])
            if ext:
                return ext
    return None


def prepare_files(raw_answer: str, user_input: str, history: Optional[List[dict]],
                  uploaded_names: set, deps: Any) -> Tuple[str, List[dict]]:
    """v1's exact direct-path file handling. Returns (answer_text, files_to_emit).

    `deps` supplies has_file_block / extract_and_generate / strip_file_blocks
    (the real core.file_generation functions, injected via PipelineDeps).
    """
    # 1. correct extensions, OR wrap prose as a file when one was requested
    if deps.has_file_block(raw_answer):
        raw_answer = correct_file_extensions(raw_answer, user_input)
    else:
        requested_ext = detect_requested_extension(user_input)
        if not requested_ext:
            requested_ext = find_historical_file_request(history)
        # don't wrap when the user uploaded something (they want it analysed, not
        # re-emitted) or when the model is asking a clarifying question
        if requested_ext and not uploaded_names and not looks_like_question(raw_answer):
            filename = make_filename(user_input, requested_ext)
            raw_answer = wrap_as_file_block(raw_answer, filename)

    # 2. extract + build files (never re-emit an uploaded file)
    files: List[dict] = []
    if deps.has_file_block(raw_answer):
        files = [f for f in deps.extract_and_generate(raw_answer)
                 if f["filename"] not in uploaded_names]

    # 3. strip blocks; suppress duplicate prose when a file was delivered
    answer = suppress_prose_if_file_delivered(deps.strip_file_blocks(raw_answer), len(files))

    # 4. description fallback when a file was delivered but no prose remains
    if files and not answer.strip():
        answer = file_description(files)

    return answer, files
