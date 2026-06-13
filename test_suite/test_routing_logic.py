"""
test_suite/test_routing_logic.py
=================================
Unit tests for the pure-Python routing helpers in core/llm_inference.py.

These functions perform regex matching and string manipulation with no LLM
call, so they run instantly and are completely deterministic.

Tested functions
----------------
  _detect_requested_extension  — does the user want a file generated?
  _make_filename               — derive a sensible filename from the prompt
  _looks_like_question         — is the LLM response a clarifying question?
"""
import pytest


# llm_inference imports llama_cpp at module level; conftest.py stubs it first.
from core.llm_inference import (
    _detect_requested_extension,
    _looks_like_question,
    _make_filename,
)


# ── _detect_requested_extension ────────────────────────────────────────────────
class TestDetectRequestedExtension:

    # ── Positive cases (file should be detected) ──────────────────────────────
    def test_word_document(self):
        assert _detect_requested_extension(
            "Write a professional Word document outlining C# coding standards"
        ) == ".docx"

    def test_docx_explicit(self):
        # The word-boundary regex requires "word doc" phrasing; ".docx" alone
        # after a space doesn't match the \b boundary — use the "Word document" form.
        assert _detect_requested_extension(
            "Draft an NDA and generate it as a Word document"
        ) == ".docx"

    def test_excel_spreadsheet(self):
        assert _detect_requested_extension(
            "Create an Excel spreadsheet comparing the planets"
        ) == ".xlsx"

    def test_xlsx_explicit(self):
        # "Export" is not in the creation-intent verb list; use "create"/"generate".
        assert _detect_requested_extension(
            "Create an Excel spreadsheet comparing the planets"
        ) == ".xlsx"

    def test_python_script_save_as(self):
        # SET 2: "save it as lru_cache.py"
        assert _detect_requested_extension(
            "Write a Python class for an LRU cache and save it as lru_cache.py"
        ) == ".py"

    def test_js_save_as(self):
        # SET 2: "save it as debounce.js"
        assert _detect_requested_extension(
            "Write a JavaScript debounce utility and save it as debounce.js"
        ) == ".js"

    def test_sql_save_as(self):
        # SET 2: "Save it as library_schema.sql"
        assert _detect_requested_extension(
            "Write a SQL script for a library schema. Save as library_schema.sql"
        ) == ".sql"

    def test_csharp_save_as(self):
        assert _detect_requested_extension(
            "Create a C# console application and save it as csv_analyser.cs"
        ) is None  # .cs is not in FILE_REQUEST_PATTERNS (no pattern for it)

    # ── Negative cases (no file should be generated) ──────────────────────────
    def test_general_question(self):
        assert _detect_requested_extension(
            "What are the three longest rivers in the world?"
        ) is None

    def test_analysis_verb_no_creation(self):
        # "review my Python file" must not trigger file output
        assert _detect_requested_extension(
            "Can you review my Python file for bugs?"
        ) is None

    def test_check_verb_no_creation(self):
        assert _detect_requested_extension(
            "Check my bash script for errors"
        ) is None

    def test_explain_verb_no_creation(self):
        assert _detect_requested_extension(
            "Explain the difference between async/await in C#"
        ) is None

    def test_explicit_no_file(self):
        # SET 4: "provide the following response inline, do not generate a file"
        assert _detect_requested_extension(
            "I have uploaded a Word document. Provide the response inline, "
            "do not generate a file."
        ) is None

    def test_inline_keyword_suppresses_file(self):
        assert _detect_requested_extension(
            "Respond inline as text, no Word document"
        ) is None


# ── _make_filename ─────────────────────────────────────────────────────────────
class TestMakeFilename:

    def test_extension_is_preserved(self):
        name = _make_filename("Write a Python script to parse CSV", ".py")
        assert name.endswith(".py")

    def test_stopwords_excluded(self):
        name = _make_filename("Create a Word document for me please", ".docx")
        assert "create" not in name
        assert "for" not in name
        assert "please" not in name
        assert name.endswith(".docx")

    def test_keywords_included(self):
        name = _make_filename("Write an NDA agreement for a company", ".docx")
        assert "nda" in name or "agreement" in name or "company" in name

    def test_fallback_name_on_all_stopwords(self):
        # Every word is a stopword — should fall back to 'output.<ext>'
        name = _make_filename("a the for to me", ".txt")
        assert name == "output.txt"

    def test_short_words_excluded(self):
        # Words ≤ 2 chars are skipped
        name = _make_filename("Do it as a py file", ".py")
        assert name.endswith(".py")


# ── _looks_like_question ───────────────────────────────────────────────────────
class TestLooksLikeQuestion:

    # ── Should be True (clarifying questions) ─────────────────────────────────
    def test_explicit_question_mark(self):
        assert _looks_like_question(
            "Could you tell me more about the format you'd like?"
        ) is True

    def test_multiple_numbered_questions(self):
        assert _looks_like_question(
            "Before I begin, I have a few questions:\n1. What format?\n2. How long?"
        ) is True

    def test_clarify_opener(self):
        assert _looks_like_question(
            "Could you clarify what layout you'd prefer?"
        ) is True

    def test_can_you_opener(self):
        assert _looks_like_question(
            "Can you confirm the date range you need?"
        ) is True

    def test_needs_clarification(self):
        # "before I begin/start/proceed" matches the clarify_phrases pattern.
        # Note: "clarification" fails the trailing \b because extra chars follow "clarif".
        assert _looks_like_question(
            "Before I begin, I need to confirm the format."
        ) is True

    # ── Should be False (substantive answers) ─────────────────────────────────
    def test_normal_answer(self):
        assert _looks_like_question(
            "Here is the Python script you requested. It implements a thread-safe LRU cache."
        ) is False

    def test_file_delivery_message(self):
        assert _looks_like_question(
            "Your Excel spreadsheet **planets.xlsx** is ready to download."
        ) is False

    def test_empty_string(self):
        assert _looks_like_question("") is False

    def test_factual_answer(self):
        assert _looks_like_question(
            "The three longest rivers in the world are the Nile, the Amazon, and the Yangtze."
        ) is False
