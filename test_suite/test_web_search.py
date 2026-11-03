"""
test_suite/test_web_search.py
=============================
WebSearchSkill's search package must be the one the install provides. It's an
optional import, so a mismatch doesn't crash anything: it only shows up when
someone asks for a search and is told it's unavailable. No network.
"""
import importlib
from pathlib import Path

import myai_skills.web_search as web_search

REQUIREMENTS = Path(__file__).resolve().parent.parent / "requirements.txt"


def test_requirements_install_the_search_package_the_skill_imports():
    pinned = [line.split("==")[0].strip().lower()
              for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines()]
    assert "ddgs" in pinned
    assert "duckduckgo_search" not in pinned


def test_search_package_loads():
    assert web_search._DDGS_AVAILABLE


def test_search_results_reach_the_answer(monkeypatch):
    class FakeDDGS:
        def __enter__(self): return self
        def __exit__(self, *exc): return False
        def text(self, query, region=None, max_results=None):
            return [{"title": "History of Poland",
                     "href": "https://en.wikipedia.org/wiki/History_of_Poland",
                     "body": "The history of Poland spans over a thousand years."}]

    monkeypatch.setattr(web_search, "DDGS", FakeDDGS)
    monkeypatch.setattr(web_search.WebSearchSkill, "_extract_content",
                        lambda self, url, query: None)
    answer = web_search.WebSearchSkill().execute("history of Poland")
    assert "History of Poland" in answer
    assert "unavailable" not in answer.lower()


class TestUnreliableSearchEngines:
    """Search goes through unofficial scrapers that fail intermittently, so one
    failed attempt mustn't lose the search."""

    def _ddgs_that_fails(self, monkeypatch, failures):
        attempts = []

        class FlakyDDGS:
            def __enter__(self): return self
            def __exit__(self, *exc): return False
            def text(self, query, region=None, max_results=None):
                attempts.append(query)
                if len(attempts) <= failures:
                    raise RuntimeError("No results found.")
                return [{"title": "A result", "href": "https://example.com", "body": "Text."}]

        monkeypatch.setattr(web_search, "DDGS", FlakyDDGS)
        monkeypatch.setattr(web_search.time, "sleep", lambda s: None)
        monkeypatch.setattr(web_search.WebSearchSkill, "_extract_content",
                            lambda self, url, query: None)
        return attempts

    def test_retries_until_a_search_succeeds(self, monkeypatch):
        attempts = self._ddgs_that_fails(monkeypatch, failures=2)
        assert "A result" in web_search.WebSearchSkill().execute("q")
        assert len(attempts) == 3

    def test_gives_up_and_says_so(self, monkeypatch):
        attempts = self._ddgs_that_fails(monkeypatch, failures=99)
        answer = web_search.WebSearchSkill().execute("q")
        assert answer.startswith("Search error")
        assert len(attempts) == web_search.SEARCH_ATTEMPTS


def test_missing_package_is_reported_at_startup_and_names_the_right_package(monkeypatch, capsys):
    import sys
    monkeypatch.setitem(sys.modules, "ddgs", None)   # makes `from ddgs import` fail
    try:
        importlib.reload(web_search)
        assert "ddgs" in capsys.readouterr().out
        assert "ddgs" in web_search.WebSearchSkill().execute("anything")
    finally:
        monkeypatch.delitem(sys.modules, "ddgs")
        importlib.reload(web_search)
