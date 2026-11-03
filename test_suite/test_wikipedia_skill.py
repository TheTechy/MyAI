"""
test_suite/test_wikipedia_skill.py
==================================
Unit tests for how WikipediaSkill works out what to look up. No network.
"""
import pytest

from myai_skills.wikipedia_skill import _extract_subject


@pytest.mark.parametrize("query, subject", [
    ("Search Wikipedia for the history of Poland.", "history of Poland"),
    ("search wikipedia for the Eiffel Tower", "Eiffel Tower"),
    ("Search on Wikipedia for Marie Curie", "Marie Curie"),
    ("Can you search Wikipedia for black holes?", "black holes"),
    ("Please check Wikipedia for Ada Lovelace", "Ada Lovelace"),
    ("What does Wikipedia say about the Roman Empire?", "Roman Empire"),
    ("Find the Battle of Hastings on Wikipedia.", "Battle of Hastings"),
    ("Search for Isaac Newton on Wikipedia", "Isaac Newton"),
    ("Look up the Eiffel Tower on Wikipedia", "Eiffel Tower"),
    ("Look Alan Turing up on Wikipedia please", "Alan Turing"),
    ("Wikipedia article on photosynthesis", "photosynthesis"),
    ("Tell me about the Moon", "Moon"),
    ("What is a neutron star?", "neutron star"),
])
def test_finds_the_subject_of_the_request(query, subject):
    assert _extract_subject(query) == subject


@pytest.mark.parametrize("subject", ["Find My iPhone", "Search engine", "St. Louis"])
def test_leaves_titles_alone_when_wikipedia_isnt_mentioned(subject):
    # "Find" and "Search" are only preamble in a request that names Wikipedia.
    assert _extract_subject(subject) == subject
