"""
test_suite/test_datetime_skill.py
==================================
Unit tests for DateTimeSkill — no LLM, no network, no Flask required.

Deterministic tests use fixed historical dates.
Time-of-day tests only verify format/structure, not exact values.
"""
import re
from datetime import date

import pytest

from myai_skills.datetime_skill import DateTimeSkill


@pytest.fixture(scope="module")
def dt():
    return DateTimeSkill()


# ── Day-of-week (historical — fully deterministic) ─────────────────────────────
class TestDayOfWeek:
    def test_d_day(self, dt):
        # 6th June 1944 — from SET 7
        result = dt.execute("What day was 6th June 1944?")
        assert "tuesday" in result.lower(), f"Expected Tuesday, got: {result}"

    def test_historical_date(self, dt):
        # 19th Jan 1873 — from SET 7
        result = dt.execute("What day was 19th Jan 1873?")
        assert "sunday" in result.lower(), f"Expected Sunday, got: {result}"

    def test_millennium(self, dt):
        # 1st January 2000 was a Saturday
        result = dt.execute("What day was 1st January 2000?")
        assert "saturday" in result.lower(), f"Expected Saturday, got: {result}"


# ── Age calculation ────────────────────────────────────────────────────────────
class TestAgeCalculation:
    def test_age_contains_number(self, dt):
        # "How old is someone born on 15th March 1990?" — from SET 7
        result = dt.execute("How old is someone born on 15th March 1990?")
        # The answer must contain an age (a number)
        assert re.search(r'\d+', result), f"No number in age response: {result}"

    def test_age_is_reasonable(self, dt):
        result = dt.execute("How old is someone born on 15th March 1990?")
        # Must be between 30 and 50 years old as of 2026
        ages = re.findall(r'\b(\d{1,3})\b', result)
        age_ints = [int(a) for a in ages if 20 <= int(a) <= 60]
        assert age_ints, f"No plausible age found in: {result}"


# ── Current time in timezone ───────────────────────────────────────────────────
class TestTimezone:
    def test_tokyo_response_has_time(self, dt):
        # "What time is it in Tokyo?" — from SET 7
        result = dt.execute("What time is it in Tokyo?")
        assert re.search(r'\d{1,2}:\d{2}', result), f"No time format in: {result}"
        assert "tokyo" in result.lower() or "jst" in result.lower() or "japan" in result.lower()

    def test_new_york_response_has_time(self, dt):
        # "What time is it in New York?" — from SET 7
        result = dt.execute("What time is it in New York?")
        assert re.search(r'\d{1,2}:\d{2}', result), f"No time format in: {result}"

    def test_london_time(self, dt):
        result = dt.execute("What time is it in London?")
        assert re.search(r'\d{1,2}:\d{2}', result), f"No time format in: {result}"


# ── Days until future event ────────────────────────────────────────────────────
class TestDaysUntil:
    def test_days_until_christmas_contains_number(self, dt):
        # "How many days until Christmas?" — from SET 7
        result = dt.execute("How many days until Christmas?")
        assert re.search(r'\d+', result), f"No number in response: {result}"

    def test_days_until_christmas_is_positive(self, dt):
        result = dt.execute("How many days until Christmas?")
        numbers = re.findall(r'\b(\d+)\b', result)
        # Christmas is always in the future (within a year), so expect a reasonable number
        ints = [int(n) for n in numbers if 0 < int(n) <= 366]
        assert ints, f"No positive day-count found in: {result}"
