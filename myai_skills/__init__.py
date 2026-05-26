"""
skills/__init__.py
==================
Skill registry — import and instantiate all skills here.
"""

from __future__ import annotations

from .base import BaseSkill
from .calculator import CalculatorSkill
from .currency import CurrencySkill
from .datetime_skill import DateTimeSkill
from .weather import WeatherSkill
from .web_search import WebSearchSkill

REGISTERED_SKILLS: list[BaseSkill] = [
    CalculatorSkill(),    # pure Python — maths, conversions, percentages
    DateTimeSkill(),      # pure Python — time, dates, timezones
    CurrencySkill(),      # Frankfurter API — live exchange rates, no key needed
    WeatherSkill(),       # OWM API — weather and forecast
    WebSearchSkill(),     # DuckDuckGo + BS4 — general real-time queries
    # ImageParserSkill(), ← future
]

__all__ = ["BaseSkill", "REGISTERED_SKILLS"]