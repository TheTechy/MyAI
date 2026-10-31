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
from .image_skill import ImageSkill
from .directions import DirectionsSkill
from .wikipedia_skill import WikipediaSkill
from .news_skill import NewsSkill
from .memory_skill import MemorySkill

REGISTERED_SKILLS: list[BaseSkill] = [
    CalculatorSkill(),    # pure Python — maths, conversions, percentages
    DateTimeSkill(),      # pure Python — time, dates, timezones
    CurrencySkill(),      # Frankfurter API — live exchange rates, no key needed
    WeatherSkill(),       # OWM API — weather and forecast
    WebSearchSkill(),     # DuckDuckGo + BS4 — general real-time queries
    ImageSkill(),         # Pillow — image manipulation and processing
    DirectionsSkill(),    # Nominatim + OSRM — driving directions + Leaflet map
    WikipediaSkill(),     # Wikipedia REST API — encyclopaedia lookups
    NewsSkill(),          # BBC + Guardian RSS — latest news headlines
    MemorySkill(),        # SQLite — stores & recalls explicit user-stated facts
]

__all__ = ["BaseSkill", "REGISTERED_SKILLS"]