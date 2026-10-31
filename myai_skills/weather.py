"""
skills/weather.py
=================
Weather skill using the OpenWeatherMap API.

Provides current conditions AND a 5-day / 3-hour forecast.
Wins over web_search for weather queries due to higher confidence scores.

Setup
-----
Add to your .env file:
    OWM_API_KEY=your_api_key_here
    OWM_UNITS=metric          # metric | imperial | standard (default: metric)
    OWM_HOME_CITY=Leeds, GB   # optional — used when no location is in the query

Free tier endpoints used
------------------------
    Geocoding: https://api.openweathermap.org/geo/1.0/direct
    Current  : https://api.openweathermap.org/data/2.5/weather
    Forecast : https://api.openweathermap.org/data/2.5/forecast
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone

from .base import BaseSkill

try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False


# ── Constants ─────────────────────────────────────────────────────────────────
OWM_BASE          = "http://api.openweathermap.org/data/2.5"
OWM_GEO_BASE      = "http://api.openweathermap.org/geo/1.0/direct"
REQUEST_TIMEOUT   = 8   # seconds


# ── Home location ─────────────────────────────────────────────────────────────
# Geocoded lazily on first use — ensures load_dotenv() has already run.
_HOME_COORDS: tuple[float, float] | None = None
_HOME_COORDS_LOADED: bool = False


def _get_home_coords(api_key: str) -> tuple[float, float] | None:
    """
    Geocode OWM_HOME_CITY on first call and cache the result.
    Called from execute() after load_dotenv() has run.
    """
    global _HOME_COORDS, _HOME_COORDS_LOADED
    if _HOME_COORDS_LOADED:
        return _HOME_COORDS

    _HOME_COORDS_LOADED = True  # mark as attempted regardless of outcome
    city = os.getenv("OWM_HOME_CITY", "").strip()
    if not city or not api_key:
        return None

    try:
        resp = requests.get(
            OWM_GEO_BASE,
            params={"q": city, "limit": 1, "appid": api_key},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        if data:
            lat, lon = data[0]["lat"], data[0]["lon"]
            print(f"[weather] home location '{city}' → ({lat:.4f}, {lon:.4f})")
            _HOME_COORDS = (lat, lon)
        else:
            print(f"[weather] could not geocode home city '{city}': empty response")
    except Exception as e:
        print(f"[weather] could not geocode home city '{city}': {e}")

    return _HOME_COORDS

# Location extraction — words to strip before sending to OWM geocoding
_STRIP_WORDS = re.compile(
    r"\b(what(?:'s| is)|whats|is|the|current|temperature|temp|weather|forecast|"
    r"conditions|like|looking|going|today|tonight|tomorrow|this|week|will|it|rain|"
    r"snow|sunny|cloudy|hot|cold|warm|chilly|freezing|outside|right now|going to|"
    r"be|being|get|getting|feel|feeling|at the moment|later|soon|weekend|"
    r"in|at|for|now|any)\b",
    re.IGNORECASE
)


class WeatherSkill(BaseSkill):
    """
    Handles weather queries using OpenWeatherMap.
    Returns current conditions + 5-day forecast summary.
    """

    name        = "weather"
    description = "Get current weather and forecast using OpenWeatherMap"

    # High confidence — always beat web_search for weather


    # ── Routing ───────────────────────────────────────────────────────────────

    # ── Execution ─────────────────────────────────────────────────────────────
    def execute(self, query: str) -> str:
        if not _REQUESTS_AVAILABLE:
            return "Weather skill unavailable: requests not installed."

        api_key = os.getenv("OWM_API_KEY", "").strip()
        if not api_key:
            return "Weather skill unavailable: OWM_API_KEY not set in .env"

        units    = os.getenv("OWM_UNITS", "metric").strip().lower()
        location = self._extract_location(query)

        # Resolve location string → (lat, lon)
        coords = self._resolve_coords(location, api_key)
        if not coords:
            return "Could not determine a location from the query."

        lat, lon = coords
        current  = self._get_current(lat, lon, api_key, units)
        forecast = self._get_forecast(lat, lon, api_key, units)

        parts = []
        if current:
            parts.append(current)
        if forecast:
            parts.append(forecast)

        return "\n\n".join(parts) if parts else "Weather data unavailable."

    # ── Coordinate resolution ─────────────────────────────────────────────────
    def _resolve_coords(self, location: str, api_key: str) -> tuple[float, float] | None:
        """
        Resolve a location string to (lat, lon).
        Falls back to home coords if location is empty.
        Returns None if resolution fails.
        """
        if not location:
            return _get_home_coords(api_key)

        # Reject strings that are too short or look like common words not cities
        if len(location) < 3 or location.lower() in {
            'looking', 'like', 'today', 'now', 'here', 'there',
            'good', 'bad', 'nice', 'fine', 'okay'
        }:
            return _get_home_coords(api_key)

        try:
            resp = requests.get(
                OWM_GEO_BASE,
                params={"q": location, "limit": 1, "appid": api_key},
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            if data:
                return data[0]["lat"], data[0]["lon"]
        except Exception as exc:
            print(f"[weather] geocode error for '{location}': {exc}")

        return None

    # ── Location extraction ───────────────────────────────────────────────────
    def _extract_location(self, text: str) -> str:
        """
        Extract an OWM-compatible location from the user query.
        Uses last preposition+city match as the most reliable signal.
        OWM prefers: 'Leeds', 'Leeds,GB', 'New York,US'
        """
        _COUNTRY_MAP = {
            "uk": "GB", "united kingdom": "GB", "england": "GB",
            "scotland": "GB", "wales": "GB", "britain": "GB",
            "us": "US", "usa": "US", "united states": "US", "america": "US",
            "australia": "AU", "canada": "CA", "france": "FR",
            "germany": "DE", "spain": "ES", "italy": "IT",
            "japan": "JP", "china": "CN", "india": "IN",
        }

        # Find all in/to/for + City matches — only match proper nouns (capitalised)
        # Require the first letter to be uppercase to avoid matching "to be", "for today" etc.
        in_m  = list(re.finditer(r'\bin\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)',  text))
        to_m  = list(re.finditer(r'\bto\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)',  text))
        for_m = list(re.finditer(r'\bfor\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)', text))

        all_matches = sorted(in_m + to_m + for_m, key=lambda m: m.start())

        location = ""  # ensure always defined

        if all_matches:
            location = all_matches[-1].group(1).strip()

            # ── Country conversion BEFORE stripping noise ──────────────────────
            words = location.split()
            for length in (2, 1):
                if len(words) >= length + 1:
                    candidate = " ".join(words[-length:]).lower()
                    if candidate in _COUNTRY_MAP:
                        city    = " ".join(words[:-length])
                        country = _COUNTRY_MAP[candidate]
                        return f"{city},{country}"

            # No country found — strip trailing noise words
            location = re.sub(
                r'\s+(today|tonight|tomorrow|this|weekend|week|right|now|please).*$',
                '', location, flags=re.IGNORECASE
            ).strip()

            # Check again after noise strip
            words = location.split()
            for length in (2, 1):
                if len(words) >= length + 1:
                    candidate = " ".join(words[-length:]).lower()
                    if candidate in _COUNTRY_MAP:
                        city    = " ".join(words[:-length])
                        country = _COUNTRY_MAP[candidate]
                        return f"{city},{country}"

        else:
            # Fallback — strip noise words from full query
            location = _STRIP_WORDS.sub(" ", text)
            location = re.sub(r"[?!.,]", " ", location)
            location = " ".join(location.split()).strip()

            # Country conversion on fallback result
            words = location.split()
            for length in (2, 1):
                if len(words) >= length + 1:
                    candidate = " ".join(words[-length:]).lower()
                    if candidate in _COUNTRY_MAP:
                        city    = " ".join(words[:-length])
                        country = _COUNTRY_MAP[candidate]
                        return f"{city},{country}"

        return location

    # ── Current weather ───────────────────────────────────────────────────────
    def _get_current(self, lat: float, lon: float, api_key: str, units: str) -> str | None:
        try:
            resp = requests.get(
                f"{OWM_BASE}/weather",
                params={
                    "lat":   lat,
                    "lon":   lon,
                    "appid": api_key,
                    "units": units,
                },
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code == 404:
                return f"Location not found at ({lat:.4f}, {lon:.4f})"
            if resp.status_code == 401:
                return "Weather API key is invalid."
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            return f"Weather API error: {exc}"

        unit_sym  = _unit_symbol(units)
        city      = data.get("name", "Unknown")
        country   = data.get("sys", {}).get("country", "")
        desc      = data.get("weather", [{}])[0].get("description", "unknown").capitalize()
        temp      = data.get("main", {}).get("temp", "?")
        feels     = data.get("main", {}).get("feels_like", "?")
        humidity  = data.get("main", {}).get("humidity", "?")
        wind_spd  = data.get("wind", {}).get("speed", "?")
        wind_unit = "m/s" if units == "metric" else "mph"
        visibility= data.get("visibility", None)
        vis_str   = f"{visibility // 1000} km" if visibility else "N/A"

        sunrise_ts = data.get("sys", {}).get("sunrise")
        sunset_ts  = data.get("sys", {}).get("sunset")
        sunrise    = _fmt_time(sunrise_ts) if sunrise_ts else "N/A"
        sunset     = _fmt_time(sunset_ts)  if sunset_ts  else "N/A"

        return (
            f"Current weather for {city}, {country}:\n"
            f"  Conditions : {desc}\n"
            f"  Temperature: {temp}{unit_sym} (feels like {feels}{unit_sym})\n"
            f"  Humidity   : {humidity}%\n"
            f"  Wind speed : {wind_spd} {wind_unit}\n"
            f"  Visibility : {vis_str}\n"
            f"  Sunrise    : {sunrise}  Sunset: {sunset}"
        )

    # ── 5-day forecast ────────────────────────────────────────────────────────
    def _get_forecast(self, lat: float, lon: float, api_key: str, units: str) -> str | None:
        try:
            resp = requests.get(
                f"{OWM_BASE}/forecast",
                params={
                    "lat":   lat,
                    "lon":   lon,
                    "appid": api_key,
                    "units": units,
                    "cnt":   40,   # 5 days × 8 slots (every 3 hours)
                },
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code in (401, 404):
                return None  # current weather already reported the error
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return None

        unit_sym = _unit_symbol(units)
        items    = data.get("list", [])
        if not items:
            return None

        # Group by day and pick the midday (or closest) slot
        days: dict[str, list] = {}
        for item in items:
            dt_txt = item.get("dt_txt", "")
            day    = dt_txt[:10]  # "YYYY-MM-DD"
            days.setdefault(day, []).append(item)

        lines = ["5-day forecast:"]
        for day, slots in list(days.items())[:5]:
            # Prefer midday slot, fallback to first
            midday = next(
                (s for s in slots if "12:00" in s.get("dt_txt", "")),
                slots[0]
            )
            desc   = midday.get("weather", [{}])[0].get("description", "").capitalize()
            t_min  = min(s["main"]["temp_min"] for s in slots)
            t_max  = max(s["main"]["temp_max"] for s in slots)
            rain   = sum(s.get("rain", {}).get("3h", 0) for s in slots)
            pop    = max(s.get("pop", 0) for s in slots) * 100  # probability of precipitation

            day_label = _fmt_day(day)
            rain_str  = f"  Rain: {rain:.1f}mm ({pop:.0f}% chance)" if rain > 0 else f"  Precip chance: {pop:.0f}%"
            lines.append(
                f"  {day_label:<12} {desc:<25} "
                f"Low: {t_min:.0f}{unit_sym}  High: {t_max:.0f}{unit_sym}  {rain_str}"
            )

        return "\n".join(lines)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _unit_symbol(units: str) -> str:
    return {"metric": "°C", "imperial": "°F", "standard": "K"}.get(units, "°C")

def _fmt_time(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H:%M UTC")

def _fmt_day(date_str: str) -> str:
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.strftime("%a %d %b")   # e.g. "Mon 20 May"
    except ValueError:
        return date_str