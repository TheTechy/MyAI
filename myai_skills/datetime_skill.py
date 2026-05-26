"""
skills/datetime_skill.py
========================
Date and time skill — pure Python stdlib, no external dependencies.

Handles
-------
  Current time     : What time is it? / What time is it in Tokyo?
  Current date     : What's today's date? / What day is it?
  Timezone convert : Convert 3pm London to New York time
  Date arithmetic  : How many days until Christmas? / How long ago was D-Day?
  Day of week      : What day was 6th June 1944?
  Age calculation  : How old is someone born on 15th March 1990?
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .base import BaseSkill


# ── Timezone aliases ──────────────────────────────────────────────────────────
# Maps common city/country names to IANA timezone strings
_TZ_ALIASES: dict[str, str] = {
    # UK / Europe
    "london": "Europe/London", "uk": "Europe/London", "gb": "Europe/London",
    "england": "Europe/London", "britain": "Europe/London",
    "paris": "Europe/Paris", "france": "Europe/Paris",
    "berlin": "Europe/Berlin", "germany": "Europe/Berlin",
    "madrid": "Europe/Madrid", "spain": "Europe/Madrid",
    "rome": "Europe/Rome", "italy": "Europe/Rome",
    "amsterdam": "Europe/Amsterdam", "netherlands": "Europe/Amsterdam",
    "brussels": "Europe/Brussels", "belgium": "Europe/Brussels",
    "zurich": "Europe/Zurich", "switzerland": "Europe/Zurich",
    "stockholm": "Europe/Stockholm", "sweden": "Europe/Stockholm",
    "oslo": "Europe/Oslo", "norway": "Europe/Oslo",
    "copenhagen": "Europe/Copenhagen", "denmark": "Europe/Copenhagen",
    "helsinki": "Europe/Helsinki", "finland": "Europe/Helsinki",
    "athens": "Europe/Athens", "greece": "Europe/Athens",
    "warsaw": "Europe/Warsaw", "poland": "Europe/Warsaw",
    "moscow": "Europe/Moscow", "russia": "Europe/Moscow",
    "istanbul": "Europe/Istanbul", "turkey": "Europe/Istanbul",
    # Americas
    "new york": "America/New_York", "nyc": "America/New_York",
    "eastern": "America/New_York", "est": "America/New_York",
    "chicago": "America/Chicago", "central": "America/Chicago",
    "denver": "America/Denver", "mountain": "America/Denver",
    "los angeles": "America/Los_Angeles", "la": "America/Los_Angeles",
    "pacific": "America/Los_Angeles", "pst": "America/Los_Angeles",
    "san francisco": "America/Los_Angeles",
    "toronto": "America/Toronto", "canada": "America/Toronto",
    "vancouver": "America/Vancouver",
    "mexico city": "America/Mexico_City", "mexico": "America/Mexico_City",
    "sao paulo": "America/Sao_Paulo", "brazil": "America/Sao_Paulo",
    "buenos aires": "America/Argentina/Buenos_Aires", "argentina": "America/Argentina/Buenos_Aires",
    # Asia / Pacific
    "dubai": "Asia/Dubai", "uae": "Asia/Dubai",
    "india": "Asia/Kolkata", "delhi": "Asia/Kolkata", "mumbai": "Asia/Kolkata",
    "kolkata": "Asia/Kolkata", "ist": "Asia/Kolkata",
    "singapore": "Asia/Singapore",
    "hong kong": "Asia/Hong_Kong",
    "beijing": "Asia/Shanghai", "shanghai": "Asia/Shanghai", "china": "Asia/Shanghai",
    "tokyo": "Asia/Tokyo", "japan": "Asia/Tokyo",
    "seoul": "Asia/Seoul", "korea": "Asia/Seoul",
    "sydney": "Australia/Sydney", "australia": "Australia/Sydney",
    "melbourne": "Australia/Melbourne",
    "auckland": "Pacific/Auckland", "new zealand": "Pacific/Auckland",
    # UTC variants
    "utc": "UTC", "gmt": "UTC",
}

# ── Date parsing helpers ───────────────────────────────────────────────────────
_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2,
    "mar": 3, "march": 3, "apr": 4, "april": 4,
    "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

_NAMED_DATES = {
    "christmas":      lambda y: date(y, 12, 25),
    "christmas day":  lambda y: date(y, 12, 25),
    "new year":       lambda y: date(y + 1, 1, 1),
    "new year's day": lambda y: date(y + 1, 1, 1),
    "halloween":      lambda y: date(y, 10, 31),
    "valentines":     lambda y: date(y, 2, 14),
    "valentine's day":lambda y: date(y, 2, 14),
}


class DateTimeSkill(BaseSkill):

    name        = "datetime"
    description = "Current time, timezone conversion, date arithmetic and day lookups"




    # ── Main execution ────────────────────────────────────────────────────────
    def execute(self, query: str) -> str:
        handlers = [
            self._try_timezone_convert,
            self._try_current_time,
            self._try_current_date,
            self._try_days_until,
            self._try_days_since,
            self._try_day_of_week,
            self._try_age,
        ]
        for handler in handlers:
            result = handler(query)
            if result is not None:
                return result

        return f"I couldn't parse '{query}' as a date/time query."

    # ── Current time ──────────────────────────────────────────────────────────
    def _try_current_time(self, text: str) -> str | None:
        if not re.search(r'\btime\b', text, re.I):
            return None

        tz_name = self._extract_timezone(text)

        if tz_name:
            try:
                tz  = ZoneInfo(tz_name)
                now = datetime.now(tz)
                label = self._tz_label(text, tz_name)
                return (
                    f"Current time in {label}:\n"
                    f"  {now.strftime('%H:%M:%S')}  ({now.strftime('%A, %d %B %Y')})\n"
                    f"  Timezone: {tz_name}  (UTC{now.strftime('%z')})"
                )
            except ZoneInfoNotFoundError:
                return f"Unknown timezone: '{tz_name}'"
        else:
            # Local time
            now = datetime.now().astimezone()
            return (
                f"Current local time:\n"
                f"  {now.strftime('%H:%M:%S')}  ({now.strftime('%A, %d %B %Y')})\n"
                f"  Timezone: {now.strftime('%Z')}  (UTC{now.strftime('%z')})"
            )

    # ── Current date ──────────────────────────────────────────────────────────
    def _try_current_date(self, text: str) -> str | None:
        if not re.search(r'\b(date|today|day)\b', text, re.I):
            return None
        if re.search(r'\b(until|till|since|ago|was|born|when)\b', text, re.I):
            return None  # let other handlers deal with it

        today = date.today()
        return (
            f"Today is {today.strftime('%A, %d %B %Y')}\n"
            f"  Week number : {today.isocalendar()[1]}\n"
            f"  Day of year : {today.timetuple().tm_yday}"
        )

    # ── Timezone conversion ───────────────────────────────────────────────────
    def _try_timezone_convert(self, text: str) -> str | None:
        # "convert 3pm London to New York" / "3pm in Tokyo, what is that in London?"
        m = re.search(
            r'(\d{1,2}(?::\d{2})?\s*(?:am|pm))\s+(?:in\s+)?([a-z\s]+?)\s+(?:to|in)\s+([a-z\s]+)',
            text, re.I
        )
        if not m:
            return None

        time_str  = m.group(1).strip()
        from_city = m.group(2).strip().lower()
        to_city   = m.group(3).strip().lower()

        from_tz = self._lookup_tz(from_city)
        to_tz   = self._lookup_tz(to_city)

        if not from_tz or not to_tz:
            return None

        try:
            # Parse time
            fmt = "%I:%M%p" if ":" in time_str else "%I%p"
            t   = datetime.strptime(time_str.replace(" ", "").upper(), fmt)
            now = date.today()
            dt  = datetime(now.year, now.month, now.day, t.hour, t.minute, tzinfo=ZoneInfo(from_tz))
            converted = dt.astimezone(ZoneInfo(to_tz))
            return (
                f"{time_str} in {from_city.title()} = "
                f"{converted.strftime('%H:%M')} in {to_city.title()}\n"
                f"  ({from_tz} → {to_tz})"
            )
        except Exception:
            return None

    # ── Days until ────────────────────────────────────────────────────────────
    def _try_days_until(self, text: str) -> str | None:
        if not re.search(r'\b(until|till|to|until)\b', text, re.I):
            return None

        today      = date.today()
        target     = self._extract_date(text, future=True)
        if not target:
            return None

        delta = (target - today).days
        if delta < 0:
            return None

        weeks, days = divmod(delta, 7)
        week_str = f"{weeks} week{'s' if weeks != 1 else ''} and " if weeks else ""
        day_str  = f"{days} day{'s' if days != 1 else ''}"

        return (
            f"Days until {target.strftime('%A, %d %B %Y')}:\n"
            f"  {delta} days  ({week_str}{day_str})"
        )

    # ── Days since ────────────────────────────────────────────────────────────
    def _try_days_since(self, text: str) -> str | None:
        if not re.search(r'\b(since|ago)\b', text, re.I):
            return None

        today  = date.today()
        target = self._extract_date(text, future=False)
        if not target:
            return None

        delta = (today - target).days
        if delta < 0:
            return None

        years  = delta // 365
        months = (delta % 365) // 30
        yr_str = f"{years} year{'s' if years != 1 else ''}, " if years else ""
        mo_str = f"{months} month{'s' if months != 1 else ''}" if months else ""

        return (
            f"Days since {target.strftime('%A, %d %B %Y')}:\n"
            f"  {delta:,} days  ({yr_str}{mo_str})"
        )

    # ── Day of week ───────────────────────────────────────────────────────────
    def _try_day_of_week(self, text: str) -> str | None:
        if not re.search(r'\bwhat\s+day\b', text, re.I):
            return None

        target = self._extract_date(text)
        if not target:
            return None

        return (
            f"{target.strftime('%d %B %Y')} was a "
            f"{target.strftime('%A')}"
        )

    # ── Age calculation ───────────────────────────────────────────────────────
    def _try_age(self, text: str) -> str | None:
        if not re.search(r'\bhow\s+old\b', text, re.I):
            return None

        dob = self._extract_date(text, future=False)
        if not dob:
            return None

        today  = date.today()
        age    = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        next_b = dob.replace(year=today.year)
        if next_b < today:
            next_b = dob.replace(year=today.year + 1)
        days_to_bday = (next_b - today).days

        return (
            f"Born on {dob.strftime('%d %B %Y')}:\n"
            f"  Age           : {age} years old\n"
            f"  Next birthday : {next_b.strftime('%d %B %Y')} ({days_to_bday} days away)"
        )

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _extract_timezone(self, text: str) -> str | None:
        """Extract an IANA timezone from natural language."""
        text_lower = text.lower()
        # Try multi-word aliases first
        for alias, tz in sorted(_TZ_ALIASES.items(), key=lambda x: -len(x[0])):
            if alias in text_lower:
                return tz
        return None

    def _lookup_tz(self, city: str) -> str | None:
        city = city.strip().lower()
        return _TZ_ALIASES.get(city)

    def _tz_label(self, text: str, tz_name: str) -> str:
        """Return a friendly label for the timezone."""
        text_lower = text.lower()
        for alias in sorted(_TZ_ALIASES.keys(), key=lambda x: -len(x)):
            if alias in text_lower and _TZ_ALIASES[alias] == tz_name:
                return alias.title()
        return tz_name

    def _extract_date(self, text: str, future: bool | None = None) -> date | None:
        """
        Parse a date from natural language text.
        Tries: named dates, DD Month YYYY, DD/MM/YYYY, Month DD YYYY.
        """
        today = date.today()

        # Named dates (Christmas, New Year, etc.)
        for name, fn in _NAMED_DATES.items():
            if name in text.lower():
                candidate = fn(today.year)
                if future is True and candidate <= today:
                    candidate = fn(today.year + 1)
                return candidate

        # "6th June 1944" / "6 June 1944" / "June 6 1944" / "June 6th 1944"
        m = re.search(
            r'(\d{1,2})(?:st|nd|rd|th)?\s+([a-z]+)\s+(\d{4})',
            text, re.I
        )
        if not m:
            m = re.search(
                r'([a-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})',
                text, re.I
            )
            if m:
                month_name = m.group(1).lower()
                day        = int(m.group(2))
                year       = int(m.group(3))
                month      = _MONTHS.get(month_name)
                if month:
                    try: return date(year, month, day)
                    except ValueError: pass

        if m and len(m.groups()) == 3:
            try:
                day        = int(m.group(1))
                month_name = m.group(2).lower()
                year       = int(m.group(3))
                month      = _MONTHS.get(month_name)
                if month:
                    return date(year, month, day)
            except ValueError:
                pass

        # DD/MM/YYYY or MM/DD/YYYY
        m = re.search(r'(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})', text)
        if m:
            try:
                return date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
            except ValueError:
                try:
                    return date(int(m.group(3)), int(m.group(1)), int(m.group(2)))
                except ValueError:
                    pass

        # "15th March 1990" style already handled above
        # Try "March 15 1990" / "March 15th"
        m = re.search(r'(\b[a-z]+\b)\s+(\d{1,2})(?:st|nd|rd|th)?(?:\s+(\d{4}))?', text, re.I)
        if m:
            month = _MONTHS.get(m.group(1).lower())
            if month:
                day  = int(m.group(2))
                year = int(m.group(3)) if m.group(3) else today.year
                try:
                    candidate = date(year, month, day)
                    if future is True and candidate <= today and not m.group(3):
                        candidate = date(today.year + 1, month, day)
                    return candidate
                except ValueError:
                    pass

        return None
