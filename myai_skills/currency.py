"""
skills/currency.py
==================
Currency conversion skill using the Frankfurter API.

https://www.frankfurter.app — free, no API key required,
European Central Bank rates, updated daily.

Handles
-------
  Simple conversion  : Convert £500 to USD
  Rate lookup        : What is the GBP to EUR rate?
  Multi-currency     : How much is £100 in euros, dollars and yen?
  Historical rates   : What was the GBP to USD rate on 1st January 2020?

Dependencies
------------
    pip install requests  (already installed for web_search)
"""

from __future__ import annotations

import re
from datetime import date, datetime

from .base import BaseSkill

try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False


# ── API ────────────────────────────────────────────────────────────────────────
_API_BASE    = "https://api.frankfurter.app"
_TIMEOUT     = 8


# ── Currency aliases ───────────────────────────────────────────────────────────
_CURRENCY_ALIASES: dict[str, str] = {
    # Symbols
    "£": "GBP", "$": "USD", "€": "EUR", "¥": "JPY", "¥": "CNY",
    "₹": "INR", "₩": "KRW", "₿": "BTC", "฿": "THB", "₣": "CHF",
    # Common names
    "pound": "GBP", "pounds": "GBP", "sterling": "GBP", "quid": "GBP",
    "dollar": "USD", "dollars": "USD", "buck": "USD", "bucks": "USD",
    "euro": "EUR", "euros": "EUR",
    "yen": "JPY", "japanese yen": "JPY",
    "yuan": "CNY", "renminbi": "CNY", "rmb": "CNY",
    "rupee": "INR", "rupees": "INR",
    "won": "KRW", "korean won": "KRW",
    "franc": "CHF", "swiss franc": "CHF", "francs": "CHF",
    "krona": "SEK", "kronor": "SEK", "swedish krona": "SEK",
    "krone": "NOK", "norwegian krone": "NOK",
    "danish krone": "DKK",
    "zloty": "PLN", "polish zloty": "PLN",
    "forint": "HUF", "hungarian forint": "HUF",
    "koruna": "CZK", "czech koruna": "CZK",
    "peso": "MXN", "mexican peso": "MXN",
    "real": "BRL", "reais": "BRL", "brazilian real": "BRL",
    "rand": "ZAR", "south african rand": "ZAR",
    "dirham": "AED", "uae dirham": "AED",
    "riyal": "SAR", "saudi riyal": "SAR",
    "baht": "THB", "thai baht": "THB",
    "ringgit": "MYR", "malaysian ringgit": "MYR",
    "dong": "VND", "vietnamese dong": "VND",
    "rupiah": "IDR", "indonesian rupiah": "IDR",
    "lira": "TRY", "turkish lira": "TRY",
    "ruble": "RUB", "russian ruble": "RUB",
    # ISO codes (uppercase and lowercase)
    "gbp": "GBP", "usd": "USD", "eur": "EUR", "jpy": "JPY",
    "cny": "CNY", "cad": "CAD", "aud": "AUD", "nzd": "NZD",
    "chf": "CHF", "sek": "SEK", "nok": "NOK", "dkk": "DKK",
    "pln": "PLN", "huf": "HUF", "czk": "CZK", "hkd": "HKD",
    "sgd": "SGD", "inr": "INR", "krw": "KRW", "mxn": "MXN",
    "brl": "BRL", "zar": "ZAR", "aed": "AED", "sar": "SAR",
    "thb": "THB", "myr": "MYR", "idr": "IDR", "try": "TRY",
    "rub": "RUB", "vnd": "VND",
}

# Currency symbols for formatting
_CURRENCY_SYMBOLS: dict[str, str] = {
    "GBP": "£", "USD": "$", "EUR": "€", "JPY": "¥",
    "CNY": "¥", "INR": "₹", "KRW": "₩", "CHF": "Fr",
    "AUD": "A$", "CAD": "C$", "NZD": "NZ$", "HKD": "HK$",
    "SGD": "S$", "MXN": "MX$", "BRL": "R$", "ZAR": "R",
    "AED": "د.إ", "SAR": "﷼", "THB": "฿", "TRY": "₺",
}

# Month name lookup for historical date parsing
_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2,
    "mar": 3, "march": 3, "apr": 4, "april": 4,
    "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


class CurrencySkill(BaseSkill):

    name        = "currency"
    description = "Live currency conversion using European Central Bank rates"




    # ── Main execution ────────────────────────────────────────────────────────
    def execute(self, query: str) -> str:
        if not _REQUESTS_AVAILABLE:
            return "Currency skill unavailable: requests not installed."

        handlers = [
            self._try_historical,
            self._try_conversion,
            self._try_rate_lookup,
        ]
        for handler in handlers:
            result = handler(query)
            if result is not None:
                return result

        return (
            f"I couldn't parse '{query}' as a currency query. "
            "Try: 'Convert £500 to USD', 'GBP to EUR rate', "
            "or 'What is £100 in euros?'"
        )

    # ── Conversion ────────────────────────────────────────────────────────────
    def _try_conversion(self, text: str) -> str | None:
        amount, from_code = self._extract_amount_and_currency(text)
        if amount is None or from_code is None:
            return None

        # Extract target currency/currencies
        to_codes = self._extract_target_currencies(text, exclude=from_code)
        if not to_codes:
            # Default to common currencies
            to_codes = ["USD", "EUR", "GBP"]
            to_codes = [c for c in to_codes if c != from_code]

        try:
            resp = requests.get(
                f"{_API_BASE}/latest",
                params={"from": from_code, "to": ",".join(to_codes)},
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            return f"Currency API error: {exc}"

        rates   = data.get("rates", {})
        updated = data.get("date", "unknown")

        sym   = _CURRENCY_SYMBOLS.get(from_code, from_code)
        lines = [f"{sym}{amount:,.2f} {from_code}  (rates as of {updated}):"]
        for code, rate in rates.items():
            converted = amount * rate
            to_sym    = _CURRENCY_SYMBOLS.get(code, code)
            lines.append(f"  {to_sym}{converted:>12,.2f}  {code}")

        return "\n".join(lines)

    # ── Rate lookup ───────────────────────────────────────────────────────────
    def _try_rate_lookup(self, text: str) -> str | None:
        # Find two currency codes
        codes = re.findall(
            r'\b(GBP|USD|EUR|JPY|CAD|AUD|NZD|CHF|SEK|NOK|DKK|PLN|HUF|CZK|HKD|SGD|INR|KRW|MXN|BRL|ZAR|AED|SAR|THB|MYR|TRY|RUB|CNY)\b',
            text, re.I
        )
        if len(codes) < 2:
            # Try name lookup
            from_code = self._name_to_code(text, position="first")
            to_code   = self._name_to_code(text, position="second")
            if from_code and to_code:
                codes = [from_code, to_code]
            else:
                return None

        from_code = codes[0].upper()
        to_code   = codes[1].upper()

        try:
            resp = requests.get(
                f"{_API_BASE}/latest",
                params={"from": from_code, "to": to_code},
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            return f"Currency API error: {exc}"

        rate    = data.get("rates", {}).get(to_code)
        updated = data.get("date", "unknown")

        if rate is None:
            return f"Could not retrieve {from_code}/{to_code} rate."

        return (
            f"Exchange rate ({updated}):\n"
            f"  1 {from_code} = {rate:,.4f} {to_code}\n"
            f"  1 {to_code} = {1/rate:,.4f} {from_code}"
        )

    # ── Historical rate ───────────────────────────────────────────────────────
    def _try_historical(self, text: str) -> str | None:
        hist_date = self._extract_date(text)
        if not hist_date:
            return None
        if hist_date >= date.today():
            return None  # not historical

        # Extract currencies from query
        codes = re.findall(
            r'\b(GBP|USD|EUR|JPY|CAD|AUD|CHF|CNY|INR|KRW)\b', text, re.I
        )
        if len(codes) < 2:
            return None

        from_code = codes[0].upper()
        to_code   = codes[1].upper()

        try:
            resp = requests.get(
                f"{_API_BASE}/{hist_date.isoformat()}",
                params={"from": from_code, "to": to_code},
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            return f"Currency API error: {exc}"

        rate = data.get("rates", {}).get(to_code)
        if rate is None:
            return None

        return (
            f"Historical rate on {hist_date.strftime('%d %B %Y')}:\n"
            f"  1 {from_code} = {rate:,.4f} {to_code}"
        )

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _extract_amount_and_currency(self, text: str) -> tuple[float | None, str | None]:
        """Extract a numeric amount and its currency from text."""
        # Symbol + number: £500, $1,000, €99.99
        m = re.search(r'([£$€¥₹₩฿])\s*([\d,]+(?:\.\d+)?)', text)
        if m:
            symbol = m.group(1)
            amount = float(m.group(2).replace(",", ""))
            code   = _CURRENCY_ALIASES.get(symbol)
            if code:
                return amount, code

        # Number + code: 500 GBP, 1000 USD
        m = re.search(r'([\d,]+(?:\.\d+)?)\s*(GBP|USD|EUR|JPY|CAD|AUD|NZD|CHF|SEK|NOK|DKK|PLN|HUF|CZK|HKD|SGD|INR|KRW|MXN|BRL|ZAR|AED|SAR|THB|MYR|TRY|RUB|CNY)\b', text, re.I)
        if m:
            amount = float(m.group(1).replace(",", ""))
            code   = m.group(2).upper()
            return amount, code

        # Number + currency name: 500 pounds, 100 euros
        m = re.search(r'([\d,]+(?:\.\d+)?)\s+(pounds?|dollars?|euros?|yen|yuan|rupees?|francs?)\b', text, re.I)
        if m:
            amount = float(m.group(1).replace(",", ""))
            code   = _CURRENCY_ALIASES.get(m.group(2).lower().rstrip("s"), "")
            if code:
                return amount, code

        return None, None

    def _extract_target_currencies(self, text: str, exclude: str = "") -> list[str]:
        """Find target currencies mentioned in the query."""
        codes = []

        # ISO codes
        found = re.findall(
            r'\b(GBP|USD|EUR|JPY|CAD|AUD|NZD|CHF|SEK|NOK|DKK|PLN|HUF|CZK|HKD|SGD|INR|KRW|MXN|BRL|ZAR|AED|SAR|THB|MYR|TRY|RUB|CNY)\b',
            text, re.I
        )
        codes += [c.upper() for c in found if c.upper() != exclude]

        # Currency names after 'to' / 'in' / 'into'
        m = re.search(r'\b(?:to|in|into)\s+(pounds?|dollars?|euros?|yen|yuan|rupees?|francs?)\b', text, re.I)
        if m:
            code = _CURRENCY_ALIASES.get(m.group(1).lower().rstrip("s"))
            if code and code not in codes and code != exclude:
                codes.append(code)

        return list(dict.fromkeys(codes))  # deduplicate preserving order

    def _name_to_code(self, text: str, position: str = "first") -> str | None:
        """Find first or second currency name/code mention in text."""
        text_lower = text.lower()
        found = []
        for alias, code in sorted(_CURRENCY_ALIASES.items(), key=lambda x: -len(x[0])):
            if alias in text_lower and code not in found:
                found.append(code)
        if position == "first" and found:
            return found[0]
        if position == "second" and len(found) > 1:
            return found[1]
        return None

    def _extract_date(self, text: str) -> date | None:
        """Parse a historical date from text."""
        # "1st January 2020" / "January 1st 2020"
        m = re.search(r'(\d{1,2})(?:st|nd|rd|th)?\s+([a-z]+)\s+(\d{4})', text, re.I)
        if m:
            month = _MONTHS.get(m.group(2).lower())
            if month:
                try:
                    return date(int(m.group(3)), month, int(m.group(1)))
                except ValueError:
                    pass

        m = re.search(r'([a-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})', text, re.I)
        if m:
            month = _MONTHS.get(m.group(1).lower())
            if month:
                try:
                    return date(int(m.group(3)), month, int(m.group(2)))
                except ValueError:
                    pass

        # YYYY-MM-DD
        m = re.search(r'(\d{4})-(\d{2})-(\d{2})', text)
        if m:
            try:
                return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError:
                pass

        return None