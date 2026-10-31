"""
skills/calculator.py
====================
Calculator skill — safe maths evaluation, unit conversions, and percentages.

No external dependencies — pure Python stdlib only.

Handles
-------
  Basic maths     : 2 + 2, 15 * 8, 100 / 4, 2 ** 10
  Percentages     : 15% of 340, what is 20% off £250
  Unit conversion : 72°F to Celsius, 100km to miles, 5kg to lbs
  Square roots    : square root of 144, sqrt(256)
  Rounding        : round 3.14159 to 2 decimal places
"""

from __future__ import annotations

import math
import re

from .base import BaseSkill


# ── Safe evaluation ───────────────────────────────────────────────────────────
# Only these names are available in eval — no builtins, no imports
_SAFE_GLOBALS = {
    "__builtins__": {},
    "abs": abs, "round": round, "min": min, "max": max,
    "sqrt": math.sqrt, "pi": math.pi, "e": math.e,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "log": math.log, "log10": math.log10, "log2": math.log2,
    "floor": math.floor, "ceil": math.ceil,
    "pow": math.pow, "factorial": math.factorial,
}

# ── Unit conversion tables ────────────────────────────────────────────────────
_CONVERSIONS = {
    # Length
    ("km",     "miles"):  0.621371,
    ("miles",  "km"):     1.60934,
    ("miles",  "ft"):     5280,
    ("km",     "m"):      1000,
    ("m",      "km"):     0.001,
    ("m",      "ft"):     3.28084,
    ("ft",     "m"):      0.3048,
    ("m",      "cm"):     100,
    ("cm",     "m"):      0.01,
    ("m",      "mm"):     1000,
    ("mm",     "m"):      0.001,
    ("m",      "yards"):  1.09361,
    ("yards",  "m"):      0.9144,
    ("cm",     "inches"): 0.393701,
    ("inches", "cm"):     2.54,
    ("cm",     "ft"):     0.0328084,
    ("ft",     "cm"):     30.48,
    ("cm",     "mm"):     10,
    ("mm",     "cm"):     0.1,
    ("mm",     "inches"): 0.0393701,
    ("inches", "mm"):     25.4,
    ("inches", "m"):      0.0254,
    # Weight
    ("kg",     "lbs"):    2.20462,
    ("lbs",    "kg"):     0.453592,
    ("kg",     "oz"):     35.274,
    ("oz",     "kg"):     0.0283495,
    ("kg",     "g"):      1000,
    ("g",      "kg"):     0.001,
    ("g",      "oz"):     0.035274,
    ("oz",     "g"):      28.3495,
    ("lbs",    "oz"):     16,
    ("oz",     "lbs"):    0.0625,
    ("stone",  "kg"):     6.35029,
    ("kg",     "stone"):  0.157473,
    # Volume
    ("litres", "gallons"):0.264172,
    ("liters", "gallons"):0.264172,
    ("gallons","litres"): 3.78541,
    ("litres", "ml"):     1000,
    ("ml",     "litres"): 0.001,
    ("ml",     "fl oz"):  0.033814,
    ("fl oz",  "ml"):     29.5735,
    # Speed
    ("mph",    "kph"):    1.60934,
    ("kph",    "mph"):    0.621371,
    ("mph",    "ms"):     0.44704,
    ("knots",  "mph"):    1.15078,
    ("mph",    "knots"):  0.868976,
    # Area
    ("sqm",    "sqft"):   10.7639,
    ("sqft",   "sqm"):    0.092903,
    ("acres",  "hectares"):0.404686,
    ("hectares","acres"): 2.47105,
    # Data
    ("kb",     "mb"):     0.001,
    ("mb",     "gb"):     0.001,
    ("gb",     "tb"):     0.001,
    ("mb",     "kb"):     1000,
    ("gb",     "mb"):     1000,
    ("tb",     "gb"):     1000,
}


class CalculatorSkill(BaseSkill):

    name        = "calculator"
    description = "Perform maths calculations, unit conversions, and percentages"

    # ── Main execution ────────────────────────────────────────────────────────
    def execute(self, query: str) -> str:
        query = query.strip()

        # Strip currency symbols — treat £, $, €, ¥ as numeric prefixes
        normalised = re.sub(r'[£$€¥]', '', query)

        # Try each handler in order
        handlers = [
            self._try_temperature,
            self._try_unit_conversion,
            self._try_percentage,
            self._try_sqrt,
            self._try_expression,
        ]

        for handler in handlers:
            result = handler(normalised)
            if result is not None:
                return result

        return (
            f"I couldn't parse '{query}' as a calculation. "
            "Try formats like: '15% of 340', '72°F to Celsius', "
            "'sqrt(144)', or '25 * 4 + 10'."
        )

    # ── Temperature conversion ────────────────────────────────────────────────
    def _try_temperature(self, text: str) -> str | None:
        # °F to °C
        m = re.search(r'([\-\d\.]+)\s*(?:°f|f|fahrenheit)\s+(?:to|in)\s+(?:°c|c|celsius|centigrade)', text, re.I)
        if m:
            f = float(m.group(1))
            c = (f - 32) * 5 / 9
            return f"{f}°F = {c:.2f}°C"

        # °C to °F
        m = re.search(r'([\-\d\.]+)\s*(?:°c|c|celsius|centigrade)\s+(?:to|in)\s+(?:°f|f|fahrenheit)', text, re.I)
        if m:
            c = float(m.group(1))
            f = (c * 9 / 5) + 32
            return f"{c}°C = {f:.2f}°F"

        # Standalone °F (assume wants °C)
        m = re.search(r'([\-\d\.]+)\s*(?:°f|fahrenheit)\b', text, re.I)
        if m and re.search(r'\b(celsius|°c|convert|to)\b', text, re.I):
            f = float(m.group(1))
            c = (f - 32) * 5 / 9
            return f"{f}°F = {c:.2f}°C"

        # Standalone °C (assume wants °F)
        m = re.search(r'([\-\d\.]+)\s*(?:°c|celsius)\b', text, re.I)
        if m and re.search(r'\b(fahrenheit|°f|convert|to)\b', text, re.I):
            c = float(m.group(1))
            f = (c * 9 / 5) + 32
            return f"{c}°C = {f:.2f}°F"

        return None

    # ── Unit conversion ───────────────────────────────────────────────────────
    _UNIT_ALIASES = {
        "kilometer": "km",  "kilometre": "km",  "kilometers": "km", "kilometres": "km",
        "mile":      "miles", "pound":  "lbs",  "kilogram": "kg",   "kilograms": "kg",
        "gram":      "g",   "grams":    "g",    "ounce":    "oz",   "ounces":    "oz",
        "litre":     "litres","litres": "litres","liter":   "liters","liters":   "liters",
        "gallon":    "gallons","gallons":"gallons","foot":   "ft",   "feet":      "ft",
        "inch":      "inches","yard":   "yards", "yards":   "yards",
        "megabyte":  "mb",  "gigabyte": "gb",   "terabyte": "tb",  "kilobyte":  "kb",
        "lb":        "lbs", "kilo":     "kg",   "meter":    "m",   "metre":     "m",
        "meters":    "m",   "metres":   "m",    "millimeter":"mm",  "millimetre":"mm",
        "millimeters":"mm", "millimetres":"mm", "centimeter":"cm",  "centimetre":"cm",
        "centimeters":"cm", "centimetres":"cm",
    }

    def _resolve_unit(self, raw: str) -> str:
        """Resolve a raw unit string to its canonical _CONVERSIONS key.

        Checks alias table first, then known conversion units, then
        strips a trailing 's' for naive plurals — but never blindly
        strips 'es' endings (e.g. 'inches' must stay 'inches').
        """
        u = raw.strip().lower().rstrip("?.,")
        # 1. Direct alias hit (handles 'feet', 'inch', 'kilometre', …)
        if u in self._UNIT_ALIASES:
            return self._UNIT_ALIASES[u]
        # 2. Already a valid key in the conversion table
        all_units = {unit for pair in _CONVERSIONS for unit in pair}
        if u in all_units:
            return u
        # 3. Try stripping a trailing 's' for simple plurals (km → km, miles → mile? no)
        #    Only strip if the result is in aliases or known units
        stripped = u[:-1] if u.endswith("s") and not u.endswith("es") else u
        if stripped != u:
            if stripped in self._UNIT_ALIASES:
                return self._UNIT_ALIASES[stripped]
            if stripped in all_units:
                return stripped
        return u  # return as-is; lookup will simply miss

    def _try_unit_conversion(self, text: str) -> str | None:
        # ── Pattern 1: compact no-space notation — "995mm in inches", "5km to miles"
        # checked first so the unit isn't swallowed into the lazy group(2) below
        m = re.search(
            r'([\d,\.]+)(mm|cm|km|m|ft|g|kg|oz|lbs?|ml)\s+(?:to|in|into|is)\s+([a-z /]+)',
            text, re.I
        )
        if m:
            try:
                value  = float(m.group(1).replace(",", ""))
                from_u = self._resolve_unit(m.group(2))
                to_u   = self._resolve_unit(m.group(3))
                factor = _CONVERSIONS.get((from_u, to_u))
                if factor:
                    return f"{value:g} {from_u} = {value * factor:,.4g} {to_u}"
            except ValueError:
                pass

        # ── Pattern 2: "how many UNIT in/are NUMBER[unit]" — "how many inches in 995mm?"
        m = re.search(
            r'how\s+many\s+([a-z]+)\s+(?:is|are|in)\s+([\d,\.]+)\s*(mm|cm|km|m|ft|g|kg|oz|lbs?|ml|[a-z]+)',
            text, re.I
        )
        if m:
            try:
                to_u   = self._resolve_unit(m.group(1))
                value  = float(m.group(2).replace(",", ""))
                from_u = self._resolve_unit(m.group(3))
                factor = _CONVERSIONS.get((from_u, to_u))
                if factor:
                    return f"{value:g} {from_u} = {value * factor:,.4g} {to_u}"
            except ValueError:
                pass

        # ── Pattern 3: general spaced — "100 km to miles", "What is 5 kg in lbs?"
        m = re.search(
            r'([\d,\.]+)\s+([a-z /]+?)\s+(?:to|in|into|is|as)\s+([a-z /]+)',
            text, re.I
        )
        if not m:
            return None

        try:
            value  = float(m.group(1).replace(",", ""))
            from_u = self._resolve_unit(m.group(2))
            to_u   = self._resolve_unit(m.group(3))
        except ValueError:
            return None

        factor = _CONVERSIONS.get((from_u, to_u))
        if factor is None:
            return None

        return f"{value:g} {from_u} = {value * factor:,.4g} {to_u}"

    # ── Percentage ────────────────────────────────────────────────────────────
    def _try_percentage(self, text: str) -> str | None:
        # "X% of Y"
        m = re.search(r'([\d\.]+)\s*%\s+of\s+([\d,\.]+)', text, re.I)
        if m:
            pct    = float(m.group(1))
            value  = float(m.group(2).replace(",", ""))
            result = value * pct / 100
            return f"{pct}% of {value:,.2f} = {result:,.2f}"

        # "X% off Y" (discount)
        m = re.search(r'([\d\.]+)\s*%\s+off\s+([\d,\.]+)', text, re.I)
        if m:
            pct      = float(m.group(1))
            value    = float(m.group(2).replace(",", ""))
            discount = value * pct / 100
            final    = value - discount
            return (
                f"{pct}% off {value:,.2f}:\n"
                f"  Discount : {discount:,.2f}\n"
                f"  Final    : {final:,.2f}"
            )

        # "Y increased by X%" / "increase X by Y%"
        m = re.search(r'([\d,\.]+)\s+(?:increased?|increase)\s+by\s+([\d\.]+)\s*%', text, re.I)
        if m:
            value    = float(m.group(1).replace(",", ""))
            pct      = float(m.group(2))
            increase = value * pct / 100
            final    = value + increase
            return (
                f"{value:,.2f} increased by {pct}%:\n"
                f"  Increase : {increase:,.2f}\n"
                f"  Final    : {final:,.2f}"
            )

        # "Y decreased by X%" / "reduce Y by X%"
        m = re.search(r'([\d,\.]+)\s+(?:decreased?|decrease|reduced?|reduce)\s+by\s+([\d\.]+)\s*%', text, re.I)
        if not m:
            m = re.search(r'(?:reduce|decrease)\s+([\d,\.]+)\s+by\s+([\d\.]+)\s*%', text, re.I)
        if m:
            value    = float(m.group(1).replace(",", ""))
            pct      = float(m.group(2))
            decrease = value * pct / 100
            final    = value - decrease
            return (
                f"{value:,.2f} decreased by {pct}%:\n"
                f"  Decrease : {decrease:,.2f}\n"
                f"  Final    : {final:,.2f}"
            )

        # "what is X% of Y" / "X percent of Y"
        m = re.search(r'([\d\.]+)\s*(?:%|percent)\s+of\s+([\d,\.]+)', text, re.I)
        if m:
            pct    = float(m.group(1))
            value  = float(m.group(2).replace(",", ""))
            result = value * pct / 100
            return f"{pct}% of {value:,.2f} = {result:,.2f}"

        return None

    # ── Square root / powers ──────────────────────────────────────────────────
    def _try_sqrt(self, text: str) -> str | None:
        m = re.search(r'(?:square root|sqrt)\s*(?:of)?\s*\(?([\d\.]+)\)?', text, re.I)
        if m:
            val    = float(m.group(1))
            result = math.sqrt(val)
            return f"√{val:g} = {result:g}"

        m = re.search(r'([\d\.]+)\s+squared', text, re.I)
        if m:
            val = float(m.group(1))
            return f"{val:g}² = {val**2:g}"

        m = re.search(r'([\d\.]+)\s+cubed', text, re.I)
        if m:
            val = float(m.group(1))
            return f"{val:g}³ = {val**3:g}"

        return None

    # ── General expression ────────────────────────────────────────────────────
    def _try_expression(self, text: str) -> str | None:
        """
        Extract and safely evaluate a mathematical expression.
        Handles word operators (times, divided by, plus, minus) and
        cases where markdown rendering stripped the * operator.
        """
        # Normalise word operators first
        expr_text = text
        expr_text = re.sub(r'\bto\s+the\s+power\s+of\b', '**', expr_text, flags=re.I)
        expr_text = re.sub(r'\btimes\b',        '*', expr_text, flags=re.I)
        expr_text = re.sub(r'\bdivided\s+by\b', '/', expr_text, flags=re.I)
        expr_text = re.sub(r'\bplus\b',         '+', expr_text, flags=re.I)
        expr_text = re.sub(r'\bminus\b',        '-', expr_text, flags=re.I)
        expr_text = re.sub(r'×', '*', expr_text)
        expr_text = re.sub(r'÷', '/', expr_text)
        expr_text = expr_text.replace("^", "**")
        expr_text = expr_text.replace(",", "")

        # Restore * stripped by markdown italic rendering:
        # "(...) 12"  → "(...) * 12"
        # "12 (...)"  → "12 * (...)"
        # "350 12"    → "350 * 12"   (two bare numbers)
        expr_text = re.sub(r'\)\s+(\d)', r') * \1', expr_text)
        expr_text = re.sub(r'(\d)\s+\(', r'\1 * (', expr_text)

        # Extract expression — must start and end with a digit or bracket
        m = re.search(r'([\(\d][\d\s\.\+\-\*\/\(\)]+[\d\)])', expr_text)
        if not m:
            return None

        expr = m.group(1).strip()

        # Must contain at least one operator
        if not re.search(r'[\+\-\*\/]', expr):
            # Check if two numbers are adjacent with no operator (markdown ate the *)
            # e.g. "347 28" — try to infer multiplication
            nums = re.findall(r'\d+\.?\d*', expr)
            if len(nums) == 2:
                expr = f"{nums[0]} * {nums[1]}"
                original = f"{nums[0]} * {nums[1]}"
            else:
                return None
        else:
            original = expr.strip()

        # Strict whitelist
        if not re.fullmatch(r'[\d\s\.\+\-\*\/\(\)]+', expr):
            return None

        try:
            result = eval(expr, _SAFE_GLOBALS, {})  # noqa: S307
            if isinstance(result, float) and result == int(result):
                result = int(result)
            if isinstance(result, (int, float)):
                return f"{original} = {result:,}"
        except Exception:
            return None

        return None