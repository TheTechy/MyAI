"""
test_suite/test_calculator.py
=============================
Unit tests for CalculatorSkill — no LLM, no network, no Flask required.
Covers every handler: temperature, unit conversion, percentage, sqrt, expression.
"""
import pytest
from myai_skills.calculator import CalculatorSkill


@pytest.fixture(scope="module")
def calc():
    return CalculatorSkill()


# ── Temperature conversion ─────────────────────────────────────────────────────
class TestTemperature:
    def test_f_to_c(self, calc):
        result = calc.execute("Convert 72°F to Celsius")
        assert "22.22" in result

    def test_c_to_f(self, calc):
        result = calc.execute("Convert 100°C to Fahrenheit")
        assert "212" in result

    def test_freezing_point(self, calc):
        result = calc.execute("0°C to Fahrenheit")
        assert "32" in result

    def test_boiling_point_c_to_f(self, calc):
        result = calc.execute("100 celsius to fahrenheit")
        assert "212" in result


# ── Unit conversions ───────────────────────────────────────────────────────────
class TestUnitConversion:
    def test_km_to_miles(self, calc):
        result = calc.execute("Convert 100km to miles")
        assert "62" in result

    def test_miles_to_km(self, calc):
        result = calc.execute("100 miles to km")
        assert "160" in result

    def test_kg_to_lbs(self, calc):
        # "How many lbs is 75kg?" — from SET 6
        result = calc.execute("How many lbs is 75kg?")
        assert "165" in result

    def test_kg_to_lbs_explicit(self, calc):
        result = calc.execute("75 kg to lbs")
        assert "165" in result


# ── Percentages ────────────────────────────────────────────────────────────────
class TestPercentage:
    def test_percent_off(self, calc):
        # "What is 35% off £899?" — from SET 6
        result = calc.execute("What is 35% off £899?")
        # 899 * 0.65 = 584.35
        assert "584" in result

    def test_increase_by_percent(self, calc):
        # "What is £85 increased by 12.5%?" — from SET 6
        result = calc.execute("What is £85 increased by 12.5%?")
        # 85 * 1.125 = 95.625
        assert "95.6" in result

    def test_percent_of(self, calc):
        result = calc.execute("What is 20% of 500?")
        assert "100" in result

    def test_percent_of_alt_phrasing(self, calc):
        result = calc.execute("15% of 200")
        assert "30" in result


# ── Arithmetic expressions ─────────────────────────────────────────────────────
class TestExpression:
    def test_grouped_expression(self, calc):
        # "What is (450 + 550) * 12?" — from SET 6
        result = calc.execute("What is (450 + 550) * 12?")
        assert "12,000" in result or "12000" in result

    def test_addition(self, calc):
        result = calc.execute("250 + 750")
        assert "1,000" in result or "1000" in result

    def test_subtraction(self, calc):
        result = calc.execute("1000 - 350")
        assert "650" in result

    def test_multiplication(self, calc):
        result = calc.execute("15 * 8")
        assert "120" in result

    def test_division(self, calc):
        result = calc.execute("100 / 4")
        assert "25" in result

    def test_power(self, calc):
        result = calc.execute("2 ** 10")
        assert "1,024" in result or "1024" in result


# ── Square root ────────────────────────────────────────────────────────────────
class TestSqrt:
    def test_sqrt_144(self, calc):
        result = calc.execute("square root of 144")
        assert "12" in result

    def test_sqrt_256(self, calc):
        result = calc.execute("sqrt(256)")
        assert "16" in result

    def test_squared(self, calc):
        result = calc.execute("12 squared")
        assert "144" in result


# ── Unknown / unparseable input ────────────────────────────────────────────────
def test_unknown_input_returns_help_text(calc):
    result = calc.execute("What is the meaning of life?")
    assert "couldn't parse" in result.lower() or "try" in result.lower()
