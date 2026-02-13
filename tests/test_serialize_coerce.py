"""Tests for format-aware number coercion and output formatting in serialize.py."""

from pdf_ocr.serialize import _coerce_value, _apply_number_format, _parse_number_format


# ─── _parse_number_format ────────────────────────────────────────────────────


def test_parse_eu_comma_decimal():
    assert _parse_number_format("#,##") == ("", ",")


def test_parse_eu_full():
    assert _parse_number_format("#.###,##") == (".", ",")


def test_parse_us_full():
    assert _parse_number_format("#,###.##") == (",", ".")


def test_parse_us_thousands_only():
    assert _parse_number_format("#,###") == (",", "")


def test_parse_plain_decimal():
    assert _parse_number_format("#.##") == ("", ".")


def test_parse_none_default():
    assert _parse_number_format(None) == (",", ".")


# ─── _coerce_value — EU comma-decimal ────────────────────────────────────────


def test_eu_comma_decimal_float():
    assert _coerce_value("55826,3", "float", "#,##") == 55826.3


def test_eu_comma_decimal_int():
    assert _coerce_value("1234,0", "int", "#,##") == 1234


def test_eu_with_period_thousands():
    assert _coerce_value("1.234,56", "float", "#.###,##") == 1234.56


def test_eu_negative_parens():
    assert _coerce_value("(1.234,56)", "float", "#.###,##") == -1234.56


# ─── _coerce_value — US / backward compat ────────────────────────────────────


def test_us_default_backward_compat():
    """No format → US default: comma is thousands separator."""
    assert _coerce_value("1,234", "int") == 1234


def test_us_explicit_format():
    assert _coerce_value("1,234.56", "float", "#,###.##") == 1234.56


def test_no_format_strips_commas():
    """Backward compatible: no format still strips commas as thousands."""
    assert _coerce_value("55826,3", "float") == 558263.0


def test_us_thousands_only_format():
    assert _coerce_value("1,234", "int", "#,###") == 1234


# ─── _apply_number_format — EU output ────────────────────────────────────────


def test_eu_output_format():
    assert _apply_number_format(1234.56, "#.###,##") == "1.234,56"


def test_eu_output_no_thousands():
    assert _apply_number_format(55826.3, "#,##") == "55826,30"


def test_eu_output_negative():
    assert _apply_number_format(-1234.56, "#.###,##") == "-1.234,56"


# ─── _apply_number_format — US output ────────────────────────────────────────


def test_us_output_format():
    assert _apply_number_format(1234.56, "#,###.##") == "1,234.56"


def test_us_output_no_decimal():
    assert _apply_number_format(1234.56, "#,###") == "1,235"


def test_us_output_sign():
    assert _apply_number_format(42, "+#") == "+42"
    assert _apply_number_format(-42, "+#") == "-42"
