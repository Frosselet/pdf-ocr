"""Serialization module: convert interpret_table() results to various formats.

Provides functions to serialize MappedTable results (from ``interpret_table()``)
to CSV, TSV, Parquet, pandas DataFrames, and polars DataFrames. All functions
validate records against the CanonicalSchema using Pydantic before serialization,
and apply output formatting specified in the schema's ``format`` field.

Usage::

    from pdf_ocr import interpret_table, CanonicalSchema, ColumnDef, to_csv, to_pandas

    result = interpret_table(compressed, schema)

    # Serialize to CSV string
    csv_str = to_csv(result, schema)

    # Write to file with page column
    to_csv(result, schema, path="output.csv", include_page=True)

    # Get pandas DataFrame
    df = to_pandas(result, schema, include_page=True)

Format Specification:
    - Dates: YYYY-MM-DD, YYYY-MM, YYYY, DD/MM/YYYY, MM/DD/YYYY, MMM YYYY,
      MMMM YYYY, YYYY-MM-DD HH:mm, YYYY-MM-DD HH:mm:ss, HH:mm, hh:mm A, etc.
    - Numbers: # (plain), #.## (decimals), #,### (thousands), #,###.## (both),
      +# (explicit sign), #% (percentage), (#) (negative in parentheses).
    - Strings: uppercase, lowercase, titlecase, capitalize, camelCase,
      PascalCase, snake_case, SCREAMING_SNAKE_CASE, kebab-case,
      SCREAMING-KEBAB-CASE, trim.
"""

from __future__ import annotations

import csv
import io
import re
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

from pydantic import BaseModel, ValidationError, create_model

from pdf_ocr.interpret import CanonicalSchema, ColumnDef

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl
    from baml_client import types as baml_types


class SerializationValidationError(Exception):
    """Raised when a record fails Pydantic validation against the schema."""

    def __init__(
        self,
        message: str,
        record_index: int,
        column_name: str | None = None,
        original_error: ValidationError | None = None,
    ):
        super().__init__(message)
        self.record_index = record_index
        self.column_name = column_name
        self.original_error = original_error


# ─── Type mappings ────────────────────────────────────────────────────────────

# Schema type -> Python type for Pydantic validation
_PYTHON_TYPE_MAP: dict[str, type] = {
    "string": str,
    "int": int,
    "float": float,
    "bool": bool,
    "date": str,  # dates stored as strings
}

# Schema type -> pandas nullable dtype
_PANDAS_DTYPE_MAP: dict[str, str] = {
    "string": "string",
    "int": "Int64",
    "float": "Float64",
    "bool": "boolean",
    "date": "string",
}

# Schema type -> polars dtype (as string for lazy import)
_POLARS_DTYPE_MAP: dict[str, str] = {
    "string": "Utf8",
    "int": "Int64",
    "float": "Float64",
    "bool": "Boolean",
    "date": "Utf8",
}

# Month name mappings
_MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
_MONTH_FULL = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]


# ─── Format application helpers ───────────────────────────────────────────────


def _apply_date_format(value: str | None, fmt: str) -> str | None:
    """Apply date/time format to a value.

    Attempts to parse the value as a date/datetime and reformat it according
    to the format specification. If parsing fails, returns the original value.

    Supported format tokens:
        YYYY - 4-digit year
        MM - 2-digit month (01-12)
        MMM - Short month name (Jan-Dec)
        MMMM - Full month name (January-December)
        DD - 2-digit day (01-31)
        HH - 24-hour hour (00-23)
        hh - 12-hour hour (01-12)
        mm - Minutes (00-59)
        ss - Seconds (00-59)
        A - AM/PM
        T - Literal 'T' separator
        Z - Literal 'Z' (UTC indicator)
    """
    if value is None:
        return None

    value = str(value).strip()
    if not value:
        return None

    # Try to parse the date value
    dt = None
    parse_formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d",
        "%Y-%m",
        "%Y",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%d/%m/%Y",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%m/%d/%Y",
        "%d-%m-%Y",
        "%m-%d-%Y",
        "%H:%M:%S",
        "%H:%M",
    ]

    for parse_fmt in parse_formats:
        try:
            dt = datetime.strptime(value, parse_fmt)
            break
        except ValueError:
            continue

    if dt is None:
        # Could not parse - return original value
        return value

    # Build output string by replacing format tokens
    result = fmt

    # Replace tokens (order matters - longer tokens first)
    result = result.replace("YYYY", f"{dt.year:04d}")
    result = result.replace("MMMM", _MONTH_FULL[dt.month - 1])
    result = result.replace("MMM", _MONTH_ABBR[dt.month - 1])
    result = result.replace("MM", f"{dt.month:02d}")
    result = result.replace("DD", f"{dt.day:02d}")
    result = result.replace("HH", f"{dt.hour:02d}")

    # 12-hour format
    hour_12 = dt.hour % 12
    if hour_12 == 0:
        hour_12 = 12
    result = result.replace("hh", f"{hour_12:02d}")

    result = result.replace("mm", f"{dt.minute:02d}")
    result = result.replace("ss", f"{dt.second:02d}")
    result = result.replace("A", "AM" if dt.hour < 12 else "PM")

    return result


def _apply_number_format(value: int | float | None, fmt: str) -> str | None:
    """Apply number format to a value.

    Supported format patterns:
        # - Plain number
        #.## - Decimal places (number of # after . determines precision)
        #,### - Thousands separator
        #,###.## - Both
        +# - Explicit sign (+ for positive, - for negative)
        #% - Percentage (appends %)
        (#) - Negative in parentheses (accounting style)
    """
    if value is None:
        return None

    # Parse format options
    use_thousands = "," in fmt
    use_sign = fmt.startswith("+")
    use_percent = fmt.endswith("%")
    use_parens = fmt.startswith("(") and fmt.endswith(")")

    # Determine decimal places
    decimal_places = 0
    if "." in fmt:
        decimal_part = fmt.split(".")[-1]
        # Count # characters after decimal point (before any % or ))
        decimal_part = decimal_part.rstrip("%)")
        decimal_places = len(decimal_part)

    # Format the number
    is_negative = value < 0
    abs_value = abs(value)

    # Round to decimal places
    if decimal_places > 0:
        formatted = f"{abs_value:.{decimal_places}f}"
    else:
        formatted = str(int(round(abs_value)))

    # Add thousands separator
    if use_thousands:
        if "." in formatted:
            int_part, dec_part = formatted.split(".")
        else:
            int_part, dec_part = formatted, None

        # Add commas to integer part
        int_part = "{:,}".format(int(int_part.replace(",", "")))

        formatted = f"{int_part}.{dec_part}" if dec_part else int_part

    # Apply sign/parentheses
    if is_negative:
        if use_parens:
            formatted = f"({formatted})"
        else:
            formatted = f"-{formatted}"
    elif use_sign:
        formatted = f"+{formatted}"

    # Add percent
    if use_percent:
        formatted = f"{formatted}%"

    return formatted


def _to_words(s: str) -> list[str]:
    """Split a string into words for case conversion.

    Handles camelCase, PascalCase, snake_case, kebab-case, and space-separated.
    """
    # Replace common separators with spaces
    s = re.sub(r"[-_]", " ", s)
    # Insert space before uppercase letters (for camelCase/PascalCase)
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    # Split on whitespace and filter empty strings
    return [w for w in s.split() if w]


def _apply_string_format(value: str | None, fmt: str) -> str | None:
    """Apply string format/case transformation to a value.

    Supported formats:
        uppercase - ALL CAPS
        lowercase - all lower
        titlecase - Title Case
        capitalize - First letter only
        camelCase - camelCase
        PascalCase - PascalCase
        snake_case - snake_case
        SCREAMING_SNAKE_CASE - SCREAMING_SNAKE_CASE
        kebab-case - kebab-case
        SCREAMING-KEBAB-CASE - SCREAMING-KEBAB-CASE
        trim - Strip whitespace
    """
    if value is None:
        return None

    value = str(value)

    if fmt == "uppercase":
        return value.upper()
    elif fmt == "lowercase":
        return value.lower()
    elif fmt == "titlecase":
        return value.title()
    elif fmt == "capitalize":
        return value.capitalize()
    elif fmt == "trim":
        return value.strip()
    elif fmt == "camelCase":
        words = _to_words(value)
        if not words:
            return value
        return words[0].lower() + "".join(w.capitalize() for w in words[1:])
    elif fmt == "PascalCase":
        words = _to_words(value)
        return "".join(w.capitalize() for w in words)
    elif fmt == "snake_case":
        words = _to_words(value)
        return "_".join(w.lower() for w in words)
    elif fmt == "SCREAMING_SNAKE_CASE":
        words = _to_words(value)
        return "_".join(w.upper() for w in words)
    elif fmt == "kebab-case":
        words = _to_words(value)
        return "-".join(w.lower() for w in words)
    elif fmt == "SCREAMING-KEBAB-CASE":
        words = _to_words(value)
        return "-".join(w.upper() for w in words)
    else:
        # Unknown format - return as-is
        return value


def _apply_format(value: Any, col: ColumnDef) -> Any:
    """Apply the column's format specification to a value.

    Args:
        value: The coerced value to format.
        col: The column definition containing type and format info.

    Returns:
        The formatted value, or the original value if no format is specified.
    """
    if col.format is None:
        return value

    if value is None:
        return None

    # Determine which formatter to use based on column type
    if col.type in ("date",) or (col.type == "string" and _is_date_format(col.format)):
        return _apply_date_format(value, col.format)
    elif col.type in ("int", "float"):
        return _apply_number_format(value, col.format)
    elif col.type == "string":
        return _apply_string_format(value, col.format)
    else:
        return value


def _is_date_format(fmt: str) -> bool:
    """Check if a format string looks like a date/time format."""
    date_tokens = ["YYYY", "MM", "DD", "HH", "hh", "mm", "ss", "MMM", "MMMM"]
    return any(token in fmt for token in date_tokens)


# ─── Internal helpers ─────────────────────────────────────────────────────────


def _build_validator_model(schema: CanonicalSchema) -> type[BaseModel]:
    """Dynamically create a Pydantic model from a CanonicalSchema.

    All fields are optional (allow None) since extracted data may have missing values.
    """
    fields: dict[str, tuple[type, Any]] = {}
    for col in schema.columns:
        py_type = _PYTHON_TYPE_MAP.get(col.type, str)
        # Make all fields optional with None default
        fields[col.name] = (py_type | None, None)

    return create_model("RecordValidator", **fields)


def _coerce_value(value: Any, col_type: str) -> Any:
    """Coerce a value to the expected type, handling common OCR artifacts.

    - Handles comma-separated numbers like "1,234" -> 1234
    - Handles percentage strings like "12.5%" -> 12.5
    - Handles whitespace and empty strings -> None
    - Handles boolean strings like "yes", "true", "1" -> True
    """
    if value is None:
        return None

    # Convert to string for processing
    if not isinstance(value, str):
        value = str(value)

    # Strip whitespace
    value = value.strip()

    # Empty string -> None
    if not value:
        return None

    if col_type == "int":
        # Remove commas, spaces, and currency symbols
        cleaned = re.sub(r"[,\s$€£¥]", "", value)
        # Remove trailing percentage
        cleaned = cleaned.rstrip("%")
        # Handle parentheses for negative numbers: (123) -> -123
        if cleaned.startswith("(") and cleaned.endswith(")"):
            cleaned = "-" + cleaned[1:-1]
        try:
            # Try parsing as float first then convert to int (handles "1.0")
            return int(float(cleaned))
        except (ValueError, TypeError):
            return None

    elif col_type == "float":
        # Remove commas, spaces, and currency symbols
        cleaned = re.sub(r"[,\s$€£¥]", "", value)
        # Remove trailing percentage
        cleaned = cleaned.rstrip("%")
        # Handle parentheses for negative numbers
        if cleaned.startswith("(") and cleaned.endswith(")"):
            cleaned = "-" + cleaned[1:-1]
        try:
            return float(cleaned)
        except (ValueError, TypeError):
            return None

    elif col_type == "bool":
        lower = value.lower()
        if lower in ("true", "yes", "1", "y", "t"):
            return True
        elif lower in ("false", "no", "0", "n", "f"):
            return False
        return None

    elif col_type in ("string", "date"):
        return value

    # Unknown type - return as string
    return value


def _flatten_and_validate(
    result: dict[int, baml_types.MappedTable] | baml_types.MappedTable,
    schema: CanonicalSchema,
    include_page: bool,
) -> list[dict[str, Any]]:
    """Flatten MappedTable result(s) to a list of dicts, validating each record.

    Args:
        result: Output from ``interpret_table()`` (page-keyed dict) or single MappedTable.
        schema: Canonical schema for validation.
        include_page: If True, add a ``page`` column with 1-indexed page number.

    Returns:
        List of validated record dicts with coerced values.

    Raises:
        SerializationValidationError: If a record fails validation.
    """
    # Import here to avoid circular dependency
    from baml_client import types as baml_types

    # Normalize input to list of (page_number, MappedTable)
    if isinstance(result, dict):
        tables = [(page, mt) for page, mt in sorted(result.items())]
    else:
        tables = [(1, result)]

    # Build validator model
    ValidatorModel = _build_validator_model(schema)

    # Build column lookups
    col_types = {col.name: col.type for col in schema.columns}
    col_by_name = {col.name: col for col in schema.columns}
    col_names = [col.name for col in schema.columns]

    records: list[dict[str, Any]] = []
    global_idx = 0

    for page_num, mapped_table in tables:
        for rec in mapped_table.records:
            # Extract raw values from MappedRecord
            raw = rec.model_dump()

            # Coerce values to expected types
            coerced: dict[str, Any] = {}
            for col_name in col_names:
                raw_value = raw.get(col_name)
                col_type = col_types.get(col_name, "string")
                coerced[col_name] = _coerce_value(raw_value, col_type)

            # Validate with Pydantic
            try:
                validated = ValidatorModel(**coerced)
                record_dict = validated.model_dump()
            except ValidationError as e:
                # Find the first failing field
                failing_col = None
                if e.errors():
                    loc = e.errors()[0].get("loc", ())
                    if loc:
                        failing_col = str(loc[0])
                raise SerializationValidationError(
                    f"Validation failed for record {global_idx}: {e}",
                    record_index=global_idx,
                    column_name=failing_col,
                    original_error=e,
                ) from e

            # Apply format specifications
            formatted: dict[str, Any] = {}
            for col_name, value in record_dict.items():
                col = col_by_name.get(col_name)
                if col is not None:
                    formatted[col_name] = _apply_format(value, col)
                else:
                    formatted[col_name] = value

            # Add page column if requested
            if include_page:
                formatted = {"page": page_num, **formatted}

            records.append(formatted)
            global_idx += 1

    return records


def _get_column_order(schema: CanonicalSchema, include_page: bool) -> list[str]:
    """Get ordered list of column names for output."""
    cols = [col.name for col in schema.columns]
    if include_page:
        cols = ["page"] + cols
    return cols


# ─── Public API ───────────────────────────────────────────────────────────────


def to_csv(
    result: dict[int, baml_types.MappedTable] | baml_types.MappedTable,
    schema: CanonicalSchema,
    *,
    path: str | Path | None = None,
    include_page: bool = False,
) -> str | None:
    """Serialize interpretation result to CSV format.

    Args:
        result: Output from ``interpret_table()`` (page-keyed dict) or single MappedTable.
        schema: Canonical schema for validation and column ordering.
        path: If provided, write to this file path and return None.
            If None, return CSV as a string.
        include_page: If True, add a ``page`` column with 1-indexed page number.

    Returns:
        CSV string if ``path`` is None, otherwise None.

    Raises:
        SerializationValidationError: If a record fails validation.
    """
    records = _flatten_and_validate(result, schema, include_page)
    columns = _get_column_order(schema, include_page)

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(records)

    csv_str = output.getvalue()

    if path is not None:
        Path(path).write_text(csv_str, encoding="utf-8")
        return None

    return csv_str


def to_tsv(
    result: dict[int, baml_types.MappedTable] | baml_types.MappedTable,
    schema: CanonicalSchema,
    *,
    path: str | Path | None = None,
    include_page: bool = False,
) -> str | None:
    """Serialize interpretation result to TSV format.

    Args:
        result: Output from ``interpret_table()`` (page-keyed dict) or single MappedTable.
        schema: Canonical schema for validation and column ordering.
        path: If provided, write to this file path and return None.
            If None, return TSV as a string.
        include_page: If True, add a ``page`` column with 1-indexed page number.

    Returns:
        TSV string if ``path`` is None, otherwise None.

    Raises:
        SerializationValidationError: If a record fails validation.
    """
    records = _flatten_and_validate(result, schema, include_page)
    columns = _get_column_order(schema, include_page)

    output = io.StringIO()
    writer = csv.DictWriter(
        output, fieldnames=columns, delimiter="\t", extrasaction="ignore"
    )
    writer.writeheader()
    writer.writerows(records)

    tsv_str = output.getvalue()

    if path is not None:
        Path(path).write_text(tsv_str, encoding="utf-8")
        return None

    return tsv_str


def to_parquet(
    result: dict[int, baml_types.MappedTable] | baml_types.MappedTable,
    schema: CanonicalSchema,
    path: str | Path,
    *,
    include_page: bool = False,
) -> None:
    """Serialize interpretation result to Parquet format.

    Requires ``pyarrow`` to be installed. Install with::

        pip install pyarrow

    or::

        pip install pdf-ocr[parquet]

    Args:
        result: Output from ``interpret_table()`` (page-keyed dict) or single MappedTable.
        schema: Canonical schema for validation and column ordering.
        path: File path to write the Parquet file.
        include_page: If True, add a ``page`` column with 1-indexed page number.

    Raises:
        ImportError: If pyarrow is not installed.
        SerializationValidationError: If a record fails validation.
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as e:
        raise ImportError(
            "pyarrow is required for Parquet export. "
            "Install it with: pip install pyarrow "
            "or: pip install pdf-ocr[parquet]"
        ) from e

    records = _flatten_and_validate(result, schema, include_page)
    columns = _get_column_order(schema, include_page)

    # Build PyArrow schema
    pa_type_map = {
        "string": pa.string(),
        "int": pa.int64(),
        "float": pa.float64(),
        "bool": pa.bool_(),
        "date": pa.string(),
    }

    pa_fields = []
    if include_page:
        pa_fields.append(pa.field("page", pa.int64()))
    for col in schema.columns:
        pa_type = pa_type_map.get(col.type, pa.string())
        pa_fields.append(pa.field(col.name, pa_type))

    pa_schema = pa.schema(pa_fields)

    # Convert records to columnar format
    column_data: dict[str, list] = {col: [] for col in columns}
    for rec in records:
        for col in columns:
            column_data[col].append(rec.get(col))

    # Create table and write
    table = pa.table(column_data, schema=pa_schema)
    pq.write_table(table, str(path))


def to_pandas(
    result: dict[int, baml_types.MappedTable] | baml_types.MappedTable,
    schema: CanonicalSchema,
    *,
    include_page: bool = False,
) -> pd.DataFrame:
    """Serialize interpretation result to a pandas DataFrame.

    Uses pandas nullable dtypes (Int64, Float64, string, boolean) for proper
    handling of missing values.

    Requires ``pandas`` to be installed. Install with::

        pip install pandas

    or::

        pip install pdf-ocr[dataframes]

    Args:
        result: Output from ``interpret_table()`` (page-keyed dict) or single MappedTable.
        schema: Canonical schema for validation and column ordering.
        include_page: If True, add a ``page`` column with 1-indexed page number.

    Returns:
        pandas DataFrame with proper nullable dtypes.

    Raises:
        ImportError: If pandas is not installed.
        SerializationValidationError: If a record fails validation.
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError(
            "pandas is required for DataFrame export. "
            "Install it with: pip install pandas "
            "or: pip install pdf-ocr[dataframes]"
        ) from e

    records = _flatten_and_validate(result, schema, include_page)
    columns = _get_column_order(schema, include_page)

    # Build dtype dict
    dtypes: dict[str, str] = {}
    if include_page:
        dtypes["page"] = "Int64"
    for col in schema.columns:
        dtypes[col.name] = _PANDAS_DTYPE_MAP.get(col.type, "string")

    # Create DataFrame
    df = pd.DataFrame(records, columns=columns)

    # Convert dtypes
    for col, dtype in dtypes.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)

    return df


def to_polars(
    result: dict[int, baml_types.MappedTable] | baml_types.MappedTable,
    schema: CanonicalSchema,
    *,
    include_page: bool = False,
) -> pl.DataFrame:
    """Serialize interpretation result to a polars DataFrame.

    Requires ``polars`` to be installed. Install with::

        pip install polars

    or::

        pip install pdf-ocr[dataframes]

    Args:
        result: Output from ``interpret_table()`` (page-keyed dict) or single MappedTable.
        schema: Canonical schema for validation and column ordering.
        include_page: If True, add a ``page`` column with 1-indexed page number.

    Returns:
        polars DataFrame with proper types.

    Raises:
        ImportError: If polars is not installed.
        SerializationValidationError: If a record fails validation.
    """
    try:
        import polars as pl
    except ImportError as e:
        raise ImportError(
            "polars is required for DataFrame export. "
            "Install it with: pip install polars "
            "or: pip install pdf-ocr[dataframes]"
        ) from e

    records = _flatten_and_validate(result, schema, include_page)
    columns = _get_column_order(schema, include_page)

    # Build polars schema
    pl_type_map = {
        "Utf8": pl.Utf8,
        "Int64": pl.Int64,
        "Float64": pl.Float64,
        "Boolean": pl.Boolean,
    }

    pl_schema: dict[str, pl.DataType] = {}
    if include_page:
        pl_schema["page"] = pl.Int64
    for col in schema.columns:
        dtype_name = _POLARS_DTYPE_MAP.get(col.type, "Utf8")
        pl_schema[col.name] = pl_type_map[dtype_name]

    # Convert records to columnar format
    column_data: dict[str, list] = {col: [] for col in columns}
    for rec in records:
        for col in columns:
            column_data[col].append(rec.get(col))

    # Create DataFrame with schema
    return pl.DataFrame(column_data, schema=pl_schema)
