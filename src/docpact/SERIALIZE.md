# Serialization

> **Module**: `serialize.py`
> **Public API**: `to_csv()`, `to_tsv()`, `to_parquet()`, `to_pandas()`, `to_polars()`

Export `interpret_table()` results to various formats with Pydantic validation and output formatting.

## Features

- **Pydantic validation**: Records validated against `CanonicalSchema` before export
- **OCR artifact coercion**: Handles common extraction artifacts automatically
- **Output formatting**: Applies format specifications from schema
- **Nullable types**: Proper nullable dtypes in pandas/polars

---

## Coercion Rules

The serializer handles common OCR and LLM extraction artifacts:

| Input | Output | Type |
|---|---|---|
| `"1,234"` | `1234` | int |
| `"(500)"` | `-500` | int (negative in parentheses) |
| `"12.5%"` | `12.5` | float |
| `"yes"`, `"true"`, `"1"` | `True` | bool |
| `"no"`, `"false"`, `"0"` | `False` | bool |
| `""`, whitespace | `None` | any |

---

## Format Specifications

Specify output format in `ColumnDef.format`:

```python
ColumnDef("date", "string", "Registration period", format="YYYY-MM")
ColumnDef("quantity", "int", "Count", format="#,###")
ColumnDef("country", "string", "Country name", format="uppercase")
```

### Date/Time Formats

| Format | Example | Description |
|---|---|---|
| `YYYY-MM-DD` | `2025-01-15` | ISO date |
| `YYYY-MM` | `2025-01` | Year-month |
| `YYYY` | `2025` | Year only |
| `DD/MM/YYYY` | `15/01/2025` | European |
| `MM/DD/YYYY` | `01/15/2025` | US |
| `MMM YYYY` | `Jan 2025` | Short month |
| `MMMM YYYY` | `January 2025` | Full month |
| `YYYY-MM-DD HH:mm` | `2025-01-15 14:30` | Date + time |
| `YYYY-MM-DD HH:mm:ss` | `2025-01-15 14:30:45` | With seconds |
| `YYYY-MM-DDTHH:mm:ssZ` | `2025-01-15T14:30:45Z` | ISO 8601 |
| `HH:mm` | `14:30` | Time (24h) |
| `HH:mm:ss` | `14:30:45` | Time with seconds |
| `hh:mm A` | `02:30 PM` | Time (12h) |

### Number Formats

| Format | Example | Description |
|---|---|---|
| `#` | `1234` | Plain integer |
| `#.##` | `1234.56` | 2 decimal places |
| `#.####` | `1234.5678` | 4 decimal places |
| `#,###` | `1,234` | Thousands separator |
| `#,###.##` | `1,234.56` | Both |
| `+#` | `+1234` / `-1234` | Explicit sign |
| `#%` | `12.5%` | Percentage |
| `(#)` | `(1234)` | Negative in parentheses |

### String Formats

| Format | Example | Description |
|---|---|---|
| `uppercase` | `HELLO WORLD` | All caps |
| `lowercase` | `hello world` | All lower |
| `titlecase` | `Hello World` | Title Case |
| `capitalize` | `Hello world` | First letter only |
| `camelCase` | `helloWorld` | camelCase |
| `PascalCase` | `HelloWorld` | PascalCase |
| `snake_case` | `hello_world` | snake_case |
| `SCREAMING_SNAKE_CASE` | `HELLO_WORLD` | SCREAMING_SNAKE |
| `kebab-case` | `hello-world` | kebab-case |
| `SCREAMING-KEBAB-CASE` | `HELLO-WORLD` | SCREAMING-KEBAB |
| `trim` | `hello` | Strip whitespace |

---

## API

### CSV/TSV

```python
from docpact import to_csv, to_tsv

# Return string
csv_str = to_csv(result, schema)
tsv_str = to_tsv(result, schema)

# Write to file
to_csv(result, schema, path="output.csv")

# Include page column
to_csv(result, schema, path="output.csv", include_page=True)
```

### Parquet

```python
from docpact import to_parquet

to_parquet(result, schema, "output.parquet")
to_parquet(result, schema, "output.parquet", include_page=True)
```

Requires `pyarrow`.

### DataFrames

```python
from docpact import to_pandas, to_polars

df = to_pandas(result, schema)
df = to_pandas(result, schema, include_page=True)

pl_df = to_polars(result, schema)
pl_df = to_polars(result, schema, include_page=True)
```

---

## Type Mappings

| Schema Type | Python | pandas dtype | polars dtype |
|---|---|---|---|
| `string` | `str \| None` | `string` | `Utf8` |
| `int` | `int \| None` | `Int64` | `Int64` |
| `float` | `float \| None` | `Float64` | `Float64` |
| `bool` | `bool \| None` | `boolean` | `Boolean` |
| `date` | `str \| None` | `string` | `Utf8` |

---

## Optional Dependencies

```bash
pip install docpact[dataframes]  # pandas + polars
pip install docpact[parquet]     # pyarrow
pip install docpact[all]         # everything
```

---

## Validation Errors

```python
from docpact import SerializationValidationError

try:
    csv_str = to_csv(result, schema)
except SerializationValidationError as e:
    print(f"Record {e.record_index} failed: {e}")
    print(f"Column: {e.column_name}")
    print(f"Original: {e.original_error}")
```

In practice, validation rarely fails because coercion converts invalid values to `None`.

---

## Example

```python
from docpact import interpret_table, to_csv, CanonicalSchema, ColumnDef

schema = CanonicalSchema(columns=[
    ColumnDef("country", "string", "Country name", format="uppercase"),
    ColumnDef("date", "string", "Period", format="YYYY-MM"),
    ColumnDef("registrations", "int", "Count", format="#,###"),
    ColumnDef("growth", "float", "YoY growth", format="+#.##%"),
])

result = interpret_table(compressed, schema)
csv = to_csv(result, schema)
```

Output:

```csv
country,date,registrations,growth
GERMANY,2025-01,"1,234,567",+12.35%
FRANCE,2024-06,"987,654",-5.67%
```
