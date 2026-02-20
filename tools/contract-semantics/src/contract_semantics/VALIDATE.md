# SHACL Validation

> **Module**: `validate.py`
> **Public API**: `records_to_graph()`, `validate_records()`, `validate_csv()`, `generate_shapes()`
> **Optional dependency**: `pyshacl` (install with `pip install contract-semantics[shacl]`)

Convert tabular records to RDF triples and validate against SHACL shapes. Catches invalid values, out-of-range numbers, and unknown category entries before data is delivered downstream.

---

## The Problem

Pipeline outputs can contain extraction errors: a metric value that doesn't exist in the schema, a negative area, a region name that's not a real Russian federal subject. These errors are invisible in a CSV but can corrupt downstream analytics. Post-extraction validation catches them at the boundary.

## The Solution

Three-step process:

```
Records (list[dict] or CSV)
  -> records_to_graph()  -- convert to RDF triples with XSD types
  -> pyshacl.validate()  -- check against SHACL shapes
  -> (conforms, report)  -- boolean + human-readable report
```

---

## API

### records_to_graph()

Convert tabular records to an rdflib Graph. Each record becomes a node of type `docpact:Record` with properties in the `docpact:` namespace.

```python
from contract_semantics.validate import records_to_graph

records = [
    {"Region": "Moscow Oblast", "Metric": "Yield", "Value": "3.5", "Year": "2025"},
]
schema = [
    {"name": "Region", "type": "string"},
    {"name": "Value", "type": "float"},
    {"name": "Year", "type": "int"},
]
graph = records_to_graph(records, schema)
```

Type mapping:

| Schema Type | XSD Datatype |
|---|---|
| `string` | `xsd:string` |
| `float` | `xsd:decimal` |
| `int` | `xsd:integer` |

Empty strings and `None` values are skipped (no triple emitted).

### validate_records()

Validate records against a SHACL shapes file.

```python
from contract_semantics.validate import validate_records

conforms, report_text, report_graph = validate_records(
    records, "shapes/harvest_output.ttl", schema
)
if not conforms:
    print(report_text)
```

### validate_csv()

Validate a CSV file against SHACL shapes (convenience wrapper).

```python
from contract_semantics.validate import validate_csv

conforms, report_text, _ = validate_csv("output.csv", "shapes/harvest_output.ttl")
```

### generate_shapes()

Auto-generate SHACL shapes from a contract's output schemas.

```python
from contract_semantics.validate import generate_shapes

results = generate_shapes("contracts/ru_ag_ministry.json", output_dir="shapes/")
# {"harvest": "shapes/harvest_output.ttl", "planting": "shapes/planting_output.ttl"}
```

Generated constraints:

| Schema Type | Generated Constraints |
|---|---|
| `string` | `sh:datatype xsd:string` |
| `string` with aliases | `sh:datatype xsd:string` + `sh:in (...)` closed list |
| `float` | `sh:minInclusive 0` |
| `int` | `sh:datatype xsd:integer` |

Template aliases (containing `{`) are excluded from `sh:in` lists since they resolve at runtime.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `contract_path` | `str \| Path` | required | Path to contract JSON |
| `output_name` | `str \| None` | `None` | Generate shapes for only this output |
| `output_dir` | `str \| Path \| None` | `shapes/` | Directory to write shape files to |

---

## Hand-Crafted Shapes

The `shapes/` directory contains hand-crafted SHACL shapes for more precise validation than auto-generation provides:

**`shapes/harvest_output.ttl`**:

- `Region`: non-empty string (`sh:minCount 1`)
- `Metric`: string, `sh:in ("Area harvested" "collected" "Yield")`
- `Value`: `sh:minInclusive 0`
- `Year`: integer, `sh:minInclusive 2000`, `sh:maxInclusive 2030`

Hand-crafted shapes can include GeoNames-grounded closed lists for region columns, range constraints that auto-generation can't infer, and cross-field validation rules.

---

## Namespaces

| Prefix | URI | Usage |
|---|---|---|
| `docpact:` | `http://docpact.dev/schema/` | Property and shape namespace |
| `record:` | `http://docpact.dev/record/` | Individual record node namespace |

---

## Test Coverage

7 tests in `tests/test_validate.py` covering RDF conversion, empty value skipping, typed literals, valid records, invalid metric detection, shape generation from contracts, and `sh:in` constraint generation.
