"""SHACL validation: convert tabular records to RDF and validate against shapes.

Converts CSV/DataFrame records into RDF triples and validates them against
SHACL shape graphs.  Requires the ``pyshacl`` optional dependency::

    pip install contract-semantics[shacl]
"""

from __future__ import annotations

import csv
import json
from io import StringIO
from pathlib import Path

from rdflib import BNode, Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, SH, XSD


# Namespace for generated record triples
DOCPACT = Namespace("http://docpact.dev/schema/")
RECORD = Namespace("http://docpact.dev/record/")


def records_to_graph(
    records: list[dict],
    schema_columns: list[dict] | None = None,
) -> Graph:
    """Convert tabular records to an RDF graph.

    Each record becomes a blank node with properties from :data:`DOCPACT` namespace.
    Column types are used for appropriate XSD datatype literals.

    Parameters:
        records: List of dicts (each dict is one row).
        schema_columns: Optional column specs for type information.

    Returns:
        An rdflib :class:`Graph` with one node per record.
    """
    g = Graph()
    g.bind("docpact", DOCPACT)
    g.bind("record", RECORD)

    # Build type map from schema
    type_map: dict[str, str] = {}
    if schema_columns:
        for col in schema_columns:
            type_map[col["name"]] = col.get("type", "string")

    for i, record in enumerate(records):
        node = RECORD[f"r{i}"]
        g.add((node, RDF.type, DOCPACT.Record))

        for key, value in record.items():
            if value is None or (isinstance(value, str) and value.strip() == ""):
                continue

            pred = DOCPACT[key]
            col_type = type_map.get(key, "string")

            if col_type == "float":
                try:
                    g.add((node, pred, Literal(float(value), datatype=XSD.decimal)))
                except (ValueError, TypeError):
                    g.add((node, pred, Literal(str(value))))
            elif col_type == "int":
                try:
                    g.add((node, pred, Literal(int(value), datatype=XSD.integer)))
                except (ValueError, TypeError):
                    g.add((node, pred, Literal(str(value))))
            else:
                g.add((node, pred, Literal(str(value))))

    return g


def validate_records(
    records: list[dict],
    shapes_path: str | Path,
    schema_columns: list[dict] | None = None,
) -> tuple[bool, str, Graph]:
    """Validate records against a SHACL shapes graph.

    Parameters:
        records: List of dicts (each dict is one row).
        shapes_path: Path to SHACL shapes file (Turtle format).
        schema_columns: Optional column specs for type information.

    Returns:
        Tuple of (conforms: bool, report_text: str, report_graph: Graph).

    Raises:
        ImportError: If ``pyshacl`` is not installed.
    """
    try:
        import pyshacl
    except ImportError:
        raise ImportError(
            "pyshacl is required for SHACL validation. "
            "Install with: pip install contract-semantics[shacl]"
        )

    data_graph = records_to_graph(records, schema_columns)

    shapes_graph = Graph()
    shapes_graph.parse(str(shapes_path), format="turtle")

    conforms, report_graph, report_text = pyshacl.validate(
        data_graph,
        shacl_graph=shapes_graph,
        inference="none",
        abort_on_first=False,
    )

    return conforms, report_text, report_graph


def validate_csv(
    csv_path: str | Path,
    shapes_path: str | Path,
    schema_columns: list[dict] | None = None,
) -> tuple[bool, str, Graph]:
    """Validate a CSV file against a SHACL shapes graph.

    Parameters:
        csv_path: Path to CSV file.
        shapes_path: Path to SHACL shapes file.
        schema_columns: Optional column specs for type information.

    Returns:
        Tuple of (conforms: bool, report_text: str, report_graph: Graph).
    """
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        records = list(reader)

    return validate_records(records, shapes_path, schema_columns)


def generate_shapes(
    contract_path: str | Path,
    output_name: str | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, str]:
    """Auto-generate SHACL shapes from a contract's output schemas.

    For each output in the contract, generates a Turtle file defining:
    - A NodeShape for Record
    - PropertyShapes for each column with appropriate constraints
    - ``sh:in`` closed lists for string columns with aliases
    - ``sh:minInclusive 0`` for float/int columns (numeric)
    - ``sh:datatype`` for typed columns

    Parameters:
        contract_path: Path to the contract JSON.
        output_name: If given, only generate shapes for this output.
        output_dir: Directory to write shape files to. Default: ``shapes/``.

    Returns:
        Dict mapping output name to file path of generated shapes.
    """
    with open(contract_path) as f:
        contract = json.load(f)

    output_dir = Path(output_dir) if output_dir else Path(__file__).resolve().parent.parent.parent / "shapes"
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, str] = {}

    for out_name, out_spec in contract.get("outputs", {}).items():
        if output_name and out_name != output_name:
            continue

        columns = out_spec.get("schema", {}).get("columns", [])
        ttl = _generate_shape_turtle(out_name, columns)

        path = output_dir / f"{out_name}_output.ttl"
        path.write_text(ttl)
        results[out_name] = str(path)

    return results


def _generate_shape_turtle(output_name: str, columns: list[dict]) -> str:
    """Generate a SHACL Turtle string for one output schema."""
    lines: list[str] = [
        "@prefix sh: <http://www.w3.org/ns/shacl#> .",
        "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
        "@prefix docpact: <http://docpact.dev/schema/> .",
        "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
        "",
        f"docpact:{output_name.capitalize()}RecordShape",
        "    a sh:NodeShape ;",
        "    sh:targetClass docpact:Record ;",
    ]

    prop_blocks: list[str] = []
    for col in columns:
        block = _generate_property_shape(col)
        if block:
            prop_blocks.append(block)

    if prop_blocks:
        for i, block in enumerate(prop_blocks):
            separator = " ;" if i < len(prop_blocks) - 1 else " ."
            lines.append(f"    sh:property {block}{separator}")
    else:
        # Remove trailing semicolon and close
        lines[-1] = lines[-1].rstrip(" ;") + " ."

    lines.append("")
    return "\n".join(lines)


def _generate_property_shape(col: dict) -> str | None:
    """Generate an inline property shape for a column."""
    name = col["name"]
    col_type = col.get("type", "string")
    aliases = col.get("aliases", [])

    parts = [f'[ sh:path docpact:{name} ;']

    if col_type == "float":
        parts.append("      sh:minInclusive 0 ;")
    elif col_type == "int":
        parts.append("      sh:datatype xsd:integer ;")
    else:
        parts.append("      sh:datatype xsd:string ;")

    # For string columns with concrete aliases (no templates), add sh:in
    concrete_aliases = [a for a in aliases if "{" not in a]
    if col_type == "string" and concrete_aliases:
        values = " ".join(f'"{a}"' for a in concrete_aliases)
        parts.append(f"      sh:in ({values}) ;")

    parts.append(f'      sh:name "{name}" ]')
    return "\n".join(parts)
