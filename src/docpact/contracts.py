"""Contract helpers: load JSON contracts and transform DataFrames.

Granular, composable building blocks for the prepare phase (contract loading)
and transform phase (DataFrame enrichment/formatting).  Each helper is
independently useful — orchestration stays in the notebook.

Usage::

    from docpact.contracts import load_contract, resolve_year_templates, enrich_dataframe, format_dataframe

    ctx = load_contract("contracts/ru_ag_ministry.json")
    # ... custom compress + classify + interpret ...
    df = to_pandas(result, ctx.outputs["harvest"].schema)
    df = enrich_dataframe(df, ctx.outputs["harvest"].enrichment, title="Wheat", report_date="June 20-21")
    df = format_dataframe(df, ctx.outputs["harvest"].col_specs)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from docpact.interpret import CanonicalSchema, ColumnDef, UnpivotStrategy
from docpact.report_date import ReportDateConfig

# Year template pattern: {YYYY}, {YYYY-1}, {YYYY+1}, etc.
_YEAR_TEMPLATE_RE = re.compile(r"\{YYYY([+-]\d+)?\}")


# ─── Data classes ────────────────────────────────────────────────────────────


@dataclass
class SemanticColumnSpec:
    """Semantic metadata for a contract column.

    Parsed from ``concept_uris``, ``resolve``, and ``semantic`` fields
    in the contract JSON.  Used by :mod:`docpact.semantics` for pre-flight
    checks and post-extraction validation.

    Attributes:
        column_name: Name of the schema column.
        concept_uris: Raw concept URI + label dicts from the contract.
        resolve_config: Raw resolve configuration dict.
        validate: Whether to validate extracted values against known labels.
    """

    column_name: str
    concept_uris: list[dict] = field(default_factory=list)
    resolve_config: dict = field(default_factory=dict)
    validate: bool = False


@dataclass
class OutputSpec:
    """Parsed output specification from a contract.

    Attributes:
        name: Output name key (e.g. ``"harvest"``).
        category: Category name for classification.
        filename: Suggested output filename (if specified).
        schema: :class:`CanonicalSchema` with LLM columns only.
        enrichment: Source-based columns (``"title"``, ``"report_date"``, ``"constant"``).
        col_specs: Raw column specs for formatting/filtering.
        semantic_columns: ``{column_name: SemanticColumnSpec}`` for columns
            with ``concept_uris`` annotations.
    """

    name: str
    category: str
    filename: str | None
    schema: CanonicalSchema
    enrichment: dict[str, dict]
    col_specs: list[dict]
    semantic_columns: dict[str, SemanticColumnSpec] = field(default_factory=dict)


@dataclass
class ContractContext:
    """Typed representation of a parsed JSON contract.

    Attributes:
        provider: Human-readable provider name.
        model: LLM model identifier.
        unpivot: Unpivot strategy or bool for backward compatibility.
        categories: Category name → keyword list.
        outputs: Output name → :class:`OutputSpec`.
        report_date_config: Declarative report-date config (if present).
        raw: Original JSON dict for custom access.
    """

    provider: str
    model: str
    unpivot: UnpivotStrategy | bool
    categories: dict[str, list[str]]
    outputs: dict[str, OutputSpec]
    report_date_config: ReportDateConfig | None
    raw: dict
    has_semantic_annotations: bool = False


# ─── Prepare helpers ─────────────────────────────────────────────────────────


def load_contract(path: str | Path) -> ContractContext:
    """Load a JSON contract and return a typed :class:`ContractContext`.

    Parses categories, schemas (separating LLM columns from enrichment),
    report-date config, and unpivot strategy.
    """
    with open(path) as f:
        contract = json.load(f)

    provider = contract.get("provider", "")
    model = contract.get("model", "openai/gpt-4o")

    # Categories
    categories: dict[str, list[str]] = {
        name: cat["keywords"]
        for name, cat in contract.get("categories", {}).items()
    }

    # Report date
    rd = contract.get("report_date")
    report_date_config = ReportDateConfig.from_dict(rd) if rd else None

    # Unpivot strategy (default: True for backward compat)
    unpivot_raw = contract.get("unpivot", True)
    if isinstance(unpivot_raw, str):
        unpivot: UnpivotStrategy | bool = UnpivotStrategy(unpivot_raw)
    else:
        unpivot = unpivot_raw  # bool passthrough

    # Outputs — separate LLM columns from enrichment columns
    outputs: dict[str, OutputSpec] = {}
    any_semantic = False
    for out_name, spec in contract.get("outputs", {}).items():
        llm_cols: list[ColumnDef] = []
        enrich: dict[str, dict] = {}
        semantic_cols: dict[str, SemanticColumnSpec] = {}
        for col in spec["schema"]["columns"]:
            if "source" in col:
                enrich[col["name"]] = col
            else:
                llm_cols.append(ColumnDef(
                    name=col["name"],
                    type=col.get("type", "string"),
                    description=col.get("description", ""),
                    aliases=col.get("aliases", []),
                    format=col.get("format"),
                ))

            # Parse semantic annotations (present on both LLM and enrichment cols)
            if "concept_uris" in col:
                any_semantic = True
                semantic_block = col.get("semantic", {})
                semantic_cols[col["name"]] = SemanticColumnSpec(
                    column_name=col["name"],
                    concept_uris=col["concept_uris"],
                    resolve_config=col.get("resolve", {}),
                    validate=semantic_block.get("validate", False),
                )

        schema = CanonicalSchema(
            description=spec["schema"].get("description", ""),
            columns=llm_cols,
        )
        outputs[out_name] = OutputSpec(
            name=out_name,
            category=spec["category"],
            filename=spec.get("filename"),
            schema=schema,
            enrichment=enrich,
            col_specs=spec["schema"]["columns"],
            semantic_columns=semantic_cols,
        )

    return ContractContext(
        provider=provider,
        model=model,
        unpivot=unpivot,
        categories=categories,
        outputs=outputs,
        report_date_config=report_date_config,
        raw=contract,
        has_semantic_annotations=any_semantic,
    )


def resolve_year_templates(aliases: list[str], pivot_years: list[str]) -> list[str]:
    """Resolve ``{YYYY}``, ``{YYYY-1}``, ``{YYYY+1}`` in aliases using pivot years.

    Supports both standalone templates (``{YYYY}``) and inline templates
    (e.g. ``"MOA Target {YYYY}"`` → ``"MOA Target 2025"``).

    - ``{YYYY}``   → latest year found in document headers
    - ``{YYYY-1}`` → one year before latest
    - ``{YYYY+1}`` → one year after latest

    When *pivot_years* is empty, aliases containing templates are stripped
    (non-template aliases are kept).
    """
    if not pivot_years:
        return [a for a in aliases if not _YEAR_TEMPLATE_RE.search(a)]
    base_year = int(pivot_years[-1])  # latest year

    def _replace(m: re.Match) -> str:
        offset = int(m.group(1)) if m.group(1) else 0
        return str(base_year + offset)

    resolved: list[str] = []
    for alias in aliases:
        if _YEAR_TEMPLATE_RE.search(alias):
            resolved.append(_YEAR_TEMPLATE_RE.sub(_replace, alias))
        else:
            resolved.append(alias)
    return resolved


# ─── Transform helpers ───────────────────────────────────────────────────────


def enrich_dataframe(
    df,
    enrichment: dict[str, dict],
    *,
    title: str | None = None,
    report_date: str | None = None,
):
    """Apply enrichment rules to a DataFrame, returning it modified in-place.

    Adds columns based on enrichment ``source`` types:

    - ``"title"`` → uses the *title* argument (falls back to ``"Unknown"``).
    - ``"report_date"`` → uses the *report_date* argument, with optional
      ``suffix`` from the enrichment spec appended.
    - ``"constant"`` → uses the ``value`` field from the enrichment spec.

    Parameters:
        df: pandas DataFrame to enrich.
        enrichment: ``{column_name: spec_dict}`` from :attr:`OutputSpec.enrichment`.
        title: Document or table title string.
        report_date: Resolved report date string.

    Returns:
        The same DataFrame (modified in-place).
    """
    for col_name, spec in enrichment.items():
        src = spec["source"]
        if src == "title":
            df[col_name] = title or "Unknown"
        elif src == "report_date":
            val = report_date or ""
            if "suffix" in spec:
                val += spec["suffix"]
            df[col_name] = val
        elif src == "constant":
            df[col_name] = spec["value"]
    return df


def format_dataframe(df, col_specs: list[dict]):
    """Apply column-level format and filter transformations to a DataFrame.

    Handles ``format`` values:
    - ``"lowercase"`` / ``"uppercase"`` / ``"titlecase"`` — case transformation.

    Handles ``filter`` values:
    - ``"latest"`` — keep only rows where the column equals its max value.
    - ``"earliest"`` — keep only rows where the column equals its min value.

    Other format values (date patterns, number patterns) are ignored here —
    those are handled by :func:`docpact.serialize.to_csv` etc.

    Parameters:
        df: pandas DataFrame to format.
        col_specs: Raw column spec dicts from the contract (all columns, including
            enrichment columns — only ``format`` and ``filter`` keys are used).

    Returns:
        The (possibly filtered) DataFrame.
    """
    for col_spec in col_specs:
        cn = col_spec["name"]
        fmt = col_spec.get("format")
        if fmt and cn in df.columns:
            if fmt == "lowercase":
                df[cn] = df[cn].astype(str).str.lower()
            elif fmt == "uppercase":
                df[cn] = df[cn].astype(str).str.upper()
            elif fmt == "titlecase":
                df[cn] = df[cn].astype(str).str.title()
        filt = col_spec.get("filter")
        if filt and filt != "all" and cn in df.columns:
            if filt == "latest":
                df = df[df[cn] == df[cn].max()]
            elif filt == "earliest":
                df = df[df[cn] == df[cn].min()]
    return df
