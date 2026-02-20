"""Semantic-aware pipeline support: pre-resolved context, pre-flight checks, validation.

This module provides data structures and functions that enable semantic awareness
in the docpact pipeline without importing from ``contract_semantics``.  The
``SemanticContext`` is built externally (by ``contract_semantics.context``) and
passed into pipeline functions as a plain data object.

Three capabilities:

1. **Runtime alias enrichment** — ``SemanticContext.aliases_for()`` returns
   ontology-resolved aliases to merge into schemas before interpretation.
2. **Pre-flight check** — ``preflight_check()`` compares document headers
   against contract aliases (manual + resolved) and reports coverage.
3. **Post-extraction validation** — ``validate_output()`` checks extracted
   DataFrame values against known valid concept labels.

Usage::

    from docpact.semantics import SemanticContext, preflight_check, validate_output

    ctx = SemanticContext.from_json("semantic_context.json")
    report = preflight_check(compressed_data, output_spec, ctx)
    val_report = validate_output(df, output_spec, ctx)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from docpact.contracts import OutputSpec
from docpact.interpret import _normalize_for_alias_match


# ─── SemanticContext ──────────────────────────────────────────────────────────


@dataclass
class SemanticContext:
    """Pre-resolved semantic data, passed into the pipeline.

    Built externally by ``contract_semantics.context.build_semantic_context()``.
    Serializable to/from JSON for caching.

    Attributes:
        resolved_aliases: ``{output_name: {column_name: [alias, ...]}}``
            Ontology-resolved aliases to merge into schemas at runtime.
        valid_values: ``{output_name: {column_name: {value, ...}}}``
            Known-valid values for post-extraction validation.
        resolved_at: ISO timestamp of when resolution was performed.
        adapter_versions: ``{adapter_name: version_string}`` for provenance.
    """

    resolved_aliases: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    valid_values: dict[str, dict[str, set[str]]] = field(default_factory=dict)
    resolved_at: str = ""
    adapter_versions: dict[str, str] = field(default_factory=dict)

    def aliases_for(self, output_name: str, column_name: str) -> list[str]:
        """Return resolved aliases for a specific output/column, or empty list."""
        return self.resolved_aliases.get(output_name, {}).get(column_name, [])

    def valid_set_for(self, output_name: str, column_name: str) -> set[str]:
        """Return the set of valid values for a specific output/column, or empty set."""
        return self.valid_values.get(output_name, {}).get(column_name, set())

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        # Convert sets to sorted lists for deterministic JSON output
        valid_values_serializable: dict[str, dict[str, list[str]]] = {}
        for out_name, cols in self.valid_values.items():
            valid_values_serializable[out_name] = {
                col_name: sorted(values) for col_name, values in cols.items()
            }
        return {
            "resolved_aliases": self.resolved_aliases,
            "valid_values": valid_values_serializable,
            "resolved_at": self.resolved_at,
            "adapter_versions": self.adapter_versions,
        }

    def to_json(self, path: str | Path) -> None:
        """Serialize to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
            f.write("\n")

    @classmethod
    def from_dict(cls, data: dict) -> SemanticContext:
        """Deserialize from a JSON-compatible dict."""
        valid_values: dict[str, dict[str, set[str]]] = {}
        for out_name, cols in data.get("valid_values", {}).items():
            valid_values[out_name] = {
                col_name: set(values) for col_name, values in cols.items()
            }
        return cls(
            resolved_aliases=data.get("resolved_aliases", {}),
            valid_values=valid_values,
            resolved_at=data.get("resolved_at", ""),
            adapter_versions=data.get("adapter_versions", {}),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> SemanticContext:
        """Deserialize from a JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))


# ─── Pre-flight check ────────────────────────────────────────────────────────


@dataclass
class PreFlightFinding:
    """A single finding from the pre-flight header check.

    Attributes:
        severity: ``"info"`` or ``"warning"``.
        column_name: Schema column the finding relates to.
        output_name: Output category name.
        message: Human-readable description of the finding.
    """

    severity: str
    column_name: str
    output_name: str
    message: str


@dataclass
class PreFlightReport:
    """Result of a pre-flight header check for one output category.

    Attributes:
        findings: Individual findings (info and warning level).
        header_coverage: Fraction of document headers matched to contract aliases.
        unmatched_headers: Document headers not matching any contract alias.
        missing_aliases: Contract aliases not found in document headers.
    """

    findings: list[PreFlightFinding] = field(default_factory=list)
    header_coverage: float = 0.0
    unmatched_headers: list[str] = field(default_factory=list)
    missing_aliases: list[str] = field(default_factory=list)


def _extract_headers_from_pipe_table(text: str) -> list[str]:
    """Extract column headers from pipe-table markdown text.

    Handles both single pipe-table strings (PDF) and multi-table text.
    Returns unique header strings preserving first-occurrence order.
    """
    headers: list[str] = []
    seen: set[str] = set()

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        # Skip separator lines (|---|---|)
        if re.match(r"^\|[\s\-:|]+\|$", stripped):
            continue
        # First pipe-table row after a non-table region or at start is a header
        cells = [c.strip() for c in stripped.split("|")[1:-1]]
        for cell in cells:
            if cell and cell not in seen:
                headers.append(cell)
                seen.add(cell)
        # Only extract from the first data-bearing pipe row per table
        break

    return headers


def _extract_all_headers(data: str | list[tuple[str, dict]]) -> list[str]:
    """Extract headers from compressed data (PDF string or DOCX table list)."""
    headers: list[str] = []
    seen: set[str] = set()

    if isinstance(data, str):
        # PDF: may have multiple tables separated by form-feeds
        for page_text in data.split("\f"):
            for h in _extract_headers_from_pipe_table(page_text):
                if h not in seen:
                    headers.append(h)
                    seen.add(h)
    else:
        # DOCX: list of (markdown, metadata) tuples
        for md, _meta in data:
            for h in _extract_headers_from_pipe_table(md):
                if h not in seen:
                    headers.append(h)
                    seen.add(h)

    return headers


def preflight_check(
    data: str | list[tuple[str, dict]],
    output_spec: OutputSpec,
    semantic_context: SemanticContext | None = None,
) -> PreFlightReport:
    """Compare document headers against contract aliases before extraction.

    Informational only — never blocks extraction.  Compares extracted
    pipe-table headers against all known aliases for the output's schema
    columns (manual aliases from the contract + resolved aliases from the
    semantic context).

    Parameters:
        data: Compressed table data (``str`` for PDF, ``list`` for DOCX).
        output_spec: Output specification from the contract.
        semantic_context: Optional pre-resolved semantic data for richer matching.

    Returns:
        :class:`PreFlightReport` with coverage and finding details.
    """
    doc_headers = _extract_all_headers(data)
    if not doc_headers:
        return PreFlightReport(
            findings=[PreFlightFinding(
                severity="warning",
                column_name="",
                output_name=output_spec.name,
                message="No headers found in document data",
            )],
        )

    # Build the full alias set per column (manual + resolved)
    all_aliases: dict[str, list[str]] = {}  # col_name -> [alias, ...]
    for col in output_spec.schema.columns:
        aliases = list(col.aliases)
        if semantic_context is not None:
            extra = semantic_context.aliases_for(output_spec.name, col.name)
            existing_lower = {a.lower() for a in aliases}
            for a in extra:
                if a.lower() not in existing_lower:
                    aliases.append(a)
                    existing_lower.add(a.lower())
        all_aliases[col.name] = aliases

    # Build normalized lookup: normalized_alias -> (col_name, original_alias)
    alias_lookup: dict[str, tuple[str, str]] = {}
    for col_name, aliases in all_aliases.items():
        for alias in aliases:
            norm = _normalize_for_alias_match(alias)
            if norm not in alias_lookup:
                alias_lookup[norm] = (col_name, alias)

    # Match document headers against aliases
    matched_headers: list[str] = []
    unmatched_headers: list[str] = []
    matched_columns: set[str] = set()

    for header in doc_headers:
        norm_header = _normalize_for_alias_match(header)
        if norm_header in alias_lookup:
            col_name, _ = alias_lookup[norm_header]
            matched_headers.append(header)
            matched_columns.add(col_name)
        else:
            unmatched_headers.append(header)

    # Find contract aliases not represented in document
    missing_aliases: list[str] = []
    findings: list[PreFlightFinding] = []

    for col_name, aliases in all_aliases.items():
        if col_name not in matched_columns:
            missing_aliases.extend(aliases[:1])  # report first alias as representative
            findings.append(PreFlightFinding(
                severity="warning",
                column_name=col_name,
                output_name=output_spec.name,
                message=f"No document header matches column '{col_name}' aliases",
            ))

    for header in unmatched_headers:
        findings.append(PreFlightFinding(
            severity="info",
            column_name="",
            output_name=output_spec.name,
            message=f"Document header '{header}' does not match any contract alias",
        ))

    coverage = len(matched_headers) / len(doc_headers) if doc_headers else 0.0

    return PreFlightReport(
        findings=findings,
        header_coverage=coverage,
        unmatched_headers=unmatched_headers,
        missing_aliases=missing_aliases,
    )


# ─── Post-extraction validation ──────────────────────────────────────────────


@dataclass
class ValidationFinding:
    """A single value validation finding.

    Attributes:
        row_index: DataFrame row index where the issue was found.
        column_name: Column containing the problematic value.
        value: The actual extracted value.
        message: Human-readable description.
        severity: ``"warning"`` (default) or ``"error"``.
    """

    row_index: int
    column_name: str
    value: str
    message: str
    severity: str = "warning"


@dataclass
class ValidationReport:
    """Result of post-extraction value validation for one output category.

    Attributes:
        output_name: Output category name.
        total_rows: Total rows in the DataFrame.
        valid_count: Number of rows where all validated columns passed.
        invalid_count: Number of rows with at least one validation finding.
        findings: Individual findings.
        column_summaries: Per-column stats ``{col: {valid, invalid, unknown_values}}``.
    """

    output_name: str
    total_rows: int = 0
    valid_count: int = 0
    invalid_count: int = 0
    findings: list[ValidationFinding] = field(default_factory=list)
    column_summaries: dict[str, dict] = field(default_factory=dict)


def _build_valid_set(
    output_spec: OutputSpec,
    column_name: str,
    semantic_context: SemanticContext | None = None,
) -> set[str]:
    """Build the set of valid values for a column.

    Combines concept_uri labels from the contract's SemanticColumnSpec
    with resolved aliases from SemanticContext.  All values are normalized
    to lowercase for case-insensitive matching.
    """
    valid: set[str] = set()

    # Baseline: concept_uri labels from contract
    sem_col = output_spec.semantic_columns.get(column_name)
    if sem_col is not None:
        for ref in sem_col.concept_uris:
            if isinstance(ref, dict):
                label = ref.get("label", "")
            else:
                label = ref
            if label:
                valid.add(label.lower())

    # Richer: resolved valid_values from SemanticContext
    if semantic_context is not None:
        for v in semantic_context.valid_set_for(output_spec.name, column_name):
            valid.add(v.lower())

    return valid


def validate_output(
    df: pd.DataFrame,
    output_spec: OutputSpec,
    semantic_context: SemanticContext | None = None,
) -> ValidationReport:
    """Validate extracted DataFrame values against known valid concept labels.

    For each column with ``validate=True`` in its :class:`SemanticColumnSpec`,
    checks every value against the combined set of concept URI labels and
    resolved aliases.  Missing/empty values are skipped (not flagged).

    Parameters:
        df: Extracted and enriched DataFrame.
        output_spec: Output specification from the contract.
        semantic_context: Optional pre-resolved semantic data for richer validation.

    Returns:
        :class:`ValidationReport` with per-row findings and column summaries.
    """
    findings: list[ValidationFinding] = []
    column_summaries: dict[str, dict] = {}
    rows_with_issues: set[int] = set()

    for col_name, sem_col in output_spec.semantic_columns.items():
        if not sem_col.validate:
            continue
        if col_name not in df.columns:
            continue

        valid_set = _build_valid_set(output_spec, col_name, semantic_context)
        if not valid_set:
            continue

        col_valid = 0
        col_invalid = 0
        unknown_values: list[str] = []

        for idx, value in df[col_name].items():
            str_val = str(value).strip()
            if not str_val or str_val.lower() in ("nan", "none", ""):
                col_valid += 1
                continue

            if str_val.lower() in valid_set:
                col_valid += 1
            else:
                col_invalid += 1
                rows_with_issues.add(idx)
                if str_val not in unknown_values:
                    unknown_values.append(str_val)
                findings.append(ValidationFinding(
                    row_index=idx,
                    column_name=col_name,
                    value=str_val,
                    message=f"Value '{str_val}' not in known valid set for '{col_name}'",
                ))

        column_summaries[col_name] = {
            "valid": col_valid,
            "invalid": col_invalid,
            "unknown_values": unknown_values,
        }

    total_rows = len(df)
    invalid_count = len(rows_with_issues)

    return ValidationReport(
        output_name=output_spec.name,
        total_rows=total_rows,
        valid_count=total_rows - invalid_count,
        invalid_count=invalid_count,
        findings=findings,
        column_summaries=column_summaries,
    )
