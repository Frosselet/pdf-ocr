"""Bridge builder: construct a docpact SemanticContext from an annotated contract.

This module resolves all concept URIs in a contract through ontology adapters
and packages the results into a :class:`docpact.semantics.SemanticContext`
data object that the pipeline can consume.

``docpact`` never imports from ``contract_semantics``.  This module imports
from both sides: it reads the contract using ``contract_semantics`` adapters
and produces a ``SemanticContext`` using ``docpact.semantics``.

Usage::

    from contract_semantics.context import build_semantic_context

    ctx = build_semantic_context("contracts/ru_ag_ministry.json")
    ctx.to_json("semantic_context.json")

Or with explicit adapters::

    from contract_semantics.agrovoc import AgrovocAdapter
    from contract_semantics.context import build_semantic_context

    agrovoc = AgrovocAdapter.from_file("data/agrovoc_core.nt")
    ctx = build_semantic_context("contracts/ru_ag_ministry.json", agrovoc=agrovoc)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from contract_semantics.models import ConceptRef, ResolveConfig
from contract_semantics.resolve import OntologyAdapter, resolve_column


def build_semantic_context(
    contract_path: str | Path,
    *,
    agrovoc: OntologyAdapter | None = None,
    geonames: OntologyAdapter | None = None,
    merge_strategy: str = "union",
    cache_path: str | Path | None = None,
) -> "SemanticContext":
    """Resolve all concept URIs in a contract and build a SemanticContext.

    For each column with ``concept_uris``, resolves labels through the
    appropriate adapter and packages them as resolved aliases and valid
    value sets.

    Parameters:
        contract_path: Path to the annotated contract JSON.
        agrovoc: AGROVOC adapter instance (offline or online).
        geonames: GeoNames adapter instance (offline or online).
        merge_strategy: How to merge resolved and manual aliases —
            ``"union"`` (default), ``"resolved_only"``, ``"manual_priority"``.
        cache_path: If given, write the built context to this JSON path.

    Returns:
        A :class:`docpact.semantics.SemanticContext` ready to pass into
        ``process_document_async()`` or ``run_pipeline_async()``.
    """
    from docpact.semantics import SemanticContext

    with open(contract_path) as f:
        contract = json.load(f)

    adapters: dict[str, OntologyAdapter] = {}
    if agrovoc is not None:
        adapters["agrovoc"] = agrovoc
    if geonames is not None:
        adapters["geonames"] = geonames

    resolved_aliases: dict[str, dict[str, list[str]]] = {}
    valid_values: dict[str, dict[str, set[str]]] = {}

    for out_name, out_spec in contract.get("outputs", {}).items():
        out_aliases: dict[str, list[str]] = {}
        out_valid: dict[str, set[str]] = {}

        for col in out_spec.get("schema", {}).get("columns", []):
            concept_uris_raw = col.get("concept_uris")
            if not concept_uris_raw:
                continue

            concept_refs = [ConceptRef(**c) for c in concept_uris_raw]
            resolve_config = ResolveConfig(**(col.get("resolve") or {}))
            manual_aliases = col.get("aliases", [])

            adapter = adapters.get(resolve_config.source)
            if adapter is None:
                continue

            result = resolve_column(
                col["name"], concept_refs, manual_aliases, resolve_config, adapter,
            )

            # Build merged alias list
            resolved_alias_strings = [ra.alias for ra in result.resolved_aliases]
            if merge_strategy == "resolved_only":
                merged = resolved_alias_strings
            else:
                # "union" and "manual_priority" both start from manual
                existing_lower = {a.lower() for a in manual_aliases}
                merged = list(manual_aliases)
                for alias in resolved_alias_strings:
                    if alias.lower() not in existing_lower:
                        merged.append(alias)
                        existing_lower.add(alias.lower())

            out_aliases[col["name"]] = merged

            # Build valid value set (concept labels + all resolved aliases)
            value_set: set[str] = set()
            for ref in concept_refs:
                if ref.label:
                    value_set.add(ref.label)
            for ra in result.resolved_aliases:
                value_set.add(ra.alias)
            if value_set:
                out_valid[col["name"]] = value_set

        if out_aliases:
            resolved_aliases[out_name] = out_aliases
        if out_valid:
            valid_values[out_name] = out_valid

    adapter_versions: dict[str, str] = {}
    if agrovoc is not None:
        adapter_versions["agrovoc"] = type(agrovoc).__name__
    if geonames is not None:
        adapter_versions["geonames"] = type(geonames).__name__

    ctx = SemanticContext(
        resolved_aliases=resolved_aliases,
        valid_values=valid_values,
        resolved_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        adapter_versions=adapter_versions,
    )

    if cache_path is not None:
        ctx.to_json(cache_path)

    return ctx
