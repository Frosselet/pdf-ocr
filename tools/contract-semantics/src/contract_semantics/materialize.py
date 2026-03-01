"""Contract materialization: annotated contract → materialized contract JSON + geo sidecar.

Reads an annotated contract (with ``concept_uris`` and ``resolve`` fields),
resolves all concept URIs through the appropriate ontology adapters, merges
aliases, and produces:

1. **Materialized contract JSON** — same format as today, aliases enriched.
2. **Geo sidecar file** (optional) — maps region names to GeoNames metadata.
"""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path

from contract_semantics.agrovoc import AgrovocAdapter
from contract_semantics.geonames import GeoNamesAdapter
from contract_semantics.models import ConceptRef, GeoEnrichment, ResolvedAlias, ResolveConfig
from contract_semantics.resolve import OntologyAdapter, resolve_column


def materialize_contract(
    annotated_path: str | Path,
    *,
    agrovoc: AgrovocAdapter | None = None,
    geonames: GeoNamesAdapter | None = None,
    merge_strategy: str = "union",
    output_path: str | Path | None = None,
    geo_sidecar_path: str | Path | None = None,
) -> dict:
    """Materialize an annotated contract into a standard contract with enriched aliases.

    Parameters:
        annotated_path: Path to the annotated contract JSON.
        agrovoc: AGROVOC adapter instance (offline or online).
        geonames: GeoNames adapter instance (offline or online).
        merge_strategy: How to merge resolved and manual aliases.
            ``"union"`` (default) — keep all manual + add resolved.
            ``"resolved_only"`` — replace manual with resolved.
            ``"manual_priority"`` — only add resolved aliases not in manual list.
        output_path: If given, write materialized contract to this path.
        geo_sidecar_path: If given, write GeoNames sidecar to this path.

    Returns:
        The materialized contract dict.
    """
    with open(annotated_path) as f:
        contract = json.load(f)

    materialized = copy.deepcopy(contract)
    geo_sidecar: dict[str, dict] = {}

    adapters: dict[str, OntologyAdapter] = {}
    if agrovoc:
        adapters["agrovoc"] = agrovoc
    if geonames:
        adapters["geonames"] = geonames

    for _out_name, out_spec in materialized.get("outputs", {}).items():
        for col in out_spec.get("schema", {}).get("columns", []):
            concept_uris_raw = col.get("concept_uris")
            resolve_raw = col.get("resolve")

            if not concept_uris_raw:
                continue

            # Parse concept refs
            concept_refs = [ConceptRef(**c) for c in concept_uris_raw]

            # Parse resolve config
            resolve_config = ResolveConfig(**(resolve_raw or {}))

            # Pick adapter
            source = resolve_config.source
            adapter = adapters.get(source)
            if adapter is None:
                continue

            # Resolve
            manual_aliases = col.get("aliases", [])
            result = resolve_column(
                col["name"], concept_refs, manual_aliases, resolve_config, adapter
            )

            # Build resolved alias lookup for provenance
            resolved_lookup: dict[str, ResolvedAlias] = {}
            for ra in result.resolved_aliases:
                resolved_lookup.setdefault(ra.alias.lower(), ra)

            # Merge aliases
            resolved_alias_strings = [ra.alias for ra in result.resolved_aliases]
            if merge_strategy == "union":
                # Manual first, then add new resolved ones
                existing_lower = {a.lower() for a in manual_aliases}
                merged = list(manual_aliases)
                for alias in resolved_alias_strings:
                    if alias.lower() not in existing_lower:
                        merged.append(alias)
                        existing_lower.add(alias.lower())
            elif merge_strategy == "resolved_only":
                merged = resolved_alias_strings
            elif merge_strategy == "manual_priority":
                existing_lower = {a.lower() for a in manual_aliases}
                merged = list(manual_aliases)
                for alias in resolved_alias_strings:
                    if alias.lower() not in existing_lower:
                        merged.append(alias)
                        existing_lower.add(alias.lower())
            else:
                merged = list(manual_aliases)

            col["aliases"] = merged

            # Build per-alias provenance metadata
            manual_lower_set = {a.lower() for a in manual_aliases}
            provenance: dict[str, dict] = {}
            for alias in merged:
                ra = resolved_lookup.get(alias.lower())
                if ra is not None and alias.lower() in manual_lower_set:
                    # Both manual and resolved
                    provenance[alias] = {
                        "source": "both",
                        "concept_uri": ra.concept_uri,
                        "language": ra.language,
                        "label_type": ra.label_type,
                    }
                elif ra is not None:
                    provenance[alias] = {
                        "source": "resolved",
                        "concept_uri": ra.concept_uri,
                        "language": ra.language,
                        "label_type": ra.label_type,
                    }
                else:
                    provenance[alias] = {"source": "manual"}
            col["_alias_provenance"] = provenance

            # Collect geo enrichment for sidecar
            if source == "geonames" and isinstance(adapter, GeoNamesAdapter):
                _collect_geo_enrichment(
                    adapter, concept_refs, resolve_config, geo_sidecar
                )

            # Remove annotation-only fields from materialized output
            col.pop("concept_uris", None)
            col.pop("resolve", None)

    if output_path:
        with open(output_path, "w") as f:
            json.dump(materialized, f, indent=2, ensure_ascii=False)
            f.write("\n")

    if geo_sidecar_path and geo_sidecar:
        sidecar = {
            "_provenance": {
                "source": "GeoNames",
                "resolved_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            },
            "regions": geo_sidecar,
        }
        with open(geo_sidecar_path, "w") as f:
            json.dump(sidecar, f, indent=2, ensure_ascii=False)
            f.write("\n")

    return materialized


def _collect_geo_enrichment(
    adapter: GeoNamesAdapter,
    concept_refs: list[ConceptRef],
    resolve_config: ResolveConfig,
    sidecar: dict[str, dict],
) -> None:
    """Collect GeoNames enrichment data for each concept ref into the sidecar dict."""
    enrich_fields = set(resolve_config.enrich_fields) if resolve_config.enrich_fields else set()
    if not enrich_fields:
        return

    for ref in concept_refs:
        try:
            enrichment: GeoEnrichment = adapter.enrich_geoname(ref.uri)
        except (KeyError, ValueError):
            continue

        entry: dict[str, str | int | float | None] = {
            "geoname_id": enrichment.geoname_id,
        }
        field_map: dict[str, str | int | float | None] = {
            "lat": enrichment.lat,
            "lng": enrichment.lng,
            "admin1Code": enrichment.admin1_code,
            "countryCode": enrichment.country_code,
            "population": enrichment.population,
        }
        for field_name in enrich_fields:
            if field_name in field_map:
                entry[field_name] = field_map[field_name]

        sidecar[ref.label] = entry
