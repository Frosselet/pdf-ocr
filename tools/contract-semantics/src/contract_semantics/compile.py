"""Compiler: JSON-LD contract → materialized contract.

Two-step process:
1. **Grounding resolution** — each column's ``grounding`` block is resolved to
   ``concept_uris`` + ``resolve`` (the annotated format).
2. **Materialization** — the annotated dict is passed to
   :func:`materialize._materialize_dict` for alias expansion, merge, and
   provenance tracking.

The compiler never touches the extraction pipeline. Its sole output is a
materialized contract JSON in the current format.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from contract_semantics.agrovoc import AgrovocAdapter
from contract_semantics.geonames import LEVEL_MAP, GeoNamesAdapter
from contract_semantics.materialize import _materialize_dict

_SUPPORTED_ONTOLOGIES = ("agrovoc", "geonames")


def compile_contract(
    jsonld_path: str | Path,
    *,
    agrovoc: AgrovocAdapter | None = None,
    geonames: GeoNamesAdapter | None = None,
    merge_strategy: str = "union",
    output_path: str | Path | None = None,
    geo_sidecar_path: str | Path | None = None,
) -> dict:
    """Compile a JSON-LD contract to a materialized contract.

    Steps:
    1. Parse and validate JSON-LD structure (``@context``, ``@type``,
       grounding blocks).
    2. For each column with a ``grounding`` block:
       - AGROVOC: ``lookup_by_label()`` for each concept → ``concept_uris``
         + ``resolve`` config.
       - GeoNames: ``enumerate_features()`` for country+level →
         ``concept_uris`` + ``resolve`` config.
    3. Map contract-level ``languages`` to per-column ``resolve.languages``.
    4. Pass the annotated dict to :func:`_materialize_dict` for alias
       resolution.
    5. Add ``_compiled_from`` metadata to output.

    Parameters:
        jsonld_path: Path to the JSON-LD contract file.
        agrovoc: AGROVOC adapter (offline or online).
        geonames: GeoNames adapter (offline or online).
        merge_strategy: Alias merge strategy for materialization.
        output_path: If given, write materialized contract to this path.
        geo_sidecar_path: If given, write GeoNames sidecar to this path.

    Returns:
        The materialized contract dict.
    """
    jsonld_path = Path(jsonld_path)
    with open(jsonld_path) as f:
        jsonld = json.load(f)

    _validate_jsonld(jsonld)

    # Step 1: resolve grounding → concept_uris + resolve
    annotated = _resolve_groundings(jsonld, agrovoc=agrovoc, geonames=geonames)

    # Step 2: materialize (alias expansion, merge, provenance)
    materialized = _materialize_dict(
        annotated,
        agrovoc=agrovoc,
        geonames=geonames,
        merge_strategy=merge_strategy,
        output_path=output_path,
        geo_sidecar_path=geo_sidecar_path,
    )

    # Step 3: add compilation metadata
    materialized["_compiled_from"] = str(jsonld_path.name)

    # Re-write if output_path was specified (to include _compiled_from)
    if output_path:
        with open(output_path, "w") as f:
            json.dump(materialized, f, indent=2, ensure_ascii=False)
            f.write("\n")

    return materialized


def _validate_jsonld(contract: dict) -> None:
    """Validate required JSON-LD structure."""
    if "@context" not in contract:
        raise ValueError(
            "JSON-LD contract must have an '@context' field. "
            'Use "@context": "https://ckir.dev/context/v1" for the default.'
        )

    if contract.get("@type") != "ExtractionContract":
        raise ValueError(
            "JSON-LD contract must have '@type': 'ExtractionContract'. "
            f"Got: {contract.get('@type')!r}"
        )

    # Validate grounding blocks
    for _out_name, out_spec in contract.get("outputs", {}).items():
        for col in out_spec.get("schema", {}).get("columns", []):
            grounding = col.get("grounding")
            if not grounding:
                continue

            ontology = grounding.get("ontology")
            if not ontology:
                raise ValueError(
                    f"Column {col['name']!r}: grounding block must have 'ontology' field."
                )
            if ontology not in _SUPPORTED_ONTOLOGIES:
                raise ValueError(
                    f"Column {col['name']!r}: unknown ontology {ontology!r}. "
                    f"Supported: {list(_SUPPORTED_ONTOLOGIES)}"
                )
            if ontology == "agrovoc" and not grounding.get("concepts"):
                raise ValueError(
                    f"Column {col['name']!r}: AGROVOC grounding must have 'concepts' list."
                )
            if ontology == "geonames":
                if not grounding.get("country"):
                    raise ValueError(
                        f"Column {col['name']!r}: GeoNames grounding must have 'country'."
                    )
                if not grounding.get("level"):
                    raise ValueError(
                        f"Column {col['name']!r}: GeoNames grounding must have 'level'."
                    )


def _resolve_groundings(
    jsonld: dict,
    *,
    agrovoc: AgrovocAdapter | None = None,
    geonames: GeoNamesAdapter | None = None,
) -> dict:
    """Transform grounding blocks into concept_uris + resolve, strip JSON-LD fields."""
    annotated = copy.deepcopy(jsonld)
    languages = annotated.get("languages", ["en"])

    for _out_name, out_spec in annotated.get("outputs", {}).items():
        for col in out_spec.get("schema", {}).get("columns", []):
            grounding = col.get("grounding")
            if not grounding:
                continue

            ontology = grounding["ontology"]

            if ontology == "agrovoc":
                _resolve_agrovoc_grounding(col, grounding, languages, agrovoc)
            elif ontology == "geonames":
                _resolve_geonames_grounding(col, grounding, languages, geonames)

            # Remove the grounding block
            del col["grounding"]

    # Strip JSON-LD metadata fields
    annotated.pop("@context", None)
    annotated.pop("@type", None)
    annotated.pop("domain", None)
    annotated.pop("sourceFormat", None)
    annotated.pop("languages", None)

    return annotated


def _resolve_agrovoc_grounding(
    col: dict,
    grounding: dict,
    languages: list[str],
    agrovoc: AgrovocAdapter | None,
) -> None:
    """Resolve AGROVOC grounding: concept labels → concept_uris + resolve."""
    if agrovoc is None:
        raise ValueError(
            f"Column {col['name']!r}: AGROVOC grounding requires an AgrovocAdapter. "
            "Load AGROVOC data first (fetch-agrovoc or provide a graph)."
        )

    concepts = grounding["concepts"]
    label_types = grounding.get("labelTypes", ["prefLabel", "altLabel"])
    include_narrower = grounding.get("includeNarrower", False)
    variants = grounding.get("variants", [])

    # Lookup each concept label → URI
    concept_uris = []
    for label in concepts:
        uri = agrovoc.lookup_by_label(label, language="en")
        concept_uris.append({"uri": uri, "label": label})

    col["concept_uris"] = concept_uris

    # Build resolve config
    resolve: dict = {
        "languages": languages,
        "label_types": label_types,
    }

    if include_narrower:
        resolve["include_narrower"] = True

    # Build prefix_patterns from variants
    if variants:
        patterns = [f"{v} {{label}}" for v in variants]
        patterns.append("{label}")
        resolve["prefix_patterns"] = patterns

    col["resolve"] = resolve


def _resolve_geonames_grounding(
    col: dict,
    grounding: dict,
    languages: list[str],
    geonames: GeoNamesAdapter | None,
) -> None:
    """Resolve GeoNames grounding: country+level → concept_uris + resolve."""
    if geonames is None:
        raise ValueError(
            f"Column {col['name']!r}: GeoNames grounding requires a GeoNamesAdapter. "
            "Load GeoNames data first (fetch-geonames or provide features)."
        )

    country = grounding["country"]
    level = grounding["level"]
    enrich = grounding.get("enrich", [])

    # Enumerate all features for this country + level
    features = geonames.enumerate_features(country, level)

    # Build concept_uris
    concept_uris = []
    for feat in features:
        uri = f"https://sws.geonames.org/{feat.geoname_id}/"
        concept_uris.append({"uri": uri, "label": feat.name})

    col["concept_uris"] = concept_uris

    # Build resolve config
    feat_class, feat_code = LEVEL_MAP[level]
    resolve: dict = {
        "source": "geonames",
        "feature_class": feat_class,
        "feature_code": feat_code,
        "country": country,
        "languages": languages,
    }

    if enrich:
        resolve["enrich_fields"] = enrich

    col["resolve"] = resolve
