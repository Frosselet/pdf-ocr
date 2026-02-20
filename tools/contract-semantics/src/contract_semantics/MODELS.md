# Models

> **Module**: `models.py`
> **Public API**: `ConceptRef`, `ResolveConfig`, `ResolvedAlias`, `ResolutionResult`, `GeoEnrichment`, `GeoSearchResult`

Pydantic data models shared across all contract-semantics modules. Every inter-module data exchange uses these types.

---

## Data Types

### ConceptRef

Reference to a concept in an external ontology. This is what the contract author writes in the `concept_uris` array.

| Field | Type | Description |
|---|---|---|
| `uri` | `str` | Ontology concept URI (e.g. `http://aims.fao.org/aos/agrovoc/c_8373`) |
| `label` | `str` | Human-readable label (e.g. `"wheat"`) |

### ResolveConfig

Controls how concept URIs are resolved into aliases. Parsed from the `resolve` field in a contract column.

| Field | Type | Default | Description |
|---|---|---|---|
| `source` | `str` | `"agrovoc"` | Ontology adapter to use (`"agrovoc"` or `"geonames"`) |
| `languages` | `list[str]` | `["en"]` | Language codes to retrieve labels for |
| `label_types` | `list[str]` | `["prefLabel", "altLabel"]` | SKOS label types to query |
| `prefix_patterns` | `list[str]` | `[]` | Expansion patterns (e.g. `"spring {label}"`) |
| `feature_class` | `str \| None` | `None` | GeoNames feature class filter (e.g. `"A"` for admin) |
| `feature_code` | `str \| None` | `None` | GeoNames feature code filter (e.g. `"ADM1"`) |
| `country` | `str \| None` | `None` | GeoNames country code filter (e.g. `"RU"`) |
| `enrich_fields` | `list[str]` | `[]` | GeoNames fields for geo sidecar (e.g. `["lat", "lng"]`) |

### ResolvedAlias

A single alias resolved from an ontology concept. Carries full provenance.

| Field | Type | Description |
|---|---|---|
| `alias` | `str` | The resolved alias string |
| `concept_uri` | `str` | Source concept URI |
| `concept_label` | `str` | English prefLabel of the source concept |
| `language` | `str` | Language code of this alias |
| `label_type` | `str` | How it was found (`"prefLabel"`, `"altLabel"`, `"name"`, `"asciiname"`) |
| `pattern` | `str \| None` | Prefix pattern used to generate this alias, if any |

### ResolutionResult

Result of resolving a column's concept URIs to aliases. Produced by `resolve_column()`.

| Field | Type | Description |
|---|---|---|
| `column_name` | `str` | Contract column name |
| `resolved_aliases` | `list[ResolvedAlias]` | All aliases found via ontology resolution |
| `manual_aliases` | `list[str]` | Original hand-curated aliases from the contract |
| `matched` | `list[str]` | Manual aliases also found via resolution |
| `manual_only` | `list[str]` | Manual aliases not found in ontology (composites, domain-specific) |
| `resolved_only` | `list[str]` | New aliases from ontology not in manual list |

**Property**: `coverage` --- fraction of manual aliases also found via resolution (`len(matched) / len(manual_aliases)`).

### GeoEnrichment

Geographic metadata from GeoNames for a resolved location. Used to build geo sidecar files.

| Field | Type | Description |
|---|---|---|
| `geoname_id` | `int` | GeoNames feature ID |
| `name` | `str` | Feature name |
| `lat` | `float` | Latitude |
| `lng` | `float` | Longitude |
| `country_code` | `str` | ISO country code |
| `admin1_code` | `str` | Admin-1 code |
| `feature_class` | `str` | GeoNames feature class |
| `feature_code` | `str` | GeoNames feature code |
| `population` | `int \| None` | Population (if available) |

### GeoSearchResult

A single GeoNames search result. Returned by `search_geonames()`.

| Field | Type | Description |
|---|---|---|
| `geoname_id` | `int` | GeoNames feature ID |
| `name` | `str` | Feature name |
| `country_code` | `str` | ISO country code |
| `feature_class` | `str` | GeoNames feature class |
| `feature_code` | `str` | GeoNames feature code |
| `admin1_code` | `str` | Admin-1 code |
| `lat` | `float` | Latitude |
| `lng` | `float` | Longitude |
