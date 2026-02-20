"""Contract recommendation engine — generates draft contract skeletons from document profiles.

Takes a DocumentProfile or MultiDocumentProfile from analyze.py and produces a
draft contract JSON with inline recommendations. Handles:
- Table grouping → category keyword proposals
- Column types → schema column definitions
- Alias suggestions from header variants
- Pivot detection → unpivot strategy
- Year columns → {YYYY} template aliases
- Section labels → dimension column aliases
- Title-based columns → source: "title" suggestions

Public functions:
    recommend_contract(profile) → dict  (draft contract JSON)
    compare_contract(profile, existing) → str  (gap report)
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from contract_semantics.analyze import (
    AlignedColumn,
    DocumentProfile,
    MultiDocumentProfile,
    TableGroup,
    TableProfile,
    _YEAR_RE,
)


# ---------------------------------------------------------------------------
# Column name normalization
# ---------------------------------------------------------------------------


def _to_snake_case(text: str) -> str:
    """Convert a header string to a plausible column name.

    "NAME OF SHIP" → "name_of_ship"
    "Area harvested" → "area_harvested"
    "Quantity (tonnes)" → "quantity"
    """
    # Strip parenthesized suffixes (unit annotations)
    text = re.sub(r"\s*\([^)]*\)\s*$", "", text)
    # Strip comma-separated unit suffix
    text = re.sub(r",\s*[^,]+$", "", text)
    # Strip compound header parts after /
    if " / " in text:
        text = text.split(" / ")[0]
    # Lowercase, replace non-alphanum with underscore, collapse
    result = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return result or "column"


def _pick_canonical_name(headers: list[str]) -> str:
    """Pick the best canonical column name from header variants.

    Prefers the most common header, falling back to the shortest one.
    """
    if not headers:
        return "column"
    counts: Counter[str] = Counter(headers)
    most_common = counts.most_common(1)[0][0]
    return most_common


# ---------------------------------------------------------------------------
# Category keyword extraction
# ---------------------------------------------------------------------------


def _extract_keywords(tables: list[TableProfile], all_tokens: set[str]) -> list[str]:
    """Extract distinctive keywords for a table group.

    Finds tokens that appear frequently in this group's headers
    and are reasonably distinctive (not ultra-common words).
    """
    _STOP_WORDS = {
        "the", "a", "an", "of", "in", "to", "for", "and", "or", "on",
        "at", "by", "is", "it", "no", "not", "with", "from", "as", "be",
        "total", "name", "date", "type", "number", "value", "code",
    }

    token_freq: Counter[str] = Counter()
    for tp in tables:
        for t in tp.header_tokens:
            token_freq[t] += 1

    # Also count tokens from titles and section labels
    for tp in tables:
        if tp.title:
            for word in tp.title.lower().split():
                cleaned = re.sub(r"[^a-z0-9]", "", word)
                if cleaned and cleaned not in _STOP_WORDS:
                    token_freq[cleaned] += 1

    # Filter: must appear in > 30% of tables, not a stop word
    min_count = max(1, len(tables) * 0.3)
    keywords = [
        token
        for token, count in token_freq.most_common(20)
        if count >= min_count and token not in _STOP_WORDS and len(token) > 2
    ]

    return keywords[:10]


# ---------------------------------------------------------------------------
# Schema column generation
# ---------------------------------------------------------------------------


def _build_column_spec(
    ac: AlignedColumn,
    *,
    is_section_dimension: bool = False,
    section_labels: list[str] | None = None,
    is_title_column: bool = False,
) -> dict:
    """Build a single column spec dict for the draft contract schema."""
    name = _to_snake_case(ac.canonical_header)

    col: dict = {
        "name": name,
        "type": ac.inferred_type,
    }

    # Recommendations (stripped during finalization)
    recs: list[str] = []

    # Aliases
    if ac.year_detected:
        col["aliases"] = ["{YYYY}"]
        recs.append("Year column detected in header. Using {YYYY} template alias.")
        col["_header_text"] = ac.canonical_header
    elif is_title_column:
        col["source"] = "title"
        recs.append(
            "Table title matches this column's role. "
            "Using source: 'title' to extract from the heading above the table."
        )
    elif is_section_dimension:
        aliases = list(section_labels) if section_labels else []
        col["aliases"] = aliases
        recs.append(
            f"Section labels detected as dimension values. "
            f"{len(aliases)} unique label(s) collected as aliases."
        )
    else:
        # Collect all unique header variants as aliases
        seen_lower: set[str] = set()
        aliases: list[str] = []
        for h in ac.all_headers:
            h_stripped = h.strip()
            if h_stripped and h_stripped.lower() not in seen_lower:
                seen_lower.add(h_stripped.lower())
                aliases.append(h_stripped)
        col["aliases"] = aliases

        if len(aliases) > 1:
            recs.append(
                f"{len(aliases)} header variants found across documents."
            )

    # Type-specific recommendations
    if ac.inferred_type == "string" and not ac.year_detected:
        # Check for low cardinality (potential enum / GeoNames grounding)
        max_unique = 0
        for h in ac.all_headers:
            # We don't have access to unique counts in AlignedColumn directly,
            # but the source tables do. This is a simplification.
            pass
        if is_section_dimension and section_labels and len(section_labels) > 10:
            recs.append(
                f"Low-cardinality string column. "
                f"{len(section_labels)} unique values — consider ontology grounding "
                f"(GeoNames for regions, AGROVOC for agricultural terms)."
            )

    if ac.unit_annotations:
        col["_unit_annotations"] = ac.unit_annotations
        recs.append(f"Unit annotation(s) in header: {', '.join(ac.unit_annotations)}")

    col["_detected_type"] = ac.inferred_type
    if recs:
        col["_recommendation"] = " ".join(recs)

    return col


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def recommend_contract(
    profile: DocumentProfile | MultiDocumentProfile,
    *,
    provider_name: str | None = None,
    model: str = "openai/gpt-4o",
) -> dict:
    """Generate a draft contract JSON from a document profile.

    The draft includes inline `_recommendation` and `_detected_*` fields
    that serve as guidance during human review. These fields are stripped
    when the contract is finalized (they are ignored by docpact's
    load_contract()).

    Args:
        profile: DocumentProfile or MultiDocumentProfile from analyze.py.
        provider_name: Optional human-readable provider name.
        model: LLM model identifier for the contract.

    Returns:
        Dict representing the draft contract JSON.
    """
    if isinstance(profile, DocumentProfile):
        # Wrap in MultiDocumentProfile for uniform handling
        from contract_semantics.analyze import merge_profiles

        multi = merge_profiles([profile])
    else:
        multi = profile

    source_docs = [Path(p).name for p in multi.doc_paths]
    contract: dict = {
        "_analyzer_version": "0.1.0",
        "_source_documents": source_docs,
        "provider": provider_name or _guess_provider_name(source_docs),
        "model": model,
    }

    # Build categories from table groups
    categories: dict[str, dict] = {}
    outputs: dict[str, dict] = {}

    for g in multi.table_groups:
        cat_name = _generate_category_name(g)
        keywords = _extract_keywords(g.tables, g.common_tokens)

        cat_entry: dict = {"keywords": keywords}
        if len(g.tables) > 1:
            cat_entry["_recommendation"] = (
                f"{len(g.tables)} tables share these header tokens"
            )
        categories[cat_name] = cat_entry

        # Build output schema
        schema_columns: list[dict] = []

        # Detect if section labels should become a dimension column
        has_section_dim = bool(g.all_section_labels)

        for ac in g.aligned_columns:
            col_spec = _build_column_spec(ac)
            schema_columns.append(col_spec)

        # Add section label dimension column if present
        if has_section_dim:
            section_col = _build_column_spec(
                AlignedColumn(
                    canonical_header="section",
                    all_headers=["section"],
                    inferred_type="string",
                ),
                is_section_dimension=True,
                section_labels=g.all_section_labels,
            )
            schema_columns.insert(0, section_col)

        # Add title column if any table has a title
        titles = [t.title for t in g.tables if t.title]
        if titles:
            # Check if we already have a column that could be title-sourced
            has_title_col = any(c.get("source") == "title" for c in schema_columns)
            if not has_title_col:
                title_col: dict = {
                    "name": "table_title",
                    "type": "string",
                    "source": "title",
                    "_recommendation": (
                        f"Table titles detected: {', '.join(sorted(set(titles))[:5])}. "
                        "Using source: 'title' to extract from heading above table."
                    ),
                }
                schema_columns.insert(0, title_col)

        # Unpivot strategy recommendation
        unpivot_str: str | None = None
        if g.layout == "pivoted":
            unpivot_str = "deterministic"
            contract.setdefault("_recommendations", []).append(
                f"Group '{cat_name}': pivoted tables detected. "
                f"Recommend unpivot: 'deterministic'."
            )

        # Report date suggestion
        report_date_suggestion = _suggest_report_date(multi)

        output_entry: dict = {
            "category": cat_name,
            "schema": {
                "columns": schema_columns,
            },
        }
        if unpivot_str:
            output_entry["_unpivot_recommendation"] = unpivot_str

        outputs[cat_name] = output_entry

    contract["categories"] = categories
    contract["outputs"] = outputs

    # Top-level unpivot strategy
    any_pivoted = any(g.layout == "pivoted" for g in multi.table_groups)
    if any_pivoted:
        contract["unpivot"] = "deterministic"

    # Report date config
    report_date_cfg = _suggest_report_date(multi)
    if report_date_cfg:
        contract["report_date"] = report_date_cfg

    return contract


def compare_contract(
    profile: DocumentProfile | MultiDocumentProfile,
    existing_path: str | Path,
) -> str:
    """Compare a document profile against an existing contract.

    Flags:
    - Headers in documents not covered by any alias
    - Aliases in contract not found in any document
    - Type mismatches between detected and declared types
    - Missing categories (tables that don't match any category)

    Args:
        profile: DocumentProfile or MultiDocumentProfile.
        existing_path: Path to existing contract JSON.

    Returns:
        Multi-line human-readable gap report.
    """
    with open(existing_path) as f:
        contract = json.load(f)

    lines: list[str] = []
    lines.append("Contract Comparison Report")
    lines.append("=" * 60)
    lines.append(f"Contract: {existing_path}")
    lines.append("")

    # Collect all document headers
    if isinstance(profile, MultiDocumentProfile):
        all_tables = [
            t for dp in profile.document_profiles for t in dp.tables
        ]
    else:
        all_tables = list(profile.tables)

    all_headers: set[str] = set()
    for tp in all_tables:
        for cp in tp.column_profiles:
            all_headers.add(cp.header_text.strip())

    # Collect all contract aliases (normalized)
    contract_aliases: set[str] = set()
    alias_to_column: dict[str, str] = {}
    for _out_name, out_spec in contract.get("outputs", {}).items():
        for col in out_spec.get("schema", {}).get("columns", []):
            for alias in col.get("aliases", []):
                # Skip template aliases
                if "{" in alias:
                    continue
                contract_aliases.add(alias.lower().strip())
                alias_to_column[alias.lower().strip()] = col["name"]

    # Headers not covered by aliases
    uncovered: list[str] = []
    for h in sorted(all_headers):
        if h.lower().strip() not in contract_aliases:
            uncovered.append(h)

    if uncovered:
        lines.append(f"Uncovered headers ({len(uncovered)}):")
        lines.append("  These document headers don't match any contract alias:")
        for h in uncovered[:30]:
            lines.append(f"    - {h!r}")
        if len(uncovered) > 30:
            lines.append(f"    ... and {len(uncovered) - 30} more")
        lines.append("")
    else:
        lines.append("All document headers are covered by contract aliases.")
        lines.append("")

    # Contract aliases not found in documents
    unused: list[str] = []
    for alias in sorted(contract_aliases):
        found = False
        for h in all_headers:
            if h.lower().strip() == alias:
                found = True
                break
        if not found:
            unused.append(alias)

    if unused:
        lines.append(f"Unused aliases ({len(unused)}):")
        lines.append("  These contract aliases don't match any document header:")
        for a in unused[:30]:
            col = alias_to_column.get(a, "?")
            lines.append(f"    - {a!r} (column: {col})")
        if len(unused) > 30:
            lines.append(f"    ... and {len(unused) - 30} more")
        lines.append("")

    # Section labels not in aliases
    all_section_labels: set[str] = set()
    for tp in all_tables:
        for label in tp.section_labels:
            all_section_labels.add(label)

    uncovered_labels = [
        l for l in sorted(all_section_labels)
        if l.lower().strip() not in contract_aliases
    ]
    if uncovered_labels:
        lines.append(f"Uncovered section labels ({len(uncovered_labels)}):")
        for l in uncovered_labels[:20]:
            lines.append(f"    - {l!r}")
        lines.append("")

    # Layout summary
    layouts: Counter[str] = Counter()
    for tp in all_tables:
        layouts[tp.layout] += 1
    if layouts:
        lines.append("Table layouts detected:")
        for layout, count in layouts.most_common():
            lines.append(f"  {layout}: {count}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _guess_provider_name(source_docs: list[str]) -> str:
    """Guess a provider name from source document filenames."""
    if not source_docs:
        return "unknown_provider"
    if len(source_docs) == 1:
        return Path(source_docs[0]).stem
    # Find common prefix
    common = source_docs[0]
    for doc in source_docs[1:]:
        while not doc.startswith(common) and common:
            common = common[:-1]
    return common.rstrip("_- .") if common else "multi_provider"


def _generate_category_name(group: TableGroup) -> str:
    """Generate a category name from a table group's tokens."""
    if group.common_tokens:
        # Pick the most descriptive token (longest)
        candidates = sorted(group.common_tokens, key=len, reverse=True)
        return candidates[0]

    # Fall back to first table's title or index
    if group.tables and group.tables[0].title:
        name = re.sub(r"[^a-z0-9]+", "_", group.tables[0].title.lower()).strip("_")
        return name[:30]

    return f"group_{group.group_id}"


def _suggest_report_date(
    profile: MultiDocumentProfile,
) -> dict | None:
    """Suggest a report_date config from temporal metadata candidates."""
    if not profile.all_temporal_candidates:
        return None

    # Prefer "as_of_date" patterns, then "period_end"
    for mc in profile.all_temporal_candidates:
        if mc.pattern_name == "as_of_date":
            return {
                "source": "content",
                "hint": f"Found: {mc.text!r}",
                "_recommendation": (
                    f"Temporal pattern detected: {mc.text!r}. "
                    "Set source to 'filename' or 'content' based on your use case."
                ),
            }

    for mc in profile.all_temporal_candidates:
        if mc.pattern_name == "period_end":
            return {
                "source": "content",
                "hint": f"Found: {mc.text!r}",
                "_recommendation": f"Period end detected: {mc.text!r}.",
            }

    # Any temporal pattern
    mc = profile.all_temporal_candidates[0]
    return {
        "source": "content",
        "hint": f"Found: {mc.text!r}",
        "_recommendation": f"Temporal pattern: {mc.text!r} ({mc.pattern_name}).",
    }


def strip_recommendations(contract: dict) -> dict:
    """Strip all _recommendation, _detected_*, _analyzer_* fields from a contract.

    Returns a clean contract JSON suitable for use with docpact's load_contract().
    """
    import copy

    result = copy.deepcopy(contract)
    _strip_underscore_keys(result)
    return result


def _strip_underscore_keys(obj: dict | list) -> None:
    """Recursively remove keys starting with '_' from dicts."""
    if isinstance(obj, dict):
        keys_to_remove = [k for k in obj if isinstance(k, str) and k.startswith("_")]
        for k in keys_to_remove:
            del obj[k]
        for v in obj.values():
            if isinstance(v, (dict, list)):
                _strip_underscore_keys(v)
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, (dict, list)):
                _strip_underscore_keys(item)
