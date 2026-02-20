"""Compare auto-generated (ontology-resolved) aliases vs manual aliases."""

from __future__ import annotations

from contract_semantics.models import ResolutionResult


def diff_aliases(result: ResolutionResult) -> str:
    """Produce a human-readable diff report for a single column's resolution.

    Parameters:
        result: A :class:`ResolutionResult` from :func:`resolve_column`.

    Returns:
        Formatted string report showing coverage, matched, manual-only,
        and resolved-only aliases.
    """
    lines: list[str] = []
    lines.append(f"Column: {result.column_name}")
    lines.append(f"Coverage: {result.coverage:.0%} ({len(result.matched)}/{len(result.manual_aliases)})")
    lines.append("")

    if result.matched:
        lines.append("Matched (manual alias also found via ontology):")
        for alias in sorted(result.matched, key=str.lower):
            lines.append(f"  + {alias}")
        lines.append("")

    if result.manual_only:
        lines.append("Manual-only (not found in ontology — composite terms, domain-specific):")
        for alias in sorted(result.manual_only, key=str.lower):
            lines.append(f"  - {alias}")
        lines.append("")

    if result.resolved_only:
        lines.append("Resolved-only (new aliases from ontology — potential additions):")
        for alias in sorted(result.resolved_only, key=str.lower):
            # Find provenance
            ra = next(
                (r for r in result.resolved_aliases if r.alias == alias), None
            )
            suffix = ""
            if ra:
                parts = [ra.language, ra.label_type]
                if ra.pattern:
                    parts.append(f'pattern="{ra.pattern}"')
                suffix = f" [{', '.join(parts)}]"
            lines.append(f"  * {alias}{suffix}")
        lines.append("")

    return "\n".join(lines)


def diff_all(results: list[ResolutionResult]) -> str:
    """Produce a combined diff report for multiple columns.

    Parameters:
        results: List of :class:`ResolutionResult` from resolving multiple columns.

    Returns:
        Combined formatted report with a summary section.
    """
    sections = [diff_aliases(r) for r in results]
    summary_lines = [
        "=" * 60,
        "Summary",
        "=" * 60,
    ]
    total_manual = sum(len(r.manual_aliases) for r in results)
    total_matched = sum(len(r.matched) for r in results)
    total_resolved_only = sum(len(r.resolved_only) for r in results)

    overall_coverage = total_matched / total_manual if total_manual > 0 else 1.0
    summary_lines.append(f"Columns resolved: {len(results)}")
    summary_lines.append(f"Overall coverage: {overall_coverage:.0%} ({total_matched}/{total_manual})")
    summary_lines.append(f"New aliases discovered: {total_resolved_only}")
    summary_lines.append("")

    return "\n".join(
        ["=" * 60]
        + ["\n" + "=" * 60 + "\n".join([""])]
        + [s for s in sections]
        + summary_lines
    )
