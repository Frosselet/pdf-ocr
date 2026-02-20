"""Click CLI for contract-semantics: analyze, materialize, diff, fetch-agrovoc, fetch-geonames."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from contract_semantics.agrovoc import AgrovocAdapter
from contract_semantics.diff import diff_aliases
from contract_semantics.geonames import GeoNamesAdapter
from contract_semantics.materialize import materialize_contract
from contract_semantics.models import ConceptRef, ResolveConfig
from contract_semantics.resolve import resolve_column

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


@click.group()
def main() -> None:
    """Contract-semantics: ontology-grounded contract authoring toolkit."""


@main.command()
@click.option("--output", "-o", default=None, help="Output directory (default: same as data/)")
def fetch_agrovoc(output: str | None) -> None:
    """Download the AGROVOC N-Triples dump."""
    import httpx

    out_dir = Path(output) if output else _DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    url = "https://agrovoc.fao.org/agrovocReleases/agrovoc_core.nt.zip"
    dest = out_dir / "agrovoc_core.nt.zip"

    click.echo(f"Downloading AGROVOC from {url} ...")
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        with client.stream("GET", url) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            with open(dest, "wb") as f:
                downloaded = 0
                for chunk in resp.iter_bytes(chunk_size=65536):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        click.echo(f"\r  {pct:.1f}% ({downloaded}/{total})", nl=False)

    click.echo(f"\nSaved to {dest}")
    click.echo("Extract with: unzip agrovoc_core.nt.zip")


@main.command()
@click.option("--country", "-c", default="RU", help="Country code (default: RU)")
@click.option("--output", "-o", default=None, help="Output directory")
def fetch_geonames(country: str, output: str | None) -> None:
    """Download a GeoNames country extract."""
    import httpx

    out_dir = Path(output) if output else _DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://download.geonames.org/export/dump/{country}.zip"
    dest = out_dir / f"{country}.zip"

    click.echo(f"Downloading GeoNames {country} extract from {url} ...")
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        with client.stream("GET", url) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            with open(dest, "wb") as f:
                downloaded = 0
                for chunk in resp.iter_bytes(chunk_size=65536):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        click.echo(f"\r  {pct:.1f}% ({downloaded}/{total})", nl=False)

    click.echo(f"\nSaved to {dest}")
    click.echo(f"Extract with: unzip {country}.zip")


@main.command()
@click.argument("contract_path", type=click.Path(exists=True))
def diff(contract_path: str) -> None:
    """Compare ontology-resolved aliases vs manual aliases for a contract."""
    with open(contract_path) as f:
        contract = json.load(f)

    # Build adapters lazily based on what the contract needs
    agrovoc: AgrovocAdapter | None = None
    geonames: GeoNamesAdapter | None = None

    results = []

    for _out_name, out_spec in contract.get("outputs", {}).items():
        for col in out_spec.get("schema", {}).get("columns", []):
            concept_uris_raw = col.get("concept_uris")
            resolve_raw = col.get("resolve")
            if not concept_uris_raw:
                continue

            concept_refs = [ConceptRef(**c) for c in concept_uris_raw]
            resolve_config = ResolveConfig(**(resolve_raw or {}))
            manual_aliases = col.get("aliases", [])

            # Pick or create adapter
            source = resolve_config.source
            if source == "geonames":
                if geonames is None:
                    # Try offline first
                    ru_path = _DATA_DIR / "RU.txt"
                    if ru_path.exists():
                        click.echo(f"Loading GeoNames data from {ru_path} ...")
                        geonames = GeoNamesAdapter.from_file(ru_path)
                    else:
                        click.echo("No local GeoNames data. Run 'fetch-geonames' first.", err=True)
                        sys.exit(1)
                adapter = geonames
            else:
                if agrovoc is None:
                    # Try offline first
                    nt_path = _DATA_DIR / "agrovoc_core.nt"
                    if nt_path.exists():
                        click.echo(f"Loading AGROVOC from {nt_path} ...")
                        agrovoc = AgrovocAdapter.from_file(nt_path)
                    else:
                        click.echo("Using AGROVOC SPARQL endpoint (online mode).")
                        agrovoc = AgrovocAdapter.online()
                adapter = agrovoc

            result = resolve_column(
                col["name"], concept_refs, manual_aliases, resolve_config, adapter
            )
            results.append(result)
            click.echo(diff_aliases(result))

    if not results:
        click.echo("No columns with concept_uris found in this contract.")


@main.command()
@click.argument("contract_path", type=click.Path(exists=True))
@click.option("--output", "-o", required=True, help="Output materialized contract path")
@click.option("--geo-sidecar", default=None, help="Output geo sidecar path")
@click.option("--merge", default="union", type=click.Choice(["union", "resolved_only", "manual_priority"]))
def materialize(contract_path: str, output: str, geo_sidecar: str | None, merge: str) -> None:
    """Materialize an annotated contract: resolve concept URIs → enriched aliases."""
    # Build adapters
    agrovoc: AgrovocAdapter | None = None
    geonames: GeoNamesAdapter | None = None

    # Detect what the contract needs
    with open(contract_path) as f:
        contract = json.load(f)

    needs_agrovoc = False
    needs_geonames = False
    for _out_name, out_spec in contract.get("outputs", {}).items():
        for col in out_spec.get("schema", {}).get("columns", []):
            if col.get("concept_uris"):
                source = (col.get("resolve") or {}).get("source", "agrovoc")
                if source == "geonames":
                    needs_geonames = True
                else:
                    needs_agrovoc = True

    if needs_agrovoc:
        nt_path = _DATA_DIR / "agrovoc_core.nt"
        if nt_path.exists():
            click.echo(f"Loading AGROVOC from {nt_path} ...")
            agrovoc = AgrovocAdapter.from_file(nt_path)
        else:
            click.echo("Using AGROVOC SPARQL endpoint (online mode).")
            agrovoc = AgrovocAdapter.online()

    if needs_geonames:
        ru_path = _DATA_DIR / "RU.txt"
        if ru_path.exists():
            click.echo(f"Loading GeoNames data from {ru_path} ...")
            geonames = GeoNamesAdapter.from_file(ru_path)
        else:
            click.echo("No local GeoNames data. Run 'fetch-geonames' first.", err=True)
            sys.exit(1)

    result = materialize_contract(
        contract_path,
        agrovoc=agrovoc,
        geonames=geonames,
        merge_strategy=merge,
        output_path=output,
        geo_sidecar_path=geo_sidecar,
    )

    total_aliases = 0
    for _out_name, out_spec in result.get("outputs", {}).items():
        for col in out_spec.get("schema", {}).get("columns", []):
            total_aliases += len(col.get("aliases", []))

    click.echo(f"Materialized contract written to {output}")
    click.echo(f"Total aliases across all columns: {total_aliases}")
    if geo_sidecar:
        click.echo(f"Geo sidecar written to {geo_sidecar}")


@main.command()
@click.argument("csv_path", type=click.Path(exists=True))
@click.argument("shapes_path", type=click.Path(exists=True))
def validate(csv_path: str, shapes_path: str) -> None:
    """Validate a CSV file against SHACL shapes."""
    from contract_semantics.validate import validate_csv

    conforms, report_text, _graph = validate_csv(csv_path, shapes_path)

    if conforms:
        click.echo("VALID: All records conform to the shapes.")
    else:
        click.echo("INVALID: Validation errors found:\n")
        click.echo(report_text)
    sys.exit(0 if conforms else 1)


@main.command("generate-shapes")
@click.argument("contract_path", type=click.Path(exists=True))
@click.option("--output-dir", "-o", default=None, help="Output directory for shapes")
@click.option("--output-name", default=None, help="Only generate for this output")
def generate_shapes_cmd(contract_path: str, output_dir: str | None, output_name: str | None) -> None:
    """Auto-generate SHACL shapes from a contract's schemas."""
    from contract_semantics.validate import generate_shapes

    results = generate_shapes(contract_path, output_name=output_name, output_dir=output_dir)

    for name, path in results.items():
        click.echo(f"Generated shapes for '{name}': {path}")


@main.command("build-context")
@click.argument("contract_path", type=click.Path(exists=True))
@click.option("--output", "-o", required=True, help="Output semantic context JSON path")
@click.option("--merge", default="union", type=click.Choice(["union", "resolved_only", "manual_priority"]))
def build_context(contract_path: str, output: str, merge: str) -> None:
    """Build a SemanticContext from an annotated contract.

    Resolves all concept URIs through ontology adapters and produces a
    SemanticContext JSON file that can be passed to the docpact pipeline
    for runtime alias enrichment, pre-flight checks, and validation.
    """
    from contract_semantics.context import build_semantic_context

    # Detect what the contract needs
    with open(contract_path) as f:
        contract = json.load(f)

    needs_agrovoc = False
    needs_geonames = False
    for _out_name, out_spec in contract.get("outputs", {}).items():
        for col in out_spec.get("schema", {}).get("columns", []):
            if col.get("concept_uris"):
                source = (col.get("resolve") or {}).get("source", "agrovoc")
                if source == "geonames":
                    needs_geonames = True
                else:
                    needs_agrovoc = True

    agrovoc: AgrovocAdapter | None = None
    geonames: GeoNamesAdapter | None = None

    if needs_agrovoc:
        nt_path = _DATA_DIR / "agrovoc_core.nt"
        if nt_path.exists():
            click.echo(f"Loading AGROVOC from {nt_path} ...")
            agrovoc = AgrovocAdapter.from_file(nt_path)
        else:
            click.echo("Using AGROVOC SPARQL endpoint (online mode).")
            agrovoc = AgrovocAdapter.online()

    if needs_geonames:
        ru_path = _DATA_DIR / "RU.txt"
        if ru_path.exists():
            click.echo(f"Loading GeoNames data from {ru_path} ...")
            geonames = GeoNamesAdapter.from_file(ru_path)
        else:
            click.echo("No local GeoNames data. Run 'fetch-geonames' first.", err=True)
            sys.exit(1)

    ctx = build_semantic_context(
        contract_path,
        agrovoc=agrovoc,
        geonames=geonames,
        merge_strategy=merge,
        cache_path=output,
    )

    total_aliases = sum(
        len(aliases)
        for cols in ctx.resolved_aliases.values()
        for aliases in cols.values()
    )
    total_valid = sum(
        len(values)
        for cols in ctx.valid_values.values()
        for values in cols.values()
    )

    click.echo(f"Semantic context written to {output}")
    click.echo(f"  Resolved aliases: {total_aliases} across all columns")
    click.echo(f"  Valid values: {total_valid} across all columns")
    click.echo(f"  Resolved at: {ctx.resolved_at}")


@main.command()
@click.argument("doc_paths", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--output", "-o", default=None, help="Output draft contract JSON path")
@click.option("--compare", default=None, type=click.Path(exists=True),
              help="Compare against existing contract and report gaps")
@click.option("--provider", "-p", default=None, help="Provider name for draft contract")
@click.option("--model", "-m", default="openai/gpt-4o", help="LLM model for draft contract")
@click.option("--strip", is_flag=True, help="Strip recommendation fields from output")
def analyze(
    doc_paths: tuple[str, ...],
    output: str | None,
    compare: str | None,
    provider: str | None,
    model: str,
    strip: bool,
) -> None:
    """Analyze documents and generate a draft contract skeleton.

    Accepts one or more PDF/DOCX files. Profiles each document's tables
    (columns, types, layouts, section labels) and produces a draft contract
    JSON with inline recommendations.

    Examples:

        contract-semantics analyze report.pdf

        contract-semantics analyze *.pdf -o draft.json

        contract-semantics analyze report.pdf --compare contracts/existing.json
    """
    from contract_semantics.analyze import (
        format_analysis_report,
        merge_profiles,
        profile_document,
    )
    from contract_semantics.recommend import (
        compare_contract,
        recommend_contract,
        strip_recommendations,
    )

    # Profile each document
    profiles = []
    for dp in doc_paths:
        click.echo(f"Profiling {dp} ...")
        prof = profile_document(dp)
        profiles.append(prof)
        click.echo(f"  {len(prof.tables)} tables, {prof.doc_format.upper()}")

    # Merge into multi-document profile
    multi = merge_profiles(profiles)

    # Print analysis report
    click.echo("")
    click.echo(format_analysis_report(multi))

    # Compare mode
    if compare:
        click.echo("")
        report = compare_contract(multi, compare)
        click.echo(report)

    # Generate draft contract
    draft = recommend_contract(multi, provider_name=provider, model=model)

    if strip:
        draft = strip_recommendations(draft)

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(draft, f, indent=2, ensure_ascii=False)
        click.echo(f"Draft contract written to {output}")
    else:
        click.echo("Draft contract JSON:")
        click.echo(json.dumps(draft, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
