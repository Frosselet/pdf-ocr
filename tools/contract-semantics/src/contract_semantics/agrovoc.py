"""AGROVOC SKOS adapter for resolving agricultural concept URIs to labels.

Supports two modes:
- **Offline**: Parse a local AGROVOC N-Triples dump into an rdflib Graph
  (with optional pickle caching for fast reload).
- **Online**: SPARQL queries to ``https://agrovoc.fao.org/sparql`` via httpx.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import httpx
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import SKOS

from contract_semantics.models import ResolvedAlias

try:
    from rapidfuzz import fuzz, process as rfprocess
except ImportError:  # pragma: no cover
    fuzz = None  # type: ignore[assignment]
    rfprocess = None  # type: ignore[assignment]

AGROVOC_SPARQL = "https://agrovoc.fao.org/sparql"

_SKOSXL = Namespace("http://www.w3.org/2008/05/skos-xl#")


class AgrovocAdapter:
    """AGROVOC SKOS concept resolver.

    Parameters:
        graph: Pre-loaded rdflib Graph (offline mode).  When *None*,
            queries are sent to the public SPARQL endpoint (online mode).
    """

    def __init__(self, graph: Graph | None = None) -> None:
        self._graph = graph

    # ── Factory helpers ──────────────────────────────────────────────────

    @classmethod
    def from_file(cls, path: str | Path) -> AgrovocAdapter:
        """Load an AGROVOC dump (N-Triples, Turtle, etc.) into an rdflib Graph.

        If a ``.pkl`` cache exists next to *path*, it is loaded instead.
        Otherwise the raw file is parsed and a pickle cache is written for
        next time.
        """
        path = Path(path)
        pkl = path.with_suffix(".pkl")
        if pkl.exists() and pkl.stat().st_mtime >= path.stat().st_mtime:
            with open(pkl, "rb") as f:
                g = pickle.load(f)
        else:
            g = Graph()
            fmt = "nt" if path.suffix in (".nt", ".ntriples") else None
            g.parse(str(path), format=fmt)
            with open(pkl, "wb") as f:
                pickle.dump(g, f, protocol=pickle.HIGHEST_PROTOCOL)
        return cls(graph=g)

    @classmethod
    def online(cls) -> AgrovocAdapter:
        """Create an adapter that queries the public AGROVOC SPARQL endpoint."""
        return cls(graph=None)

    # ── Core resolution ──────────────────────────────────────────────────

    def resolve_concept(
        self,
        uri: str,
        *,
        languages: list[str] | None = None,
        label_types: list[str] | None = None,
        include_narrower: bool = False,
        narrower_depth: int = 1,
    ) -> list[ResolvedAlias]:
        """Resolve a concept URI to multilingual labels.

        Parameters:
            uri: AGROVOC concept URI (e.g. ``http://aims.fao.org/aos/agrovoc/c_8373``).
            languages: Language codes to retrieve (default ``["en"]``).
            label_types: ``"prefLabel"`` and/or ``"altLabel"`` (default both).
            include_narrower: Whether to traverse ``skos:narrower`` children.
            narrower_depth: Maximum depth for narrower traversal.

        Returns:
            List of :class:`ResolvedAlias` with one entry per label found.
        """
        languages = languages or ["en"]
        label_types = label_types or ["prefLabel", "altLabel"]

        if self._graph is not None:
            return self._resolve_offline(
                uri, languages, label_types, include_narrower, narrower_depth
            )
        return self._resolve_online(
            uri, languages, label_types, include_narrower, narrower_depth
        )

    # ── Reverse lookup ─────────────────────────────────────────────────

    def lookup_by_label(
        self,
        label: str,
        *,
        language: str = "en",
    ) -> str:
        """Reverse lookup: concept label → concept URI.

        Searches for the label among ``skos:prefLabel`` entries (case-insensitive).

        Parameters:
            label: Human-readable concept label (e.g. ``"wheat"``).
            language: Language code to search in (default ``"en"``).

        Returns:
            The concept URI string.

        Raises:
            KeyError: If label not found (with fuzzy-match suggestions).
            ValueError: If label is ambiguous (multiple distinct concept URIs match).
        """
        if self._graph is not None:
            return self._lookup_offline(label, language)
        return self._lookup_online(label, language)

    def _lookup_offline(self, label: str, language: str) -> str:
        g = self._graph
        assert g is not None
        label_lower = label.lower()

        # Search for matching prefLabels (case-insensitive)
        matching_uris: set[str] = set()
        for subj, _p, obj in g.triples((None, SKOS.prefLabel, None)):
            if not isinstance(obj, Literal):
                continue
            if (obj.language or "en") != language:
                continue
            if str(obj).lower() == label_lower:
                matching_uris.add(str(subj))

        if len(matching_uris) == 1:
            return matching_uris.pop()

        if len(matching_uris) > 1:
            raise ValueError(
                f"Ambiguous label {label!r} (language={language!r}) matches "
                f"{len(matching_uris)} concepts: {sorted(matching_uris)}"
            )

        # No match — collect all prefLabels in this language for suggestions
        all_labels: list[str] = []
        for _s, _p, obj in g.triples((None, SKOS.prefLabel, None)):
            if isinstance(obj, Literal) and (obj.language or "en") == language:
                all_labels.append(str(obj))

        suggestions = self._fuzzy_suggestions(label, all_labels)
        msg = f"Label {label!r} not found in AGROVOC (language={language!r})."
        if suggestions:
            msg += f" Similar labels: {suggestions}"
        raise KeyError(msg)

    def _lookup_online(self, label: str, language: str) -> str:
        query = f"""
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
        SELECT DISTINCT ?concept WHERE {{
            ?concept skos:prefLabel ?label .
            FILTER(LCASE(STR(?label)) = LCASE("{label}"))
            FILTER(lang(?label) = "{language}")
        }}
        """
        uris = self._sparql_select(query)

        if len(uris) == 1:
            return uris[0]

        if len(uris) > 1:
            raise ValueError(
                f"Ambiguous label {label!r} (language={language!r}) matches "
                f"{len(uris)} concepts: {sorted(uris)}"
            )

        raise KeyError(
            f"Label {label!r} not found in AGROVOC (language={language!r})."
        )

    @staticmethod
    def _fuzzy_suggestions(query: str, candidates: list[str], limit: int = 5) -> list[str]:
        """Return fuzzy-matched suggestions from candidates."""
        if not candidates or rfprocess is None:
            return []
        matches = rfprocess.extract(query, candidates, scorer=fuzz.ratio, limit=limit)
        return [m[0] for m in matches if m[1] > 50]

    def _resolve_offline(
        self,
        uri: str,
        languages: list[str],
        label_types: list[str],
        include_narrower: bool,
        narrower_depth: int,
    ) -> list[ResolvedAlias]:
        g = self._graph
        assert g is not None
        concept = URIRef(uri)
        results: list[ResolvedAlias] = []

        # Collect labels for this concept
        results.extend(self._labels_for(concept, uri, languages, label_types))

        # Optionally traverse narrower concepts
        if include_narrower and narrower_depth > 0:
            for _s, _p, child in g.triples((concept, SKOS.narrower, None)):
                child_uri = str(child)
                results.extend(self._labels_for(child, child_uri, languages, label_types))
                if narrower_depth > 1:
                    results.extend(
                        self._resolve_offline(
                            child_uri,
                            languages,
                            label_types,
                            include_narrower=True,
                            narrower_depth=narrower_depth - 1,
                        )
                    )

        return results

    def _labels_for(
        self,
        concept: URIRef,
        concept_uri: str,
        languages: list[str],
        label_types: list[str],
    ) -> list[ResolvedAlias]:
        g = self._graph
        assert g is not None
        results: list[ResolvedAlias] = []

        predicates = {
            "prefLabel": SKOS.prefLabel,
            "altLabel": SKOS.altLabel,
        }

        # Determine the English prefLabel for concept_label fallback
        concept_label = concept_uri.rsplit("/", 1)[-1]
        for _s, _p, obj in g.triples((concept, SKOS.prefLabel, None)):
            if isinstance(obj, Literal) and obj.language == "en":
                concept_label = str(obj)
                break

        for lt in label_types:
            pred = predicates.get(lt)
            if pred is None:
                continue
            for _s, _p, obj in g.triples((concept, pred, None)):
                if not isinstance(obj, Literal):
                    continue
                lang = obj.language or "en"
                if lang in languages:
                    results.append(
                        ResolvedAlias(
                            alias=str(obj),
                            concept_uri=concept_uri,
                            concept_label=concept_label,
                            language=lang,
                            label_type=lt,
                        )
                    )
        return results

    def _resolve_online(
        self,
        uri: str,
        languages: list[str],
        label_types: list[str],
        include_narrower: bool,
        narrower_depth: int,
    ) -> list[ResolvedAlias]:
        lang_filter = " || ".join(f'lang(?label) = "{lg}"' for lg in languages)
        label_preds = []
        if "prefLabel" in label_types:
            label_preds.append("skos:prefLabel")
        if "altLabel" in label_types:
            label_preds.append("skos:altLabel")

        if not label_preds:
            return []

        pred_union = " UNION ".join(
            f'{{ <{uri}> {p} ?label . BIND("{p.split(":")[-1]}" AS ?lt) }}'
            for p in label_preds
        )

        query = f"""
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
        SELECT ?label ?lt WHERE {{
            {{ {pred_union} }}
            FILTER({lang_filter})
        }}
        """

        results = self._sparql_query(query, uri)

        if include_narrower and narrower_depth > 0:
            narrower_query = f"""
            PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
            SELECT ?child WHERE {{
                <{uri}> skos:narrower ?child .
            }}
            """
            children = self._sparql_select(narrower_query)
            for child_uri in children:
                results.extend(
                    self._resolve_online(
                        child_uri,
                        languages,
                        label_types,
                        include_narrower=True,
                        narrower_depth=narrower_depth - 1,
                    )
                )

        return results

    def _sparql_query(self, query: str, concept_uri: str) -> list[ResolvedAlias]:
        """Execute a SPARQL SELECT and return ResolvedAlias list."""
        with httpx.Client(timeout=30) as client:
            resp = client.get(
                AGROVOC_SPARQL,
                params={"query": query},
                headers={"Accept": "application/sparql-results+json"},
            )
            resp.raise_for_status()
            data = resp.json()

        results: list[ResolvedAlias] = []
        for binding in data.get("results", {}).get("bindings", []):
            label_val = binding["label"]["value"]
            lang = binding["label"].get("xml:lang", "en")
            lt = binding["lt"]["value"]
            results.append(
                ResolvedAlias(
                    alias=label_val,
                    concept_uri=concept_uri,
                    concept_label=concept_uri.rsplit("/", 1)[-1],
                    language=lang,
                    label_type=lt,
                )
            )
        return results

    def _sparql_select(self, query: str) -> list[str]:
        """Execute a SPARQL SELECT and return a flat list of first-column values."""
        with httpx.Client(timeout=30) as client:
            resp = client.get(
                AGROVOC_SPARQL,
                params={"query": query},
                headers={"Accept": "application/sparql-results+json"},
            )
            resp.raise_for_status()
            data = resp.json()

        bindings = data.get("results", {}).get("bindings", [])
        if not bindings:
            return []
        key = list(bindings[0].keys())[0]
        return [b[key]["value"] for b in bindings]
