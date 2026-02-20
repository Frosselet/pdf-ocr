"""GeoNames adapter for resolving geographic concept URIs to labels and metadata.

Supports two modes:
- **Offline**: Parse a GeoNames country extract TSV (e.g. ``RU.txt``) into a
  lightweight in-memory lookup by geoname ID.
- **Online**: REST API at ``http://api.geonames.org/``.
"""

from __future__ import annotations

import csv
from pathlib import Path

import httpx

from contract_semantics.models import GeoEnrichment, GeoSearchResult, ResolvedAlias

GEONAMES_API = "http://api.geonames.org"

# GeoNames TSV column indices (allCountries.txt / country extracts)
_COL_ID = 0
_COL_NAME = 1
_COL_ASCIINAME = 2
_COL_ALTERNATENAMES = 3
_COL_LAT = 4
_COL_LNG = 5
_COL_FEATURE_CLASS = 6
_COL_FEATURE_CODE = 7
_COL_COUNTRY_CODE = 8
_COL_ADMIN1 = 10
_COL_POPULATION = 14

# Alternate names TSV columns
_ALT_ID = 0
_ALT_GEONAMEID = 1
_ALT_LANG = 2
_ALT_NAME = 3


def _parse_geonames_uri(uri: str) -> int:
    """Extract geoname ID from a GeoNames URI like ``https://sws.geonames.org/524894/``."""
    parts = uri.rstrip("/").rsplit("/", 1)
    return int(parts[-1])


class GeoNamesAdapter:
    """GeoNames geographic concept resolver.

    Parameters:
        features: Pre-loaded dict mapping geoname_id to feature dicts (offline).
        alternate_names: Pre-loaded dict mapping geoname_id to list of (lang, name) tuples.
        username: GeoNames API username for online mode.
    """

    def __init__(
        self,
        features: dict[int, dict] | None = None,
        alternate_names: dict[int, list[tuple[str, str]]] | None = None,
        username: str | None = None,
    ) -> None:
        self._features = features or {}
        self._alt_names = alternate_names or {}
        self._username = username

    @classmethod
    def from_file(
        cls,
        features_path: str | Path,
        alt_names_path: str | Path | None = None,
    ) -> GeoNamesAdapter:
        """Load a GeoNames country extract TSV into memory.

        Parameters:
            features_path: Path to the main features file (e.g. ``RU.txt``).
            alt_names_path: Optional path to alternate names file
                (e.g. ``alternateNamesV2.txt`` or a filtered subset).
        """
        features: dict[int, dict] = {}
        with open(features_path, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            for row in reader:
                if len(row) < 15:
                    continue
                gid = int(row[_COL_ID])
                pop_str = row[_COL_POPULATION] if len(row) > _COL_POPULATION else ""
                features[gid] = {
                    "geoname_id": gid,
                    "name": row[_COL_NAME],
                    "asciiname": row[_COL_ASCIINAME],
                    "alternatenames": row[_COL_ALTERNATENAMES],
                    "lat": float(row[_COL_LAT]),
                    "lng": float(row[_COL_LNG]),
                    "feature_class": row[_COL_FEATURE_CLASS],
                    "feature_code": row[_COL_FEATURE_CODE],
                    "country_code": row[_COL_COUNTRY_CODE],
                    "admin1_code": row[_COL_ADMIN1],
                    "population": int(pop_str) if pop_str else None,
                }

        alt_names: dict[int, list[tuple[str, str]]] = {}
        if alt_names_path:
            with open(alt_names_path, encoding="utf-8") as f:
                reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
                for row in reader:
                    if len(row) < 4:
                        continue
                    gid = int(row[_ALT_GEONAMEID])
                    lang = row[_ALT_LANG]
                    name = row[_ALT_NAME]
                    if lang and name:
                        alt_names.setdefault(gid, []).append((lang, name))

        return cls(features=features, alternate_names=alt_names)

    @classmethod
    def online(cls, username: str) -> GeoNamesAdapter:
        """Create an adapter that queries the GeoNames REST API."""
        return cls(username=username)

    # ── OntologyAdapter protocol ───────────────────────────────────────

    def resolve_concept(
        self,
        uri: str,
        *,
        languages: list[str] | None = None,
        label_types: list[str] | None = None,
    ) -> list[ResolvedAlias]:
        """Conform to :class:`OntologyAdapter` protocol by delegating to resolve_geoname."""
        return self.resolve_geoname(uri, languages=languages)

    # ── Resolve: aliases from alternate names ────────────────────────────

    def resolve_geoname(
        self,
        uri_or_id: str | int,
        *,
        languages: list[str] | None = None,
    ) -> list[ResolvedAlias]:
        """Return alternate names for a GeoNames feature in requested languages.

        Parameters:
            uri_or_id: GeoNames URI (``https://sws.geonames.org/XXX/``) or integer ID.
            languages: Language codes to retrieve (default ``["en"]``).

        Returns:
            List of :class:`ResolvedAlias` entries.
        """
        languages = languages or ["en"]
        gid = _parse_geonames_uri(uri_or_id) if isinstance(uri_or_id, str) else uri_or_id

        if self._features or self._alt_names:
            return self._resolve_offline(gid, languages)
        return self._resolve_online(gid, languages)

    def _resolve_offline(self, gid: int, languages: list[str]) -> list[ResolvedAlias]:
        results: list[ResolvedAlias] = []
        feature = self._features.get(gid)
        concept_label = feature["name"] if feature else str(gid)
        concept_uri = f"https://sws.geonames.org/{gid}/"

        # Primary name is always "en"
        if feature and "en" in languages:
            results.append(
                ResolvedAlias(
                    alias=feature["name"],
                    concept_uri=concept_uri,
                    concept_label=concept_label,
                    language="en",
                    label_type="name",
                )
            )
            # ASCII name if different
            if feature["asciiname"] and feature["asciiname"] != feature["name"]:
                results.append(
                    ResolvedAlias(
                        alias=feature["asciiname"],
                        concept_uri=concept_uri,
                        concept_label=concept_label,
                        language="en",
                        label_type="asciiname",
                    )
                )

        # Inline alternate names from the main features file
        if feature and feature.get("alternatenames"):
            for alt in feature["alternatenames"].split(","):
                alt = alt.strip()
                if alt and alt != feature["name"]:
                    # Inline alternates have no language tag; include if "en" requested
                    if "en" in languages:
                        results.append(
                            ResolvedAlias(
                                alias=alt,
                                concept_uri=concept_uri,
                                concept_label=concept_label,
                                language="en",
                                label_type="altLabel",
                            )
                        )

        # Language-tagged alternate names from separate file
        for lang, name in self._alt_names.get(gid, []):
            if lang in languages:
                results.append(
                    ResolvedAlias(
                        alias=name,
                        concept_uri=concept_uri,
                        concept_label=concept_label,
                        language=lang,
                        label_type="altLabel",
                    )
                )

        return results

    def _resolve_online(self, gid: int, languages: list[str]) -> list[ResolvedAlias]:
        if not self._username:
            raise ValueError("GeoNames API requires a username. Use GeoNamesAdapter.online(username=...)")

        concept_uri = f"https://sws.geonames.org/{gid}/"

        with httpx.Client(timeout=30) as client:
            resp = client.get(
                f"{GEONAMES_API}/getJSON",
                params={"geonameId": gid, "username": self._username},
            )
            resp.raise_for_status()
            data = resp.json()

        concept_label = data.get("name", str(gid))
        results: list[ResolvedAlias] = []

        if "en" in languages:
            results.append(
                ResolvedAlias(
                    alias=concept_label,
                    concept_uri=concept_uri,
                    concept_label=concept_label,
                    language="en",
                    label_type="name",
                )
            )

        # Fetch alternate names
        with httpx.Client(timeout=30) as client:
            resp = client.get(
                f"{GEONAMES_API}/searchJSON",
                params={
                    "geonameId": gid,
                    "username": self._username,
                    "style": "FULL",
                },
            )
            if resp.status_code == 200:
                full_data = resp.json()
                for geoname in full_data.get("geonames", []):
                    for alt in geoname.get("alternateNames", []):
                        lang = alt.get("lang", "")
                        name = alt.get("name", "")
                        if lang in languages and name:
                            results.append(
                                ResolvedAlias(
                                    alias=name,
                                    concept_uri=concept_uri,
                                    concept_label=concept_label,
                                    language=lang,
                                    label_type="altLabel",
                                )
                            )

        return results

    # ── Enrich: geographic metadata ──────────────────────────────────────

    def enrich_geoname(self, uri_or_id: str | int) -> GeoEnrichment:
        """Return geographic metadata for a GeoNames feature.

        Parameters:
            uri_or_id: GeoNames URI or integer ID.

        Returns:
            :class:`GeoEnrichment` with coordinates, admin codes, etc.
        """
        gid = _parse_geonames_uri(uri_or_id) if isinstance(uri_or_id, str) else uri_or_id

        if self._features:
            return self._enrich_offline(gid)
        return self._enrich_online(gid)

    def _enrich_offline(self, gid: int) -> GeoEnrichment:
        feat = self._features.get(gid)
        if feat is None:
            raise KeyError(f"GeoNames ID {gid} not found in local data")
        return GeoEnrichment(
            geoname_id=gid,
            name=feat["name"],
            lat=feat["lat"],
            lng=feat["lng"],
            country_code=feat["country_code"],
            admin1_code=feat["admin1_code"],
            feature_class=feat["feature_class"],
            feature_code=feat["feature_code"],
            population=feat["population"],
        )

    def _enrich_online(self, gid: int) -> GeoEnrichment:
        if not self._username:
            raise ValueError("GeoNames API requires a username")

        with httpx.Client(timeout=30) as client:
            resp = client.get(
                f"{GEONAMES_API}/getJSON",
                params={"geonameId": gid, "username": self._username},
            )
            resp.raise_for_status()
            data = resp.json()

        return GeoEnrichment(
            geoname_id=gid,
            name=data.get("name", ""),
            lat=float(data.get("lat", 0)),
            lng=float(data.get("lng", 0)),
            country_code=data.get("countryCode", ""),
            admin1_code=data.get("adminCode1", ""),
            feature_class=data.get("fcl", ""),
            feature_code=data.get("fcode", ""),
            population=data.get("population"),
        )

    # ── Search ───────────────────────────────────────────────────────────

    def search_geonames(
        self,
        query: str,
        *,
        country: str | None = None,
        feature_class: str | None = None,
        feature_code: str | None = None,
        max_results: int = 10,
    ) -> list[GeoSearchResult]:
        """Search GeoNames by name.

        Offline mode filters the in-memory feature dict by substring match.
        Online mode uses the GeoNames search API.

        Parameters:
            query: Name to search for.
            country: Restrict to country code (e.g. ``"RU"``).
            feature_class: Filter by feature class (e.g. ``"A"`` for admin).
            feature_code: Filter by feature code (e.g. ``"ADM1"``).
            max_results: Max results to return.

        Returns:
            List of :class:`GeoSearchResult` entries.
        """
        if self._features:
            return self._search_offline(query, country, feature_class, feature_code, max_results)
        return self._search_online(query, country, feature_class, feature_code, max_results)

    def _search_offline(
        self,
        query: str,
        country: str | None,
        feature_class: str | None,
        feature_code: str | None,
        max_results: int,
    ) -> list[GeoSearchResult]:
        query_lower = query.lower()
        results: list[GeoSearchResult] = []
        for feat in self._features.values():
            if country and feat["country_code"] != country:
                continue
            if feature_class and feat["feature_class"] != feature_class:
                continue
            if feature_code and feat["feature_code"] != feature_code:
                continue
            if query_lower in feat["name"].lower() or query_lower in feat["asciiname"].lower():
                results.append(
                    GeoSearchResult(
                        geoname_id=feat["geoname_id"],
                        name=feat["name"],
                        country_code=feat["country_code"],
                        feature_class=feat["feature_class"],
                        feature_code=feat["feature_code"],
                        admin1_code=feat["admin1_code"],
                        lat=feat["lat"],
                        lng=feat["lng"],
                    )
                )
                if len(results) >= max_results:
                    break
        return results

    def _search_online(
        self,
        query: str,
        country: str | None,
        feature_class: str | None,
        feature_code: str | None,
        max_results: int,
    ) -> list[GeoSearchResult]:
        if not self._username:
            raise ValueError("GeoNames API requires a username")

        params: dict[str, str | int] = {
            "q": query,
            "maxRows": max_results,
            "username": self._username,
        }
        if country:
            params["country"] = country
        if feature_class:
            params["featureClass"] = feature_class
        if feature_code:
            params["featureCode"] = feature_code

        with httpx.Client(timeout=30) as client:
            resp = client.get(f"{GEONAMES_API}/searchJSON", params=params)
            resp.raise_for_status()
            data = resp.json()

        results: list[GeoSearchResult] = []
        for g in data.get("geonames", []):
            results.append(
                GeoSearchResult(
                    geoname_id=g.get("geonameId", 0),
                    name=g.get("name", ""),
                    country_code=g.get("countryCode", ""),
                    feature_class=g.get("fcl", ""),
                    feature_code=g.get("fcode", ""),
                    admin1_code=g.get("adminCode1", ""),
                    lat=float(g.get("lat", 0)),
                    lng=float(g.get("lng", 0)),
                )
            )
        return results
