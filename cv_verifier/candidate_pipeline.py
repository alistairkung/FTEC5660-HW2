import re
import logging
from typing import Any


class CandidatePipeline:
    """Investigate ranked candidates within a fixed budget; never score CV claims."""

    MAX_PROFILE_ATTEMPTS = 8

    def __init__(self, extractor: Any, validator: Any, retriever: Any,
                 prioritizer: Any, ranking_validator: Any,
                 identity_resolver: Any, identity_validator: Any):
        self.extractor = extractor
        self.validator = validator
        self.retriever = retriever
        self.prioritizer = prioritizer
        self.ranking_validator = ranking_validator
        self.identity_resolver = identity_resolver
        self.identity_validator = identity_validator

    async def ainvoke(self, cv: str) -> dict[str, Any]:
        # Both extraction and validation remain the pre-search extraction stage
        # in the inspection result, preserving the existing failure contract.
        for attempt in range(2):
            try:
                extracted = await self.extractor.extract(cv)
                claims = self.validator.validate(extracted)
                break
            except (ValueError, TypeError) as error:
                logging.getLogger(__name__).warning("Extraction contract attempt %s failed: %s", attempt + 1, error)
                if attempt == 1:
                    return self._build_result(errors=[{"stage": "extraction", "message": str(error)}])
            except Exception as error:
                return self._build_result(errors=[{"stage": "extraction", "message": str(error)}])

        hits, search_errors, search_succeeded, searches = await self._search_candidates(claims)
        if not search_succeeded:
            return self._build_result(claims=claims, errors=search_errors,
                                      discovery_searches=searches,
                                      broader_search=bool(self._location_filters(claims)))
        ranking, shortlist, ranking_errors = await self._prioritize(claims, hits)
        return await self._investigate(
            claims, hits, ranking, shortlist, search_errors + ranking_errors,
            discovery_searches=searches,
            broader_search=bool(self._location_filters(claims)),
        )

    async def _search_candidates(
        self, claims: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool, list[dict[str, Any]]]:
        filters = self._location_filters(claims) + [None]
        combined = []
        known_ids = set()
        errors = []
        searches = []
        succeeded = False
        for location in filters:
            scope = "city_filtered" if location else "name_only"
            hits, error, search = await self._search_once(claims["name"]["value"], location, scope)
            new_hits = self._new_hits(hits, known_ids)
            combined.extend(new_hits)
            search["new_candidate_count"] = len(new_hits)
            searches.append(search)
            if error is not None:
                errors.append(error)
            else:
                succeeded = True

        supplementary = self._supplementary_query(claims, combined, searches)
        if supplementary is not None:
            location, industry = supplementary
            hits, error, search = await self._search_once(
                claims["name"]["value"], location, "industry_filtered", industry,
            )
            new_hits = self._new_hits(hits, known_ids)
            combined.extend(new_hits)
            search["new_candidate_count"] = len(new_hits)
            searches.append(search)
            if error is not None:
                errors.append(error)
        return combined, errors, succeeded, searches

    async def _search_once(
        self, name: str, location: str | None, scope: str, industry: str | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None, dict[str, Any]]:
        search = {"scope": scope, "location": location, "industry": industry, "returned_count": None}
        try:
            if industry is None:
                hits = await self.retriever.search(name, location)
            else:
                hits = await self.retriever.search(name, location, industry=industry)
            search["returned_count"] = len(hits)
            return hits, None, search
        except Exception as error:
            return [], {"stage": "search", "scope": scope, "message": str(error)}, search

    def _supplementary_query(
        self, claims: dict[str, Any], hits: list[dict[str, Any]], searches: list[dict[str, Any]],
    ) -> tuple[str | None, str] | None:
        saturated = [search for search in searches
                     if search["returned_count"] is not None
                     and search["returned_count"] >= self.retriever.MAX_SEARCH_RESULTS]
        if not saturated:
            return None
        industry = self._industry_hint(claims, hits)
        if industry is None:
            return None
        locations = [search["location"] for search in saturated if search["location"] is not None]
        # An uncapped city pool is already exhausted; diversify the capped name pool instead.
        location = locations[0] if locations else None
        return location, industry

    def _industry_hint(self, claims: dict[str, Any], hits: list[dict[str, Any]]) -> str | None:
        """Use observed labels with explicit CV support; hints never exclude original hits."""
        categories = [
            [job[field] for job in claims["jobs"] for field in ("company", "title")],
            [education[field] for education in claims["education"] for field in ("school", "degree", "field")],
            [skill["name"] for skill in claims["skills"]],
        ]
        wording = [[self._words(value) for value in category if isinstance(value, str)]
                   for category in categories]
        best_hint = None
        best_support = 0
        seen = set()
        for hit in hits:
            industry = hit.get("industry")
            if not isinstance(industry, str):
                continue
            label = self._words(industry)
            if not label.strip() or label in seen:
                continue
            seen.add(label)
            support = sum(any(label in value for value in category) for category in wording)
            if support > best_support:
                best_hint, best_support = industry.strip(), support
        return best_hint

    def _words(self, value: str) -> str:
        """Bound normalized phrases with spaces to avoid substring matches such as AI/AIA."""
        return " " + " ".join(re.findall(r"\w+", value.casefold())) + " "

    def _location_filters(self, claims: dict[str, Any]) -> list[str]:
        """Use current/unspecified cities as discovery hints; always search the name too."""
        locations = []
        seen = set()
        for candidate in claims["city"]["candidates"]:
            if candidate["role"] not in ("current", "unspecified"):
                continue
            city = candidate["city"]
            key = city.strip().casefold()
            if key not in seen:
                locations.append(city)
                seen.add(key)
        return locations

    async def _investigate(
        self, claims: dict[str, Any], hits: list[dict[str, Any]],
        ranking: list[int], shortlist: list[int], errors: list[dict[str, Any]], broader_search: bool,
        discovery_searches: list[dict[str, Any]],
    ) -> dict[str, Any]:
        hits = self._new_hits(hits, set())
        ranking = list(dict.fromkeys(ranking))
        profiles = []
        attempted_hits = []
        attempted_ids = set()
        decision = {"candidate_id": None, "reason": "No successfully retrieved profiles to examine."}
        stop_reason = "candidates_exhausted"

        while len(attempted_hits) < self.MAX_PROFILE_ATTEMPTS:
            pending_shortlist = [candidate_id for candidate_id in shortlist if candidate_id not in attempted_ids]
            batch = self._next_batch(hits, pending_shortlist or ranking, attempted_ids)
            if not batch:
                break

            remaining = self.MAX_PROFILE_ATTEMPTS - len(attempted_hits)
            batch = batch[:remaining]
            attempted_hits.extend(batch)
            attempted_ids.update(hit["id"] for hit in batch)
            retrieved, retrieval_errors = await self.retriever.retrieve_profiles(batch)
            profiles.extend(retrieved)
            errors.extend(retrieval_errors)
            if not set(shortlist).issubset(attempted_ids):
                continue

            decision, resolution_errors = await self._resolve_identity(claims, profiles)
            errors.extend(resolution_errors)
            if decision["candidate_id"] is not None:
                stop_reason = "identity_resolved"
                break
        else:
            stop_reason = "budget_exhausted"

        return self._build_result(
            claims=claims, search_hits=hits, ranking=ranking, shortlist=shortlist, profiles=profiles,
            selected_hits=attempted_hits, errors=errors, decision=decision,
            broader_search=broader_search, stop_reason=stop_reason,
            discovery_searches=discovery_searches,
        )

    def _next_batch(self, hits: list[dict[str, Any]], ranking: list[int],
                    attempted_ids: set[int]) -> list[dict[str, Any]]:
        hits_by_id = {hit["id"]: hit for hit in hits}
        remaining_ids = [candidate_id for candidate_id in ranking if candidate_id not in attempted_ids]
        return [hits_by_id[candidate_id] for candidate_id in remaining_ids[:self.retriever.MAX_PROFILES]]

    def _new_hits(self, hits: list[dict[str, Any]], known_ids: set[int]) -> list[dict[str, Any]]:
        unique_hits = []
        for hit in hits:
            if hit["id"] not in known_ids:
                unique_hits.append(hit)
                known_ids.add(hit["id"])
        return unique_hits

    async def _resolve_identity(
        self, claims: dict[str, Any], profiles: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        if not profiles:
            return {"candidate_id": None, "reason": "No successfully retrieved profiles to examine."}, []
        try:
            proposed = await self.identity_resolver.resolve(claims, list(profiles))
            return self.identity_validator.validate(proposed, profiles, claims), []
        except Exception as error:
            return {"candidate_id": None, "reason": "Identity decision failed; identity remains unresolved."}, [
                {"stage": "identity", "message": str(error)},
            ]

    def _build_result(
        self, *, claims: dict[str, Any] | None = None,
        search_hits: list[dict[str, Any]] | None = None,
        ranking: list[int] | None = None,
        shortlist: list[int] | None = None,
        discovery_searches: list[dict[str, Any]] | None = None,
        profiles: list[dict[str, Any]] | None = None,
        selected_hits: list[dict[str, Any]] | None = None,
        errors: list[dict[str, Any]] | None = None,
        decision: dict[str, Any] | None = None,
        broader_search: bool = False, stop_reason: str | None = None,
    ) -> dict[str, Any]:
        """Assemble complete or partial output; None means retrieval was not reached."""
        hits = search_hits if search_hits is not None else []
        retrieved = profiles if profiles is not None else []
        identity = decision or {"candidate_id": None, "reason": "Identity investigation has not started."}
        return {
            "claims": claims,
            "search_hits": hits,
            "discovery_searches": discovery_searches if discovery_searches is not None else [],
            "ranked_candidate_ids": ranking if ranking is not None else [],
            "shortlisted_candidate_ids": shortlist if shortlist is not None else [],
            "shortlist_covered": shortlist is not None and selected_hits is not None
                and set(shortlist).issubset({hit["id"] for hit in selected_hits}),
            "profiles": retrieved,
            "identity": "resolved" if identity["candidate_id"] is not None else "unresolved",
            "selected_candidate_id": identity["candidate_id"],
            "identity_reason": identity["reason"],
            "attempted_candidate_ids": [hit["id"] for hit in selected_hits] if selected_hits is not None else [],
            "broader_search_performed": broader_search,
            "investigation_stop_reason": stop_reason,
            "retrieval_complete": selected_hits is not None and len(retrieved) == len(selected_hits),
            "retrieval_sampled": selected_hits is not None and len(hits) > len(selected_hits),
            "errors": errors if errors is not None else [],
        }

    async def _prioritize(self, claims: dict[str, Any], hits: list[dict[str, Any]]) -> tuple[list[int], list[int], list]:
        if not hits:
            return [], [], []
        try:
            proposed = await self.prioritizer.prioritize(claims, hits)
            shortlist = self.ranking_validator.validate(proposed, hits)
            selected_ids = set(shortlist)
            remaining_ids = [hit["id"] for hit in self._fallback_order(claims, hits) if hit["id"] not in selected_ids]
            return shortlist + remaining_ids, shortlist, []
        except Exception as error:
            return [hit["id"] for hit in self._fallback_order(claims, hits)], [], [{"stage": "prioritization", "message": str(error)}]

    def _fallback_order(self, claims: dict[str, Any], hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Rank returned metadata as search hints only; stable ties preserve coverage."""
        categories = [
            [job[field] for job in claims["jobs"] for field in ("company", "title")],
            [entry[field] for entry in claims["education"] for field in ("school", "degree", "field")],
            [skill["name"] for skill in claims["skills"]],
        ]
        wording = [[self._words(value) for value in group if isinstance(value, str)] for group in categories]
        locations = [self._words(city) for city in self._location_filters(claims)]
        def support(hit):
            industry = self._words(hit["industry"]) if isinstance(hit.get("industry"), str) else ""
            industry_support = sum(any(industry in value for value in group) for group in wording) if industry.strip() else 0
            location = self._words(hit["location"]) if isinstance(hit.get("location"), str) else ""
            location_support = any(city in location for city in locations)
            return industry_support, location_support
        return sorted(hits, key=support, reverse=True)
