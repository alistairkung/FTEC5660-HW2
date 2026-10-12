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
        try:
            extracted = await self.extractor.extract(cv)
            claims = self.validator.validate(extracted)
        except Exception as error:
            return self._build_result(errors=[{"stage": "extraction", "message": str(error)}])

        hits, search_errors, search_succeeded = await self._search_candidates(claims)
        if not search_succeeded:
            return self._build_result(claims=claims, errors=search_errors,
                                      broader_search=bool(self._location_filters(claims)))
        ranking, ranking_errors = await self._prioritize(claims, hits)
        return await self._investigate(
            claims, hits, ranking, search_errors + ranking_errors,
            broader_search=bool(self._location_filters(claims)),
        )

    async def _search_candidates(
        self, claims: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
        filters = self._location_filters(claims) + [None]
        combined = []
        known_ids = set()
        errors = []
        succeeded = False
        for location in filters:
            try:
                hits = await self.retriever.search(claims["name"]["value"], location)
                combined.extend(self._new_hits(hits, known_ids))
                succeeded = True
            except Exception as error:
                errors.append({"stage": "search", "scope": "city_filtered" if location else "name_only",
                               "message": str(error)})
        return combined, errors, succeeded

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
        ranking: list[int], errors: list[dict[str, Any]], broader_search: bool,
    ) -> dict[str, Any]:
        hits = self._new_hits(hits, set())
        ranking = list(dict.fromkeys(ranking))
        profiles = []
        attempted_hits = []
        attempted_ids = set()
        decision = {"candidate_id": None, "reason": "No successfully retrieved profiles to examine."}
        stop_reason = "candidates_exhausted"

        while len(attempted_hits) < self.MAX_PROFILE_ATTEMPTS:
            batch = self._next_batch(hits, ranking, attempted_ids)
            if not batch:
                break

            remaining = self.MAX_PROFILE_ATTEMPTS - len(attempted_hits)
            batch = batch[:remaining]
            attempted_hits.extend(batch)
            attempted_ids.update(hit["id"] for hit in batch)
            retrieved, retrieval_errors = await self.retriever.retrieve_profiles(batch)
            profiles.extend(retrieved)
            errors.extend(retrieval_errors)
            decision, resolution_errors = await self._resolve_identity(claims, profiles)
            errors.extend(resolution_errors)
            if decision["candidate_id"] is not None:
                stop_reason = "identity_resolved"
                break
        else:
            stop_reason = "budget_exhausted"

        return self._build_result(
            claims=claims, search_hits=hits, ranking=ranking, profiles=profiles,
            selected_hits=attempted_hits, errors=errors, decision=decision,
            broader_search=broader_search, stop_reason=stop_reason,
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
            return self.identity_validator.validate(proposed, profiles), []
        except Exception as error:
            return {"candidate_id": None, "reason": "Identity decision failed; identity remains unresolved."}, [
                {"stage": "identity", "message": str(error)},
            ]

    def _build_result(
        self, *, claims: dict[str, Any] | None = None,
        search_hits: list[dict[str, Any]] | None = None,
        ranking: list[int] | None = None,
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
            "ranked_candidate_ids": ranking if ranking is not None else [],
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

    async def _prioritize(self, claims: dict[str, Any], hits: list[dict[str, Any]]) -> tuple[list[int], list]:
        if not hits:
            return [], []
        try:
            proposed = await self.prioritizer.prioritize(claims, hits)
            shortlist = self.ranking_validator.validate(proposed, hits)
            selected_ids = set(shortlist)
            remaining_ids = [hit["id"] for hit in hits if hit["id"] not in selected_ids]
            return shortlist + remaining_ids, []
        except Exception as error:
            return [hit["id"] for hit in hits], [{"stage": "prioritization", "message": str(error)}]
