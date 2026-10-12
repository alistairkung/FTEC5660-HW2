from typing import Any


class CandidatePipeline:
    """Compose extraction, validation, and retrieval without resolving identity."""

    def __init__(self, extractor: Any, validator: Any, retriever: Any,
                 prioritizer: Any, ranking_validator: Any):
        self.extractor = extractor
        self.validator = validator
        self.retriever = retriever
        self.prioritizer = prioritizer
        self.ranking_validator = ranking_validator

    async def ainvoke(self, cv: str) -> dict[str, Any]:
        # Both extraction and validation remain the pre-search extraction stage
        # in the inspection result, preserving the existing failure contract.
        try:
            extracted = await self.extractor.extract(cv)
            claims = self.validator.validate(extracted)
        except Exception as error:
            return self._build_result(errors=[{"stage": "extraction", "message": str(error)}])

        try:
            hits = await self.retriever.search(claims["name"]["value"], claims["city"]["value"])
        except Exception as error:
            return self._build_result(
                claims=claims, errors=[{"stage": "search", "message": str(error)}],
            )
        ranking, ranking_errors = await self._prioritize(claims, hits)

        hits_by_id = {hit["id"]: hit for hit in hits}
        selected_hits = [hits_by_id[candidate_id] for candidate_id in ranking[:self.retriever.MAX_PROFILES]]
        profiles, errors = await self.retriever.retrieve_profiles(selected_hits)
        return self._build_result(
            claims=claims, search_hits=hits, ranking=ranking, profiles=profiles,
            selected_hits=selected_hits, errors=ranking_errors + errors,
        )

    def _build_result(
        self, *, claims: dict[str, Any] | None = None,
        search_hits: list[dict[str, Any]] | None = None,
        ranking: list[int] | None = None,
        profiles: list[dict[str, Any]] | None = None,
        selected_hits: list[dict[str, Any]] | None = None,
        errors: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Assemble complete or partial output; None means retrieval was not reached."""
        hits = search_hits if search_hits is not None else []
        retrieved = profiles if profiles is not None else []
        return {
            "claims": claims,
            "search_hits": hits,
            "ranked_candidate_ids": ranking if ranking is not None else [],
            "profiles": retrieved,
            "identity": "unresolved",
            "retrieval_complete": selected_hits is not None and len(retrieved) == len(selected_hits),
            "retrieval_sampled": selected_hits is not None and len(hits) > len(selected_hits),
            "errors": errors if errors is not None else [],
        }

    async def _prioritize(self, claims: dict[str, Any], hits: list[dict[str, Any]]) -> tuple[list[int], list]:
        if not hits:
            return [], []
        try:
            proposed = await self.prioritizer.prioritize(claims, hits)
            return self.ranking_validator.validate(proposed, hits), []
        except Exception as error:
            return [hit["id"] for hit in hits], [{"stage": "prioritization", "message": str(error)}]
