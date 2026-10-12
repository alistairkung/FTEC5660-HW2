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
        result = {"claims": None, "search_hits": [], "ranked_candidate_ids": [], "profiles": [],
                  "identity": "unresolved", "retrieval_complete": False,
                  "retrieval_sampled": False, "errors": []}
        # Both extraction and validation remain the pre-search extraction stage
        # in the inspection result, preserving the existing failure contract.
        try:
            extracted = await self.extractor.extract(cv)
            claims = self.validator.validate(extracted)
            result["claims"] = claims
        except Exception as error:
            result["errors"].append({"stage": "extraction", "message": str(error)})
            return result

        try:
            hits = await self.retriever.search(claims["name"]["value"], claims["city"]["value"])
        except Exception as error:
            result["errors"].append({"stage": "search", "message": str(error)})
            return result
        result["search_hits"] = hits
        ranking, ranking_errors = await self._prioritize(claims, hits)
        result["ranked_candidate_ids"] = ranking
        result["errors"].extend(ranking_errors)

        hits_by_id = {hit["id"]: hit for hit in hits}
        selected_hits = [hits_by_id[candidate_id] for candidate_id in ranking[:self.retriever.MAX_PROFILES]]
        profiles, errors = await self.retriever.retrieve_profiles(selected_hits)
        result["profiles"] = profiles
        result["errors"].extend(errors)
        result["retrieval_complete"] = len(profiles) == len(selected_hits)
        result["retrieval_sampled"] = len(hits) > len(selected_hits)
        return result

    async def _prioritize(self, claims: dict[str, Any], hits: list[dict[str, Any]]) -> tuple[list[int], list]:
        if not hits:
            return [], []
        try:
            proposed = await self.prioritizer.prioritize(claims, hits)
            return self.ranking_validator.validate(proposed, hits), []
        except Exception as error:
            return [hit["id"] for hit in hits], [{"stage": "prioritization", "message": str(error)}]
