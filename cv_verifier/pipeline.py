from typing import Any


class CandidatePipeline:
    """Compose extraction, validation, and retrieval without resolving identity."""

    def __init__(self, extractor: Any, validator: Any, retriever: Any):
        self.extractor = extractor
        self.validator = validator
        self.retriever = retriever

    async def ainvoke(self, cv: str) -> dict[str, Any]:
        result = {"claims": None, "search_hits": [], "profiles": [],
                  "identity": "unresolved", "retrieval_complete": False,
                  "retrieval_sampled": False, "errors": []}
        # Both extraction and validation remain the pre-search extraction stage
        # in the inspection result, preserving the existing failure contract.
        try:
            extracted = await self.extractor.ainvoke({"cv": cv})
            claims = self.validator.validate(extracted)
            result["claims"] = claims
        except Exception as error:
            result["errors"].append({"stage": "extraction", "message": str(error)})
            return result

        result.update(await self.retriever.retrieve(
            claims["name"]["value"], claims["city"]["value"],
        ))
        return result
