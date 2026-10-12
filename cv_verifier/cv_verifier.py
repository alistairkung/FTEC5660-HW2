from typing import Any

from .identity_decision_validator import IdentityDecisionValidator


class CVVerifier:
    """Compose identity, validated comparison and scoring for one CV."""

    def __init__(self, identity_pipeline: Any, comparator: Any, comparison_validator: Any, scorer: Any):
        self.identity_pipeline = identity_pipeline
        self.comparator = comparator
        self.comparison_validator = comparison_validator
        self.scorer = scorer

    async def ainvoke(self, cv: str) -> dict[str, Any]:
        try:
            result = dict(await self.identity_pipeline.ainvoke(cv))
        except Exception as error:
            return {"identity": "unresolved", "findings": [], "score": 0.5,
                    "errors": [{"stage": "identity", "message": str(error)}]}
        result["findings"] = []
        result["score"] = 0.5
        try:
            profile = self._selected_profile(result)
        except Exception as error:
            result["identity"] = "unresolved"
            result["selected_candidate_id"] = None
            result["identity_reason"] = "Identity selection could not be validated."
            result["errors"].append({"stage": "identity", "message": str(error)})
            return result
        if profile is None:
            return result

        try:
            proposed = await self.comparator.compare(result["claims"], profile)
            findings = self.comparison_validator.validate(proposed, result["claims"], profile)
            result["findings"] = findings
            result["score"] = self.scorer.score(findings, identity_resolved=True)
        except Exception as error:
            result["errors"].append({"stage": "comparison", "message": str(error)})
        return result

    def _selected_profile(self, result: dict[str, Any]) -> dict[str, Any] | None:
        if result["identity"] != "resolved":
            return None
        decision = {"candidate_id": result["selected_candidate_id"], "reason": result["identity_reason"]}
        IdentityDecisionValidator().validate(decision, result["profiles"])
        if decision["candidate_id"] is None:
            raise ValueError("Resolved identity must have a selected candidate ID")
        profiles = [profile for profile in result["profiles"] if profile["id"] == decision["candidate_id"]]
        if len(profiles) != 1:
            raise ValueError("Selected identity must correspond to exactly one retrieved profile")
        return profiles[0]
