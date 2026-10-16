from typing import Any
from copy import deepcopy

from .identity_decision_validator import IdentityDecisionValidator


class CVVerifier:
    """Compose identity, validated comparison and scoring for one CV."""

    def __init__(self, identity_pipeline: Any, comparator: Any, comparison_validator: Any,
                 scorer: Any, reflector: Any):
        self.identity_pipeline = identity_pipeline
        self.comparator = comparator
        self.comparison_validator = comparison_validator
        self.scorer = scorer
        self.reflector = reflector

    async def ainvoke(self, cv: str) -> dict[str, Any]:
        try:
            result = dict(await self.identity_pipeline.ainvoke(cv))
        except Exception as error:
            return {"identity": "unresolved", "findings": [], "score": 0.5,
                    "errors": [{"stage": "identity", "message": str(error)}]}
        result["findings"] = []
        result["score"] = 0.5
        result["reflection"] = {"attempted": False, "corrections": []}
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
        if result["score"] == 0.9 or (result["score"] == 0.5 and result["findings"]
                and self.comparison_validator.reflectable_unknowns(result["findings"], result["claims"], profile)):
            await self._reflect_result(result, profile)
        return result

    async def _reflect_result(self, result: dict[str, Any], profile: dict[str, Any]) -> None:
        result["reflection"]["attempted"] = True
        baseline = result["findings"]
        baseline_score = result["score"]
        try:
            proposed = await self.reflector.reflect(
                deepcopy(result["claims"]), deepcopy(profile), deepcopy(baseline),
            )
            findings = self.comparison_validator.validate(proposed, result["claims"], profile)
            score = self.scorer.score(findings, identity_resolved=True)
            if score not in (0.1, baseline_score):
                raise ValueError("Reflection must preserve the baseline score or establish a discrepancy")
            originals = {finding["claim_id"]: finding for finding in baseline}
            corrections = [finding for finding in findings if finding != originals[finding["claim_id"]]]
            result["findings"] = findings
            result["score"] = score
            result["reflection"]["corrections"] = corrections
        except Exception as error:
            result["errors"].append({"stage": "reflection", "message": str(error)})

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
