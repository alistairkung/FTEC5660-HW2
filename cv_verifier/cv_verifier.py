from typing import Any
from copy import deepcopy
import asyncio

from .identity_decision_validator import IdentityDecisionValidator
from .title_equivalence_reviewer import TitleInput
from .skill_equivalence_reviewer import SkillInput


class CVVerifier:
    """Compose identity, validated comparison and scoring for one CV."""

    def __init__(self, identity_pipeline: Any, comparator: Any, comparison_validator: Any,
                 scorer: Any, reflector: Any, title_reviewer: Any = None, cv_timeout_seconds: float = 180,
                 skill_reviewer: Any = None):
        self.identity_pipeline = identity_pipeline
        self.comparator = comparator
        self.comparison_validator = comparison_validator
        self.scorer = scorer
        self.reflector = reflector
        self.title_reviewer = title_reviewer
        self.cv_timeout_seconds = cv_timeout_seconds
        self.skill_reviewer = skill_reviewer

    async def ainvoke(self, cv: str) -> dict[str, Any]:
        deadline = asyncio.get_running_loop().time() + self.cv_timeout_seconds
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
            if self.skill_reviewer is not None:
                await self._review_skills(result, profile, deadline)
            result["score"] = self.scorer.score(result['findings'], identity_resolved=True)
        except Exception as error:
            result["errors"].append({"stage": "comparison", "message": str(error)})
        if result["score"] == 0.9:
            await self._reflect_positive_result(result, profile)
        if result["score"] == 0.9 and self.title_reviewer is not None:
            await self._audit_positive_titles(result, profile, deadline)
        return result

    async def _review_skills(self, result: dict[str, Any], profile: dict[str, Any], deadline: float) -> None:
        """One optional batch, <=15s; reserve 60s reflection, 60s title audit and 1s finalization."""
        audit = result['skill_review'] = {'attempted': False, 'corrections': []}
        try:
            references = self.comparison_validator.eligible_skill_reviews(result['findings'], result['claims'], profile)
            audit['eligible_claims'] = references
            if not references:
                return
            remaining = min(15, deadline - asyncio.get_running_loop().time() - 121)
            if remaining <= 0:
                audit['skipped'] = 'Insufficient remaining CV deadline for optional skill review'
                return
            sources = []
            for reference in references:
                skill = result['claims']['skills'][int(reference.split('.')[1])]
                sources.append(SkillInput(skill['raw'], skill['name'], deepcopy(profile.get('skills'))))
            audit['attempted'] = True
            decisions = await asyncio.wait_for(self.skill_reviewer.review_many(sources), timeout=remaining)
            audit['decisions'] = decisions
            baseline = result['findings']
            findings = self.comparison_validator.apply_skill_reviews(
                references, decisions, baseline, result['claims'], profile)
            audit['corrections'] = [after for before, after in zip(baseline, findings) if before != after]
            result['findings'] = findings
        except Exception as error:
            result['errors'].append({'stage': 'skill_review', 'message': str(error) or type(error).__name__})

    async def _audit_positive_titles(self, result: dict[str, Any], profile: dict[str, Any], deadline: float) -> None:
        """Optional semantic audit: at most one title, after reflection, inside the CV deadline."""
        audit = result['title_review'] = {'attempted': False, 'corrections': []}
        try:
            candidates = self.comparison_validator.eligible_title_reviews(result['findings'], result['claims'], profile)
            audit['eligible_claims'] = [candidate['claim_id'] for candidate in candidates]
            if not candidates:
                return
            remaining = min(60, deadline - asyncio.get_running_loop().time() - 1)
            if remaining <= 0:
                audit['skipped'] = 'Insufficient remaining CV deadline for optional audit'
                return
            candidate = candidates[0]  # Stable catalog order; one-call budget, never a title hierarchy.
            job = result['claims']['jobs'][int(candidate['claim_id'].split('.')[1])]
            record = profile['experience'][candidate['record_index']]
            source = TitleInput(job['raw'], job['title'], job.get('seniority'),
                                record['title'], record.get('seniority'))
            audit.update(attempted=True, claim_id=candidate['claim_id'], record_index=candidate['record_index'])
            decision = await asyncio.wait_for(self.title_reviewer.review(source), timeout=remaining)
            audit['decision'] = decision
            if decision['status'] != 'MISMATCH':
                return
            correction = {'claim_id': candidate['claim_id'], 'status': 'MISMATCH',
                          'reason': decision['reason'], 'evidence_paths': candidate['evidence_paths']}
            findings = self.comparison_validator.apply_corrections(
                {'corrections': [correction]}, result['findings'], result['claims'], profile)
            score = self.scorer.score(findings, identity_resolved=True)
            if score != 0.1:
                raise ValueError('Title correction must establish a validated discrepancy')
            result['findings'], result['score'] = findings, score
            audit['corrections'] = [f for f in findings if f['claim_id'] == candidate['claim_id']]
        except Exception as error:
            result['errors'].append({'stage': 'title_review', 'message': str(error) or type(error).__name__})

    async def _reflect_positive_result(self, result: dict[str, Any], profile: dict[str, Any]) -> None:
        result["reflection"]["attempted"] = True
        baseline = result["findings"]
        try:
            proposed = await self.reflector.reflect(
                deepcopy(result["claims"]), deepcopy(profile), deepcopy(baseline),
            )
            findings = self.comparison_validator.validate(proposed, result["claims"], profile)
            score = self.scorer.score(findings, identity_resolved=True)
            if score not in (0.1, 0.9):
                raise ValueError("Reflection cannot downgrade a positive result to uncertainty")
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
