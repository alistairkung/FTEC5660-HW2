import re
from typing import Any


class IdentityDecisionValidator:
    """Validate membership and reject broad matches with wholly unsupported career histories."""

    def validate(self, decision: Any, profiles: list[dict[str, Any]],
                 claims: dict[str, Any] | None = None) -> dict[str, Any]:
        if not isinstance(decision, dict):
            raise ValueError("Identity decision must be an object")
        if set(decision) != {"candidate_id", "reason"}:
            raise ValueError("Identity decision requires exactly candidate_id and reason")
        candidate_id = decision["candidate_id"]
        if candidate_id is not None and type(candidate_id) is not int:
            raise ValueError("Selected candidate ID must be an integer or null")
        if not isinstance(decision["reason"], str) or not decision["reason"].strip():
            raise ValueError("Identity reason must be a nonempty string")
        if candidate_id is not None and candidate_id not in {profile["id"] for profile in profiles}:
            raise ValueError("Selected candidate ID was not successfully retrieved")
        if candidate_id is not None and claims is not None:
            selected = next(profile for profile in profiles if profile["id"] == candidate_id)
            self._validate_career_support(claims, selected)
        return decision

    def _validate_career_support(self, claims: dict[str, Any], profile: dict[str, Any]) -> None:
        jobs, education = claims["jobs"], claims["education"]
        experience, schools = profile.get("experience"), profile.get("education")
        # This narrow rejection needs two available career histories. Sparse or
        # unavailable evidence remains a semantic decision for the existing resolver.
        if not (jobs and education and isinstance(experience, list) and experience
                and isinstance(schools, list) and schools):
            return
        if (any(not job.get("company") for job in jobs)
                or any(not entry.get("school") for entry in education)
                or any(not record.get("company") for record in experience)
                or any(not record.get("school") for record in schools)):
            return
        if any(self._related(job.get("company"), record.get("company"))
               for job in jobs for record in experience):
            return
        if any(self._related(entry.get("school"), record.get("school"))
               for entry in education for record in schools):
            return
        # A fabricated employer/school can still leave a distinctive dated record.
        for job in jobs:
            for record in experience:
                if (job.get("start_year") is not None and job.get("end_year") is not None
                        and job["start_year"] == record.get("start_year")
                        and job["end_year"] == record.get("end_year")
                        and self._related(job.get("title"), record.get("title"))):
                    return
        for entry in education:
            for record in schools:
                if (entry.get("graduation_year") is not None
                        and entry["graduation_year"] == record.get("end_year")
                        and (self._related(entry.get("degree"), record.get("degree"))
                             or self._related(entry.get("field"), record.get("field")))):
                    return
        raise ValueError("Selected identity lacks employment or education support beyond broad personal matches")

    def _related(self, left: Any, right: Any) -> bool:
        if not isinstance(left, str) or not isinstance(right, str):
            return False
        a, b = re.findall(r"\w+", left.casefold()), re.findall(r"\w+", right.casefold())
        if a and a == b:
            return True
        generic = {"the", "of", "and", "company", "corporation", "inc", "ltd", "limited",
                   "university", "college", "school", "institute", "group"}
        return bool({word for word in a if len(word) > 2 and word not in generic}
                    & {word for word in b if len(word) > 2 and word not in generic})
