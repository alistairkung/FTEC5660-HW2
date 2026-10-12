from typing import Any


class IdentityDecisionValidator:
    """Validate identity output shape and membership without assessing evidence."""

    def validate(self, decision: Any, profiles: list[dict[str, Any]]) -> dict[str, Any]:
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
        return decision
