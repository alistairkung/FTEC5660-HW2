from typing import Any


class CandidateRankingValidator:
    """Accept only an exact permutation of supplied IDs, without repairing it."""

    def validate(self, ranking: Any, hits: list[dict[str, Any]]) -> list[int]:
        if not isinstance(ranking, list):
            raise ValueError("Candidate ranking must be a list")
        if any(type(candidate_id) is not int for candidate_id in ranking):
            raise ValueError("Candidate ranking IDs must be integers")
        supplied_ids = [hit["id"] for hit in hits]
        if len(set(supplied_ids)) != len(supplied_ids):
            raise ValueError("Search hits contain duplicate candidate IDs")
        if len(set(ranking)) != len(ranking):
            raise ValueError("Candidate ranking contains duplicate IDs")
        if len(ranking) != len(supplied_ids) or set(ranking) != set(supplied_ids):
            raise ValueError("Candidate ranking must contain every supplied ID exactly once")
        return ranking
