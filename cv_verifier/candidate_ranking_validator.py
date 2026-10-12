from typing import Any


class CandidateRankingValidator:
    """Validate a bounded investigation shortlist without repairing it."""

    MAX_CANDIDATES = 5

    def validate(self, ranking: Any, hits: list[dict[str, Any]]) -> list[int]:
        if not isinstance(ranking, list):
            raise ValueError("Candidate ranking must be a list")
        if any(type(candidate_id) is not int for candidate_id in ranking):
            raise ValueError("Candidate ranking IDs must be integers")
        if len(ranking) > self.MAX_CANDIDATES:
            raise ValueError("Candidate shortlist must contain at most five IDs")
        if hits and not ranking:
            raise ValueError("Candidate shortlist must not be empty for nonempty search results")
        supplied_ids = [hit["id"] for hit in hits]
        if len(set(supplied_ids)) != len(supplied_ids):
            raise ValueError("Search hits contain duplicate candidate IDs")
        if len(set(ranking)) != len(ranking):
            raise ValueError("Candidate ranking contains duplicate IDs")
        if not set(ranking).issubset(supplied_ids):
            raise ValueError("Candidate shortlist contains unknown IDs")
        return ranking
