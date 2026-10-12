from typing import Any


class CVScorer:
    """Aggregate validated findings; one mismatch outweighs any number of matches."""

    def score(self, findings: Any, *, identity_resolved: bool) -> float:
        if identity_resolved is not True:
            return 0.5
        statuses = self._finding_statuses(findings)
        if not statuses:
            return 0.5
        if "MISMATCH" in statuses:
            return 0.1
        if "UNKNOWN" in statuses:
            return 0.5
        return 0.9

    def _finding_statuses(self, findings: Any) -> list[str]:
        if not isinstance(findings, list) or not findings:
            return []
        statuses = []
        seen = set()
        for finding in findings:
            if not isinstance(finding, dict) or set(finding) != {
                "claim_id", "cv_claim", "status", "evidence", "reason",
            }:
                return []
            reference = finding["claim_id"]
            if not isinstance(reference, str) or not reference or reference in seen:
                return []
            if finding["status"] not in ("MATCH", "MISMATCH", "UNKNOWN"):
                return []
            if not isinstance(finding["cv_claim"], dict) or set(finding["cv_claim"]) != {"value", "raw"}:
                return []
            if not isinstance(finding["evidence"], list):
                return []
            if finding["status"] != "UNKNOWN" and not finding["evidence"]:
                return []
            if not isinstance(finding["reason"], str) or not finding["reason"].strip():
                return []
            seen.add(reference)
            statuses.append(finding["status"])
        return statuses
