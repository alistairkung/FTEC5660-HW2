import json
from typing import Any


class ClaimComparisonValidator:
    """Declare stable comparison references and validate complete, grounded findings."""

    JOB_FIELDS = ("company", "title", "seniority", "start_year", "end_year", "is_current")
    EDUCATION_FIELDS = ("school", "degree", "field", "graduation_year")

    def claim_catalog(self, claims: dict[str, Any]) -> dict[str, dict[str, Any]]:
        catalog = {}
        for field in ("name", "city"):
            value = claims[field]["value"] if claims[field]["value"] is not None else claims[field]["raw"]
            self._add_claim(catalog, field, value, claims[field]["raw"])
        for collection, fields in (("jobs", self.JOB_FIELDS), ("education", self.EDUCATION_FIELDS)):
            for index, entry in enumerate(claims[collection]):
                prefix = f"{collection}.{index}"
                for field in fields:
                    self._add_claim(catalog, f"{prefix}.{field}", entry[field], entry["raw"])
                # Preserve an explicit but unparsed date claim instead of dropping it.
                if entry["dates_raw"] and self._has_unparsed_dates(collection, entry):
                    self._add_claim(catalog, f"{prefix}.dates", entry["dates_raw"], entry["dates_raw"])
                if entry["raw"] and not any(key.startswith(prefix + ".") for key in catalog):
                    self._add_claim(catalog, f"{prefix}.record", entry["raw"], entry["raw"])
        for index, skill in enumerate(claims["skills"]):
            value = skill["name"] if skill["name"] is not None else skill["raw"]
            self._add_claim(catalog, f"skills.{index}.name", value, skill["raw"])
        return catalog

    def comparison_profile(self, profile: dict[str, Any]) -> dict[str, Any]:
        """Expose only permitted evidence, preserving missing fields as missing."""
        selected = {field: profile[field] for field in ("name", "city") if field in profile}
        for collection, fields in (
            ("experience", self.JOB_FIELDS),
            ("education", ("school", "degree", "field", "end_year")),
            ("skills", ("name",)),
        ):
            if collection not in profile:
                continue
            entries = profile[collection]
            selected[collection] = [
                {field: entry[field] for field in fields if field in entry} for entry in entries
            ] if isinstance(entries, list) else entries
        return selected

    def validate(self, findings: Any, claims: dict[str, Any], profile: dict[str, Any]) -> list[dict[str, Any]]:
        if not isinstance(findings, list):
            raise ValueError("Comparison findings must be a list")
        catalog = self.claim_catalog(claims)
        evidence = self._evidence_catalog(self.comparison_profile(profile))
        seen = set()
        for finding in findings:
            self._validate_finding(finding, catalog, evidence, seen)
        if seen != set(catalog):
            raise ValueError(f"Missing comparison claims: {sorted(set(catalog) - seen)}")
        return findings

    def _add_claim(self, catalog: dict, reference: str, value: Any, raw: str | None) -> None:
        if value is not None:
            catalog[reference] = {"value": value, "raw": raw}

    def _has_unparsed_dates(self, collection: str, entry: dict[str, Any]) -> bool:
        if collection == "education":
            return entry["graduation_year"] is None
        if entry["start_year"] is None:
            return True
        return entry["end_year"] is None and entry["is_current"] is not True

    def _evidence_catalog(self, profile: dict[str, Any]) -> dict[str, Any]:
        evidence = {}
        for field, value in profile.items():
            evidence[field] = value
            if isinstance(value, list):
                for index, entry in enumerate(value):
                    for key, scalar in entry.items():
                        evidence[f"{field}.{index}.{key}"] = scalar
        return evidence

    def _validate_finding(self, finding: Any, catalog: dict, evidence: dict, seen: set) -> None:
        required = {"claim_id", "cv_claim", "status", "evidence", "reason"}
        if not isinstance(finding, dict) or set(finding) != required:
            raise ValueError("Finding requires claim_id, cv_claim, status, evidence and reason")
        reference = finding["claim_id"]
        if not isinstance(reference, str) or reference not in catalog:
            raise ValueError("Finding refers to a nonexistent CV claim")
        if reference in seen:
            raise ValueError("Duplicate comparison claim reference")
        claim = finding["cv_claim"]
        if not isinstance(claim, dict) or set(claim) != {"value", "raw"}:
            raise ValueError("CV claim requires exactly value and raw")
        for field in ("value", "raw"):
            if type(claim[field]) is not type(catalog[reference][field]) or claim[field] != catalog[reference][field]:
                raise ValueError("Finding does not preserve the referenced CV claim")
        if finding["status"] not in ("MATCH", "MISMATCH", "UNKNOWN"):
            raise ValueError("Unsupported comparison status")
        if not isinstance(finding["reason"], str) or not finding["reason"].strip():
            raise ValueError("Comparison reason must be a nonempty string")
        citations = finding["evidence"]
        if not isinstance(citations, list):
            raise ValueError("Comparison evidence must be a list")
        if finding["status"] != "UNKNOWN" and not citations:
            raise ValueError("MATCH and MISMATCH require supplied evidence")
        for citation in citations:
            if not isinstance(citation, dict) or set(citation) != {"path", "value"}:
                raise ValueError("Evidence requires exactly path and value")
            path = citation["path"]
            if not isinstance(path, str) or path not in evidence:
                raise ValueError("Evidence refers to a nonexistent or excluded profile field")
            if evidence[path] is None:
                raise ValueError("Missing profile values are not supporting or contradictory evidence")
            if json.dumps(citation["value"], sort_keys=True) != json.dumps(evidence[path], sort_keys=True):
                raise ValueError("Evidence value differs from the supplied profile")
        seen.add(reference)
