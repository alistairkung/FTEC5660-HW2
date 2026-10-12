import json
import re
from copy import deepcopy
from typing import Any


class ClaimComparisonValidator:
    """Declare stable comparison references and validate complete, grounded findings."""

    JOB_FIELDS = ("company", "title", "seniority", "start_year", "end_year", "is_current")
    EDUCATION_FIELDS = ("school", "degree", "field", "graduation_year")
    REFLECTION_FIELDS = {
        "jobs": ("title", "seniority", "start_year", "end_year", "is_current"),
        "education": ("degree", "graduation_year"),
    }

    def claim_catalog(self, claims: dict[str, Any]) -> dict[str, dict[str, Any]]:
        catalog = {}
        name = claims["name"]
        self._add_claim(catalog, "name", name["value"] if name["value"] is not None else name["raw"], name["raw"])
        location = claims["city"]
        if location["raw"] is not None or location["value"] is not None or location["candidates"]:
            self._add_claim(catalog, "city", {
                "value": location["value"], "candidates": deepcopy(location["candidates"]),
            }, location["raw"])
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

    def apply_corrections(self, output: Any, baseline: list[dict[str, Any]],
                          claims: dict[str, Any], profile: dict[str, Any]) -> list[dict[str, Any]]:
        """Apply sparse, grounded MATCH-to-MISMATCH corrections atomically."""
        self.validate(baseline, claims, profile)
        if not isinstance(output, dict) or set(output) != {"corrections"}:
            raise ValueError("Reflection output requires exactly a corrections envelope")
        if not isinstance(output["corrections"], list):
            raise ValueError("Reflection corrections must be a list")
        catalog = self.claim_catalog(claims)
        evidence = self._evidence_catalog(self.comparison_profile(profile))
        originals = {finding["claim_id"]: finding for finding in baseline}
        replacements = {}
        for decision in output["corrections"]:
            finding = self._reconstruct_finding(decision, catalog, evidence)
            reference = finding["claim_id"]
            self._validate_reflection_reference(reference)
            if reference in replacements:
                raise ValueError("Duplicate reflection claim reference")
            if originals[reference]["status"] != "MATCH":
                raise ValueError("Reflection may correct only existing MATCH findings")
            if finding["status"] != "MISMATCH":
                raise ValueError("Reflection requires a concrete MISMATCH, not uncertainty")
            if reference.startswith("jobs."):
                self._validate_employment_correction(finding, baseline, claims, profile)
            if not self._has_relevant_contradiction(finding):
                raise ValueError("Reflection requires relevant contradictory evidence")
            replacements[reference] = finding
        merged = [deepcopy(replacements.get(finding["claim_id"], finding)) for finding in baseline]
        return self.validate(merged, claims, profile)

    def _validate_reflection_reference(self, reference: str) -> None:
        parts = reference.split(".")
        if len(parts) != 3 or parts[0] not in self.REFLECTION_FIELDS:
            raise ValueError("Claim is not eligible for reflection corrections")
        if parts[2] not in self.REFLECTION_FIELDS[parts[0]]:
            raise ValueError("Claim is not eligible for reflection corrections")

    def _validate_employment_correction(self, finding: dict[str, Any], baseline: list[dict[str, Any]],
                                        claims: dict[str, Any], profile: dict[str, Any]) -> None:
        _, job_index, field = finding["claim_id"].split(".")
        prefix = f"jobs.{job_index}."
        records = set()
        for original in baseline:
            if original["claim_id"].startswith(prefix):
                for citation in original["evidence"]:
                    parts = citation["path"].split(".")
                    if len(parts) == 3 and parts[0] == "experience":
                        records.add(int(parts[1]))
        if len(records) != 1:
            raise ValueError("Reflection requires one baseline-aligned employment record")
        record_index = records.pop()
        record_prefix = f"experience.{record_index}."
        if any(not citation["path"].startswith(record_prefix) for citation in finding["evidence"]):
            raise ValueError("Employment correction cites an unrelated or unaligned record")
        if field in ("title", "seniority"):
            job = claims["jobs"][int(job_index)]
            record = profile["experience"][record_index]
            self._validate_title_seniority_correction(finding, field, job, record)

    def _validate_title_seniority_correction(self, finding: dict[str, Any], field: str,
                                            job: dict[str, Any], record: dict[str, Any]) -> None:
        claim_title, title_level = self._title_parts(job.get("title"))
        profile_title, profile_title_level = self._title_parts(record.get("title"))
        claim_level = self._seniority(job.get("seniority"))
        if field == "title":
            claim_level = title_level or claim_level
        profile_level = self._seniority(record.get("seniority")) or profile_title_level
        cited_fields = {citation["path"].split(".")[-1] for citation in finding["evidence"]}
        level_source = "seniority" if self._seniority(record.get("seniority")) else "title"
        level_contradicts = (claim_level is not None and profile_level is not None
                             and claim_level != profile_level and level_source in cited_fields)
        if field == "title":
            title_contradicts = (bool(claim_title) and bool(profile_title)
                                 and claim_title != profile_title and "title" in cited_fields)
            if title_contradicts or level_contradicts:
                return
        elif level_contradicts:
            return
        raise ValueError("Reflection must contradict the claimed title or explicit seniority")

    def _seniority(self, value: str | None) -> str | None:
        if not value or not value.strip():
            return None
        normalized = value.strip().casefold().rstrip(".")
        return {"sr": "senior", "jr": "junior", "mid-level": "mid"}.get(normalized, normalized)

    def _title_parts(self, title: str | None) -> tuple[str, str | None]:
        """Recognize explicit qualifiers/abbreviations; never derive grade from a role."""
        words = (title or "").casefold().replace(".", "").split()
        qualifiers = {"senior": "senior", "sr": "senior", "junior": "junior", "jr": "junior",
                      "mid": "mid", "mid-level": "mid"}
        level = qualifiers.get(words[0]) if words else None
        if level:
            words = words[1:]
        return " ".join("manager" if word == "mgr" else word for word in words), level

    def _has_relevant_contradiction(self, finding: dict[str, Any]) -> bool:
        """Check evidence relevance and reject unchanged scalar values; meaning stays with the LLM."""
        reference = finding["claim_id"]
        parts = reference.split(".")
        collection = "experience" if parts[0] == "jobs" else "education"
        field = parts[-1]
        if field in ("title", "seniority"):
            fields = ("title", "seniority")
        elif field == "graduation_year":
            fields = ("end_year",)
        else:
            fields = (field,)
        for citation in finding["evidence"]:
            path = citation["path"]
            if path == collection and isinstance(citation["value"], list):
                return True
            path_parts = path.split(".")
            relevant = (
                len(path_parts) == 3 and path_parts[0] == collection and path_parts[-1] in fields
            )
            if relevant and citation["value"] != finding["cv_claim"]["value"]:
                return True
        return False

    def reconstruct(self, output: Any, claims: dict[str, Any], profile: dict[str, Any]) -> list[dict[str, Any]]:
        """Resolve model decisions against canonical inputs, then validate findings."""
        if not isinstance(output, dict) or set(output) != {"findings"}:
            raise ValueError("Comparison output requires exactly a findings envelope")
        if not isinstance(output["findings"], list):
            raise ValueError("Comparison findings must be a list")
        catalog = self.claim_catalog(claims)
        evidence = self._evidence_catalog(self.comparison_profile(profile))
        findings = [self._reconstruct_finding(decision, catalog, evidence)
                    for decision in output["findings"]]
        return self.adjudicate_city(findings, claims, profile)

    def adjudicate_city(self, findings: Any, claims: dict[str, Any],
                        profile: dict[str, Any]) -> list[dict[str, Any]]:
        """Override only supported city judgments, never repair invalid model output."""
        self.validate(findings, claims, profile)
        city = profile.get("city")
        if not isinstance(city, str) or not city.strip():
            return findings
        normalized = self._normalize_city(city, profile.get("country"))
        if not normalized:
            return findings
        for candidate in claims["city"]["candidates"]:
            if candidate["role"] not in ("current", "unspecified"):
                continue
            if self._normalize_city(candidate["city"], profile.get("country")) != normalized:
                continue
            replacement = {
                "claim_id": "city", "cv_claim": deepcopy(self.claim_catalog(claims)["city"]),
                "status": "MATCH", "evidence": [{"path": "city", "value": city}],
                "reason": f"Profile city {city!r} matches the CV's {candidate['role']} location "
                          f"candidate {candidate['city']!r} after conservative normalization.",
            }
            merged = [replacement if finding["claim_id"] == "city" and finding["status"] != "MATCH"
                      else finding for finding in findings]
            return self.validate(merged, claims, profile)
        return findings

    def _normalize_city(self, value: str, country: Any) -> str:
        """Strip only an explicitly delimited qualifier matching supplied country metadata."""
        parts = [part for part in re.split(r"[,()|/]", value) if part.strip()]
        if isinstance(country, str) and len(parts) > 1:
            qualifier = self._city_words(parts[-1])
            if qualifier and qualifier == self._city_words(country):
                value = " ".join(parts[:-1])
        return self._city_words(value)

    def _city_words(self, value: str) -> str:
        return " ".join(re.findall(r"[^\W_]+", value.casefold()))

    def _reconstruct_finding(self, decision: Any, catalog: dict, evidence: dict) -> dict[str, Any]:
        required = {"claim_id", "status", "evidence_paths", "reason"}
        if not isinstance(decision, dict) or set(decision) != required:
            raise ValueError("Decision requires claim_id, status, evidence_paths and reason")
        reference = decision["claim_id"]
        if not isinstance(reference, str) or reference not in catalog:
            raise ValueError("Finding refers to a nonexistent CV claim")
        paths = decision["evidence_paths"]
        if not isinstance(paths, list):
            raise ValueError("Evidence paths must be a list")
        citations = []
        seen = set()
        for path in paths:
            if not isinstance(path, str) or path not in evidence:
                raise ValueError("Evidence refers to a nonexistent or excluded profile field")
            if path in seen:
                raise ValueError("Duplicate evidence path")
            if evidence[path] is None:
                raise ValueError("Missing profile values are not supporting or contradictory evidence")
            seen.add(path)
            citations.append({"path": path, "value": deepcopy(evidence[path])})
        return {"claim_id": reference, "cv_claim": deepcopy(catalog[reference]),
                "status": decision["status"], "evidence": citations, "reason": decision["reason"]}

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
        if reference == "city":
            self._validate_city_finding(finding, catalog[reference])
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

    def _validate_city_finding(self, finding: dict[str, Any], claim: dict[str, Any]) -> None:
        candidates = claim["value"]["candidates"]
        if not any(candidate["role"] in ("current", "unspecified") for candidate in candidates):
            if finding["status"] != "UNKNOWN":
                raise ValueError("Without plausible residence candidates the city finding must be UNKNOWN")
        if any(not isinstance(citation, dict) or citation.get("path") != "city" for citation in finding["evidence"]):
            raise ValueError("City findings may cite only the profile city")
