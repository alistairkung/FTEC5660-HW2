from typing import Any

from cv_verifier.claim_comparison_validator import ClaimComparisonValidator


class DeterministicClaimVerifier:
    """Verify objective claims using validated alignments; leave semantic cases undecided."""

    def verify(self, claims: dict[str, Any], profile: dict[str, Any], alignments: dict[str, Any],
               *, skills_authoritative: bool = False) -> list[dict[str, Any]]:
        contract = ClaimComparisonValidator()
        decisions = [self._verify_claim(reference, claim["value"], claims, profile, alignments, skills_authoritative)
                     for reference, claim in contract.claim_catalog(claims).items()]
        # Reconstruction enforces canonical values, full coverage, evidence rules and city adjudication.
        return contract.reconstruct({"findings": decisions}, claims, profile)

    def _verify_claim(self, reference, value, claims, profile, alignments, skills_authoritative):
        if reference == "name":
            return self._exact_text(reference, value, profile.get("name"), "name")
        if reference == "city":
            city = profile.get("city")
            reason = "Missing profile city evidence." if not isinstance(city, str) or not city.strip() else "Semantic city review required."
            return self._decision(reference, "UNKNOWN", [], reason)
        if reference.startswith("skills."):
            return self._verify_skill(reference, value, profile, skills_authoritative)
        collection, index, field = reference.split(".")
        alignment = next(entry for entry in alignments[collection] if entry["cv_index"] == int(index))
        if alignment["profile_index"] is None:
            return self._decision(reference, "UNKNOWN", [], "Record alignment is ambiguous or unavailable.")
        source = "experience" if collection == "jobs" else "education"
        record_index = alignment["profile_index"]
        record = profile[source][record_index]
        prefix = f"{source}.{record_index}"
        if field in ("dates", "record"):
            return self._decision(reference, "UNKNOWN", [], "Unparsed claim requires semantic review.")
        if collection == "jobs":
            return self._verify_job(reference, value, field, claims[collection][int(index)], record, prefix)
        evidence_field = "end_year" if field == "graduation_year" else field
        actual, path = record.get(evidence_field), f"{prefix}.{evidence_field}"
        if field == "graduation_year":
            return self._exact_scalar(reference, value, actual, path, int)
        return self._exact_text(reference, value, actual, path)

    def _verify_skill(self, reference, value, profile, authoritative):
        skills = profile.get("skills")
        if not isinstance(skills, list):
            return self._decision(reference, "UNKNOWN", [], "Missing profile skills evidence.")
        for index, skill in enumerate(skills):
            if self._equal_text(value, skill.get("name")):
                return self._decision(reference, "MATCH", [f"skills.{index}.name"], "Exact normalized skill match.")
        if not skills and authoritative is True:
            return self._decision(reference, "MISMATCH", ["skills"], "Authoritative empty skills collection contradicts the claim.")
        if not skills:
            return self._decision(reference, "UNKNOWN", [], "Missing established skills completeness.")
        if not any(isinstance(skill.get("name"), str) and skill["name"].strip() for skill in skills):
            return self._decision(reference, "UNKNOWN", [], "Missing usable profile skill names.")
        return self._decision(reference, "UNKNOWN", [], "Nonexact skill names require semantic review.")

    def _verify_job(self, reference, value, field, job, record, prefix):
        if field in ("start_year", "end_year", "is_current"):
            expected_type = bool if field == "is_current" else int
            return self._exact_scalar(reference, value, record.get(field), f"{prefix}.{field}", expected_type)
        if field == "company":
            return self._exact_text(reference, value, record.get(field), f"{prefix}.{field}")
        if field == "seniority":
            return self._verify_seniority(reference, self._level(value), record, prefix)
        base, title_level = self._title_parts(value)
        actual_base, _ = self._title_parts(record.get("title"))
        claim_level = title_level or self._level(job.get("seniority"))
        if claim_level is not None:
            seniority = self._verify_seniority(reference, claim_level, record, prefix)
            if seniority["status"] != "MATCH":
                return seniority
        else:
            seniority = None
        if base and actual_base and base == actual_base:
            paths = [f"{prefix}.title"]
            if seniority is not None:
                paths.extend(path for path in seniority["evidence_paths"] if path not in paths)
            return self._decision(reference, "MATCH", paths, "Exact base title; explicit claimed seniority is compatible.")
        reason = "Missing profile title evidence." if not actual_base else "Semantic title equivalence requires review."
        return self._decision(reference, "UNKNOWN", [], reason)

    def _verify_seniority(self, reference, claim_level, record, prefix):
        actual_level = self._level(record.get("seniority"))
        _, title_level = self._title_parts(record.get("title"))
        if actual_level is not None and title_level is not None and actual_level != title_level:
            return self._decision(reference, "UNKNOWN", [], "Conflicting profile seniority evidence requires semantic review.")
        path = f"{prefix}.seniority"
        if actual_level is None:
            actual_level = title_level
            path = f"{prefix}.title"
        if claim_level is None or actual_level is None:
            return self._decision(reference, "UNKNOWN", [], "Missing explicit seniority evidence.")
        if claim_level == actual_level:
            return self._decision(reference, "MATCH", [path], "Explicit seniority agrees.")
        if claim_level in ("senior", "junior", "mid") and actual_level in ("senior", "junior", "mid"):
            return self._decision(reference, "MISMATCH", [path], "Explicit claimed and profile seniority contradict.")
        return self._decision(reference, "UNKNOWN", [], "Semantic seniority equivalence requires review.")

    def _title_parts(self, value):
        words = self._text(value).split()
        qualifiers = {"senior": "senior", "sr": "senior", "sr.": "senior", "junior": "junior",
                      "jr": "junior", "jr.": "junior", "mid": "mid", "mid-level": "mid"}
        level = qualifiers.get(words[0]) if words else None
        return " ".join(words[1:] if level else words), level

    def _level(self, value):
        normalized = self._text(value)
        return {"sr": "senior", "sr.": "senior", "jr": "junior", "jr.": "junior",
                "mid level": "mid", "mid-level": "mid"}.get(normalized, normalized) or None

    def _exact_text(self, reference, value, actual, path):
        if self._equal_text(value, actual):
            return self._decision(reference, "MATCH", [path], "Exact normalized text match.")
        reason = "Missing profile text evidence." if not isinstance(actual, str) or not actual.strip() else "Semantic text equivalence requires review."
        return self._decision(reference, "UNKNOWN", [], reason)

    def _equal_text(self, value, actual):
        return bool(self._text(value)) and self._text(value) == self._text(actual)

    def _text(self, value):
        # Punctuation can distinguish skills/degrees (C++ versus C#); leave aliases to semantic review.
        return " ".join(value.casefold().split()) if isinstance(value, str) else ""

    def _exact_scalar(self, reference, value, actual, path, expected_type):
        if type(actual) is not expected_type:
            return self._decision(reference, "UNKNOWN", [], "Missing or invalid typed profile evidence.")
        status = "MATCH" if value == actual else "MISMATCH"
        return self._decision(reference, status, [path], "Explicit typed values agree." if status == "MATCH" else "Explicit typed values contradict.")

    def _decision(self, reference, status, paths, reason):
        return {"claim_id": reference, "status": status, "evidence_paths": paths, "reason": reason}
