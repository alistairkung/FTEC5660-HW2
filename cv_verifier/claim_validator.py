from typing import Any

CLAIM_SCHEMA = {
    "name": {"raw": str, "value": str},
    "city": {"raw": str, "value": str, "candidates": [{"city": str, "role": str}]},
    "jobs": [{"raw": str, "company": str, "title": str, "seniority": str,
              "dates_raw": str, "start_year": int, "end_year": int, "is_current": bool}],
    "education": [{"raw": str, "school": str, "degree": str, "field": str,
                   "dates_raw": str, "graduation_year": int}],
    "skills": [{"raw": str, "name": str}],
}


class ClaimValidator:
    """Validate extraction shape and location consistency without repairing claims.

    All schema keys are required; collections must be lists of shaped objects.
    Claim scalars may be null; candidate cities and roles must be present.
    Scalar types must match exactly. Definite city values require one current candidate.
    The candidate name must be a nonempty, non-whitespace string.
    Return valid claims unchanged; raise ValueError for invalid claims.
    No semantic equivalence, date inference, or identity assessment is performed.
    """

    def validate(self, claims: Any) -> dict[str, Any]:
        self._validate_shape(claims)
        name = claims["name"]["value"]
        if not name or not name.strip():
            raise ValueError("Candidate name is missing")
        self._validate_city(claims["city"])
        return claims

    def _validate_city(self, location: dict[str, Any]) -> None:
        raw, value, candidates = location["raw"], location["value"], location["candidates"]
        if raw is not None and not raw.strip():
            raise ValueError("Location wording must be nonempty or null")
        if (value is not None or candidates) and raw is None:
            raise ValueError("Location candidates require original wording")
        seen = set()
        current = set()
        for candidate in candidates:
            city, role = candidate["city"], candidate["role"]
            if not isinstance(city, str) or not city.strip():
                raise ValueError("Location candidate city must be a nonempty string")
            if role not in ("current", "hometown", "previous", "unspecified"):
                raise ValueError("Unsupported location role")
            normalized = city.strip().casefold()
            if (normalized, role) in seen:
                raise ValueError("Duplicate location candidate")
            seen.add((normalized, role))
            if role == "current":
                current.add(normalized)
        if value is not None:
            if not value.strip() or current != {value.strip().casefold()}:
                raise ValueError("Location value must identify the single current candidate")
        elif len(current) == 1:
            raise ValueError("A single current candidate requires a matching location value")

    def _validate_shape(self, value: Any, schema: Any = CLAIM_SCHEMA, path: str = "claims") -> None:
        """Reject malformed extraction before using it to query evidence."""
        if isinstance(schema, dict):
            if not isinstance(value, dict) or set(value) != set(schema):
                raise ValueError(f"{path}: expected keys {list(schema)}")
            for key, expected in schema.items():
                self._validate_shape(value[key], expected, f"{path}.{key}")
        elif isinstance(schema, list):
            if not isinstance(value, list):
                raise ValueError(f"{path}: expected a list")
            for index, item in enumerate(value):
                self._validate_shape(item, schema[0], f"{path}[{index}]")
        elif value is not None and type(value) is not schema:
            raise ValueError(f"{path}: expected {schema.__name__} or null")
