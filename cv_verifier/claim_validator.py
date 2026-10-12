from typing import Any

CLAIM_SCHEMA = {
    "name": {"raw": str, "value": str},
    "city": {"raw": str, "value": str},
    "jobs": [{"raw": str, "company": str, "title": str, "seniority": str,
              "dates_raw": str, "start_year": int, "end_year": int, "is_current": bool}],
    "education": [{"raw": str, "school": str, "degree": str, "field": str,
                   "dates_raw": str, "graduation_year": int}],
    "skills": [{"raw": str, "name": str}],
}


class ClaimValidator:
    """Validate extraction shape without repairing or interpreting claims.

    All schema keys are required; collections must be lists of shaped objects.
    Scalar values may be null, otherwise their exact Python type must match.
    The candidate name must be a nonempty, non-whitespace string.
    Return valid claims unchanged; raise ValueError for invalid claims.
    No semantic equivalence, date inference, or identity assessment is performed.
    """

    def validate(self, claims: Any) -> dict[str, Any]:
        self._validate_shape(claims)
        name = claims["name"]["value"]
        if not name or not name.strip():
            raise ValueError("Candidate name is missing")
        return claims

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
