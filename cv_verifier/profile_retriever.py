import json
from typing import Any


class ProfileRetriever:
    """Search and fetch at most two supplied profiles; never establish identity.

    Search preserves returned metadata. Retrieval follows supplied order and keeps
    partial failures. The pipeline owns selection and coverage flags.
    """

    MAX_PROFILES = 2

    def __init__(self, tools: list[Any]):
        self.tools = {tool.name: tool for tool in tools}

    async def search(self, name: str, city: str | None) -> list[dict[str, Any]]:
        blocks = await self.tools["search_linkedin_people"].ainvoke({
            "q": name, "location": city, "limit": 20, "fuzzy": True,
        })
        hits = self._decode_mcp_result(blocks)
        if not isinstance(hits, list):
            raise ValueError("Search hits must be a list")
        for hit in hits:
            if not isinstance(hit, dict):
                raise ValueError("Search hit must be an object")
            if type(hit.get("id")) is not int:
                raise ValueError("Search hit ID must be an integer")
        return hits

    async def retrieve_profiles(
        self, hits: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        profiles = []
        errors = []
        for hit in hits[:self.MAX_PROFILES]:
            person_id = hit["id"]
            try:
                profiles.append(await self._get_profile(person_id))
            except Exception as error:
                errors.append({"stage": "profile", "person_id": person_id,
                               "message": str(error)})
        return profiles, errors

    async def _get_profile(self, person_id: int) -> dict[str, Any]:
        blocks = await self.tools["get_linkedin_profile"].ainvoke({"person_id": person_id})
        profile = self._decode_mcp_result(blocks)
        self._validate_profile(profile, person_id)
        return profile

    def _validate_profile(self, profile: Any, expected_id: int) -> None:
        if not isinstance(profile, dict):
            raise ValueError("Profile must be an object")
        if type(profile.get("id")) is not int:
            raise ValueError("Profile ID must be an integer")
        if profile["id"] != expected_id:
            raise ValueError("Profile ID does not match requested candidate")
        if not isinstance(profile.get("name"), str):
            raise ValueError("Profile name must be a string")
        for field in ("experience", "education", "skills"):
            if not isinstance(profile.get(field), list):
                raise ValueError(f"Profile {field} must be a list")

    def _decode_mcp_result(self, blocks: Any) -> Any:
        """Decode the server's JSON text content and surface server errors."""
        if not isinstance(blocks, list):
            raise ValueError("Expected MCP content blocks")
        if not blocks:
            return []
        text = "".join(b["text"] for b in blocks
                       if isinstance(b, dict) and b.get("type") == "text")
        result = json.loads(text)
        if isinstance(result, dict) and "error" in result:
            raise ValueError(str(result["error"]))
        return result
