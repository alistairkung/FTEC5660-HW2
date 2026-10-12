import json
from typing import Any


class ProfileRetriever:
    """Search and fetch an arbitrary two-profile sample; never establish identity.

    retrieval_complete means every selected lookup succeeded, including zero hits.
    retrieval_sampled means returned search hits were omitted by the two-profile cap.
    Neither flag implies exhaustive search coverage beyond the server result cap.
    """

    def __init__(self, tools: list[Any]):
        self.tools = {tool.name: tool for tool in tools}

    async def retrieve(self, name: str, city: str | None) -> dict[str, Any]:
        result = {"search_hits": [], "profiles": [], "retrieval_complete": False,
                  "retrieval_sampled": False, "errors": []}
        try:
            blocks = await self.tools["search_linkedin_people"].ainvoke({
                "q": name, "location": city,
                "limit": 20, "fuzzy": True,
            })
            hits = self._decode_mcp_result(blocks)
            if not isinstance(hits, list) or any(
                not isinstance(hit, dict) or type(hit.get("id")) is not int for hit in hits
            ):
                raise ValueError("Expected search hits with integer IDs")
            result["search_hits"] = hits
        except Exception as error:
            result["errors"].append({"stage": "search", "message": str(error)})
            return result

        result["retrieval_sampled"] = len(hits) > 2
        # Search order is arbitrary. These profiles are a bounded sample, not a ranking.
        for hit in hits[:2]:
            try:
                blocks = await self.tools["get_linkedin_profile"].ainvoke({"person_id": hit["id"]})
                profile = self._decode_mcp_result(blocks)
                if (not isinstance(profile, dict) or type(profile.get("id")) is not int
                        or profile["id"] != hit["id"]
                        or not isinstance(profile.get("name"), str)
                        or any(not isinstance(profile.get(key), list)
                               for key in ("experience", "education", "skills"))):
                    raise ValueError("Expected a profile with the requested ID")
                result["profiles"].append(profile)
            except Exception as error:
                result["errors"].append({"stage": "profile", "person_id": hit["id"],
                                         "message": str(error)})
        result["retrieval_complete"] = len(result["profiles"]) == min(2, len(hits))
        return result

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
