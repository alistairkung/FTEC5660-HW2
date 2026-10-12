import json
from typing import Any

from .extraction import validate_claims

def decode_mcp_result(blocks: Any) -> Any:
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


class CandidateRetrieval:
    """Contract-proving slice; retrieved profiles never establish identity."""

    def __init__(self, extractor: Any, tools: list[Any]):
        self.extractor = extractor
        self.tools = {tool.name: tool for tool in tools}

    async def ainvoke(self, cv: str) -> dict[str, Any]:
        result = {"claims": None, "search_hits": [], "profiles": [],
                  "identity": "unresolved", "retrieval_complete": False, "errors": []}
        stage = "extraction"
        try:
            claims = await self.extractor.ainvoke({"cv": cv})
            validate_claims(claims)
            name = claims["name"]["value"]
            if not name or not name.strip():
                raise ValueError("Candidate name is missing")
            result["claims"] = claims
            stage = "search"
            blocks = await self.tools["search_linkedin_people"].ainvoke({
                "q": name, "location": claims["city"]["value"],
                "limit": 20, "fuzzy": True,
            })
            hits = decode_mcp_result(blocks)
            if not isinstance(hits, list) or any(
                not isinstance(hit, dict) or type(hit.get("id")) is not int for hit in hits
            ):
                raise ValueError("Expected search hits with integer IDs")
            result["search_hits"] = hits
        except Exception as error:
            result["errors"].append({"stage": stage, "message": str(error)})
            return result

        # Search order is arbitrary. These profiles are a bounded sample, not a ranking.
        for hit in hits[:2]:
            try:
                blocks = await self.tools["get_linkedin_profile"].ainvoke({"person_id": hit["id"]})
                profile = decode_mcp_result(blocks)
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
        result["retrieval_complete"] = len(result["profiles"]) == len(hits)
        return result


