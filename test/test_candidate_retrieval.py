import json
import unittest
from copy import deepcopy
from unittest.mock import AsyncMock, patch

from cv_verifier.retrieval import CandidateRetrieval
from hw2 import build_agent


CLAIMS = {
    "name": {"raw": "Example Person", "value": "Example Person"},
    "city": {"raw": "Hong Kong", "value": "Hong Kong"},
    "jobs": [], "education": [], "skills": [],
}


def full_profile(person_id):
    return {"id": person_id, "name": "Example Person",
            "experience": [], "education": [], "skills": []}


def blocks(value):
    return [{"type": "text", "text": json.dumps(value)}]


class RetrievalTests(unittest.IsolatedAsyncioTestCase):
    retrieval_class = CandidateRetrieval
    agent_factory = staticmethod(build_agent)

    async def test_real_extraction_chain_parses_fake_model_json(self):
        from langchain_core.messages import AIMessage
        from langchain_core.runnables import RunnableLambda

        captured = []

        def respond(prompt):
            captured.extend(prompt.to_messages())
            return AIMessage(content=json.dumps(CLAIMS))

        with patch('langchain_deepseek.ChatDeepSeek', return_value=RunnableLambda(respond)):
            agent = self.agent_factory([])
            claims = await agent.extractor.ainvoke({"cv": "Example Person CV"})
        self.assertEqual(claims, CLAIMS)
        self.assertEqual(captured[0].type, "system")
        self.assertEqual(captured[1].content, "Example Person CV")

    def pipeline(self, claims=None, hits=None, profile_results=None):
        extractor = AsyncMock()
        extractor.ainvoke.return_value = deepcopy(CLAIMS if claims is None else claims)
        search = AsyncMock()
        search.name = "search_linkedin_people"
        search.ainvoke.return_value = blocks([] if hits is None else hits)
        profile = AsyncMock()
        profile.name = "get_linkedin_profile"
        profile.ainvoke.side_effect = profile_results or []
        return self.retrieval_class(extractor, [search, profile]), search, profile

    async def test_bounded_retrieval_preserves_all_search_hits_and_explicit_claims(self):
        claims = deepcopy(CLAIMS)
        claims["jobs"] = [{"raw": "Engineer 2019–Present", "company": None,
                           "title": "Engineer", "seniority": None,
                           "dates_raw": "2019–Present", "start_year": 2019,
                           "end_year": None, "is_current": True}]
        hits = [{"id": n, "name": "Example Person"} for n in range(1, 5)]
        agent, search, profile = self.pipeline(claims, hits, [blocks(full_profile(1)), blocks(full_profile(2))])
        result = await agent.ainvoke("CV text")
        self.assertEqual(result["claims"], claims)
        self.assertEqual(result["search_hits"], hits)
        self.assertEqual(result["profiles"], [full_profile(1), full_profile(2)])
        self.assertEqual(result["identity"], "unresolved")
        self.assertFalse(result["retrieval_complete"])
        self.assertEqual(profile.ainvoke.await_count, 2)
        search.ainvoke.assert_awaited_once_with(
            {"q": "Example Person", "location": "Hong Kong", "limit": 20, "fuzzy": True})

    async def test_malformed_or_nameless_extraction_stops_before_search(self):
        nameless = deepcopy(CLAIMS)
        nameless["name"]["value"] = None
        for claims in ({"name": "wrong shape"}, nameless):
            with self.subTest(claims=claims):
                agent, search, profile = self.pipeline(claims)
                result = await agent.ainvoke("CV text")
                self.assertEqual(result["errors"][0]["stage"], "extraction")
                search.ainvoke.assert_not_awaited()
                profile.ainvoke.assert_not_awaited()

    async def test_empty_search_is_success_and_identity_remains_unresolved(self):
        agent, search, profile = self.pipeline()
        search.ainvoke.return_value = []
        result = await agent.ainvoke("CV text")
        self.assertEqual(result["errors"], [])
        self.assertTrue(result["retrieval_complete"])
        self.assertEqual(result["identity"], "unresolved")
        profile.ainvoke.assert_not_awaited()

    async def test_failed_profile_preserves_search_and_other_profile(self):
        hits = [{"id": 1}, {"id": 2}]
        for failure in (RuntimeError("network failed"), blocks({"error": "unknown id"}), blocks({"id": 99})):
            with self.subTest(failure=failure):
                agent, _, _ = self.pipeline(hits=hits, profile_results=[failure, blocks(full_profile(2))])
                result = await agent.ainvoke("CV text")
                self.assertEqual(result["search_hits"], hits)
                self.assertEqual(result["profiles"], [full_profile(2)])
                self.assertEqual(result["errors"][0]["person_id"], 1)
                self.assertFalse(result["retrieval_complete"])

    async def test_search_error_preserves_claims_without_profile_calls(self):
        agent, search, profile = self.pipeline()
        search.ainvoke.return_value = blocks({"error": "unavailable"})
        result = await agent.ainvoke("CV text")
        self.assertEqual(result["claims"], CLAIMS)
        self.assertEqual(result["errors"][0]["stage"], "search")
        profile.ainvoke.assert_not_awaited()
