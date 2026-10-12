import json
import unittest
from unittest.mock import AsyncMock, call

from cv_verifier.profile_retriever import ProfileRetriever


def full_profile(person_id):
    return {"id": person_id, "name": "Example Person",
            "experience": [], "education": [], "skills": []}


def blocks(value):
    return [{"type": "text", "text": json.dumps(value)}]


class ProfileRetrieverTests(unittest.IsolatedAsyncioTestCase):
    def retriever(self, hits=None, profile_results=None):
        search = AsyncMock()
        search.name = "search_linkedin_people"
        search.ainvoke.return_value = blocks([] if hits is None else hits)
        profile = AsyncMock()
        profile.name = "get_linkedin_profile"
        profile.ainvoke.side_effect = profile_results or []
        return ProfileRetriever([search, profile]), search, profile

    async def test_bounded_retrieval_preserves_all_search_hits(self):
        hits = [{"id": n, "name": "Example Person"} for n in range(1, 5)]
        agent, search, profile = self.retriever(hits, [blocks(full_profile(1)), blocks(full_profile(2))])
        result = await agent.retrieve("Example Person", "Hong Kong")
        self.assertEqual(result["search_hits"], hits)
        self.assertEqual(result["profiles"], [full_profile(1), full_profile(2)])
        self.assertTrue(result["retrieval_complete"])
        self.assertTrue(result["retrieval_sampled"])
        self.assertEqual(profile.ainvoke.await_args_list,
                         [call({"person_id": 1}), call({"person_id": 2})])
        search.ainvoke.assert_awaited_once_with(
            {"q": "Example Person", "location": "Hong Kong", "limit": 20, "fuzzy": True})

    async def test_empty_search_is_success_without_profile_calls(self):
        agent, search, profile = self.retriever()
        search.ainvoke.return_value = []
        result = await agent.retrieve("Example Person", "Hong Kong")
        self.assertEqual(result["errors"], [])
        self.assertTrue(result["retrieval_complete"])
        self.assertFalse(result["retrieval_sampled"])
        profile.ainvoke.assert_not_awaited()

    async def test_failed_profile_preserves_search_and_other_profile(self):
        hits = [{"id": 1}, {"id": 2}]
        for failure in (RuntimeError("network failed"), blocks({"error": "unknown id"}), blocks({"id": 99})):
            with self.subTest(failure=failure):
                agent, _, _ = self.retriever(hits=hits, profile_results=[failure, blocks(full_profile(2))])
                result = await agent.retrieve("Example Person", "Hong Kong")
                self.assertEqual(result["search_hits"], hits)
                self.assertEqual(result["profiles"], [full_profile(2)])
                self.assertEqual(result["errors"][0]["person_id"], 1)
                self.assertFalse(result["retrieval_complete"])

    async def test_search_error_prevents_profile_calls(self):
        agent, search, profile = self.retriever()
        search.ainvoke.return_value = blocks({"error": "unavailable"})
        result = await agent.retrieve("Example Person", "Hong Kong")
        self.assertEqual(result["errors"][0]["stage"], "search")
        profile.ainvoke.assert_not_awaited()

    async def test_malformed_search_results_fail_without_profile_calls(self):
        for response in (blocks({}), blocks([{"id": True}]), blocks([{}]),
                         [{"type": "text", "text": "not JSON"}]):
            with self.subTest(response=response):
                retriever, search, profile = self.retriever()
                search.ainvoke.return_value = response
                result = await retriever.retrieve("Person", None)
                self.assertEqual(result["errors"][0]["stage"], "search")
                self.assertFalse(result["retrieval_complete"])
                profile.ainvoke.assert_not_awaited()

    async def test_sampling_and_lookup_failure_are_independent(self):
        for hit_count in (2, 4):
            for failed in (False, True):
                with self.subTest(hit_count=hit_count, failed=failed):
                    retriever, _, _ = self.retriever(
                        hits=[{"id": n} for n in range(1, hit_count + 1)],
                        profile_results=[RuntimeError("failed") if failed else blocks(full_profile(1)),
                                         blocks(full_profile(2))],
                    )
                    result = await retriever.retrieve("Person", None)
                    self.assertEqual(result["retrieval_complete"], not failed)
                    self.assertEqual(result["retrieval_sampled"], hit_count > 2)
