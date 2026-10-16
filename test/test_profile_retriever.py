import json
import unittest
from unittest.mock import AsyncMock, call
from cv_verifier.profile_retriever import ProfileRetriever


def profile(person_id):
    return {'id': person_id, 'name': 'Person', 'experience': [], 'education': [], 'skills': []}


def blocks(value):
    return [{'type': 'text', 'text': json.dumps(value)}]


class ProfileRetrieverTests(unittest.IsolatedAsyncioTestCase):
    def retriever(self):
        search, lookup = AsyncMock(), AsyncMock()
        search.name, lookup.name = 'search_linkedin_people', 'get_linkedin_profile'
        return ProfileRetriever([search, lookup]), search, lookup

    async def test_search_preserves_all_metadata_and_uses_expected_arguments(self):
        retriever, search, _ = self.retriever()
        hits = [{'id': n, 'headline': 'example'} for n in range(20)]
        search.ainvoke.return_value = blocks(hits)
        self.assertEqual(await retriever.search('Person', 'London'), hits)
        search.ainvoke.assert_awaited_once_with(
            {'q': 'Person', 'location': 'London', 'limit': 20, 'fuzzy': True})

    async def test_empty_search_returns_empty_list(self):
        for response in ([], blocks([])):
            with self.subTest(response=response):
                retriever, search, _ = self.retriever()
                search.ainvoke.return_value = response
                self.assertEqual(await retriever.search('Person', None), [])

    async def test_industry_qualified_search_uses_supported_optional_filter(self):
        retriever, search, _ = self.retriever()
        hits = [{'id': 1, 'industry': 'Software'}]
        search.ainvoke.return_value = blocks(hits)
        self.assertEqual(await retriever.search('Person', 'London', industry='Software'), hits)
        search.ainvoke.assert_awaited_once_with({
            'q': 'Person', 'location': 'London', 'industry': 'Software', 'limit': 20, 'fuzzy': True})

    async def test_invalid_search_results_surface(self):
        for response in (blocks({}), blocks([{'id': True}]), blocks([{}]),
                         blocks({'error': 'unavailable'}), [{'type': 'text', 'text': 'not JSON'}]):
            with self.subTest(response=response), self.assertRaises(ValueError):
                retriever, search, _ = self.retriever()
                search.ainvoke.return_value = response
                await retriever.search('Person', None)

    async def test_retrieval_is_sequential_in_supplied_order_and_capped_at_two(self):
        retriever, _, lookup = self.retriever()
        lookup.ainvoke.side_effect = [blocks(profile(3)), blocks(profile(1))]
        profiles, errors = await retriever.retrieve_profiles([{'id': n} for n in (3, 1, 2)])
        self.assertEqual(profiles, [profile(3), profile(1)])
        self.assertEqual(errors, [])
        self.assertEqual(lookup.ainvoke.await_args_list, [call({'person_id': 3}), call({'person_id': 1})])

    async def test_empty_retrieval_makes_no_tool_calls(self):
        retriever, _, lookup = self.retriever()
        self.assertEqual(await retriever.retrieve_profiles([]), ([], []))
        lookup.ainvoke.assert_not_awaited()

    async def test_failed_lookup_preserves_other_profile(self):
        for failure in (RuntimeError('network failed'), blocks({'error': 'unknown ID'})):
            with self.subTest(failure=failure):
                retriever, _, lookup = self.retriever()
                lookup.ainvoke.side_effect = [failure, blocks(profile(2))]
                profiles, errors = await retriever.retrieve_profiles([{'id': 1}, {'id': 2}])
                self.assertEqual(profiles, [profile(2)])
                self.assertEqual(errors[0]['person_id'], 1)
                self.assertEqual(errors[0]['stage'], 'profile')

    async def test_profile_contract_failures_preserve_other_profile(self):
        invalid = [[], {**profile(1), 'id': True}, profile(99), {**profile(1), 'name': None}]
        invalid.extend({**profile(1), field: None} for field in ('experience', 'education', 'skills'))
        for candidate in invalid:
            with self.subTest(candidate=candidate):
                retriever, _, lookup = self.retriever()
                lookup.ainvoke.side_effect = [blocks(candidate), blocks(profile(2))]
                profiles, errors = await retriever.retrieve_profiles([{'id': 1}, {'id': 2}])
                self.assertEqual(profiles, [profile(2)])
                self.assertEqual(errors[0]['person_id'], 1)
