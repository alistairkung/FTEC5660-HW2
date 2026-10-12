import unittest
from copy import deepcopy
from unittest.mock import AsyncMock, call
from cv_verifier.candidate_pipeline import CandidatePipeline
from cv_verifier.claim_validator import ClaimValidator
from cv_verifier.candidate_ranking_validator import CandidateRankingValidator
from cv_verifier.identity_decision_validator import IdentityDecisionValidator

CLAIMS = {'name': {'raw': 'Person', 'value': 'Person'},
          'city': {'raw': 'London', 'value': 'London'}, 'jobs': [], 'education': [], 'skills': []}
UNRESOLVED = {'candidate_id': None, 'reason': 'Candidates remain ambiguous.'}


def profile(candidate_id):
    return {'id': candidate_id, 'name': 'Person', 'experience': [], 'education': [], 'skills': []}


class InvestigationTests(unittest.IsolatedAsyncioTestCase):
    def pipeline(self, filtered, broad=(), decisions=None, failed=()):
        extractor, retriever, prioritizer, resolver = (AsyncMock() for _ in range(4))
        extractor.extract.return_value = deepcopy(CLAIMS)
        retriever.MAX_PROFILES = 2
        retriever.search.side_effect = [[{'id': n} for n in filtered], [{'id': n} for n in broad]]
        def retrieve(hits):
            profiles = [profile(h['id']) for h in hits if h['id'] not in failed]
            errors = [{'stage': 'profile', 'person_id': h['id'], 'message': 'failed'}
                      for h in hits if h['id'] in failed]
            return profiles, errors
        retriever.retrieve_profiles.side_effect = retrieve
        prioritizer.prioritize.side_effect = lambda claims, hits: [h['id'] for h in hits]
        resolver.resolve.side_effect = decisions
        resolver.resolve.return_value = UNRESOLVED
        pipeline = CandidatePipeline(extractor, ClaimValidator(), retriever, prioritizer,
                                     CandidateRankingValidator(), resolver, IdentityDecisionValidator())
        return pipeline, retriever, resolver

    async def test_progressive_resolution_uses_all_successful_profiles(self):
        pipeline, retriever, resolver = self.pipeline(
            [1, 2, 3, 4, 5], decisions=[UNRESOLVED, {'candidate_id': 3, 'reason': 'Wider history supports 3.'}])
        result = await pipeline.ainvoke('CV')
        self.assertEqual(result['selected_candidate_id'], 3)
        self.assertEqual(result['identity'], 'resolved')
        self.assertEqual(result['attempted_candidate_ids'], [1, 2, 3, 4])
        self.assertEqual([[p['id'] for p in c.args[1]] for c in resolver.resolve.await_args_list],
                         [[1, 2], [1, 2, 3, 4]])
        self.assertEqual(retriever.search.await_count, 2)
        self.assertTrue(result['retrieval_complete'])
        self.assertTrue(result['retrieval_sampled'])

    async def test_two_searches_combine_and_deduplicate_before_investigation(self):
        pipeline, retriever, resolver = self.pipeline(
            [1, 2], broad=[1, 2, 3, 3, 4], failed=[1],
            decisions=[UNRESOLVED, {'candidate_id': 4, 'reason': 'Employment and education identify 4.'}])
        result = await pipeline.ainvoke('CV')
        self.assertEqual(result['attempted_candidate_ids'], [1, 2, 3, 4])
        self.assertEqual(result['ranked_candidate_ids'], [1, 2, 3, 4])
        self.assertEqual(retriever.search.await_args_list, [call('Person', 'London'), call('Person', None)])
        self.assertEqual([[p['id'] for p in c.args[1]] for c in resolver.resolve.await_args_list],
                         [[2], [2, 3, 4]])
        self.assertEqual(result['selected_candidate_id'], 4)
        self.assertFalse(result['retrieval_complete'])
        self.assertTrue(result['broader_search_performed'])
        self.assertEqual(result['errors'][0]['person_id'], 1)

    async def test_budget_counts_failed_attempts_and_never_exceeds_four_batches(self):
        pipeline, retriever, resolver = self.pipeline(range(1, 21), failed=range(1, 21))
        result = await pipeline.ainvoke('CV')
        self.assertEqual(result['attempted_candidate_ids'], list(range(1, 9)))
        self.assertEqual(retriever.retrieve_profiles.await_count, 4)
        self.assertEqual(retriever.search.await_count, 2)
        resolver.resolve.assert_not_awaited()
        self.assertIsNone(result['selected_candidate_id'])
        self.assertEqual(result['investigation_stop_reason'], 'budget_exhausted')
        self.assertEqual(len(result['errors']), 8)

    async def test_successful_budget_limit_remains_ambiguous(self):
        pipeline, retriever, resolver = self.pipeline(range(1, 21))
        result = await pipeline.ainvoke('CV')
        self.assertEqual(resolver.resolve.await_count, 4)
        self.assertEqual(len(result['profiles']), 8)
        self.assertEqual(result['identity'], 'unresolved')
        self.assertTrue(result['retrieval_sampled'])
        self.assertEqual(result['identity_reason'], UNRESOLVED['reason'])

    async def test_budget_is_shared_with_broader_search(self):
        pipeline, retriever, resolver = self.pipeline([1, 2, 3, 4, 5], broad=range(1, 21))
        result = await pipeline.ainvoke('CV')
        self.assertEqual(result['attempted_candidate_ids'], list(range(1, 9)))
        self.assertEqual([len(c.args[0]) for c in retriever.retrieve_profiles.await_args_list], [2, 2, 2, 2])
        self.assertEqual(resolver.resolve.await_count, 4)
        self.assertEqual(result['investigation_stop_reason'], 'budget_exhausted')

    async def test_invalid_decision_or_resolver_error_cannot_resolve_identity(self):
        for invalid in ({'candidate_id': 99, 'reason': 'invented'}, {'candidate_id': True, 'reason': 'bad'},
                        {'candidate_id': 1}, RuntimeError('failed')):
            with self.subTest(invalid=invalid):
                pipeline, _, _ = self.pipeline([1, 2], decisions=[invalid, UNRESOLVED], broad=[3])
                result = await pipeline.ainvoke('CV')
                self.assertEqual(result['identity'], 'unresolved')
                self.assertIsNone(result['selected_candidate_id'])
                self.assertEqual(result['errors'][0]['stage'], 'identity')
                self.assertEqual(result['attempted_candidate_ids'], [1, 2, 3])

    async def test_invented_failed_profile_id_cannot_be_selected(self):
        pipeline, _, _ = self.pipeline([1, 2], failed=[1], decisions=[{'candidate_id': 1, 'reason': 'wrong'}])
        result = await pipeline.ainvoke('CV')
        self.assertIsNone(result['selected_candidate_id'])
        self.assertEqual(result['errors'][0]['stage'], 'profile')
        self.assertEqual(result['errors'][1]['stage'], 'identity')

    async def test_empty_filtered_search_still_performs_one_broader_search(self):
        pipeline, retriever, _ = self.pipeline([], broad=[3])
        result = await pipeline.ainvoke('CV')
        self.assertEqual(result['attempted_candidate_ids'], [3])
        self.assertEqual(retriever.search.await_count, 2)
        self.assertEqual(result['identity'], 'unresolved')

    async def test_no_city_filter_does_not_repeat_identical_search(self):
        pipeline, retriever, _ = self.pipeline([])
        pipeline.extractor.extract.return_value['city']['value'] = None
        result = await pipeline.ainvoke('CV')
        self.assertEqual(retriever.search.await_count, 1)
        self.assertFalse(result['broader_search_performed'])

    async def test_broader_search_failure_preserves_examined_profiles(self):
        pipeline, retriever, _ = self.pipeline([1, 2])
        retriever.search.side_effect = [[{'id': 1}, {'id': 2}], RuntimeError('search failed')]
        result = await pipeline.ainvoke('CV')
        self.assertEqual([p['id'] for p in result['profiles']], [1, 2])
        self.assertEqual(result['errors'][0]['scope'], 'name_only')
        self.assertIsNone(result['selected_candidate_id'])

    async def test_resolves_to_earlier_profile_after_more_evidence(self):
        pipeline, _, _ = self.pipeline([1, 2, 3], decisions=[
            UNRESOLVED, {'candidate_id': 1, 'reason': 'Additional alternatives make candidate 1 strongest.'}])
        result = await pipeline.ainvoke('CV')
        self.assertEqual(result['selected_candidate_id'], 1)
        self.assertEqual(result['attempted_candidate_ids'], [1, 2, 3])

    async def test_duplicate_filtered_hits_do_not_repeat_lookups(self):
        pipeline, retriever, _ = self.pipeline([1, 1, 2])
        result = await pipeline.ainvoke('CV')
        self.assertEqual(result['attempted_candidate_ids'], [1, 2])
        self.assertEqual(retriever.retrieve_profiles.await_count, 1)
        self.assertEqual(result['errors'], [])

    async def test_combined_pool_is_ranked_once_before_retrieval_with_city_metadata_preferred(self):
        pipeline, retriever, _ = self.pipeline([1, 2], broad=[2, 3])
        filtered = [{'id': 1, 'headline': 'filtered'}, {'id': 2, 'headline': 'preferred'}]
        broad = [{'id': 2, 'headline': 'duplicate'}, {'id': 3, 'headline': 'new'}]
        retriever.search.side_effect = [filtered, broad]
        async def retrieve(hits):
            self.assertEqual(retriever.search.await_count, 2)
            return [profile(hit['id']) for hit in hits], []
        retriever.retrieve_profiles.side_effect = retrieve
        result = await pipeline.ainvoke('CV')
        combined = filtered + [broad[1]]
        self.assertEqual(result['search_hits'], combined)
        pipeline.prioritizer.prioritize.assert_awaited_once_with(CLAIMS, combined)

    async def test_either_search_failure_keeps_successful_results(self):
        for failed_index in (0, 1):
            with self.subTest(failed_index=failed_index):
                pipeline, retriever, _ = self.pipeline([1])
                responses = [[{'id': 1}], [{'id': 2}]]
                responses[failed_index] = RuntimeError('failed')
                retriever.search.side_effect = responses
                result = await pipeline.ainvoke('CV')
                self.assertEqual(result['attempted_candidate_ids'], [2 if failed_index == 0 else 1])
                self.assertEqual(result['errors'][0]['stage'], 'search')
                self.assertEqual(retriever.search.await_count, 2)

    async def test_both_search_failures_remain_unresolved_without_model_calls(self):
        pipeline, retriever, resolver = self.pipeline([])
        retriever.search.side_effect = [RuntimeError('filtered failed'), RuntimeError('name failed')]
        result = await pipeline.ainvoke('CV')
        self.assertEqual([error['scope'] for error in result['errors']], ['city_filtered', 'name_only'])
        self.assertEqual(result['identity'], 'unresolved')
        self.assertFalse(result['retrieval_complete'])
        pipeline.prioritizer.prioritize.assert_not_awaited()
        retriever.retrieve_profiles.assert_not_awaited()
        resolver.resolve.assert_not_awaited()
