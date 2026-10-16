import unittest
import json
from copy import deepcopy
from unittest.mock import AsyncMock, call
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from cv_verifier.identity_resolver import IdentityResolver
from cv_verifier.candidate_pipeline import CandidatePipeline
from cv_verifier.claim_validator import ClaimValidator
from cv_verifier.candidate_ranking_validator import CandidateRankingValidator
from cv_verifier.identity_decision_validator import IdentityDecisionValidator

CLAIMS = {'name': {'raw': 'Person', 'value': 'Person'},
          'city': {'raw': 'London', 'value': 'London',
                   'candidates': [{'city': 'London', 'role': 'current'}]}, 'jobs': [], 'education': [], 'skills': []}
UNRESOLVED = {'candidate_id': None, 'reason': 'Candidates remain ambiguous.'}


def profile(candidate_id):
    return {'id': candidate_id, 'name': 'Person', 'experience': [], 'education': [], 'skills': []}


class InvestigationTests(unittest.IsolatedAsyncioTestCase):
    async def test_ambiguous_location_hints_search_upfront_and_keep_name_only_coverage(self):
        pipeline, retriever, _ = self.pipeline([])
        claims = pipeline.extractor.extract.return_value
        claims['city'] = {'raw': 'Beijing, China | Hong Kong', 'value': None, 'candidates': [
            {'city': 'Beijing', 'role': 'unspecified'}, {'city': 'Hong Kong', 'role': 'unspecified'}]}
        retriever.search.side_effect = [
            [{'id': 1, 'headline': 'first location'}],
            [{'id': 1, 'headline': 'duplicate'}, {'id': 2}], [{'id': 3}],
        ]
        async def retrieve(hits):
            self.assertEqual(retriever.search.await_count, 3)
            return [profile(h['id']) for h in hits], []
        retriever.retrieve_profiles.side_effect = retrieve
        result = await pipeline.ainvoke('CV')
        self.assertEqual(retriever.search.await_args_list,
                         [call('Person', 'Beijing'), call('Person', 'Hong Kong'), call('Person', None)])
        self.assertEqual(result['search_hits'], [{'id': 1, 'headline': 'first location'}, {'id': 2}, {'id': 3}])
        pipeline.prioritizer.prioritize.assert_awaited_once_with(claims, result['search_hits'])
        self.assertEqual(result['attempted_candidate_ids'], [1, 2, 3])
        self.assertIsNone(result['claims']['city']['value'])
        self.assertTrue(result['broader_search_performed'])

    async def test_hometown_and_previous_locations_never_filter_discovery(self):
        for role in ('hometown', 'previous'):
            pipeline, retriever, _ = self.pipeline([1])
            pipeline.extractor.extract.return_value['city'] = {
                'raw': f'{role}: London', 'value': None, 'candidates': [{'city': 'London', 'role': role}]}
            result = await pipeline.ainvoke('CV')
            retriever.search.assert_awaited_once_with('Person', None)
            self.assertFalse(result['broader_search_performed'])

    async def test_location_filters_deduplicated_and_still_bound_to_eight_profile_attempts(self):
        pipeline, retriever, _ = self.pipeline([])
        pipeline.extractor.extract.return_value['city'] = {
            'raw': 'Current Beijing; Beijing and Hong Kong also mentioned', 'value': 'Beijing',
            'candidates': [{'city': 'Beijing', 'role': 'current'}, {'city': 'beijing', 'role': 'unspecified'},
                           {'city': 'Hong Kong', 'role': 'unspecified'}]}
        retriever.search.side_effect = [[{'id': n} for n in range(1, 21)],
                                      RuntimeError('second city failed'), [{'id': 21}]]
        result = await pipeline.ainvoke('CV')
        self.assertEqual(retriever.search.await_args_list,
                         [call('Person', 'Beijing'), call('Person', 'Hong Kong'), call('Person', None)])
        self.assertEqual(result['attempted_candidate_ids'], list(range(1, 9)))
        self.assertEqual(retriever.retrieve_profiles.await_count, 5)
        self.assertEqual(result['search_hits'][-1], {'id': 21})
        self.assertEqual(result['errors'][0]['stage'], 'search')

    def pipeline(self, filtered, broad=(), decisions=None, failed=()):
        extractor, retriever, prioritizer, resolver = (AsyncMock() for _ in range(4))
        extractor.extract.return_value = deepcopy(CLAIMS)
        retriever.MAX_PROFILES = 2
        retriever.MAX_SEARCH_RESULTS = 20
        retriever.search.side_effect = [[{'id': n} for n in filtered], [{'id': n} for n in broad]]
        def retrieve(hits):
            profiles = [profile(h['id']) for h in hits if h['id'] not in failed]
            errors = [{'stage': 'profile', 'person_id': h['id'], 'message': 'failed'}
                      for h in hits if h['id'] in failed]
            return profiles, errors
        retriever.retrieve_profiles.side_effect = retrieve
        prioritizer.prioritize.side_effect = lambda claims, hits: [h['id'] for h in hits[:5]]
        resolver.resolve.side_effect = decisions
        resolver.resolve.return_value = UNRESOLVED
        pipeline = CandidatePipeline(extractor, ClaimValidator(), retriever, prioritizer,
                                     CandidateRankingValidator(), resolver, IdentityDecisionValidator())
        return pipeline, retriever, resolver

    async def test_progressive_resolution_uses_all_successful_profiles(self):
        pipeline, retriever, resolver = self.pipeline(
            [1, 2, 3, 4, 5], decisions=[UNRESOLVED, {'candidate_id': 3, 'reason': 'Wider history supports 3.'}])
        pipeline.prioritizer.prioritize.return_value = [1, 2]
        pipeline.prioritizer.prioritize.side_effect = None
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
            decisions=[{'candidate_id': 4, 'reason': 'Employment and education identify 4.'}])
        result = await pipeline.ainvoke('CV')
        self.assertEqual(result['attempted_candidate_ids'], [1, 2, 3, 4])
        self.assertEqual(result['ranked_candidate_ids'], [1, 2, 3, 4])
        self.assertEqual(retriever.search.await_args_list, [call('Person', 'London'), call('Person', None)])
        self.assertEqual([[p['id'] for p in c.args[1]] for c in resolver.resolve.await_args_list],
                         [[2, 3, 4]])
        self.assertEqual(result['selected_candidate_id'], 4)
        self.assertFalse(result['retrieval_complete'])
        self.assertTrue(result['broader_search_performed'])
        self.assertEqual(result['errors'][0]['person_id'], 1)

    async def test_budget_counts_failed_attempts_and_keeps_batches_bounded(self):
        pipeline, retriever, resolver = self.pipeline(range(1, 21), failed=range(1, 21))
        result = await pipeline.ainvoke('CV')
        self.assertEqual(result['attempted_candidate_ids'], list(range(1, 9)))
        self.assertEqual([len(c.args[0]) for c in retriever.retrieve_profiles.await_args_list], [2, 2, 1, 2, 1])
        self.assertEqual(retriever.search.await_count, 2)
        resolver.resolve.assert_not_awaited()
        self.assertIsNone(result['selected_candidate_id'])
        self.assertEqual(result['investigation_stop_reason'], 'budget_exhausted')
        self.assertEqual(len(result['errors']), 8)

    async def test_successful_budget_limit_remains_ambiguous(self):
        pipeline, retriever, resolver = self.pipeline(range(1, 21))
        result = await pipeline.ainvoke('CV')
        self.assertEqual(resolver.resolve.await_count, 3)
        self.assertEqual(len(result['profiles']), 8)
        self.assertEqual(result['identity'], 'unresolved')
        self.assertTrue(result['retrieval_sampled'])
        self.assertEqual(result['identity_reason'], UNRESOLVED['reason'])

    async def test_budget_is_shared_with_broader_search(self):
        pipeline, retriever, resolver = self.pipeline([1, 2, 3, 4, 5], broad=range(1, 21))
        result = await pipeline.ainvoke('CV')
        self.assertEqual(result['attempted_candidate_ids'], list(range(1, 9)))
        self.assertEqual([len(c.args[0]) for c in retriever.retrieve_profiles.await_args_list], [2, 2, 1, 2, 1])
        self.assertEqual(resolver.resolve.await_count, 3)
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
        pipeline.extractor.extract.return_value['city'] = {'raw': None, 'value': None, 'candidates': []}
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
        pipeline.prioritizer.prioritize.side_effect = None
        pipeline.prioritizer.prioritize.return_value = [1, 2]
        result = await pipeline.ainvoke('CV')
        self.assertEqual(result['selected_candidate_id'], 1)
        self.assertEqual(result['attempted_candidate_ids'], [1, 2, 3])

    async def test_fifth_shortlisted_identity_prevents_early_weak_selection(self):
        pipeline, retriever, _ = self.pipeline(range(1, 11))
        calls = []
        def respond(prompt):
            context = json.loads(prompt.to_messages()[1].content)
            ids = [p['id'] for p in context['profiles']]
            calls.append(ids)
            self.assertEqual([h['id'] for c in retriever.retrieve_profiles.await_args_list
                              for h in c.args[0]], [1, 2, 3, 4, 5])
            candidate_id = 5 if 5 in ids else 4
            return AIMessage(content=json.dumps({
                'candidate_id': candidate_id, 'reason': 'Distinctive employment and education support 5.'}))
        pipeline.identity_resolver = IdentityResolver(RunnableLambda(respond))
        result = await pipeline.ainvoke('CV')
        self.assertEqual(calls, [[1, 2, 3, 4, 5]])
        self.assertEqual(result['selected_candidate_id'], 5)
        self.assertEqual(result['shortlisted_candidate_ids'], [1, 2, 3, 4, 5])
        self.assertTrue(result['shortlist_covered'])
        self.assertEqual(result['attempted_candidate_ids'], [1, 2, 3, 4, 5])
        self.assertEqual([len(c.args[0]) for c in retriever.retrieve_profiles.await_args_list], [2, 2, 1])

    async def test_short_shortlists_resolve_after_exact_coverage(self):
        for length in range(1, 5):
            with self.subTest(length=length):
                pipeline, _, resolver = self.pipeline(range(1, 11), decisions=[
                    {'candidate_id': length, 'reason': 'Wider identifying evidence supports this profile.'}])
                pipeline.prioritizer.prioritize.side_effect = None
                pipeline.prioritizer.prioritize.return_value = list(range(1, length + 1))
                result = await pipeline.ainvoke('CV')
                self.assertEqual(result['attempted_candidate_ids'], list(range(1, length + 1)))
                self.assertTrue(result['shortlist_covered'])
                resolver.resolve.assert_awaited_once()

    async def test_failed_shortlist_lookups_count_as_coverage_and_preserve_errors(self):
        pipeline, _, resolver = self.pipeline(range(1, 11), failed=[2, 5], decisions=[
            {'candidate_id': 3, 'reason': 'Successfully retrieved history supports 3.'}])
        result = await pipeline.ainvoke('CV')
        self.assertEqual(result['attempted_candidate_ids'], [1, 2, 3, 4, 5])
        self.assertEqual([p['id'] for p in resolver.resolve.await_args.args[1]], [1, 3, 4])
        self.assertEqual([e['person_id'] for e in result['errors']], [2, 5])
        self.assertTrue(result['shortlist_covered'])
        self.assertFalse(result['retrieval_complete'])

    async def test_ambiguous_shortlist_continues_with_all_profiles_until_budget(self):
        pipeline, _, _ = self.pipeline(range(1, 11))
        examined = []
        def respond(prompt):
            context = json.loads(prompt.to_messages()[1].content)
            examined.append([p['id'] for p in context['profiles']])
            return AIMessage(content=json.dumps(UNRESOLVED))
        pipeline.identity_resolver = IdentityResolver(RunnableLambda(respond))
        result = await pipeline.ainvoke('CV')
        self.assertEqual(examined, [list(range(1, 6)), list(range(1, 8)), list(range(1, 9))])
        self.assertEqual(result['identity'], 'unresolved')
        self.assertEqual(result['investigation_stop_reason'], 'budget_exhausted')
        self.assertEqual(result['attempted_candidate_ids'], list(range(1, 9)))

    async def test_prioritization_failure_preserves_original_order_resolution(self):
        pipeline, _, resolver = self.pipeline(range(1, 11), decisions=[
            {'candidate_id': 2, 'reason': 'Identifying evidence supports 2.'}])
        pipeline.prioritizer.prioritize.side_effect = RuntimeError('Prioritization failed')
        result = await pipeline.ainvoke('CV')
        self.assertEqual(result['ranked_candidate_ids'], list(range(1, 11)))
        self.assertEqual(result['shortlisted_candidate_ids'], [])
        self.assertEqual(result['attempted_candidate_ids'], [1, 2])
        self.assertEqual(result['selected_candidate_id'], 2)
        self.assertEqual(result['errors'][0]['stage'], 'prioritization')
        resolver.resolve.assert_awaited_once()

    async def test_isolated_false_claim_does_not_block_identity_after_coverage(self):
        pipeline, retriever, _ = self.pipeline(range(1, 6))
        claims = pipeline.extractor.extract.return_value
        claims['jobs'] = [{'raw': 'Analyst at Netflix 2016-present', 'company': 'Netflix',
                          'title': 'Analyst', 'seniority': None, 'dates_raw': '2016-present',
                          'start_year': 2016, 'end_year': None, 'is_current': True}]
        async def retrieve(hits):
            profiles = [profile(h['id']) for h in hits]
            for p in profiles:
                if p['id'] == 5:
                    p['experience'] = [{'company': 'Meta', 'title': 'Analyst', 'start_year': 2016}]
            return profiles, []
        retriever.retrieve_profiles.side_effect = retrieve
        def respond(prompt):
            context = json.loads(prompt.to_messages()[1].content)
            self.assertEqual(context['claims'], claims)
            self.assertEqual(len(context['profiles']), 5)
            self.assertEqual(context['profiles'][-1]['experience'][0]['company'], 'Meta')
            return AIMessage(content=json.dumps({'candidate_id': 5,
                'reason': 'Other identifying evidence supports identity despite the false employer.'}))
        pipeline.identity_resolver = IdentityResolver(RunnableLambda(respond))
        result = await pipeline.ainvoke('CV')
        self.assertEqual(result['selected_candidate_id'], 5)
        self.assertEqual(result['claims'], claims)

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
