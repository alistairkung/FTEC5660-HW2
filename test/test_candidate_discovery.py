import json
import unittest
from copy import deepcopy
from unittest.mock import AsyncMock, call

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from cv_verifier.candidate_pipeline import CandidatePipeline
from cv_verifier.candidate_prioritizer import CandidatePrioritizer
from cv_verifier.candidate_ranking_validator import CandidateRankingValidator
from cv_verifier.claim_extractor import ClaimExtractor
from cv_verifier.claim_validator import ClaimValidator
from cv_verifier.identity_decision_validator import IdentityDecisionValidator
from cv_verifier.identity_resolver import IdentityResolver
from cv_verifier.profile_retriever import ProfileRetriever


CLAIMS = {
    'name': {'raw': 'Person', 'value': 'Person'},
    'city': {'raw': 'London', 'value': 'London', 'candidates': [{'city': 'London', 'role': 'current'}]},
    'jobs': [{'raw': 'Analyst at Example 2016-present', 'company': 'Example', 'title': 'Analyst',
              'seniority': None, 'dates_raw': '2016-present', 'start_year': 2016,
              'end_year': None, 'is_current': True}],
    'education': [{'raw': 'BSc Software at Example University 2004', 'school': 'Example University',
                   'degree': 'BSc', 'field': 'Software', 'dates_raw': '2004', 'graduation_year': 2004}],
    'skills': [{'raw': 'Docker', 'name': 'Docker'}],
}


def hits(start, industry='Software', count=20):
    return [{'id': n, 'industry': industry} for n in range(start, start + count)]


class CandidateDiscoveryTests(unittest.IsolatedAsyncioTestCase):
    def pipeline(self, responses, claims=None, shortlist=None, selected_id=None):
        claims = deepcopy(CLAIMS if claims is None else claims)
        extractor = ClaimExtractor(RunnableLambda(lambda _: AIMessage(content=json.dumps(claims))))
        search, lookup = AsyncMock(), AsyncMock()
        search.name, lookup.name = 'search_linkedin_people', 'get_linkedin_profile'
        search.ainvoke.side_effect = [response if isinstance(response, Exception) else
                                    [{'type': 'text', 'text': json.dumps(response)}] for response in responses]
        lookup.ainvoke.side_effect = lambda args: [{'type': 'text', 'text': json.dumps({
            'id': args['person_id'], 'name': 'Person', 'experience': [], 'education': [], 'skills': [],
        })}]
        ranking_inputs, identity_inputs = [], []
        def prioritize(prompt):
            context = json.loads(prompt.to_messages()[1].content)
            ranking_inputs.append(context)
            self.assertEqual(search.ainvoke.await_count, len(responses))
            proposed = shortlist if shortlist is not None else [h['id'] for h in context['search_hits'][:5]]
            return AIMessage(content=json.dumps(proposed))
        def resolve(prompt):
            identity_inputs.append(json.loads(prompt.to_messages()[1].content))
            return AIMessage(content=json.dumps({'candidate_id': selected_id,
                'reason': 'Wider identifying evidence supports the decision; industry alone is insufficient.'}))
        pipeline = CandidatePipeline(extractor, ClaimValidator(), ProfileRetriever([search, lookup]),
                                     CandidatePrioritizer(RunnableLambda(prioritize)), CandidateRankingValidator(),
                                     IdentityResolver(RunnableLambda(resolve)), IdentityDecisionValidator())
        return pipeline, search, lookup, ranking_inputs, identity_inputs

    async def test_saturation_recovers_missing_fifth_identity_and_deduplicates_before_ranking(self):
        original, broad = hits(1), hits(21, 'Finance')
        supplemental = [{'id': 3, 'industry': 'Software', 'headline': 'duplicate'},
                        {'id': 99, 'industry': 'Software'}, {'id': 99, 'headline': 'duplicate within supplement'}]
        pipeline, search, lookup, ranking, decisions = self.pipeline(
            [original, broad, supplemental], shortlist=[1, 2, 3, 4, 99], selected_id=99)
        result = await pipeline.ainvoke('CV')
        search.ainvoke.assert_has_awaits([
            call({'q': 'Person', 'location': 'London', 'limit': 20, 'fuzzy': True}),
            call({'q': 'Person', 'location': None, 'limit': 20, 'fuzzy': True}),
            call({'q': 'Person', 'location': 'London', 'industry': 'Software', 'limit': 20, 'fuzzy': True}),
        ])
        self.assertEqual(result['search_hits'], original + broad + [supplemental[1]])
        self.assertEqual(len(ranking), 1)
        self.assertEqual(ranking[0]['search_hits'], result['search_hits'])
        self.assertEqual(result['selected_candidate_id'], 99)
        self.assertEqual(result['attempted_candidate_ids'], [1, 2, 3, 4, 99])
        self.assertTrue(result['shortlist_covered'])
        self.assertEqual(len(decisions), 1)
        self.assertEqual(len(decisions[0]['profiles']), 5)
        self.assertEqual(lookup.ainvoke.await_count, 5)
        self.assertEqual([q['returned_count'] for q in result['discovery_searches']], [20, 20, 3])
        self.assertEqual(result['discovery_searches'][-1]['new_candidate_count'], 1)

    async def test_combined_twenty_results_do_not_imply_individual_saturation(self):
        pipeline, search, _, _, _ = self.pipeline([hits(1, count=10), hits(21, count=10)])
        result = await pipeline.ainvoke('CV')
        self.assertEqual(search.ainvoke.await_count, 2)
        self.assertEqual(len(result['search_hits']), 20)

    async def test_uncapped_city_does_not_constrain_diversification_of_capped_name_pool(self):
        pipeline, search, _, _, _ = self.pipeline([hits(1, count=10), hits(21), [{'id': 99}]])
        result = await pipeline.ainvoke('CV')
        self.assertEqual(search.ainvoke.await_args.args[0]['industry'], 'Software')
        self.assertIsNone(search.ainvoke.await_args.args[0]['location'])
        self.assertEqual(result['search_hits'][-1]['id'], 99)

    async def test_failed_city_search_keeps_name_only_results_and_diversifies_without_city(self):
        pipeline, search, _, _, _ = self.pipeline([RuntimeError('City search failed'), hits(21), []])
        result = await pipeline.ainvoke('CV')
        self.assertEqual(result['search_hits'], hits(21))
        self.assertEqual(search.ainvoke.await_args.args[0]['industry'], 'Software')
        self.assertIsNone(search.ainvoke.await_args.args[0]['location'])
        self.assertEqual(result['errors'][0]['scope'], 'city_filtered')

    async def test_no_usable_hint_skips_extra_search(self):
        for industry in ('Finance', None, ''):
            with self.subTest(industry=industry):
                pipeline, search, _, _, _ = self.pipeline([hits(1, industry), hits(21, industry)])
                await pipeline.ainvoke('CV')
                self.assertEqual(search.ainvoke.await_count, 2)

    async def test_hint_can_come_from_each_structured_claim_category(self):
        for category in ('employment', 'education', 'skills'):
            with self.subTest(category=category):
                claims = deepcopy(CLAIMS)
                claims['education'][0]['field'] = None
                if category == 'employment':
                    claims['jobs'][0]['title'] = 'Software Engineer'
                elif category == 'education':
                    claims['education'][0]['field'] = 'Software'
                else:
                    claims['skills'] = [{'raw': 'Software development', 'name': 'Software development'}]
                pipeline, search, _, _, _ = self.pipeline([hits(1), hits(21), []], claims=claims)
                await pipeline.ainvoke('CV')
                self.assertEqual(search.ainvoke.await_args.kwargs, {})
                self.assertEqual(search.ainvoke.await_args.args[0]['industry'], 'Software')

    async def test_most_supported_label_wins_with_search_order_tiebreak(self):
        for extra_support, expected in ((False, 'Finance'), (True, 'Software')):
            with self.subTest(extra_support=extra_support):
                claims = deepcopy(CLAIMS)
                claims['jobs'][0]['title'] = 'Finance Analyst'
                if extra_support:
                    claims['skills'] = [{'raw': 'Software testing', 'name': 'Software testing'}]
                pipeline, search, _, _, _ = self.pipeline([hits(1, 'Finance'), hits(21), []], claims=claims)
                await pipeline.ainvoke('CV')
                self.assertEqual(search.ainvoke.await_args.args[0]['industry'], expected)

    async def test_industry_vocabulary_is_observed_and_matching_uses_whole_words(self):
        for title, industry, expect_extra in (('AIA Analyst', 'AI', False),
                                             ('Marine Research Scientist', 'Marine Research', True)):
            with self.subTest(title=title):
                claims = deepcopy(CLAIMS)
                claims['education'][0]['field'] = None
                claims['jobs'][0]['title'] = title
                responses = [hits(1, industry), hits(21, industry)] + ([[]] if expect_extra else [])
                pipeline, search, _, _, _ = self.pipeline(responses, claims=claims)
                await pipeline.ainvoke('CV')
                self.assertEqual(search.ainvoke.await_count, 3 if expect_extra else 2)

    async def test_false_claims_never_remove_name_only_coverage(self):
        for field in ('employer', 'degree', 'location', 'skill'):
            with self.subTest(field=field):
                claims = deepcopy(CLAIMS)
                if field == 'employer':
                    claims['jobs'][0]['company'] = 'Wrong Company'
                elif field == 'degree':
                    claims['education'][0]['degree'] = 'PhD'
                elif field == 'location':
                    claims['city'] = {'raw': 'Jakarta', 'value': 'Jakarta',
                                      'candidates': [{'city': 'Jakarta', 'role': 'current'}]}
                else:
                    claims['skills'][0] = {'raw': 'Unrelated skill', 'name': 'Unrelated skill'}
                pipeline, search, _, ranking, _ = self.pipeline([hits(1), hits(21), []], claims=claims)
                result = await pipeline.ainvoke('CV')
                self.assertEqual(search.ainvoke.await_args_list[1].args[0]['location'], None)
                self.assertEqual(result['search_hits'], hits(1) + hits(21))
                self.assertEqual(ranking[0]['claims'], claims)

    async def test_zero_results_or_search_failure_preserve_original_pool(self):
        for response in ([], RuntimeError('Supplementary search failed')):
            with self.subTest(response=response):
                pipeline, search, lookup, _, _ = self.pipeline([hits(1), hits(21), response])
                result = await pipeline.ainvoke('CV')
                self.assertEqual(result['search_hits'], hits(1) + hits(21))
                self.assertEqual(search.ainvoke.await_count, 3)
                self.assertEqual(lookup.ainvoke.await_count, 8)
                self.assertEqual(result['identity'], 'unresolved')
                self.assertEqual(result['discovery_searches'][-1]['new_candidate_count'], 0)
                if isinstance(response, Exception):
                    self.assertEqual(result['errors'][0]['scope'], 'industry_filtered')
                    self.assertEqual(result['errors'][0]['stage'], 'search')
                else:
                    self.assertEqual(result['errors'], [])

    async def test_multiple_saturated_locations_and_supplement_still_add_only_one_search(self):
        claims = deepcopy(CLAIMS)
        claims['city'] = {'raw': 'London | Paris', 'value': None, 'candidates': [
            {'city': 'London', 'role': 'unspecified'}, {'city': 'Paris', 'role': 'unspecified'}]}
        pipeline, search, lookup, _, decisions = self.pipeline(
            [hits(1), hits(21), hits(41), hits(61)], claims=claims)
        result = await pipeline.ainvoke('CV')
        self.assertEqual(search.ainvoke.await_count, 4)
        self.assertEqual(search.ainvoke.await_args.args[0]['location'], 'London')
        self.assertEqual(lookup.ainvoke.await_count, 8)
        self.assertEqual(len(set(result['attempted_candidate_ids'])), 8)
        self.assertTrue(result['shortlist_covered'])
        self.assertEqual(result['identity'], 'unresolved')
        self.assertEqual([len(c['profiles']) for c in decisions], [5, 7, 8])

    async def test_no_location_uses_industry_hint_with_name_only_query(self):
        claims = deepcopy(CLAIMS)
        claims['city'] = {'raw': None, 'value': None, 'candidates': []}
        pipeline, search, _, _, _ = self.pipeline([hits(1), []], claims=claims)
        await pipeline.ainvoke('CV')
        self.assertEqual(search.ainvoke.await_count, 2)
        self.assertIsNone(search.ainvoke.await_args.args[0]['location'])
        self.assertEqual(search.ainvoke.await_args.args[0]['industry'], 'Software')
