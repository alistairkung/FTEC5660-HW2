import json
import unittest
from unittest.mock import AsyncMock, Mock, call, patch
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from cv_verifier.candidate_pipeline import CandidatePipeline
from cv_verifier.claim_validator import ClaimValidator
from cv_verifier.profile_retriever import ProfileRetriever
from cv_verifier.claim_extractor import ClaimExtractor
from cv_verifier.candidate_prioritizer import CandidatePrioritizer
from cv_verifier.candidate_ranking_validator import CandidateRankingValidator
from hw2 import build_agent

CLAIMS = {'name': {'raw': 'Person', 'value': 'Person'}, 'city': {'raw': None, 'value': None},
          'jobs': [], 'education': [], 'skills': []}


def blocks(value):
    return [{'type': 'text', 'text': json.dumps(value)}]


class CandidatePipelineTests(unittest.IsolatedAsyncioTestCase):
    def pipeline(self, ranking=None, hits=None):
        extractor = AsyncMock()
        extractor.extract.return_value = CLAIMS
        retriever = AsyncMock()
        retriever.MAX_PROFILES = 2
        retriever.search.return_value = [{'id': n} for n in (1, 2, 3)] if hits is None else hits
        retriever.retrieve_profiles.return_value = ([], [{'stage': 'profile', 'message': 'failed'}])
        prioritizer = AsyncMock()
        prioritizer.prioritize.return_value = [3, 2, 1] if ranking is None else ranking
        pipeline = CandidatePipeline(extractor, ClaimValidator(), retriever,
                                     prioritizer, CandidateRankingValidator())
        return pipeline, extractor, retriever, prioritizer

    async def test_real_components_retrieve_in_ranked_order(self):
        extractor = ClaimExtractor(RunnableLambda(lambda _: AIMessage(content=json.dumps(CLAIMS))))
        prioritizer = CandidatePrioritizer(RunnableLambda(lambda _: AIMessage(content='[3, 1, 2]')))
        search, lookup = AsyncMock(), AsyncMock()
        search.name, lookup.name = 'search_linkedin_people', 'get_linkedin_profile'
        hits = [{'id': n} for n in (1, 2, 3)]
        search.ainvoke.return_value = blocks(hits)
        profiles = [{'id': n, 'name': 'Person', 'experience': [], 'education': [], 'skills': []}
                    for n in (3, 1)]
        lookup.ainvoke.side_effect = [blocks(p) for p in profiles]
        pipeline = CandidatePipeline(extractor, ClaimValidator(), ProfileRetriever([search, lookup]),
                                     prioritizer, CandidateRankingValidator())
        result = await pipeline.ainvoke('CV')
        self.assertEqual(result['claims'], CLAIMS)
        self.assertEqual(result['search_hits'], hits)
        self.assertEqual(result['ranked_candidate_ids'], [3, 1, 2])
        self.assertEqual(result['profiles'], profiles)
        self.assertEqual(result['identity'], 'unresolved')
        self.assertTrue(result['retrieval_complete'])
        self.assertTrue(result['retrieval_sampled'])
        self.assertEqual(result['errors'], [])
        lookup.ainvoke.assert_has_awaits([call({'person_id': 3}), call({'person_id': 1})])
        self.assertEqual(lookup.ainvoke.await_count, 2)

    async def test_invalid_ranking_or_model_failure_falls_back_and_keeps_partial_errors(self):
        for ranking in ([1], [1, 1, 2], [1, 2, 99], 'malformed', RuntimeError('failed')):
            with self.subTest(ranking=ranking):
                pipeline, _, retriever, prioritizer = self.pipeline(ranking=ranking)
                if isinstance(ranking, Exception):
                    prioritizer.prioritize.side_effect = ranking
                result = await pipeline.ainvoke('CV')
                self.assertEqual(result['ranked_candidate_ids'], [1, 2, 3])
                retriever.retrieve_profiles.assert_awaited_once_with([{'id': 1}, {'id': 2}])
                self.assertEqual([e['stage'] for e in result['errors']], ['prioritization', 'profile'])
                self.assertEqual(result['identity'], 'unresolved')
                self.assertFalse(result['retrieval_complete'])
                self.assertEqual(prioritizer.prioritize.await_count, 1)

    async def test_empty_search_skips_prioritization(self):
        pipeline, _, retriever, prioritizer = self.pipeline(hits=[])
        retriever.retrieve_profiles.return_value = ([], [])
        result = await pipeline.ainvoke('CV')
        prioritizer.prioritize.assert_not_awaited()
        self.assertEqual(result['ranked_candidate_ids'], [])
        self.assertEqual(result['errors'], [])
        self.assertTrue(result['retrieval_complete'])
        self.assertFalse(result['retrieval_sampled'])

    async def test_extraction_validation_and_search_failures_stop_later_stages(self):
        for stage in ('extraction', 'validation', 'search'):
            with self.subTest(stage=stage):
                pipeline, extractor, retriever, prioritizer = self.pipeline()
                if stage == 'extraction':
                    extractor.extract.side_effect = ValueError('failed')
                elif stage == 'validation':
                    extractor.extract.return_value = {}
                else:
                    retriever.search.side_effect = ValueError('failed')
                result = await pipeline.ainvoke('CV')
                self.assertEqual(result['errors'][0]['stage'], 'search' if stage == 'search' else 'extraction')
                self.assertEqual(result['claims'], CLAIMS if stage == 'search' else None)
                self.assertEqual(result['search_hits'], [])
                self.assertEqual(result['ranked_candidate_ids'], [])
                self.assertEqual(result['profiles'], [])
                self.assertEqual(result['identity'], 'unresolved')
                self.assertFalse(result['retrieval_complete'])
                self.assertFalse(result['retrieval_sampled'])
                prioritizer.prioritize.assert_not_awaited()
                retriever.retrieve_profiles.assert_not_awaited()

    def test_build_agent_disables_retries_only_for_prioritization(self):
        model = RunnableLambda(lambda _: AIMessage(content='{}'))
        with patch('langchain_deepseek.ChatDeepSeek', return_value=model) as factory:
            agent = build_agent([])
        self.assertEqual(factory.call_args_list, [
            call(model='deepseek-v4-flash', timeout=60, max_retries=2),
            call(model='deepseek-v4-flash', timeout=60, max_retries=0)])
        self.assertIsInstance(agent.prioritizer, CandidatePrioritizer)
        self.assertIsInstance(agent.ranking_validator, CandidateRankingValidator)
