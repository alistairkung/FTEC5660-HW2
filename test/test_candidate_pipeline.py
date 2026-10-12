import unittest
import json
from unittest.mock import AsyncMock, Mock, patch

from cv_verifier.pipeline import CandidatePipeline
from cv_verifier.validation import ClaimValidator
from cv_verifier.retrieval import ProfileRetriever
from hw2 import build_agent
from cv_verifier.extraction import build_extraction_chain


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_components_compose_with_fake_model_and_mcp(self):
        from langchain_core.messages import AIMessage
        from langchain_core.runnables import RunnableLambda

        claims = {"name": {"raw": "Person", "value": "Person"},
                  "city": {"raw": None, "value": None}, "jobs": [], "education": [], "skills": []}
        extractor = build_extraction_chain(RunnableLambda(
            lambda _: AIMessage(content=json.dumps(claims))))
        search = AsyncMock()
        search.name = "search_linkedin_people"
        search.ainvoke.return_value = [{"type": "text", "text": '[{"id": 1}]'}]
        lookup = AsyncMock()
        lookup.name = "get_linkedin_profile"
        profile = {"id": 1, "name": "Person", "experience": [], "education": [], "skills": []}
        lookup.ainvoke.return_value = [{"type": "text", "text": json.dumps(profile)}]
        pipeline = CandidatePipeline(extractor, ClaimValidator(), ProfileRetriever([search, lookup]))
        result = await pipeline.ainvoke("CV")
        self.assertEqual(result["claims"], claims)
        self.assertEqual(result["profiles"], [profile])
        self.assertEqual(result["identity"], "unresolved")
        self.assertEqual(result["errors"], [])
        search.ainvoke.assert_awaited_once_with(
            {"q": "Person", "location": None, "limit": 20, "fuzzy": True})
        lookup.ainvoke.assert_awaited_once_with({"person_id": 1})

    async def test_composes_components_and_preserves_partial_results(self):
        claims = {'name': {'raw': 'Person', 'value': 'Person'},
                  'city': {'raw': None, 'value': None}, 'jobs': [], 'education': [], 'skills': []}
        extractor = AsyncMock()
        extractor.ainvoke.return_value = claims
        retriever = AsyncMock()
        retrieval = {'search_hits': [{'id': 1}], 'profiles': [], 'retrieval_complete': False,
                     'retrieval_sampled': False,
                     'errors': [{'stage': 'profile', 'person_id': 1, 'message': 'failed'}]}
        retriever.retrieve.return_value = retrieval
        result = await CandidatePipeline(extractor, ClaimValidator(), retriever).ainvoke('CV')
        extractor.ainvoke.assert_awaited_once_with({'cv': 'CV'})
        retriever.retrieve.assert_awaited_once_with('Person', None)
        self.assertEqual(result, {'claims': claims, 'identity': 'unresolved', **retrieval})

    async def test_extraction_or_validation_failure_prevents_retrieval(self):
        for stage in ('extraction', 'validation'):
            with self.subTest(stage=stage):
                extractor, retriever = AsyncMock(), AsyncMock()
                validator = Mock()
                if stage == 'extraction':
                    extractor.ainvoke.side_effect = ValueError('failed')
                else:
                    validator.validate.side_effect = ValueError('failed')
                result = await CandidatePipeline(extractor, validator, retriever).ainvoke('CV')
                retriever.retrieve.assert_not_awaited()
                self.assertEqual(result['identity'], 'unresolved')
                self.assertEqual(result['errors'], [{'stage': 'extraction', 'message': 'failed'}])
                if stage == 'extraction':
                    validator.validate.assert_not_called()

    def test_build_agent_wires_real_validator_and_retriever(self):
        with patch('hw2.build_extractor', return_value=object()) as factory:
            agent = build_agent([])
        factory.assert_called_once_with('deepseek-v4-flash')
        self.assertIsInstance(agent, CandidatePipeline)
        self.assertIsInstance(agent.validator, ClaimValidator)
        self.assertIsInstance(agent.retriever, ProfileRetriever)
        self.assertIs(agent.extractor, factory.return_value)
