import json
import unittest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from cv_verifier.candidate_prioritizer import CandidatePrioritizer


class CandidatePrioritizerTests(unittest.IsolatedAsyncioTestCase):
    async def test_supplies_all_claims_and_search_metadata_and_preserves_order(self):
        claims = {'jobs': [{'raw': 'Senior Manager', 'title': 'Senior Manager'}]}
        hits = [{'id': 1, 'industry': 'Legal', 'headline': 'Lawyer', 'years_experience': 0},
                {'id': 2, 'industry': 'Logistics', 'location': 'London', 'extra': 'preserved'}]
        captured = []
        def respond(prompt):
            captured.extend(prompt.to_messages())
            return AIMessage(content='[2]')
        result = await CandidatePrioritizer(RunnableLambda(respond)).prioritize(claims, hits)
        self.assertEqual(result, [2])
        self.assertEqual(json.loads(captured[1].content), {'claims': claims, 'search_hits': hits})
        self.assertIn('untrusted clues', captured[0].content)
        self.assertIn('up to five', captured[0].content)
        self.assertIn('you do not need to rank every candidate', captured[0].content)

    async def test_empty_candidates_skip_model(self):
        def fail(_):
            raise AssertionError('Model must not be called')
        self.assertEqual(await CandidatePrioritizer(RunnableLambda(fail)).prioritize({}, []), [])

    async def test_malformed_output_or_model_failure_surfaces_without_retry(self):
        for response in ('not JSON', '[1', '```json\n[1]\n```', RuntimeError('unavailable')):
            calls = []
            def respond(_):
                calls.append(1)
                if isinstance(response, Exception):
                    raise response
                return AIMessage(content=response)
            with self.subTest(response=response), self.assertRaises((ValueError, RuntimeError)):
                await CandidatePrioritizer(RunnableLambda(respond)).prioritize({}, [{'id': 1}])
            self.assertEqual(len(calls), 1)
