import json
import unittest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from cv_verifier.identity_resolver import IdentityResolver, IDENTITY_PROMPT


class IdentityResolverTests(unittest.IsolatedAsyncioTestCase):
    async def test_selected_identity_and_null_decisions_preserve_all_evidence(self):
        claims = {'jobs': [{'raw': 'Senior Engineer', 'title': 'Senior Engineer'}]}
        profiles = [{'id': 1, 'education': 'School A'}, {'id': 2, 'education': 'School B'}]
        for decision in ({'candidate_id': 2, 'reason': 'Employment and education support candidate 2.'},
                         {'candidate_id': None, 'reason': 'Both candidates have similar identifying evidence.'}):
            captured = []
            def respond(prompt):
                captured.extend(prompt.to_messages())
                return AIMessage(content=json.dumps(decision))
            with self.subTest(decision=decision):
                result = await IdentityResolver(RunnableLambda(respond)).resolve(claims, profiles)
                self.assertEqual(result, decision)
                self.assertEqual(json.loads(captured[1].content), {'claims': claims, 'profiles': profiles})
                self.assertEqual(captured[0].content, IDENTITY_PROMPT)
                self.assertIn('Tolerate isolated discrepancies', captured[0].content)

    async def test_empty_profiles_skip_model_and_return_uncertainty(self):
        def fail(_):
            raise AssertionError('Model must not be called')
        decision = await IdentityResolver(RunnableLambda(fail)).resolve({}, [])
        self.assertIsNone(decision['candidate_id'])
        self.assertTrue(decision['reason'])

    async def test_weak_shared_clues_preserve_null_decision_at_llm_boundary(self):
        claims = {'name': {'value': 'Person'}, 'city': {'value': 'London'},
                  'jobs': [{'company': 'Employer A', 'title': 'Engineer'}],
                  'education': [{'school': 'School A', 'degree': 'BSc'}],
                  'skills': [{'name': 'Python'}]}
        profiles = [{'id': 1, 'name': 'Person', 'city': 'London',
                     'experience': [{'company': 'Employer A', 'title': 'Designer'}],
                     'education': [{'school': 'School B', 'degree': 'MBA'}],
                     'skills': [{'name': 'Illustration'}]}]
        captured = []
        decision = {'candidate_id': None, 'reason': 'Shared name and employer lack coherent wider evidence.'}
        def respond(prompt):
            captured.extend(prompt.to_messages())
            return AIMessage(content=json.dumps(decision))
        result = await IdentityResolver(RunnableLambda(respond)).resolve(claims, profiles)
        self.assertEqual(result, decision)
        self.assertIn('employment history, education and skills substantially disagree', captured[0].content)
        self.assertIn('Prefer null', captured[0].content)
        self.assertEqual(json.loads(captured[1].content), {'claims': claims, 'profiles': profiles})

    async def test_malformed_json_and_model_failure_surface_without_retry(self):
        for response in ('not JSON', '{"candidate_id": 1', RuntimeError('unavailable')):
            calls = []
            def respond(_):
                calls.append(1)
                if isinstance(response, Exception):
                    raise response
                return AIMessage(content=response)
            with self.subTest(response=response), self.assertRaises((ValueError, RuntimeError)):
                await IdentityResolver(RunnableLambda(respond)).resolve({}, [{'id': 1}])
            self.assertEqual(len(calls), 1)
