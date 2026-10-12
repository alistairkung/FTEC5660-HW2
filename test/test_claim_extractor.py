import json
import unittest

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from cv_verifier.claim_extractor import EXTRACTION_PROMPT, ClaimExtractor
from cv_verifier.claim_validator import ClaimValidator


class ClaimExtractorTests(unittest.IsolatedAsyncioTestCase):
    async def test_location_contract_with_fake_model(self):
        cases = [
            ('Beijing, China | Hong Kong', {'raw': 'Beijing, China | Hong Kong', 'value': None, 'candidates': [
                {'city': 'Beijing', 'role': 'unspecified'}, {'city': 'Hong Kong', 'role': 'unspecified'}]}),
            ('Current city Beijing; hometown Hong Kong', {'raw': 'Current city Beijing; hometown Hong Kong',
             'value': 'Beijing', 'candidates': [{'city': 'Beijing', 'role': 'current'},
                                               {'city': 'Hong Kong', 'role': 'hometown'}]}),
            ('London, UK', {'raw': 'London, UK', 'value': 'London',
                           'candidates': [{'city': 'London', 'role': 'current'}]}),
            ('NYC / New York, USA | HK / Hong Kong', {'raw': 'NYC / New York, USA | HK / Hong Kong',
             'value': None, 'candidates': [{'city': 'New York', 'role': 'unspecified'},
                                         {'city': 'Hong Kong', 'role': 'unspecified'}]}),
            ('Education: The University of Hong Kong', {'raw': None, 'value': None, 'candidates': []}),
            ('No location specified', {'raw': None, 'value': None, 'candidates': []}),
        ]
        for text, location in cases:
            claims = {'name': {'raw': 'Person', 'value': 'Person'}, 'city': location,
                      'jobs': [], 'education': [], 'skills': []}
            captured = []
            def respond(prompt):
                captured.extend(prompt.to_messages())
                return AIMessage(content=json.dumps(claims))
            with self.subTest(text=text):
                output = await ClaimExtractor(RunnableLambda(respond)).extract(text)
                self.assertEqual(ClaimValidator().validate(output)['city'], location)
                self.assertEqual(captured[1].content, text)
                self.assertIn('Never infer current residence from list order', captured[0].content)
                self.assertIn('university names', captured[0].content)

    async def test_prompt_and_each_runtime_cv_sent_to_fake_model(self):
        captured = []
        def respond(prompt):
            captured.append(prompt.to_messages())
            return AIMessage(content='{}')
        chain = ClaimExtractor(RunnableLambda(respond))
        for cv in ('First CV', 'Second CV'):
            await chain.extract(cv)
        self.assertEqual([messages[1].content for messages in captured], ['First CV', 'Second CV'])
        for messages in captured:
            self.assertEqual(messages[0].type, 'system')
            self.assertEqual(messages[0].content, EXTRACTION_PROMPT)

    async def test_parses_json_without_validating_claims(self):
        # The extractor must not perform the validator's job.
        extracted = {'name': None}
        chain = ClaimExtractor(RunnableLambda(lambda _: AIMessage(content=json.dumps(extracted))))
        self.assertEqual(await chain.extract('CV'), extracted)

    async def test_invalid_json_fails(self):
        chain = ClaimExtractor(RunnableLambda(lambda _: AIMessage(content='not JSON')))
        with self.assertRaises(ValueError):
            await chain.extract('CV')
