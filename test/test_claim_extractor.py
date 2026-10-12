import json
import unittest

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from cv_verifier.claim_extractor import EXTRACTION_PROMPT, ClaimExtractor


class ClaimExtractorTests(unittest.IsolatedAsyncioTestCase):
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
