import json
import unittest

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from cv_verifier.extraction import EXTRACTION_PROMPT, build_extraction_chain


class ExtractorTests(unittest.IsolatedAsyncioTestCase):
    async def test_prompt_and_each_runtime_cv_sent_to_fake_model(self):
        captured = []
        def respond(prompt):
            captured.append(prompt.to_messages())
            return AIMessage(content='{}')
        chain = build_extraction_chain(RunnableLambda(respond))
        for cv in ('First CV', 'Second CV'):
            await chain.ainvoke({'cv': cv})
        self.assertEqual([messages[1].content for messages in captured], ['First CV', 'Second CV'])
        for messages in captured:
            self.assertEqual(messages[0].type, 'system')
            self.assertEqual(messages[0].content, EXTRACTION_PROMPT)

    async def test_parses_json_without_validating_claims(self):
        # The extractor must not perform the validator's job.
        extracted = {'name': None}
        chain = build_extraction_chain(RunnableLambda(lambda _: AIMessage(content=json.dumps(extracted))))
        self.assertEqual(await chain.ainvoke({'cv': 'CV'}), extracted)

    async def test_invalid_json_fails(self):
        chain = build_extraction_chain(RunnableLambda(lambda _: AIMessage(content='not JSON')))
        with self.assertRaises(ValueError):
            await chain.ainvoke({'cv': 'CV'})
