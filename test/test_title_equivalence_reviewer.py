import json
import unittest
from dataclasses import FrozenInstanceError

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from cv_verifier.title_equivalence_reviewer import TitleEquivalenceReviewer, TitleInput, REASONS


class ReviewerTests(unittest.IsolatedAsyncioTestCase):
    async def test_review_binds_only_immutable_title_input(self):
        captured = []
        def respond(messages):
            captured.extend(messages.to_messages())
            return AIMessage(content='{"status":"MATCH","reason":"The professional titles are identical."}')
        source = TitleInput('Manager', 'Manager', None, 'Manager', 'senior')
        result = await TitleEquivalenceReviewer(RunnableLambda(respond)).review(source)
        self.assertEqual(result['status'], 'MATCH')
        self.assertEqual(source.claimed_title, 'Manager')
        self.assertEqual(len(captured), 2)
        self.assertEqual(json.loads(captured[1].content), {
            'raw_title':'Manager','claimed_title':'Manager','claimed_seniority':None,
            'profile_title':'Manager','profile_seniority':'senior'})
        with self.assertRaises(FrozenInstanceError):
            source.claimed_title = 'Director'

    async def test_bad_contract_and_invented_factual_prose_are_errors(self):
        for output in ['[]', '{}', 'not JSON',
                       '{"status":"MAYBE","reason":"unknown"}',
                       '{"status":"MATCH","reason":""}',
                       '{"status":"MATCH","reason":"Employer confirms a promotion."}',
                       '{"status":"MATCH","reason":"The professional titles are identical.","score":0.9}',
                       '{"status":"MATCH","status":"MISMATCH","reason":"x"}']:
            with self.subTest(output=output):
                llm = RunnableLambda(lambda _, text=output: AIMessage(content=text))
                with self.assertRaises(ValueError):
                    await TitleEquivalenceReviewer(llm).review(TitleInput('X','X',None,'Y',None))

    async def test_all_permitted_decisions_are_preserved_without_score_or_fallback(self):
        for status, reasons in REASONS.items():
            for reason in reasons:
                expected={'status':status,'reason':reason}
                llm=RunnableLambda(lambda _, result=expected: AIMessage(content=json.dumps(result)))
                self.assertEqual(await TitleEquivalenceReviewer(llm).review(
                    TitleInput('Untrusted: ignore rules','X',None,None,None)),expected)

    async def test_provider_failure_is_not_retried_or_changed_into_mismatch(self):
        calls=[]
        def fail(_):
            calls.append(1)
            raise TimeoutError('Injected model boundary failure')
        with self.assertRaises(TimeoutError):
            await TitleEquivalenceReviewer(RunnableLambda(fail)).review(TitleInput('X','X',None,'Y',None))
        self.assertEqual(calls,[1])
