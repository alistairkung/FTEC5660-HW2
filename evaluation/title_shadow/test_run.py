import unittest
from langchain_core.runnables import RunnableLambda
from evaluation.title_shadow.run import review_one


class ShadowBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_unsupported_correspondence_never_calls_reviewer(self):
        def forbidden(_):
            raise AssertionError('No review is permitted without independent correspondence')
        result=await review_one({'sample_id':'unsupported','case_id':'unsupported',
                                 'binding':{'status':'NOT_EVALUATED'}},RunnableLambda(forbidden))
        self.assertEqual(result['status'],'NOT_EVALUATED')
        self.assertEqual(result['calls'],[])
