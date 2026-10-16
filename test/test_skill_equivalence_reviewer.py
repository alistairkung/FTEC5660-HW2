"""Offline tests at the model boundary; never call the provider."""
import json
import unittest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from cv_verifier.skill_equivalence_reviewer import SkillEquivalenceReviewer, SkillInput, REASONS


class SkillContractTests(unittest.IsolatedAsyncioTestCase):
    async def ask(self, decisions, sources=None, raw=None):
        captured = []
        def answer(prompt):
            captured.append(prompt.to_messages())
            return AIMessage(content=raw if raw is not None else json.dumps({'reviews': decisions}))
        result = await SkillEquivalenceReviewer(RunnableLambda(answer)).review_many(sources or [
            SkillInput('Microsoft PowerPoint', 'Microsoft PowerPoint', [{'name':'PowerPoint','proficiency':3}])])
        return result,captured

    def decision(self,status='MATCH',index=0):
        return {'status':status,'matched_skill_index':index,'reason':REASONS[status][0]}

    async def test_valid_indices_and_minimal_inputs(self):
        result,messages=await self.ask([self.decision()])
        self.assertEqual(result,[self.decision()])
        inputs=json.loads(messages[0][1].content)
        self.assertEqual(set(inputs[0]),{'raw_claimed_skill','structured_claimed_skill','linkedin_skills'})
        self.assertEqual(inputs[0]['linkedin_skills'],[{'name':'PowerPoint','proficiency':3}])

    async def test_bad_indices_are_not_evidence(self):
        for index in [True,False,-1,1,1.0,'0',None]:
            with self.subTest(index=index),self.assertRaises(ValueError):
                await self.ask([self.decision(index=index)])

    async def test_absence_requires_interpretable_collection_and_null_index(self):
        for skills in [None,[{}],[{'name':None}],[{'name':''}]]:
            with self.subTest(skills=skills),self.assertRaises(ValueError):
                await self.ask([self.decision('MISMATCH',None)],[SkillInput('Docker','Docker',skills)])
        result,_=await self.ask([self.decision('MISMATCH',None)],[SkillInput('Docker','Docker',[])])
        self.assertEqual(result[0]['status'],'MISMATCH')
        with self.assertRaises(ValueError):await self.ask([self.decision('MISMATCH',0)])

    async def test_unknown_has_no_index(self):
        result,_=await self.ask([self.decision('UNKNOWN',None)],[SkillInput('Docker','Docker',None)])
        self.assertEqual(result[0]['status'],'UNKNOWN')
        with self.assertRaises(ValueError):await self.ask([self.decision('UNKNOWN',0)])

    async def test_invalid_structure_and_invented_reason_rejected(self):
        for raw in ['{}','[]','{"reviews":null}','{"reviews":[],"score":0.1}',
                    '{"reviews":[],"reviews":[]}', 'not JSON']:
            with self.subTest(raw=raw),self.assertRaises(ValueError):await self.ask([],raw=raw)
        for decision in [dict(self.decision(),score=.1),dict(self.decision(),reason='Employer implies skill'),
                         dict(self.decision(),status='DISCREPANT')]:
            with self.subTest(decision=decision),self.assertRaises(ValueError):await self.ask([decision])

    async def test_batch_cannot_drop_or_add_decisions(self):
        for decisions in [[],[self.decision(),self.decision()]]:
            with self.assertRaises(ValueError):await self.ask(decisions)

    async def test_service_errors_propagate_without_decision(self):
        async def fail(_):raise TimeoutError('deadline')
        with self.assertRaises(TimeoutError):
            await SkillEquivalenceReviewer(RunnableLambda(fail)).review(SkillInput('Docker','Docker',[]))


if __name__=='__main__':unittest.main()
