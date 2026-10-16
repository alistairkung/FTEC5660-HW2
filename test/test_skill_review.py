"""Canonical skill correction and failure isolation through the real verifier."""
import asyncio
from copy import deepcopy
import json
import unittest

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from cv_verifier.claim_comparator import ClaimComparator
from cv_verifier.claim_comparison_validator import ClaimComparisonValidator
from cv_verifier.claim_reflector import ClaimReflector
from cv_verifier.cv_scorer import CVScorer
from cv_verifier.cv_verifier import CVVerifier
from cv_verifier.skill_equivalence_reviewer import REASONS, SkillEquivalenceReviewer
from cv_verifier.title_equivalence_reviewer import TitleEquivalenceReviewer
import test_title_adjudication


def decision(status, index=None):
    return dict(status=status, matched_skill_index=index, reason=REASONS[status][0])


class SkillReviewTests(unittest.IsolatedAsyncioTestCase):
    async def run_case(self, claim='Docker', names=('Kubernetes','Python'), original='UNKNOWN',
                       response=None, change=None, enabled=True, deadline=180):
        fixture=test_title_adjudication.TitleAdjudicationTests()
        claims,profile=fixture.inputs('Manager','Manager',None)
        claims['skills']=[{'raw':claim,'name':claim}]
        profile['skills']=None if names is None else [{'name':n,'proficiency':3} for n in names]
        if change:change(claims,profile)
        output=fixture.decisions(claims,profile)
        skill=next(f for f in output['findings'] if f['claim_id']=='skills.0.name')
        skill.update(status=original,evidence_paths=[] if original=='UNKNOWN' else ['skills'])
        events=[]
        async def respond(prompt):
            system=prompt.to_messages()[0].content
            if system.startswith('Judge only whether each'):
                events.append('skill_review')
                if isinstance(response,Exception):raise response
                value=response if response is not None else [decision('MISMATCH')]
                return AIMessage(content=value if isinstance(value,str) else json.dumps({'reviews':value}))
            if system.startswith('Audit'):
                events.append('reflection');return AIMessage(content='{"corrections":[]}')
            if system.startswith('Judge only'):
                events.append('title_review')
                return AIMessage(content=json.dumps({'status':'UNKNOWN','reason':
                    'Relevant title or qualifier evidence is missing or ambiguous.'}))
            events.append('comparison');return AIMessage(content=json.dumps(output))
        model=RunnableLambda(respond)
        selected=dict(deepcopy(profile),id=7)
        identity={'claims':deepcopy(claims),'profiles':[selected],'identity':'resolved',
                  'selected_candidate_id':7,'identity_reason':'Injected selection','errors':[]}
        verifier=CVVerifier(RunnableLambda(lambda _:deepcopy(identity)),ClaimComparator(model),
            ClaimComparisonValidator(),CVScorer(),ClaimReflector(model),
            TitleEquivalenceReviewer(model),cv_timeout_seconds=deadline,
            skill_reviewer=SkillEquivalenceReviewer(model) if enabled else None)
        result=await verifier.ainvoke('CV text')
        ClaimComparisonValidator().validate(result['findings'],claims,selected)
        return result,events,claims,selected

    async def test_absence_repairs_unknown_and_incorrect_match(self):
        for status in ['UNKNOWN','MATCH']:
            with self.subTest(status=status):
                result,events,_,profile=await self.run_case(original=status)
                finding=next(f for f in result['findings'] if f['claim_id']=='skills.0.name')
                self.assertEqual((finding['status'],result['score']),('MISMATCH',.1))
                self.assertEqual(finding['evidence'],[{'path':'skills','value':[{'name':n} for n in ('Kubernetes','Python')]}])
                self.assertEqual(events,['comparison','skill_review'])

    async def test_aliases_match_canonical_nonzero_index_and_reflection_still_runs(self):
        for claim,names,index in [('UI/UX Design',('Python','UI/UX'),1),
                                  ('Microsoft PowerPoint',('PowerPoint',),0),
                                  ('React',('React.js',),0)]:
            with self.subTest(claim=claim):
                result,events,_,profile=await self.run_case(claim,names,response=[decision('MATCH',index)])
                finding=next(f for f in result['findings'] if f['claim_id']=='skills.0.name')
                self.assertEqual((finding['status'],result['score']),('MATCH',.9))
                self.assertEqual(finding['evidence'],[{'path':f'skills.{index}.name','value':names[index]}])
                self.assertEqual(events,['comparison','skill_review','reflection'])

    async def test_missing_and_empty_are_distinct(self):
        result,events,_,_=await self.run_case(names=None)
        self.assertEqual(result['score'],.5);self.assertNotIn('skill_review',events)
        result,_,_,_=await self.run_case(names=())
        self.assertEqual(result['score'],.1)
        finding=next(f for f in result['findings'] if f['claim_id']=='skills.0.name')
        self.assertEqual(finding['evidence'],[{'path':'skills','value':[]}])

    async def test_unknown_errors_and_invalid_output_preserve_findings(self):
        baseline,_,_,_=await self.run_case(enabled=False)
        for response in [[decision('UNKNOWN')],TimeoutError('deadline'),RuntimeError('service'),
                         '{}',[decision('MATCH',True)],[decision('MATCH',99)]]:
            with self.subTest(response=response):
                result,_,_,_=await self.run_case(response=response)
                self.assertEqual(result['score'],baseline['score'])
                self.assertEqual(result['findings'],baseline['findings'])

    async def test_unrelated_claims_and_existing_discrepancies_are_preserved(self):
        for original in ['UNKNOWN','MATCH','MISMATCH']:
            with self.subTest(original=original):
                before,_,claims,profile=await self.run_case(original=original,enabled=False)
                after,events,after_claims,after_profile=await self.run_case(original=original)
                self.assertEqual((claims,profile),(after_claims,after_profile))
                self.assertEqual([f for f in before['findings'] if not f['claim_id'].startswith('skills.')],
                                 [f for f in after['findings'] if not f['claim_id'].startswith('skills.')])
                if original=='MISMATCH':
                    self.assertEqual(before['findings'],after['findings']);self.assertNotIn('skill_review',events)

    async def test_no_skill_correction_can_replace_grounded_contradiction(self):
        before,_,claims,profile=await self.run_case(original='MISMATCH',enabled=False)
        with self.assertRaises(ValueError):
            ClaimComparisonValidator().apply_skill_reviews(['skills.0.name'],[decision('MATCH',0)],
                before['findings'],claims,profile)

    async def test_exact_match_skips_call_but_unknown_exact_cannot_be_rejected(self):
        result,events,_,_=await self.run_case('Python',('Python',),original='MATCH')
        self.assertEqual(result['score'],.9);self.assertNotIn('skill_review',events)
        result,_,_,_=await self.run_case('Python',('Python',))
        self.assertEqual(result['score'],.5)
        self.assertEqual(result['errors'][-1]['stage'],'skill_review')

    async def test_time_reservation_skips_optional_review(self):
        result,events,_,_=await self.run_case(deadline=120)
        self.assertEqual(result['score'],.5);self.assertNotIn('skill_review',events)
        self.assertIn('skipped',result['skill_review'])

    async def test_actual_timeout_cancels_call_and_preserves_previous_findings(self):
        baseline,_,claims,profile=await self.run_case(enabled=False)
        cancelled=[]
        async def hang(sources):
            try:await asyncio.sleep(1)
            finally:cancelled.append(True)
        reviewer=type('SlowReviewer',(),{'review_many':staticmethod(hang)})()
        verifier=CVVerifier(None,None,ClaimComparisonValidator(),CVScorer(),None,skill_reviewer=reviewer)
        result=deepcopy(baseline)
        await verifier._review_skills(result,profile,asyncio.get_running_loop().time()+121.01)
        self.assertEqual(result['findings'],baseline['findings'])
        self.assertEqual(cancelled,[True]);self.assertEqual(result['errors'][-1]['message'],'TimeoutError')

    async def test_all_eligible_skills_use_one_atomic_batch(self):
        def two(claims,profile):claims['skills'].append({'raw':'React','name':'React'})
        result,events,_,_=await self.run_case(change=two,response=[decision('MISMATCH'),decision('MISMATCH')])
        self.assertEqual(events.count('skill_review'),1)
        self.assertEqual(len(result['skill_review']['decisions']),2)
        before,_,_,_=await self.run_case(change=two,enabled=False)
        after,_,_,_=await self.run_case(change=two,response=[decision('MISMATCH'),decision('MATCH',99)])
        self.assertEqual(before['findings'],after['findings'])


if __name__=='__main__':unittest.main()
