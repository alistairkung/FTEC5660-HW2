"""Prompt/data contracts only; live title judgments are evaluated separately."""
import json
import unittest
from copy import deepcopy

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from cv_verifier.claim_comparator import ClaimComparator
from cv_verifier.claim_comparison_validator import ClaimComparisonValidator
from cv_verifier.cv_scorer import CVScorer
from cv_verifier.claim_reflector import ClaimReflector
from cv_verifier.cv_verifier import CVVerifier
from test_claim_comparator import CLAIMS, PROFILE


class TitleSemanticsTests(unittest.IsolatedAsyncioTestCase):
    def inputs(self, title='Director', profile_title='Manager', grade='senior'):
        claims, profile = deepcopy(CLAIMS), deepcopy(PROFILE)
        claims['jobs'][0].update(title=title, seniority=None,
                                raw=f'{title}, Company A, 2020–Present')
        profile['experience'][1].update(title=profile_title, seniority=grade)
        return claims, profile

    def decisions(self, claims, status):
        decisions = []
        for reference in ClaimComparisonValidator().claim_catalog(claims):
            if reference.startswith('jobs.'):
                _, index, field = reference.split('.')
                path = f'experience.{1-int(index)}.{field}'
            elif reference.startswith('education.'):
                path = reference.replace('graduation_year', 'end_year')
            else:
                path = reference
            target = reference == 'jobs.0.title'
            decisions.append({'claim_id': reference, 'status': status if target else 'MATCH',
                              'evidence_paths': [] if target and status == 'UNKNOWN' else [path],
                              'reason': 'External model boundary fixture, not a semantic-model test'})
        return {'findings': decisions}

    async def test_model_receives_correspondence_and_title_support_as_separate_requirements(self):
        claims, profile = self.inputs()
        captured = []

        def respond(prompt):
            captured.extend(prompt.to_messages())
            return AIMessage(content=json.dumps(self.decisions(claims, 'MISMATCH')))

        findings = await ClaimComparator(RunnableLambda(respond)).compare(claims, profile)
        system = captured[0].content
        self.assertIn('Record correspondence is not title equivalence.', system)
        self.assertIn('Director and Manager are not automatically interchangeable', system)
        self.assertIn('genuinely ambiguous organization-specific title mappings', system)
        self.assertIn('One false date or employer does not make an otherwise supported title false.', system)
        self.assertIn('Once a corresponding record is established', system)
        context = json.loads(captured[1].content)
        self.assertEqual(context['claims']['jobs'][0]['title'], 'Director')
        self.assertEqual(context['profile']['experience'][1]['title'], 'Manager')
        self.assertEqual(context['profile']['experience'][1]['seniority'], 'senior')
        self.assertEqual(CVScorer().score(findings, identity_resolved=True), .1)

    async def test_skill_alias_matching_precedes_absence_checks(self):
        claims, profile = self.inputs('Manager')
        claims['skills'][0] = {'name': 'Microsoft Excel', 'raw': 'Microsoft Excel'}
        profile['skills'] = [{'name': 'Excel'}]
        captured = []

        def respond(prompt):
            captured.extend(prompt.to_messages())
            return AIMessage(content=json.dumps(self.decisions(claims, 'MATCH')))

        findings = await ClaimComparator(RunnableLambda(respond)).compare(claims, profile)
        self.assertIn('Match skill aliases before checking absence', captured[0].content)
        finding = next(f for f in findings if f['claim_id'] == 'skills.0.name')
        self.assertEqual(finding['status'], 'MATCH')
        self.assertEqual(finding['evidence'], [{'path': 'skills.0.name', 'value': 'Excel'}])
        self.assertEqual(CVScorer().score(findings, identity_resolved=True), .9)

    def verifier(self, claims, profile, responses, reflection=None):
        profile = {**deepcopy(profile), 'id': 7}
        identity = {'claims': deepcopy(claims), 'profiles': [profile], 'identity': 'resolved',
                    'selected_candidate_id': 7, 'identity_reason': 'Fixed supported test identity', 'errors': []}
        pending = iter(responses)
        calls = []

        def compare(prompt):
            calls.append(prompt.to_messages())
            return AIMessage(content=json.dumps(next(pending)))

        def reflect(_):
            return AIMessage(content=json.dumps({'corrections': []} if reflection is None else reflection))

        return CVVerifier(RunnableLambda(lambda _: deepcopy(identity)),
                          ClaimComparator(RunnableLambda(compare)), ClaimComparisonValidator(),
                          CVScorer(), ClaimReflector(RunnableLambda(reflect))), calls

    async def test_title_outcomes_preserve_canonical_correspondence_aliases_and_ambiguity(self):
        # Fixed external model outcomes test the production contracts, not model compliance.
        cases = [
            ('Director', 'Manager', 'senior', 'MISMATCH', .1),
            ('Manager', 'Manager', 'senior', 'MATCH', .9),
            ('Manager', 'Manager', 'junior', 'MATCH', .9),
            ('Analyst', 'Analyst', 'junior', 'MATCH', .9),
            ('Engineer', 'Engineer', 'mid', 'MATCH', .9),
            ('Senior Manager', 'Manager', 'senior', 'MATCH', .9),
            ('Manager', 'Mgr.', 'senior', 'MATCH', .9),
            ('Director', 'Director', 'senior', 'MATCH', .9),
            ('Manager', 'Director', 'senior', 'MISMATCH', .1),
            ('Associate Director', 'Senior Manager', 'senior', 'UNKNOWN', .5),
        ]
        for title, supplied, grade, status, score in cases:
            with self.subTest(title=title, supplied=supplied):
                claims, profile = self.inputs(title, supplied, grade)
                verifier, _ = self.verifier(claims, profile, [self.decisions(claims, status)])
                result = await verifier.ainvoke('Untrusted CV data')
                finding = next(f for f in result['findings'] if f['claim_id'] == 'jobs.0.title')
                self.assertEqual(result['score'], score)
                self.assertEqual(finding['status'], status)
                self.assertEqual(finding['evidence'], [] if status == 'UNKNOWN' else [
                    {'path': 'experience.1.title', 'value': supplied}])
                self.assertEqual(result['reflection']['attempted'], score == .9)
                self.assertEqual(result['errors'], [])

    async def test_explicit_grade_guard_still_corrects_a_wrong_model_match(self):
        claims, profile = self.inputs('Senior Analyst', 'Analyst', 'junior')
        verifier, _ = self.verifier(claims, profile, [self.decisions(claims, 'MATCH')])
        result = await verifier.ainvoke('CV')
        finding = next(f for f in result['findings'] if f['claim_id'] == 'jobs.0.title')
        self.assertEqual(result['score'], .1)
        self.assertEqual(finding['status'], 'MISMATCH')
        self.assertIn({'path': 'experience.1.seniority', 'value': 'junior'}, finding['evidence'])
        self.assertFalse(result['reflection']['attempted'])

    async def test_bounded_comparison_recovery_preserves_a_valid_title_discrepancy(self):
        claims, profile = self.inputs()
        first = self.decisions(claims, 'MISMATCH')
        next(f for f in first['findings'] if f['claim_id'] == 'city')['evidence_paths'] = ['missing.city']
        verifier, calls = self.verifier(claims, profile, [first, self.decisions(claims, 'MATCH')])
        result = await verifier.ainvoke('CV')
        finding = next(f for f in result['findings'] if f['claim_id'] == 'jobs.0.title')
        self.assertEqual(result['score'], .1)
        self.assertEqual(finding['status'], 'MISMATCH')
        self.assertEqual(finding['evidence'], [{'path': 'experience.1.title', 'value': 'Manager'}])
        self.assertEqual(len(calls), 2)
        self.assertFalse(result['reflection']['attempted'])

    async def test_reflection_can_correct_a_distinct_title_but_rejects_wrong_record_evidence(self):
        for title, supplied, evidence_path, expected, error_stage in [
            ('Director', 'Manager', 'experience.1.title', .1, None),
            ('Manager', 'Mgr.', 'experience.0.title', .9, 'reflection'),
        ]:
            with self.subTest(title=title, evidence_path=evidence_path):
                claims, profile = self.inputs(title, supplied)
                correction = {'corrections': [{'claim_id': 'jobs.0.title', 'status': 'MISMATCH',
                              'evidence_paths': [evidence_path], 'reason': 'A supplied different role title'}]}
                verifier, _ = self.verifier(claims, profile, [self.decisions(claims, 'MATCH')], correction)
                result = await verifier.ainvoke('CV')
                self.assertEqual(result['score'], expected)
                self.assertTrue(result['reflection']['attempted'])
                self.assertEqual([e['stage'] for e in result['errors']], [] if error_stage is None else [error_stage])

    async def test_reflector_checks_actual_title_support_not_the_comparator_rationale(self):
        claims, profile = self.inputs()
        baseline = ClaimComparisonValidator().reconstruct(self.decisions(claims, 'MATCH'), claims, profile)
        captured = []

        def respond(prompt):
            captured.extend(prompt.to_messages())
            return AIMessage(content=json.dumps({'corrections': []}))

        await ClaimReflector(RunnableLambda(respond)).reflect(claims, profile, baseline)
        self.assertIn('A baseline MATCH or its rationale is not evidence of title equivalence.', captured[0].content)
        self.assertIn('An unqualified same-base title must not be corrected', captured[0].content)
        context = json.loads(captured[1].content)
        self.assertEqual(context['claims']['jobs'][0]['title'], 'Director')
        self.assertEqual(context['profile']['experience'][1]['title'], 'Manager')

    async def test_subset_skills_do_not_excuse_a_claim_absent_from_profile_evidence(self):
        claims, profile = self.inputs('Manager')
        claims['skills'][0] = {'name': 'Rust', 'raw': 'Rust'}
        decisions = self.decisions(claims, 'MATCH')
        target = next(f for f in decisions['findings'] if f['claim_id'] == 'skills.0.name')
        target.update(status='MISMATCH', evidence_paths=['skills'], reason='Rust is absent from the complete skill list')
        captured = []

        def respond(prompt):
            captured.extend(prompt.to_messages())
            return AIMessage(content=json.dumps(decisions))

        findings = await ClaimComparator(RunnableLambda(respond)).compare(claims, profile)
        self.assertIn('Each claimed skill must still be present or semantically equivalent', captured[0].content)
        finding = next(f for f in findings if f['claim_id'] == 'skills.0.name')
        self.assertEqual(finding['status'], 'MISMATCH')
        self.assertEqual(finding['evidence'], [{'path': 'skills', 'value': profile['skills']}])
        self.assertEqual(CVScorer().score(findings, identity_resolved=True), .1)
