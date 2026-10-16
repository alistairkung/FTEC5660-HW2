"""Behavior at the actual model, reconstruction and verifier boundaries."""
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


class TitleAdjudicationTests(unittest.IsolatedAsyncioTestCase):
    def inputs(self, title='Director', supplied='Manager', grade='senior'):
        claims, profile = deepcopy(CLAIMS), deepcopy(PROFILE)
        claims['jobs'][0].update(title=title, seniority=None,
                                raw=f'{title}, Company A, 2020–Present')
        profile['experience'][1].update(title=supplied, seniority=grade)
        return claims, profile

    def decisions(self, claims, profile, overrides=None):
        contract = ClaimComparisonValidator()
        permitted = contract.permitted_evidence_paths(profile)
        findings = []
        for reference in contract.claim_catalog(claims):
            if reference.startswith('jobs.'):
                _, index, field = reference.split('.')
                path = f'experience.{1-int(index)}.{field}'
            else:
                path = reference.replace('graduation_year', 'end_year')
            finding = {'claim_id': reference, 'status': 'MATCH' if path in permitted else 'UNKNOWN',
                       'evidence_paths': [path] if path in permitted else [],
                       'reason': 'Injected external model judgment'}
            finding.update((overrides or {}).get(reference, {}))
            findings.append(finding)
        return {'findings': findings}

    async def compare(self, claims, profile, decisions=None):
        output = decisions if decisions is not None else self.decisions(claims, profile)
        model = RunnableLambda(lambda _: AIMessage(content=json.dumps(output)))
        return await ClaimComparator(model).compare(claims, profile)

    async def test_wrong_director_match_becomes_a_canonical_discrepancy(self):
        claims, profile = self.inputs()
        findings = await self.compare(claims, profile)
        title = next(f for f in findings if f['claim_id'] == 'jobs.0.title')
        self.assertEqual(title['status'], 'MISMATCH')
        self.assertEqual(CVScorer().score(findings, identity_resolved=True), .1)
        self.assertIn({'path': 'experience.1.title', 'value': 'Manager'}, title['evidence'])
        self.assertIn({'path': 'experience.1.company', 'value': 'Company A'}, title['evidence'])
        self.assertIn({'path': 'experience.1.start_year', 'value': 2020}, title['evidence'])
        self.assertIn({'path': 'experience.1.is_current', 'value': True}, title['evidence'])
        ClaimComparisonValidator().validate(findings, claims, profile)

    async def test_exact_abbreviated_and_same_base_titles_are_preserved(self):
        for title, supplied, grade, expected in [
            ('Manager', 'Manager', 'senior', 'MATCH'),
            ('Manager', 'Mgr.', 'senior', 'MATCH'),
            ('Senior Manager', 'Manager', 'senior', 'MATCH'),
            ('Director', 'Director', 'senior', 'MATCH'),
            ('Manager', 'Manager', 'junior', 'MATCH'),
            ('Senior Analyst', 'Analyst', 'junior', 'MISMATCH'),
        ]:
            with self.subTest(title=title, supplied=supplied, grade=grade):
                claims, profile = self.inputs(title, supplied, grade)
                findings = await self.compare(claims, profile)
                self.assertEqual(next(f['status'] for f in findings
                                      if f['claim_id'] == 'jobs.0.title'), expected)

    async def test_unknown_and_out_of_policy_judgments_are_not_rewritten(self):
        for title, supplied in [('Manager', 'Director'), ('Senior Analyst', 'Manager'),
                                ('Associate Director', 'Senior Manager'),
                                ('Head of Sales', 'Sales Manager'), ('Senior Director', 'Manager'),
                                ('Director', 'Senior Manager'), ('Director', 'Manager')]:
            for status in ('MATCH', 'UNKNOWN', 'MISMATCH'):
                if (title, supplied, status) == ('Director', 'Manager', 'MATCH'):
                    continue
                with self.subTest(title=title, supplied=supplied, status=status):
                    claims, profile = self.inputs(title, supplied)
                    output = self.decisions(claims, profile, {'jobs.0.title': {
                        'status': status, 'evidence_paths': [] if status == 'UNKNOWN' else ['experience.1.title']}})
                    findings = await self.compare(claims, profile, output)
                    self.assertEqual(next(f['status'] for f in findings
                                          if f['claim_id'] == 'jobs.0.title'), status)

    async def test_incomplete_duplicate_aliased_or_uncertain_correspondence_abstains(self):
        cases = []
        for field in ('start_year', 'is_current'):
            claims, profile = self.inputs()
            claims['jobs'][0][field] = None
            cases.append((f'missing claim {field}', claims, profile, {}))
        for field in ('company', 'title', 'start_year', 'end_year', 'is_current'):
            claims, profile = self.inputs()
            profile['experience'][1].pop(field)
            cases.append((f'missing profile {field}', claims, profile, {}))
        for partial in (False, True):
            claims, profile = self.inputs()
            duplicate = deepcopy(profile['experience'][1])
            duplicate['title'] = 'Director'  # Uniqueness must not depend on the disputed role.
            if partial:
                duplicate.pop('start_year')
            profile['experience'].append(duplicate)
            cases.append((f'duplicate partial={partial}', claims, profile, {}))
        claims, profile = self.inputs()
        claims['jobs'][0]['company'] = 'Company A Ltd'
        cases.append(('employer alias remains semantic', claims, profile, {}))
        claims, profile = self.inputs()
        cases.append(('wrong record citation', claims, profile, {
            'jobs.0.start_year': {'evidence_paths': ['experience.0.start_year']}}))
        claims, profile = self.inputs()
        cases.append(('uncertain independent evidence', claims, profile, {
            'jobs.0.is_current': {'status': 'UNKNOWN', 'evidence_paths': []}}))
        claims, profile = self.inputs()
        claims['jobs'][0].update(is_current=False, end_year=None)
        profile['experience'][1]['is_current'] = False
        cases.append(('unfinished historical period', claims, profile, {}))
        for name, claims, profile, changes in cases:
            with self.subTest(case=name):
                findings = await self.compare(claims, profile, self.decisions(claims, profile, changes))
                self.assertNotEqual(next(f['status'] for f in findings
                                         if f['claim_id'] == 'jobs.0.title'), 'MISMATCH')

    async def test_only_target_changes_and_false_employer_does_not_invalidate_title(self):
        claims, profile = self.inputs()
        original_inputs = deepcopy((claims, profile))
        baseline = await self.compare(claims, profile, self.decisions(claims, profile, {
            'jobs.0.title': {'status': 'UNKNOWN', 'evidence_paths': []}}))
        corrected = await self.compare(claims, profile)
        self.assertEqual([f for f in baseline if f['claim_id'] != 'jobs.0.title'],
                         [f for f in corrected if f['claim_id'] != 'jobs.0.title'])
        self.assertEqual((claims, profile), original_inputs)
        claims, profile = self.inputs('Senior Manager')
        claims['jobs'][0]['company'] = 'Fabricated Employer'
        findings = await self.compare(claims, profile, self.decisions(claims, profile, {
            'jobs.0.company': {'status': 'MISMATCH'}}))
        self.assertEqual(next(f['status'] for f in findings if f['claim_id'] == 'jobs.0.title'), 'MATCH')
        self.assertEqual(next(f['status'] for f in findings if f['claim_id'] == 'jobs.0.company'), 'MISMATCH')
        claims, profile = self.inputs()
        claims['jobs'][0].update(is_current=False, end_year=2023)
        profile['experience'][1].update(is_current=False, end_year=2023)
        findings = await self.compare(claims, profile)
        self.assertEqual(next(f['status'] for f in findings if f['claim_id'] == 'jobs.0.title'), 'MISMATCH')

    async def test_invalid_citations_are_rejected_and_bounded_retry_still_works(self):
        claims, profile = self.inputs()
        invalid = self.decisions(claims, profile, {'jobs.0.title': {'evidence_paths': ['invented.title']}})
        with self.assertRaises(ValueError):
            await self.compare(claims, profile, invalid)
        broken = self.decisions(claims, profile, {'city': {'evidence_paths': ['missing.city']}})
        pending = iter([broken, self.decisions(claims, profile)])
        model = RunnableLambda(lambda _: AIMessage(content=json.dumps(next(pending))))
        findings = await ClaimComparator(model).compare(claims, profile)
        self.assertEqual(CVScorer().score(findings, identity_resolved=True), .1)

    def verifier(self, claims, profile, resolved=True, reflection_error=False):
        selected = dict(deepcopy(profile), id=7)
        identity = {'claims': deepcopy(claims), 'profiles': [selected],
                    'identity': 'resolved' if resolved else 'unresolved',
                    'selected_candidate_id': 7 if resolved else None,
                    'identity_reason': 'Injected upstream selection', 'errors': []}
        output = self.decisions(claims, profile)
        def respond(prompt):
            if prompt.to_messages()[0].content.startswith('Audit'):
                if reflection_error:
                    raise RuntimeError('Injected external model failure')
                return AIMessage(content='{"corrections": []}')
            return AIMessage(content=json.dumps(output))
        llm = RunnableLambda(respond)
        return CVVerifier(RunnableLambda(lambda _: deepcopy(identity)), ClaimComparator(llm),
                          ClaimComparisonValidator(), CVScorer(), ClaimReflector(llm))

    async def test_reflection_failure_preserves_validated_positive_baseline(self):
        claims, profile = self.inputs('Manager')
        baseline = await self.compare(claims, profile)
        result = await self.verifier(claims, profile, reflection_error=True).ainvoke('CV text')
        self.assertEqual(result['score'], .9)
        self.assertEqual(result['findings'], baseline)
        self.assertEqual([e['stage'] for e in result['errors']], ['reflection'])

    async def test_verifier_skips_unresolved_identity_and_reflection_on_corrected_negative(self):
        claims, profile = self.inputs()
        unresolved = await self.verifier(claims, profile, resolved=False).ainvoke('CV text')
        self.assertEqual(unresolved['score'], .5)
        self.assertEqual(unresolved['findings'], [])
        result = await self.verifier(claims, profile, reflection_error=True).ainvoke('CV text')
        self.assertEqual(result['score'], .1)
        self.assertFalse(result['reflection']['attempted'])
        self.assertEqual(result['errors'], [])
