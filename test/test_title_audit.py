"""Optional audit behavior through the real verifier and injected model boundary."""
import json
import unittest
from copy import deepcopy

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from cv_verifier.claim_comparator import ClaimComparator
from cv_verifier.claim_comparison_validator import ClaimComparisonValidator
from cv_verifier.claim_reflector import ClaimReflector
from cv_verifier.cv_scorer import CVScorer
from cv_verifier.cv_verifier import CVVerifier
from cv_verifier.title_equivalence_reviewer import TitleEquivalenceReviewer, REASONS
import test_title_adjudication


class TitleAuditTests(unittest.IsolatedAsyncioTestCase):
    async def run_case(self, title='Senior Analyst', supplied='Manager', grade='senior',
                       decision='MISMATCH', change=None, reflection=None, resolved=True, cv_timeout_seconds=180):
        fixture = test_title_adjudication.TitleAdjudicationTests()
        claims, profile = fixture.inputs(title, supplied, grade)
        if change:
            change(claims, profile)
        output = fixture.decisions(claims, profile)
        events = []
        def respond(prompt):
            system = prompt.to_messages()[0].content
            if system.startswith('Judge only'):
                events.append('title_review')
                if isinstance(decision, Exception):
                    raise decision
                content = (json.dumps({'status': decision, 'reason': REASONS[decision][0]})
                           if decision in REASONS else decision)
            elif system.startswith('Audit'):
                events.append('reflection')
                if isinstance(reflection, Exception):
                    raise reflection
                content = json.dumps(reflection or {'corrections': []})
            else:
                events.append('comparison')
                content = json.dumps(output)
            return AIMessage(content=content)
        model = RunnableLambda(respond)
        selected = dict(deepcopy(profile), id=7)
        identity = {'claims': claims, 'profiles': [selected], 'identity': 'resolved' if resolved else 'unresolved',
                    'selected_candidate_id': 7 if resolved else None, 'identity_reason': 'Injected selection', 'errors': []}
        verifier = CVVerifier(RunnableLambda(lambda _: deepcopy(identity)), ClaimComparator(model),
                              ClaimComparisonValidator(), CVScorer(), ClaimReflector(model),
                              title_reviewer=TitleEquivalenceReviewer(model), cv_timeout_seconds=cv_timeout_seconds)
        result = await verifier.ainvoke('CV text')
        return result, events, claims, selected

    async def test_material_substitution_is_corrected_after_reflection(self):
        result, events, claims, profile = await self.run_case()
        self.assertEqual(events, ['comparison', 'reflection', 'title_review'])
        self.assertEqual(result['score'], .1)
        title = next(f for f in result['findings'] if f['claim_id'] == 'jobs.0.title')
        self.assertEqual(title['status'], 'MISMATCH')
        self.assertIn({'path': 'experience.1.title', 'value': 'Manager'}, title['evidence'])
        ClaimComparisonValidator().validate(result['findings'], claims, profile)

    async def test_existing_director_discrepancy_skips_reflection_and_reviewer(self):
        result, events, _, _ = await self.run_case(title='Director')
        self.assertEqual(result['score'], .1)
        self.assertEqual(events, ['comparison'])

    async def test_unresolved_and_uncertain_results_skip_reviewer(self):
        result, events, _, _ = await self.run_case(resolved=False)
        self.assertEqual((result['score'], events), (.5, []))
        def missing_name(claims, profile):
            profile.pop('name')
        result, events, _, _ = await self.run_case(change=missing_name)
        self.assertEqual((result['score'], events), (.5, ['comparison']))

    async def test_exact_abbreviated_and_supported_qualifiers_need_no_call(self):
        for title, supplied, grade in [('Manager','Manager',None), ('Manager','Mgr.',None),
                                       ('Senior Manager','Manager','senior'), ('Manager','Manager','senior')]:
            with self.subTest(title=title, supplied=supplied, grade=grade):
                result, events, _, _ = await self.run_case(title, supplied, grade)
                self.assertEqual(result['score'], .9)
                self.assertNotIn('title_review', events)

    async def test_conventional_occupational_alias_survives_semantic_match(self):
        result, events, _, _ = await self.run_case('Chief Financial Officer', 'CFO', None, decision='MATCH')
        self.assertEqual(result['score'], .9)
        self.assertEqual(events[-1], 'title_review')

    async def test_unknown_malformed_timeout_and_service_failure_preserve_baseline(self):
        baseline, _, _, _ = await self.run_case(decision='MATCH')
        for decision in ('UNKNOWN', '{}', TimeoutError('deadline'), RuntimeError('service')):
            with self.subTest(decision=decision):
                result, events, _, _ = await self.run_case(decision=decision)
                self.assertEqual(result['score'], .9)
                self.assertEqual(result['findings'], baseline['findings'])
                self.assertEqual(len(result['errors']), 0 if decision == 'UNKNOWN' else 1)
                self.assertEqual(events.count('title_review'), 1)

    async def test_reflection_failure_is_preserved_and_optional_audit_can_still_correct(self):
        result, events, _, _ = await self.run_case(reflection=RuntimeError('reflector service'))
        self.assertEqual(result['score'], .1)
        self.assertEqual([error['stage'] for error in result['errors']], ['reflection'])
        self.assertEqual(events, ['comparison', 'reflection', 'title_review'])

    async def test_reflection_discrepancy_remains_and_prevents_title_audit(self):
        correction = {'corrections': [{'claim_id': 'jobs.0.title', 'status': 'MISMATCH',
                                      'evidence_paths': ['experience.1.title'], 'reason': 'Material role substitution'}]}
        result, events, _, _ = await self.run_case(reflection=correction)
        self.assertEqual(result['score'], .1)
        self.assertEqual(events, ['comparison', 'reflection'])

    async def test_duplicate_partial_and_alias_competitors_skip_review(self):
        for update in ({}, {'company':'Company A Ltd'}, {'company':'Company A Ltd', 'start_year':None},
                       {'company':'Other', 'start_year':None, 'end_year':None, 'is_current':None}):
            with self.subTest(update=update):
                def duplicate(claims, profile):
                    profile['experience'].append(dict(profile['experience'][1], **update))
                result, events, _, _ = await self.run_case(change=duplicate)
                self.assertEqual(result['score'], .9)
                self.assertNotIn('title_review', events)

    async def test_missing_corroboration_skips_review(self):
        for field in ('company', 'start_year', 'is_current'):
            with self.subTest(field=field):
                def partial(claims, profile):
                    profile['experience'][1].pop(field)
                result, events, _, _ = await self.run_case(change=partial)
                self.assertNotIn('title_review', events)

    async def test_only_canonical_target_changes_and_inputs_are_immutable(self):
        baseline, _, claims, profile = await self.run_case(decision='MATCH')
        corrected, _, after_claims, after_profile = await self.run_case()
        self.assertEqual((claims, profile), (after_claims, after_profile))
        self.assertEqual([f for f in baseline['findings'] if f['claim_id'] != 'jobs.0.title'],
                         [f for f in corrected['findings'] if f['claim_id'] != 'jobs.0.title'])
        self.assertEqual(corrected['title_review']['corrections'][0]['cv_claim'],
                         next(f['cv_claim'] for f in baseline['findings'] if f['claim_id'] == 'jobs.0.title'))

    async def test_invalid_title_correction_preserves_positive(self):
        # A false MISMATCH on an absent grade cannot supply a concrete contradiction.
        result, events, _, _ = await self.run_case('Senior Scientist', 'Scientist', None)
        self.assertEqual(result['score'], .9)
        self.assertEqual(result['errors'][-1]['stage'], 'title_review')
        self.assertEqual(result['title_review']['corrections'], [])

    async def test_one_call_budget_when_two_titles_are_eligible(self):
        def second(claims, profile):
            claims['jobs'][1].update(title='Senior Analyst', raw='Senior Analyst, Company B, 2012–2018')
            profile['experience'][0]['title'] = 'Manager'
        result, events, _, _ = await self.run_case(change=second, decision='UNKNOWN')
        self.assertEqual(events.count('title_review'), 1)
        self.assertEqual(result['title_review']['eligible_claims'], ['jobs.0.title', 'jobs.1.title'])

    async def test_exhausted_audit_budget_keeps_validated_positive(self):
        result, events, _, _ = await self.run_case(cv_timeout_seconds=1)
        self.assertEqual(result['score'], .9)
        self.assertEqual(events, ['comparison', 'reflection'])
        self.assertIn('deadline', result['title_review']['skipped'])
