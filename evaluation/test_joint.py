import json
import asyncio
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from cv_verifier.cv_scorer import CVScorer
from evaluation.cases import load_cases
from evaluation.joint_comparison import JointComparisonValidator, JointRecordComparator
from evaluation.run_joint import evaluate, grounding


def fixture_output(case):
    """Independent fixture correspondence/statuses for contract tests, never inference."""
    records = {}
    for collection, indexes in case['alignment_expectations'].items():
        records[collection] = [
            {'cv_index': i, 'profile_index': j if type(j) is int else None,
             'state': 'ALIGNED' if type(j) is int else 'ABSENT' if j is None else 'UNRESOLVED',
             'reason': 'Fixed correspondence from the saved test fixture.'}
            for i, j in enumerate(indexes)]
    findings = []
    for reference, status in case['expected_claims'].items():
        parts = reference.split('.')
        if status == 'UNKNOWN':
            paths = []
        elif parts[0] in ('jobs', 'education'):
            source = 'experience' if parts[0] == 'jobs' else 'education'
            index = case['alignment_expectations'][parts[0]][int(parts[1])]
            field = 'end_year' if parts[2] == 'graduation_year' else parts[2]
            paths = [source] if index is None else [f'{source}.{index}.{field}']
            if parts[2] == 'title' and type(index) is int:
                paths.append(f'{source}.{index}.seniority')
        elif parts[0] == 'skills':
            paths = ['skills'] if status == 'MISMATCH' else [reference]
        else:
            paths = [reference]
        findings.append({'claim_id': reference, 'status': status, 'evidence_paths': paths,
                         'reason': 'Fixed independently specified fixture decision.'})
    return {'records': records, 'findings': findings}


class JointContractTests(unittest.TestCase):
    def setUp(self):
        self.cases = {case['id']: case for case in load_cases()}
        self.validator = JointComparisonValidator()

    def test_correct_whole_record_with_false_year_keeps_grounded_discrepancy(self):
        case = self.cases['start_year_contradiction']
        output = fixture_output(case)
        original = deepcopy((output, case))
        result = self.validator.validate(output, case['claims'], case['profile'])
        finding = next(f for f in result['findings'] if f['claim_id'] == 'jobs.0.start_year')
        self.assertEqual(finding['status'], 'MISMATCH')
        self.assertEqual(finding['evidence'], [{'path': 'experience.1.start_year', 'value': 2020}])
        self.assertEqual(result['records']['jobs'][0]['profile_index'], 1)
        self.assertEqual(CVScorer().score(result['findings'], identity_resolved=True), 0.1)
        self.assertEqual((output, case), original)

    def test_conflicting_indexes_rejected_even_when_claim_status_is_plausible(self):
        case = self.cases['mba_alias']
        output = fixture_output(case)
        finding = next(f for f in output['findings'] if f['claim_id'] == 'jobs.0.title')
        finding['evidence_paths'] = ['experience.1.title', 'experience.0.seniority']
        with self.assertRaisesRegex(ValueError, 'corresponding record'):
            self.validator.validate(output, case['claims'], case['profile'])

    def test_incorrect_typed_year_status_rejected_without_repair(self):
        case = self.cases['start_year_contradiction']
        output = fixture_output(case)
        next(f for f in output['findings'] if f['claim_id'] == 'jobs.0.start_year')['status'] = 'MATCH'
        with self.assertRaisesRegex(ValueError, 'typed'):
            self.validator.validate(output, case['claims'], case['profile'])

    def test_explicit_seniority_contradiction_rejects_match_but_accepts_mismatch(self):
        case = self.cases['explicit_seniority_contradiction']
        output = fixture_output(case)
        self.assertEqual(CVScorer().score(self.validator.validate(output, case['claims'], case['profile'])['findings'],
                                         identity_resolved=True), 0.1)
        next(f for f in output['findings'] if f['claim_id'] == 'jobs.0.title')['status'] = 'MATCH'
        with self.assertRaisesRegex(ValueError, 'seniority'):
            self.validator.validate(output, case['claims'], case['profile'])

    def test_qualified_profile_title_can_supply_seniority_without_duplicate_citation(self):
        case = deepcopy(self.cases['mba_alias'])
        case['profile']['experience'][1]['title'] = 'Senior Manager'
        output = fixture_output(case)
        next(f for f in output['findings'] if f['claim_id'] == 'jobs.0.title')['evidence_paths'] = ['experience.1.title']
        result = self.validator.validate(output, case['claims'], case['profile'])
        self.assertEqual(next(f for f in result['findings'] if f['claim_id'] == 'jobs.0.title')['status'], 'MATCH')

    def test_fixture_aliases_reordering_absence_and_uncertainty(self):
        for name in ('mba_alias', 'employer_alias', 'school_alias', 'reordered_employment',
                     'unqualified_title_extra_grade', 'additional_fabricated_job',
                     'ambiguous_record_details', 'missing_skills_evidence'):
            case = self.cases[name]
            with self.subTest(case=name):
                result = self.validator.validate(fixture_output(case), case['claims'], case['profile'])
                self.assertEqual({f['claim_id']: f['status'] for f in result['findings']}, case['expected_claims'])

    def test_unresolved_cannot_issue_definite_findings_from_arbitrary_records(self):
        case = self.cases['mba_alias']
        output = fixture_output(case)
        output['records']['jobs'][0].update(profile_index=None, state='UNRESOLVED')
        with self.assertRaisesRegex(ValueError, 'only UNKNOWN'):
            self.validator.validate(output, case['claims'], case['profile'])
        for finding in output['findings']:
            if finding['claim_id'].startswith('jobs.0.'):
                finding.update(status='UNKNOWN', evidence_paths=[])
        result = self.validator.validate(output, case['claims'], case['profile'])
        self.assertEqual(CVScorer().score(result['findings'], identity_resolved=True), 0.5)

    def test_absence_needs_collection_and_cannot_borrow_record_fields(self):
        case = self.cases['additional_fabricated_job']
        output = fixture_output(case)
        next(f for f in output['findings'] if f['claim_id'] == 'jobs.2.company')['evidence_paths'] = ['experience.1.company']
        with self.assertRaises(ValueError):
            self.validator.validate(output, case['claims'], case['profile'])
        unavailable = deepcopy(case)
        unavailable['profile']['experience'] = None
        with self.assertRaises(ValueError):
            self.validator.validate(fixture_output(unavailable), unavailable['claims'], unavailable['profile'])

    def test_irrelevant_missing_null_and_fabricated_evidence_rejected(self):
        case = self.cases['mba_alias']
        for paths in (['name'], [], ['experience.99.start_year'], ['experience.1.end_year']):
            output = fixture_output(case)
            next(f for f in output['findings'] if f['claim_id'] == 'jobs.0.start_year')['evidence_paths'] = paths
            with self.subTest(paths=paths), self.assertRaises(ValueError):
                self.validator.validate(output, case['claims'], case['profile'])
        output = fixture_output(case)
        output['findings'][0]['evidence'] = [{'path': 'name', 'value': 'Invented'}]
        with self.assertRaises(ValueError):
            self.validator.validate(output, case['claims'], case['profile'])

    def test_record_contract_bounds_coverage_states_and_reasons(self):
        case = self.cases['mba_alias']
        for change in ({'cv_index': True}, {'profile_index': True}, {'profile_index': 99},
                       {'state': 'CONFIDENT'}, {'reason': ''}):
            output = fixture_output(case)
            output['records']['jobs'][0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.validator.validate(output, case['claims'], case['profile'])
        output = fixture_output(case)
        output['records']['jobs'].pop()
        with self.assertRaises(ValueError):
            self.validator.validate(output, case['claims'], case['profile'])

    def test_malformed_envelope_and_incomplete_or_duplicate_claims_rejected(self):
        case = self.cases['mba_alias']
        for malformed in (None, {}, {'records': {}, 'findings': [], 'score': 0.9}):
            with self.assertRaises(ValueError):
                self.validator.validate(malformed, case['claims'], case['profile'])
        for operation in ('missing', 'duplicate'):
            output = fixture_output(case)
            if operation == 'missing':
                output['findings'].pop()
            else:
                output['findings'].append(deepcopy(output['findings'][0]))
            with self.assertRaises(ValueError):
                self.validator.validate(output, case['claims'], case['profile'])

    def test_overlapping_cv_records_may_share_a_profile_index(self):
        case = deepcopy(self.cases['mba_alias'])
        case['claims']['jobs'].append(deepcopy(case['claims']['jobs'][0]))
        case['alignment_expectations']['jobs'].append(1)
        for reference, status in list(case['expected_claims'].items()):
            if reference.startswith('jobs.0.'):
                case['expected_claims'][reference.replace('jobs.0.', 'jobs.2.')] = status
        result = self.validator.validate(fixture_output(case), case['claims'], case['profile'])
        self.assertEqual([r['profile_index'] for r in result['records']['jobs']], [1, 0, 1])

    def test_coherent_but_wrong_correspondence_can_pass_internal_consistency(self):
        case = self.cases['mba_alias']
        output = fixture_output(case)
        for entry in output['records']['jobs']:
            entry['profile_index'] = 1 - entry['profile_index']
        for finding in output['findings']:
            if finding['claim_id'].startswith('jobs.'):
                index = int(finding['claim_id'].split('.')[1])
                old, new = [1, 0][index], [0, 1][index]
                finding['evidence_paths'] = [p.replace(f'experience.{old}.', f'experience.{new}.') for p in finding['evidence_paths']]
                finding['status'] = 'MISMATCH'
                if finding['claim_id'] == 'jobs.1.end_year':
                    finding.update(status='UNKNOWN', evidence_paths=[])
        result = self.validator.validate(output, case['claims'], case['profile'])
        self.assertEqual(CVScorer().score(result['findings'], identity_resolved=True), 0.1)
        self.assertEqual(case['expected_label'], 1)

    def test_unknown_is_not_mechanically_promoted_from_equal_typed_evidence(self):
        case = self.cases['mba_alias']
        output = fixture_output(case)
        next(f for f in output['findings'] if f['claim_id'] == 'jobs.0.start_year').update(status='UNKNOWN', evidence_paths=[])
        result = self.validator.validate(output, case['claims'], case['profile'])
        self.assertEqual(next(f for f in result['findings'] if f['claim_id'] == 'jobs.0.start_year')['status'], 'UNKNOWN')

    def test_unqualified_title_cannot_be_false_merely_for_extra_profile_grade(self):
        case = self.cases['unqualified_title_extra_grade']
        output = fixture_output(case)
        next(f for f in output['findings'] if f['claim_id'] == 'jobs.0.title')['status'] = 'MISMATCH'
        with self.assertRaisesRegex(ValueError, 'unqualified'):
            self.validator.validate(output, case['claims'], case['profile'])

    def test_definite_title_match_requires_role_and_claimed_grade_evidence(self):
        case = self.cases['mba_alias']
        output = fixture_output(case)
        next(f for f in output['findings'] if f['claim_id'] == 'jobs.0.title')['evidence_paths'] = ['experience.1.seniority']
        with self.assertRaises(ValueError):
            self.validator.validate(output, case['claims'], case['profile'])
        missing = deepcopy(case)
        missing['profile']['experience'][1]['seniority'] = None
        output = fixture_output(missing)
        next(f for f in output['findings'] if f['claim_id'] == 'jobs.0.title')['evidence_paths'] = ['experience.1.title']
        with self.assertRaisesRegex(ValueError, 'grade'):
            self.validator.validate(output, missing['claims'], missing['profile'])

    def test_existing_whole_skills_evidence_is_relevant_for_positive_matching(self):
        case = self.cases['mba_alias']
        output = fixture_output(case)
        for finding in output['findings']:
            if finding['claim_id'].startswith('skills.'):
                finding['evidence_paths'] = ['skills']
        result = self.validator.validate(output, case['claims'], case['profile'])
        self.assertTrue(all(f['status'] == 'MATCH' for f in result['findings'] if f['claim_id'].startswith('skills.')))

    def test_unqualified_title_and_separate_seniority_have_independent_citations(self):
        case = self.cases['title_plus_seniority_equivalence']
        output = fixture_output(case)
        next(f for f in output['findings'] if f['claim_id'] == 'jobs.0.title')['evidence_paths'] = ['experience.1.title']
        result = self.validator.validate(output, case['claims'], case['profile'])
        self.assertEqual(next(f for f in result['findings'] if f['claim_id'] == 'jobs.0.title')['status'], 'MATCH')
        self.assertEqual(next(f for f in result['findings'] if f['claim_id'] == 'jobs.0.seniority')['evidence'],
                         [{'path': 'experience.1.seniority', 'value': 'senior'}])


class JointComparatorTests(unittest.IsolatedAsyncioTestCase):
    async def run_cases(self, cases, model):
        folder = Path(__file__).with_name('results')
        folder.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder) as directory:
            output = Path(directory) / 'pair.json'
            payload = await evaluate(cases, model, output)
            self.assertEqual(json.loads(output.read_text()), payload)
            return payload

    async def test_one_inference_identical_claim_profile_evidence_and_canonical_values(self):
        case = load_cases(1)[0]
        prompts = []
        def model(prompt):
            prompts.append(json.loads(prompt.to_messages()[1].content))
            return AIMessage(content=json.dumps(fixture_output(case)))
        original = deepcopy(case)
        result = await JointRecordComparator(RunnableLambda(model)).compare(case['claims'], case['profile'])
        self.assertEqual(len(prompts), 1)
        self.assertEqual(set(prompts[0]), {'claims', 'claim_catalog', 'profile'})
        self.assertEqual(prompts[0]['claims'], case['claims'])
        self.assertEqual(CVScorer().score(result['findings'], identity_resolved=True), 0.9)
        self.assertEqual(case, original)

    async def test_paired_runner_keeps_inputs_equal_labels_outside_prompts_and_counts_calls(self):
        case = load_cases(1)[0]
        case['explanation'] = 'SECRET GRADING ONLY'
        seen = []
        def model(prompt):
            system, context = prompt.to_messages()[0].content, json.loads(prompt.to_messages()[1].content)
            seen.append(context)
            output = ({'corrections': []} if system.startswith('Audit') else fixture_output(case))
            if 'For EVERY CV job' not in system and not system.startswith('Audit'):
                output = {'findings': output['findings']}
            return AIMessage(content=json.dumps(output), usage_metadata={'input_tokens': 10, 'output_tokens': 5, 'total_tokens': 15})
        folder = Path(__file__).with_name('results')
        folder.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder) as directory:
            result = await evaluate([case], RunnableLambda(model), Path(directory) / 'pair.json')
        self.assertTrue(result['complete'])
        self.assertNotIn('SECRET GRADING ONLY', json.dumps(seen))
        contexts = [c for c in seen if 'claim_catalog' in c and 'findings' not in c]
        self.assertEqual(contexts[0], contexts[1])
        self.assertEqual(result['summary']['paths']['production']['llm_calls'], 2)
        self.assertEqual(result['summary']['paths']['joint']['llm_calls'], 1)
        self.assertEqual(result['summary']['paths']['joint']['grounded_correct_claims'], len(case['expected_claims']))

    async def test_contract_failure_is_not_verified_discrepancy_and_does_not_retry(self):
        case = load_cases(2)[1]
        def model(prompt):
            return AIMessage(content='not JSON' if 'For EVERY CV job' in prompt.to_messages()[0].content else
                             json.dumps({'findings': fixture_output(case)['findings']}))
        payload = await self.run_cases([case], RunnableLambda(model))
        result = payload['records'][0]['joint']
        self.assertEqual(len(result['calls']), 1)
        self.assertEqual(result['result']['score'], 0.5)
        self.assertEqual(result['result']['errors'][0]['kind'], 'contract')
        self.assertFalse(result['grounding']['verified_expected_discrepancy'])
        self.assertTrue(result['grading']['correct'])

    async def test_valid_ambiguity_is_separate_from_model_failure_and_other_cases_finish(self):
        cases = [next(c for c in load_cases() if c['id'] == 'ambiguous_record_details'), load_cases(1)[0]]
        def model(prompt):
            system = prompt.to_messages()[0].content
            context = json.loads(prompt.to_messages()[1].content)
            case = cases[0] if context['claims']['jobs'][0]['company'] is None else cases[1]
            if system.startswith('Audit'):
                return AIMessage(content='{"corrections": []}')
            if case is cases[1] and 'For EVERY CV job' in system:
                raise RuntimeError('Injected transport failure')
            output = fixture_output(case)
            return AIMessage(content=json.dumps(output if 'For EVERY CV job' in system else {'findings': output['findings']}))
        payload = await self.run_cases(cases, RunnableLambda(model))
        valid, failed = payload['records']
        self.assertEqual(valid['joint']['result']['errors'], [])
        self.assertEqual(valid['joint']['result']['score'], 0.5)
        self.assertEqual(failed['joint']['result']['errors'][0]['kind'], 'model')
        self.assertEqual(len(failed['joint']['calls']), 1)
        self.assertEqual(failed['production']['result']['score'], 0.9)

    async def test_runner_keeps_three_pairs_in_flight_and_restricts_artifact_path(self):
        base = load_cases(1)[0]
        cases = [dict(deepcopy(base), id=f'bounded_{i}') for i in range(5)]
        active = maximum = 0
        async def model(prompt):
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            await asyncio.sleep(0.01)
            active -= 1
            system = prompt.to_messages()[0].content
            output = {'corrections': []} if system.startswith('Audit') else fixture_output(base)
            if 'For EVERY CV job' not in system and not system.startswith('Audit'):
                output = {'findings': output['findings']}
            return AIMessage(content=json.dumps(output))
        await self.run_cases(cases, RunnableLambda(model))
        self.assertEqual(maximum, 3)
        with self.assertRaises(ValueError):
            await evaluate(cases, RunnableLambda(model), Path('/tmp/disallowed_joint.json'))

    def test_grading_does_not_accept_wrong_record_status_agreement_as_grounded(self):
        case = load_cases(1)[0]
        accepted = JointComparisonValidator().validate(fixture_output(case), case['claims'], case['profile'])
        accepted['errors'] = []
        next(f for f in accepted['findings'] if f['claim_id'] == 'jobs.0.company')['evidence'] = [
            {'path': 'experience.0.company', 'value': 'Carousell'}]
        result = next(c for c in grounding(case, accepted)['claims'] if c['claim_id'] == 'jobs.0.company')
        self.assertTrue(result['wrong_record_cited'])
        self.assertEqual(result['outcome'], 'unsafe_definite')


if __name__ == '__main__':
    unittest.main()
