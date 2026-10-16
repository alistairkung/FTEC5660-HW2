import json
import tempfile
import unittest
import asyncio
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from evaluation.cases import CASES, SAVED_INPUTS, load_cases
from evaluation.run_controlled import build_components, evaluate, grade, aggregate
from cv_verifier.claim_comparison_validator import ClaimComparisonValidator


class ControlledEvaluationTests(unittest.IsolatedAsyncioTestCase):
    def fake_model(self, fail_name=None):
        prompts = []
        def respond(prompt):
            messages = prompt.to_messages()
            context = json.loads(messages[1].content)
            prompts.append(context)
            if fail_name is not None and context.get('claims', {}).get('name', {}).get('value') == fail_name:
                raise RuntimeError('Fake model unavailable')
            system = messages[0].content
            if system.startswith('Audit'):
                output = {'corrections': []}
            else:
                findings = []
                for reference in context['claim_catalog']:
                    parts = reference.split('.')
                    if parts[0] == 'jobs':
                        field = parts[-1]
                        path = f"experience.{[1, 0][int(parts[1])]}.{field}"
                    elif parts[0] == 'education':
                        path = f"education.0.{'end_year' if parts[-1] == 'graduation_year' else parts[-1]}"
                    else:
                        path = reference
                    findings.append({'claim_id': reference, 'status': 'MATCH', 'evidence_paths': [path],
                                     'reason': 'Fixed fake finding for harness contract test'})
                output = {'findings': findings}
            return AIMessage(content=json.dumps(output), usage_metadata={'input_tokens': 10, 'output_tokens': 5, 'total_tokens': 15})
        return RunnableLambda(respond), prompts

    async def run_cases(self, cases, model):
        folder = Path(__file__).with_name('results')
        folder.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder) as directory:
            output = Path(directory) / 'results.json'
            payload = await evaluate(cases, build_components(model), output)
            self.assertEqual(json.loads(output.read_text()), payload)
            self.assertTrue(payload['complete'])
            return payload

    def test_three_case_mutations_and_ground_truth_are_explicit_and_immutable(self):
        before = SAVED_INPUTS.read_bytes()
        specs = deepcopy(CASES)
        cases = load_cases(limit=3)
        self.assertEqual([c['expected_label'] for c in cases], [1, 0, 0])
        self.assertEqual(cases[0]['claims']['education'][0]['degree'], 'Master of Business Administration')
        self.assertEqual(cases[1]['claims']['jobs'][0]['start_year'], 2021)
        self.assertIn('2021', cases[1]['claims']['jobs'][0]['raw'])
        self.assertEqual(cases[2]['claims']['jobs'][0]['company'], 'Siemens')
        self.assertNotIn('Siemens', [job['company'] for job in cases[2]['profile']['experience']])
        self.assertEqual(cases[2]['completeness'], ['experience'])
        cases[0]['claims']['name']['value'] = 'Changed test copy'
        self.assertEqual(SAVED_INPUTS.read_bytes(), before)
        self.assertEqual(CASES, specs)
        self.assertNotEqual(load_cases(limit=1)[0]['claims']['name']['value'], 'Changed test copy')

    async def test_production_inputs_no_labels_canonical_results_and_full_persistence(self):
        cases = load_cases(limit=1)
        cases[0]['explanation'] = 'SECRET GRADING EXPLANATION'
        original = deepcopy(cases)
        model, prompts = self.fake_model()
        payload = await self.run_cases(cases, model)
        record = payload['records'][0]
        self.assertEqual(cases, original)
        self.assertNotIn('SECRET GRADING EXPLANATION', json.dumps(prompts))
        self.assertTrue(all('expected_label' not in prompt for prompt in prompts))
        comparator = next(p for p in prompts if 'claim_catalog' in p and 'claims' in p)
        self.assertEqual(comparator['claims'], cases[0]['claims'])
        result = record['production']['result']
        ClaimComparisonValidator().validate(result['findings'], cases[0]['claims'], cases[0]['profile'])
        self.assertEqual(result['score'], 0.9)
        self.assertEqual(payload['summary']['production']['llm_calls'], 2)

    async def test_three_case_slice_reports_grade_failures_without_relabelling_mutations(self):
        model, _ = self.fake_model()
        payload = await self.run_cases(load_cases(limit=3), model)
        self.assertEqual([r['case']['id'] for r in payload['records']], [c['id'] for c in CASES[:3]])
        self.assertFalse(payload['records'][1]['production']['grading']['correct'])
        self.assertFalse(payload['records'][2]['production']['grading']['correct'])
        self.assertTrue(payload['records'][2]['production']['grading']['claim_failures'])

    async def test_per_case_model_errors_do_not_stop_other_cases(self):
        original = load_cases(limit=1)[0]
        cases = [deepcopy(original), deepcopy(original)]
        cases[0]['id'] = 'failing_test_case'
        cases[0]['claims']['name']['value'] = cases[0]['claims']['name']['raw'] = 'Fail case'
        cases[0]['profile']['name'] = 'Fail case'
        model, _ = self.fake_model(fail_name='Fail case')
        payload = await self.run_cases(cases, model)
        self.assertEqual(len(payload['records']), 2)
        self.assertEqual(payload['records'][0]['production']['result']['score'], 0.5)
        self.assertEqual(payload['records'][0]['production']['result']['errors'][0]['kind'], 'model')
        self.assertEqual(payload['records'][1]['production']['result']['score'], 0.9)

    def test_unknown_labels_threshold_and_metric_denominators(self):
        case = load_cases(limit=1)[0]
        case['expected_label'] = None
        case['expected_claims'] = {'name': 'UNKNOWN', 'city': None}
        result = {'score': 0.5, 'findings': [{'claim_id': 'name', 'status': 'UNKNOWN'}], 'errors': []}
        grading = grade(case, result)
        self.assertFalse(grading['binary_eligible'])
        self.assertIsNone(grading['correct'])
        self.assertEqual(grading['prediction'], 0)
        row = {'case': case, 'production': {'result': result, 'grading': grading, 'calls': [], 'seconds': 1}}
        summary = aggregate([row])
        self.assertEqual(summary['production']['binary_cases'], 0)
        self.assertEqual(summary['production']['graded_claims'], 1)
        self.assertEqual(summary['production']['correct_claims'], 1)
        self.assertEqual(summary['production']['unknown_claims'], 1)
        case['expected_label'] = 1
        self.assertFalse(grade(case, result)['correct'])
        result['score'] = 0.9
        self.assertTrue(grade(case, result)['correct'])

    def test_final_case_set_metadata_redactions_reordering_and_ground_truth(self):
        before = SAVED_INPUTS.read_bytes()
        cases = load_cases()
        self.assertEqual(len(cases), 25)
        self.assertEqual(len({c['id'] for c in cases}), 25)
        self.assertEqual(len({json.dumps([c['claims'], c['profile']], sort_keys=True) for c in cases}), 25)
        indexed = {c['id']: c for c in cases}
        self.assertEqual(indexed['reordered_employment']['claims']['jobs'][0]['company'], 'Carousell')
        self.assertEqual(indexed['reordered_employment']['alignment_expectations']['jobs'], [0, 1])
        self.assertIsNone(indexed['missing_skills_evidence']['profile']['skills'])
        self.assertTrue(indexed['mba_alias']['profile']['skills'])
        self.assertEqual(indexed['additional_fabricated_job']['alignment_expectations']['jobs'], [1, 0, None])
        self.assertEqual(indexed['ambiguous_record_details']['expected_claims']['jobs.0.record'], 'UNKNOWN')
        self.assertEqual(sum(c['expected_label'] is None for c in cases), 2)
        for case in cases:
            catalog = ClaimComparisonValidator().claim_catalog(case['claims'])
            self.assertEqual(set(case['expected_claims']), set(catalog))
            self.assertTrue(case['explanation'])
            self.assertTrue(case['baseline_basis'])
        self.assertEqual(SAVED_INPUTS.read_bytes(), before)
        with self.assertRaises(ValueError):
            load_cases(limit=-1)

    def test_citation_grading_distinguishes_wrong_record_from_ambiguous_anchor(self):
        case = load_cases(limit=1)[0]
        result = {'score': 0.5, 'errors': [], 'findings': [{'claim_id': 'jobs.0.company', 'status': 'MATCH',
                   'evidence': [{'path': 'experience.0.company', 'value': 'Carousell'}]}]}
        grading = grade(case, result)
        self.assertEqual(len(grading['off_anchor_citations']), 1)
        case['alignment_expectations']['jobs'] = ['ambiguous', 'ambiguous']
        grading = grade(case, result)
        self.assertEqual(grading['off_anchor_citations'], [])

    def test_metric_aggregation_counts_false_predictions_contracts_tokens_and_corrections(self):
        case = load_cases(limit=1)[0]
        records = []
        for expected, production_score in ((1, 0.5), (0, 0.9), (None, 0.5)):
            specimen = deepcopy(case)
            specimen['expected_label'] = expected
            specimen['expected_claims'] = {'name': 'MATCH'}
            specimen['targets'] = {'name': 'MATCH'}
            row = {'case': specimen}
            result = {'score': production_score, 'findings': [{'claim_id': 'name', 'status': 'MATCH'}], 'errors': [],
                      'reflection': {'corrections': [{'claim_id': 'name'}]}}
            if expected == 1:
                result['errors'] = [{'stage': 'comparison', 'kind': 'contract', 'message': 'Invalid fake output'}]
            row['production'] = {'result': result, 'grading': grade(specimen, result), 'seconds': 1,
                         'calls': [{'stage': 'comparison', 'usage': {'input_tokens': 10, 'output_tokens': 5, 'total_tokens': 15}}]}
            records.append(row)
        summary = aggregate(records)
        self.assertEqual(summary['production']['binary_cases'], 2)
        self.assertEqual(summary['production']['false_valid'], 1)
        self.assertEqual(summary['production']['false_discrepant'], 1)
        self.assertEqual(summary['production']['classification_errors_with_processing_error'], 1)
        self.assertEqual(summary['production']['reflection_corrections'], 3)
        self.assertEqual(summary['production']['input_tokens'], 30)
        self.assertEqual(summary['production']['output_tokens'], 15)
        self.assertEqual(summary['production']['accuracy'], 0.0)
        self.assertEqual(summary['production']['claim_accuracy'], 1.0)
        self.assertEqual(summary['production']['target_accuracy'], 1.0)

    def test_missing_binary_label_is_excluded(self):
        case = load_cases(limit=1)[0]
        del case['expected_label']
        result = {'score': 0.5, 'findings': [], 'errors': []}
        self.assertFalse(grade(case, result)['binary_eligible'])

    async def test_timeout_isolation_and_output_path_protection(self):
        case = load_cases(limit=1)[0]
        model, _ = self.fake_model()
        async def stalled(*_):
            await asyncio.sleep(0.05)
        with patch('evaluation.run_controlled.production', side_effect=stalled), \
             patch('evaluation.run_controlled.PER_CV_TIMEOUT_SECONDS', 0.001):
            payload = await self.run_cases([case], model)
        self.assertEqual(payload['records'][0]['production']['result']['errors'][0]['kind'], 'timeout')
        with self.assertRaises(ValueError):
            await evaluate([case], build_components(model), Path('results.csv'))

    async def test_contract_failures_are_persisted_and_scoring_checkpoint_is_never_called(self):
        model = RunnableLambda(lambda _: AIMessage(content='not JSON'))
        with patch('hw2._write_checkpoint') as checkpoint:
            payload = await self.run_cases(load_cases(limit=1), model)
        self.assertEqual(payload['records'][0]['production']['result']['errors'][0]['kind'], 'contract')
        self.assertEqual(payload['records'][0]['production']['result']['score'], 0.5)
        checkpoint.assert_not_called()

    async def test_production_concurrency_never_exceeds_three(self):
        original = load_cases(limit=1)[0]
        cases = [dict(deepcopy(original), id=f'fake_case_{i}') for i in range(7)]
        active = peak = 0
        async def measured(*_):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            try:
                await asyncio.sleep(0.005)
                return {'score': 0.5, 'findings': [], 'errors': []}
            finally:
                active -= 1
        model, _ = self.fake_model()
        with patch('evaluation.run_controlled.production', side_effect=measured):
            payload = await self.run_cases(cases, model)
        self.assertEqual(peak, 3)
        self.assertEqual(len(payload['records']), 7)
