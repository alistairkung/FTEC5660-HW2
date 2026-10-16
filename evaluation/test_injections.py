import asyncio
import json
import tempfile
import unittest
from collections import Counter
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from evaluation.experimental.evidence_aligner import EvidenceAlignmentValidator
from evaluation.cases import SAVED_INPUTS
from evaluation.injection_cases import INJECTIONS, load_injections
from evaluation.run_controlled import build_components, neutral_record_reasons
from evaluation.run_injections import evaluate, outcome, verify_records


def unknown_model(prompt):
    context = json.loads(prompt.to_messages()[1].content)
    return AIMessage(content=json.dumps({'findings': [
        {'claim_id': reference, 'status': 'UNKNOWN', 'evidence_paths': [], 'reason': 'Fixed fake abstention'}
        for reference in context['requested_claim_ids']]}),
        usage_metadata={'input_tokens': 10, 'output_tokens': 5, 'total_tokens': 15})


class InjectionTests(unittest.IsolatedAsyncioTestCase):
    def fake_recovery(self, case, *, force_wrong_anchor=False, override_objective=False):
        """Known replies exercise contracts only; they are not semantic evidence."""
        prompts = []
        def respond(prompt):
            context = json.loads(prompt.to_messages()[1].content)
            prompts.append(context)
            output = []
            for reference in context['requested_claim_ids']:
                parts = reference.split('.')
                expected = case['expected_claims'][reference]
                if parts[0] in ('jobs', 'education'):
                    source = 'experience' if parts[0] == 'jobs' else 'education'
                    index = case['original_alignment'][parts[0]][int(parts[1])]['profile_index']
                    field = 'end_year' if parts[-1] == 'graduation_year' else parts[-1]
                    paths = [f'{source}.{index}.{field}']
                    if field == 'title':
                        paths.append(f'{source}.{index}.seniority')
                elif parts[0] == 'skills':
                    index = case['original_alignment']['skills'][int(parts[1])]['profile_index']
                    paths = ['skills'] if index is None else [f'skills.{index}.name']
                else:
                    paths = [reference]
                allowed = context['evidence_path_contracts'][reference][
                    'allowed_match_evidence_paths' if expected == 'MATCH' else 'allowed_mismatch_evidence_paths']
                status = expected if force_wrong_anchor or set(paths) <= set(allowed) else 'UNKNOWN'
                output.append({'claim_id': reference, 'status': status,
                               'evidence_paths': paths if status != 'UNKNOWN' else [], 'reason': 'Fixed fake canonical decision'})
            if override_objective:
                output.append({'claim_id': 'jobs.0.start_year', 'status': 'MATCH',
                               'evidence_paths': ['experience.1.start_year'], 'reason': 'Forbidden override'})
            return AIMessage(content=json.dumps({'findings': output}),
                             usage_metadata={'input_tokens': 10, 'output_tokens': 5, 'total_tokens': 15})
        return RunnableLambda(respond), prompts

    async def run_cases(self, cases, model):
        folder = Path(__file__).with_name('results')
        folder.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder) as directory:
            output = Path(directory) / 'injections.json'
            result = await evaluate(cases, model, output)
            self.assertTrue(result['complete'])
            self.assertEqual(json.loads(output.read_text()), result)
            return result

    def test_fixed_injections_validate_and_do_not_mutate_sources(self):
        saved, specs = SAVED_INPUTS.read_bytes(), deepcopy(INJECTIONS)
        cases = load_injections()
        sources = {case['source_case']: (case['claims'], case['profile']) for case in cases}
        validator = EvidenceAlignmentValidator()
        self.assertEqual(len(cases), 13)
        self.assertEqual(Counter(case['injection_type'] for case in cases),
                         {'false_unmatched': 4, 'false_ambiguous': 4, 'wrong_confident': 2, 'skill_isolation': 3})
        self.assertEqual([case['injection_type'] for case in cases[:3]],
                         ['false_unmatched', 'false_ambiguous', 'wrong_confident'])
        for case in cases:
            for alignment in (case['original_alignment'], case['injected_alignment']):
                self.assertEqual(validator.validate(alignment, case['claims'], case['profile']), alignment)
            self.assertEqual((case['claims'], case['profile']), sources[case['source_case']])
            self.assertTrue(case['targets'])
            self.assertTrue(case['ground_truth_basis'])
            for change in case['changes_to_alignment']:
                self.assertNotEqual(change['before'], change['after'])
            if case['injection_type'] == 'wrong_confident':
                original = [entry['profile_index'] for entry in case['original_alignment']['jobs']]
                injected = [entry['profile_index'] for entry in case['injected_alignment']['jobs']]
                self.assertEqual(injected, original[::-1])
                self.assertEqual(len(injected), len(set(injected)))
        self.assertEqual(SAVED_INPUTS.read_bytes(), saved)
        self.assertEqual(INJECTIONS, specs)
        for bad in (-1, 0, 14, True):
            with self.assertRaises(ValueError):
                load_injections(bad)

    async def test_slice_pair_inputs_canonical_recovery_and_metadata_isolation(self):
        cases = load_injections(3)
        cases[0]['ground_truth_basis'] = 'SECRET INJECTION GROUND TRUTH'
        original = deepcopy(cases)
        model, prompts = self.fake_recovery(cases[0])
        observed = []
        async def capture(claims, profile, alignment, components):
            observed.append((deepcopy(claims), deepcopy(profile)))
            return await verify_records(claims, profile, alignment, components)
        with patch('evaluation.run_injections.verify_records', side_effect=capture):
            result = await self.run_cases(cases, model)
        self.assertEqual(observed, [(cases[0]['claims'], cases[0]['profile'])] * 6)
        self.assertEqual(cases, original)
        self.assertNotIn('SECRET INJECTION GROUND TRUTH', json.dumps(prompts))
        for context in prompts:
            for field in ('expected_claims', 'expected_label', 'injection_type', 'source_case', 'targets', 'ground_truth_basis'):
                self.assertNotIn(field, context)
            self.assertNotIn('UNMATCHED', json.dumps(context))
            self.assertNotIn('AMBIGUOUS', json.dumps(context))
        for row in result['records']:
            for name in ('control', 'injected'):
                for call in row[name]['calls']:
                    context = json.loads(call['input_messages'][1]['content'])
                    for record_context in context['record_context'].values():
                        cv_record = record_context['cv_record']
                        self.assertTrue(cv_record in row['case']['claims']['jobs'] or cv_record in row['case']['claims']['education'])
            self.assertEqual(row['case']['claims'], cases[result['records'].index(row)]['claims'])
        self.assertEqual(result['records'][0]['injected']['result']['score'], 0.9)
        self.assertEqual(result['summary']['recovery']['false_unmatched']['outcomes'], {'recovered': 4})
        self.assertEqual(result['summary']['recovery']['false_unmatched']['semantic_correspondence_recoveries'], 4)
        self.assertEqual(result['summary']['recovery']['false_ambiguous']['outcomes'], {'recovered': 4})
        self.assertEqual(result['summary']['paths']['control']['model_call_attempts'], 3)
        self.assertEqual(result['summary']['paths']['injected']['model_call_attempts'], 3)
        self.assertEqual(result['summary']['paths']['control']['total_tokens'], 45)
        self.assertEqual(result['summary']['alignment_model_calls'], 0)
        findings = result['records'][0]['injected']['result']['findings']
        company = next(finding for finding in findings if finding['claim_id'] == 'jobs.0.company')
        self.assertEqual(company['evidence'], [{'path': 'experience.1.company', 'value': 'Deloitte'}])

    async def test_wrong_confidence_preserves_incorrect_objective_findings(self):
        case = load_injections(3)[2]
        model, _ = self.fake_recovery(case)
        payload = await self.run_cases([case], model)
        row = payload['records'][0]
        result = row['injected']['result']
        final = {finding['claim_id']: finding for finding in result['findings']}
        for finding in result['deterministic_findings']:
            if finding['status'] != 'UNKNOWN':
                self.assertEqual(final[finding['claim_id']], finding)
        self.assertEqual(result['score'], 0.1)
        self.assertFalse(row['grading']['whole_cv']['injected']['classification_correct'])
        self.assertFalse(row['grading']['whole_cv']['injected']['verified_expected_discrepancy'])
        metrics = payload['summary']['recovery']['wrong_confident']
        self.assertEqual(metrics['incorrect_immutable_findings'], 4)
        self.assertEqual(metrics['outcomes']['unsafe_error'], 4)
        self.assertEqual(metrics['outcomes']['safe_abstention'], 5)
        self.assertEqual(metrics['cross_record_recovery_prohibited_claims'], 5)

    async def test_cross_record_semantic_recovery_is_rejected(self):
        case = load_injections(3)[2]
        model, _ = self.fake_recovery(case, force_wrong_anchor=True)
        payload = await self.run_cases([case], model)
        result = payload['records'][0]['injected']['result']
        self.assertEqual(result['errors'][0]['kind'], 'contract')
        self.assertIn('realign', result['errors'][0]['message'])
        self.assertEqual(result['findings'], result['deterministic_findings'])
        self.assertEqual(payload['summary']['recovery']['wrong_confident']['outcomes']['processing_failure'], 5)
        self.assertEqual(payload['summary']['recovery']['wrong_confident']['outcomes']['unsafe_error'], 4)

    async def test_correct_status_with_wrong_record_evidence_is_not_recovery(self):
        case = load_injections(3)[2]
        def respond(prompt):
            context = json.loads(prompt.to_messages()[1].content)
            decisions = []
            for reference in context['requested_claim_ids']:
                paths = context['evidence_path_contracts'][reference]['allowed_match_evidence_paths']
                decisions.append({'claim_id': reference, 'status': 'MATCH' if paths else 'UNKNOWN',
                                  'evidence_paths': paths, 'reason': 'Fake label agrees but indexed evidence may be unrelated'})
            return AIMessage(content=json.dumps({'findings': decisions}))
        payload = await self.run_cases([case], RunnableLambda(respond))
        targets = payload['records'][0]['grading']['targets']
        employer = next(target for target in targets if target['claim_id'] == 'jobs.0.company')
        self.assertEqual(employer['status_outcome'], 'recovered')
        self.assertEqual(employer['outcome'], 'unsafe_error')
        self.assertTrue(employer['wrong_record_cited'])
        self.assertFalse(employer['semantic_correspondence_recovered'])
        evidence_changes = payload['records'][0]['grading']['claim_evidence_changes']
        self.assertIn('jobs.0.title', {change['claim_id'] for change in evidence_changes})
        metrics = payload['summary']['recovery']['wrong_confident']
        self.assertEqual(metrics['correct_status_wrong_record_citations'], 4)
        self.assertEqual(metrics['outcomes']['unsafe_error'], 8)
        self.assertEqual(metrics['outcomes'].get('recovered', 0), 0)

    async def test_attempted_objective_override_does_not_replace_findings(self):
        case = load_injections(1)[0]
        model, _ = self.fake_recovery(case, override_objective=True)
        records = neutral_record_reasons(EvidenceAlignmentValidator().record_alignments(
            case['original_alignment'], case['claims'], case['profile']))
        path = await verify_records(case['claims'], case['profile'], records, build_components(model))
        self.assertEqual(path['result']['errors'][0]['kind'], 'contract')
        self.assertEqual(path['result']['findings'], path['result']['deterministic_findings'])

    def test_outcome_definitions_and_no_processing_fallback_recovery(self):
        for expected, actual, errors, label in (
            ('MATCH', 'MATCH', [], 'recovered'), ('MISMATCH', 'MISMATCH', [], 'recovered'),
            ('MATCH', 'UNKNOWN', [], 'safe_abstention'), ('MISMATCH', 'UNKNOWN', [], 'missed_discrepancy'),
            ('MATCH', 'MISMATCH', [], 'unsafe_error'), ('MISMATCH', 'MATCH', [], 'unsafe_error'),
            ('MATCH', 'UNKNOWN', [{}], 'processing_failure'), ('MISMATCH', None, [{}], 'processing_failure'),
        ):
            self.assertEqual(outcome(expected, {'status': actual} if actual else None, errors), label)

    async def test_control_failure_is_separate_from_injection_denominator(self):
        case = load_injections(3)[1]
        recovery, _ = self.fake_recovery(case)
        count = 0
        async def fail_first(prompt):
            nonlocal count
            count += 1
            if count == 1:
                raise RuntimeError('Fake control model unavailable')
            return await recovery.ainvoke(prompt)
        payload = await self.run_cases([case], RunnableLambda(fail_first))
        metrics = payload['summary']['recovery']['false_ambiguous']
        self.assertEqual(metrics['target_claims'], 4)
        self.assertEqual(metrics['control_failed_targets'], 1)
        self.assertEqual(metrics['control_correct_targets'], 3)
        self.assertEqual(metrics['outcomes'], {'recovered': 3})
        self.assertEqual(metrics['all_target_outcomes'], {'recovered': 4})
        self.assertEqual(payload['summary']['paths']['control']['processing_error_cases'], 1)

    async def test_skill_only_injections_replay_identical_tasks_without_new_inference(self):
        for case in load_injections()[10:]:
            with self.subTest(case=case['id']):
                before = deepcopy(case)
                model, prompts = self.fake_recovery(case)
                payload = await self.run_cases([case], model)
                row = payload['records'][0]
                self.assertEqual(row['control']['result'], row['injected']['result'])
                self.assertEqual(row['injected']['isolation'], {'inputs_identical': True, 'response_replayed': True, 'outcome': 'passed'})
                self.assertEqual(payload['summary']['skill_isolation'], {'passed': 1})
                self.assertEqual(row['grading']['targets'][0]['outcome'], 'isolation_control')
                self.assertEqual(row['grading']['targets'][0]['status_outcome'], 'status_agreement')
                self.assertEqual(payload['summary']['paths']['control']['model_call_attempts'], 1)
                self.assertEqual(row['injected']['calls'], [])
                self.assertEqual(len(prompts), 1)
                self.assertEqual(case, before)
                self.assertTrue(all(value['target_claims'] == 0 for value in payload['summary']['recovery'].values()))

    async def test_failed_skill_control_is_inconclusive_without_extra_model_call(self):
        case = load_injections()[10]
        count = 0
        async def fail(_):
            nonlocal count
            count += 1
            raise TimeoutError('Fake timeout')
        payload = await self.run_cases([case], RunnableLambda(fail))
        row = payload['records'][0]
        self.assertEqual(count, 1)
        self.assertEqual(row['control']['result'], row['injected']['result'])
        self.assertEqual(row['injected']['isolation']['outcome'], 'inconclusive_control_processing_failure')
        self.assertEqual(row['injected']['result']['errors'][0]['kind'], 'timeout')

    async def test_skill_timeout_before_model_invocation_remains_visible(self):
        case = load_injections()[10]
        with patch('evaluation.run_injections.PER_CV_TIMEOUT_SECONDS', 0):
            payload = await self.run_cases([case], RunnableLambda(unknown_model))
        row = payload['records'][0]
        self.assertEqual(row['control']['calls'], [])
        self.assertEqual(row['injected']['isolation']['outcome'], 'inconclusive_control_processing_failure')
        self.assertEqual(row['control']['result'], row['injected']['result'])

    async def test_timeout_missing_output_isolation_and_full_persistence(self):
        cases = load_injections(3)
        async def stalled(_):
            await asyncio.sleep(0.2)
        with patch('evaluation.run_injections.PER_CV_TIMEOUT_SECONDS', 0.03), \
             patch('hw2._write_checkpoint') as checkpoint:
            payload = await self.run_cases(cases, RunnableLambda(stalled))
        checkpoint.assert_not_called()
        self.assertEqual(len(payload['records']), 3)
        first = payload['records'][0]['injected']
        self.assertIsNone(first['calls'][0]['output'])
        self.assertEqual(first['result']['errors'][0]['kind'], 'timeout')
        self.assertEqual(first['result']['findings'], first['result']['deterministic_findings'])
        self.assertEqual(payload['summary']['recovery']['false_unmatched']['outcomes'], {'processing_failure': 4})
        self.assertEqual(payload['summary']['paths']['injected']['missing_token_usage_calls'], 3)

    async def test_invalid_json_contracts_do_not_count_as_recovery(self):
        case = load_injections(1)[0]
        payload = await self.run_cases([case], RunnableLambda(lambda _: AIMessage(content='not JSON')))
        self.assertEqual(payload['summary']['recovery']['false_unmatched']['outcomes'], {'processing_failure': 4})
        self.assertEqual(payload['summary']['paths']['injected']['contract_failure_cases'], 1)
        self.assertEqual(payload['records'][0]['injected']['result']['score'], 0.5)

    async def test_bounded_concurrency_all_cases_path_guard_and_input_immutability(self):
        active = peak = 0
        async def measured(prompt):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            try:
                await asyncio.sleep(0.005)
                return unknown_model(prompt)
            finally:
                active -= 1
        cases = load_injections()
        before = deepcopy(cases)
        payload = await self.run_cases(cases, RunnableLambda(measured))
        self.assertEqual(peak, 3)
        self.assertEqual(cases, before)
        self.assertEqual(payload['summary']['injections'], 13)
        self.assertEqual(payload['summary']['paths']['control']['model_call_attempts'], 13)
        self.assertEqual(payload['summary']['paths']['injected']['model_call_attempts'], 10)
        self.assertEqual(payload['summary']['skill_isolation'], {'passed': 3})
        self.assertEqual(sum(value['target_claims'] for value in payload['summary']['recovery'].values()), 51)
        with self.assertRaises(ValueError):
            await evaluate(cases, RunnableLambda(unknown_model), Path('results.csv'))
        with self.assertRaises(ValueError):
            await evaluate([cases[0], cases[0]], RunnableLambda(unknown_model), Path('evaluation/results/duplicate.json'))
