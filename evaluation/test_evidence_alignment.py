import asyncio
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from evaluation.cases import load_cases, SAVED_INPUTS
from evaluation.evidence_alignment import correspondence_expectations, grade_correspondence
from evaluation.run_controlled import build_components, evaluate


class EvidenceEvaluationTests(unittest.IsolatedAsyncioTestCase):
    def model(self, *, invalid=False, diagnostic_unmatched=False):
        prompts = []
        def respond(prompt):
            messages = prompt.to_messages()
            context = json.loads(messages[1].content)
            prompts.append((messages[0].content, context))
            if 'requested_claim_ids' in context:
                output = {'findings': [{'claim_id': reference, 'status': 'MATCH',
                                       'evidence_paths': ['education.0.degree'], 'reason': 'MBA alias'}
                                      for reference in context['requested_claim_ids']]}
            else:
                output = {'jobs': [{'cv_index': 0, 'profile_index': 1, 'reason': 'Deloitte'},
                                    {'cv_index': 1, 'profile_index': 0, 'reason': 'Carousell'}],
                          'education': [{'cv_index': 0, 'profile_index': 0, 'reason': 'Stanford'}]}
                if 'individual skills' in messages[0].content:
                    for entries in output.values():
                        for entry in entries:
                            entry['status'] = 'ALIGNED'
                    output['skills'] = [{'cv_index': i, 'profile_index': i, 'status': 'ALIGNED', 'reason': 'Original exact name'}
                                        for i in range(3)]
                    if invalid:
                        output['skills'][0]['profile_index'] = 99
                    elif diagnostic_unmatched:
                        output['skills'][0].update(profile_index=None, status='UNMATCHED', reason='Deliberately wrong diagnostic')
            return AIMessage(content=json.dumps(output), usage_metadata={'input_tokens': 10, 'output_tokens': 5, 'total_tokens': 15})
        return RunnableLambda(respond), prompts

    async def run_pair(self, cases, llm):
        folder = Path(__file__).with_name('results')
        folder.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder) as directory:
            output = Path(directory) / 'alignment.json'
            result = await evaluate(cases, build_components(llm), output,
                                    evidence_components=build_components(llm, broad_alignment=True))
            self.assertEqual(json.loads(output.read_text()), result)
            self.assertTrue(result['complete'])
            return result

    def test_all_25_expectations_have_complete_independent_index_coverage(self):
        before = SAVED_INPUTS.read_bytes()
        cases = load_cases()
        original = deepcopy(cases)
        expected = {case['id']: correspondence_expectations(case) for case in cases}
        for case in cases:
            for collection in ('jobs', 'education', 'skills'):
                self.assertEqual(len(expected[case['id']][collection]), len(case['claims'][collection]))
        self.assertEqual(expected['skill_alias']['skills'][2], ('ALIGNED', 2))
        self.assertEqual(expected['bsc_alias']['skills'][0], ('ALIGNED', 0))
        self.assertEqual(expected['fabricated_skill']['skills'][0], ('UNMATCHED', None))
        self.assertEqual(expected['missing_skills_evidence']['skills'], [('AMBIGUOUS', None)] * 3)
        self.assertEqual(expected['ambiguous_record_details']['jobs'][0], ('AMBIGUOUS', None))
        self.assertEqual(expected['additional_fabricated_job']['jobs'][-1], ('UNMATCHED', None))
        self.assertEqual(cases, original)
        self.assertEqual(SAVED_INPUTS.read_bytes(), before)

    async def test_paired_inputs_labels_and_same_downstream_evidence(self):
        cases = load_cases(limit=1)
        cases[0]['explanation'] = 'SECRET EXPECTATION'
        before = deepcopy(cases)
        llm, prompts = self.model()
        payload = await self.run_pair(cases, llm)
        record = payload['records'][0]
        self.assertEqual(cases, before)
        self.assertNotIn('SECRET EXPECTATION', json.dumps(prompts))
        for _, context in prompts:
            self.assertNotIn('expected_label', context)
            self.assertNotIn('alignment_expectations', context)
            self.assertNotIn('expected_claims', context)
            self.assertNotIn('skill_expectations', context)
        alignment_contexts = [context for _, context in prompts if set(context) == {'claims', 'profile'}]
        self.assertEqual(alignment_contexts[0], alignment_contexts[1])
        semantic_contexts = [context for _, context in prompts if 'requested_claim_ids' in context]
        self.assertEqual(semantic_contexts[0], semantic_contexts[1])
        self.assertEqual(record['record']['result']['findings'], record['evidence']['result']['findings'])
        self.assertEqual(record['record']['result']['alignments'], record['evidence']['result']['alignments'])
        self.assertNotIn('skills', record['evidence']['result']['alignments'])
        self.assertIn('skills', record['evidence']['result']['evidence_alignments'])
        self.assertEqual(payload['summary']['alignment']['evidence']['collections']['skills']['correct'], 3)
        for name in ('record', 'evidence'):
            self.assertEqual(payload['summary'][name]['llm_calls'], 2)
            self.assertEqual(payload['summary'][name]['total_tokens'], 30)
        self.assertEqual(payload['summary']['alignment']['claim_status_differences'], [])

    async def test_wrong_skill_correspondence_never_changes_verification(self):
        llm, _ = self.model(diagnostic_unmatched=True)
        payload = await self.run_pair(load_cases(limit=1), llm)
        record = payload['records'][0]
        self.assertEqual(record['record']['result']['findings'], record['evidence']['result']['findings'])
        self.assertEqual(record['evidence']['result']['score'], 0.9)
        self.assertEqual(payload['summary']['alignment']['evidence']['collections']['skills']['correct'], 2)
        self.assertEqual(payload['summary']['alignment']['claim_status_differences'], [])

    async def test_invalid_alignment_counts_as_failure_without_hiding_denominator(self):
        llm, _ = self.model(invalid=True)
        payload = await self.run_pair(load_cases(limit=1), llm)
        broad = payload['summary']['alignment']['evidence']
        self.assertEqual(broad['invalid_output_cases'], 1)
        self.assertEqual(broad['invalid_output_frequency'], 1)
        self.assertEqual(broad['collections']['skills']['graded'], 3)
        self.assertEqual(broad['collections']['skills']['correct'], 0)
        self.assertEqual(broad['collections']['skills']['missing'], 3)
        self.assertEqual(payload['records'][0]['record']['result']['score'], 0.9)
        self.assertEqual(payload['records'][0]['evidence']['result']['score'], 0.5)
        self.assertEqual(payload['summary']['evidence']['call_stages'], {'alignment': 1})
        for difference in payload['summary']['alignment']['record_mapping_differences']:
            self.assertTrue(difference['record_mapping_present'])
            self.assertFalse(difference['evidence_mapping_present'])

    def test_ambiguous_controls_and_absence_are_graded_separately(self):
        cases = {case['id']: case for case in load_cases()}
        case = cases['missing_skills_evidence']
        result = {'evidence_alignments': {collection: [
            {'cv_index': i, 'profile_index': index, 'status': status, 'reason': 'Fixed contract result'}
            for i, (status, index) in enumerate(expectations)]
            for collection, expectations in correspondence_expectations(case).items()}}
        metrics = grade_correspondence(case, result, broad=True)['skills']
        self.assertEqual(metrics['graded'], 0)
        self.assertEqual(metrics['uncertain_expected'], 3)
        self.assertEqual(metrics['uncertain_correct'], 3)
        self.assertEqual(metrics['ambiguous'], 3)
        result['evidence_alignments']['skills'][0]['status'] = 'UNMATCHED'
        metrics = grade_correspondence(case, result, broad=True)['skills']
        self.assertEqual(metrics['uncertain_correct'], 2)
        self.assertEqual(len(metrics['failures']), 1)

    def test_alias_and_unmatched_diagnostics_use_independent_expectations(self):
        cases = {case['id']: case for case in load_cases()}
        for case_id, aliases, unmatched in (('skill_alias', 1, 0), ('bsc_alias', 1, 0), ('fabricated_skill', 0, 1)):
            case = cases[case_id]
            result = {'evidence_alignments': {collection: [
                {'cv_index': i, 'profile_index': index, 'status': status, 'reason': 'Fixed test correspondence'}
                for i, (status, index) in enumerate(expectations)]
                for collection, expectations in correspondence_expectations(case).items()}}
            metrics = grade_correspondence(case, result, broad=True)['skills']
            self.assertEqual(metrics['alias_correct'], aliases)
            self.assertEqual(metrics['alias_graded'], aliases)
            self.assertEqual(metrics['unmatched_correct'], unmatched)
            self.assertEqual(metrics['unmatched_graded'], unmatched)

    async def test_alternating_order_and_three_case_concurrency_remain_bounded(self):
        llm, _ = self.model()
        active = peak = 0
        invocation_order = {}
        async def slow(prompt):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            try:
                context = json.loads(prompt.to_messages()[1].content)
                if 'claims' in context:
                    invocation_order.setdefault(context['claims']['name']['value'], []).append(
                        'evidence' if 'individual skills' in prompt.to_messages()[0].content else 'record')
                await asyncio.sleep(0.01)
                return await llm.ainvoke(prompt)
            finally:
                active -= 1
        cases = [deepcopy(load_cases(limit=1)[0]) for _ in range(7)]
        for i, case in enumerate(cases):
            case['id'] = f'case_{i}'
            case['claims']['name'] = {'raw': f'Person {i}', 'value': f'Person {i}'}
            case['profile']['name'] = f'Person {i}'
        payload = await self.run_pair(cases, RunnableLambda(slow))
        self.assertEqual(peak, 3)
        self.assertEqual(payload['max_concurrency'], 3)
        for i in range(7):
            self.assertEqual(invocation_order[f'Person {i}'], ['record', 'evidence'] if i % 2 == 0 else ['evidence', 'record'])
