import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from cv_verifier.deterministic_claim_verifier import DeterministicClaimVerifier
from cv_verifier.evidence_aligner import EvidenceAlignmentValidator
from cv_verifier.record_aligner import RecordAligner
from evaluation.correspondence_gate import CorrespondencePlausibilityGate
from evaluation.gate_cases import load_gate_cases, probe_candidate_rules
from evaluation.run_controlled import build_components
from evaluation.run_gate import evaluate, grounding
from evaluation.test_injections import unknown_model


class GateTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.cases = {case['id']: case for case in load_gate_cases()}
        self.gate = CorrespondencePlausibilityGate()

    def apply(self, case):
        return self.gate.apply(case['claims'], case['profile'], case['input_alignment'])

    def test_saved_correct_correspondences_preserved(self):
        for case in self.cases.values():
            if case['group'] == 'saved_control':
                with self.subTest(case=case['id']):
                    self.assertEqual(self.apply(case)['alignments'], case['input_alignment'])

    def test_gate_reports_all_three_decisions_and_keeps_weak_evidence_unadjudicated(self):
        plausible = self.apply(self.cases['start_year_contradiction'])
        self.assertEqual(plausible['diagnostics'][0]['decision'], 'PLAUSIBLE')
        self.assertEqual(plausible['alignments']['jobs'], self.cases['start_year_contradiction']['input_alignment']['jobs'])

        sparse = deepcopy(self.cases['mba_alias'])
        for field in CorrespondencePlausibilityGate.FIELDS['jobs']:
            if field != 'company':
                sparse['claims']['jobs'][0][field] = None
        undecided = self.apply(sparse)
        self.assertEqual(undecided['diagnostics'][0]['decision'], 'UNADJUDICATED')
        self.assertIn('not independently verified', undecided['diagnostics'][0]['reason'])
        self.assertEqual(undecided['alignments']['jobs'][0], sparse['input_alignment']['jobs'][0])

        rejected = self.apply(self.cases['wrong_jobs_logistics'])
        self.assertEqual(rejected['diagnostics'][0]['decision'], 'IMPLAUSIBLE')
        self.assertIsNone(rejected['alignments']['jobs'][0]['profile_index'])

    def test_offline_candidate_rules_expose_required_control_costs(self):
        rules = probe_candidate_rules()['summary']
        self.assertIn('start_year_contradiction:jobs.0', rules['all_fields_agree']['rejected_ids'])
        self.assertIn('probe_sparse_company_only:jobs.0', rules['match_count_2']['rejected_ids'])
        self.assertIn('school_alias:education.0', rules['match_count_3']['rejected_ids'])
        self.assertIn('probe_duplicate_context:jobs.0', rules['higher_match_count_alternative']['rejected_ids'])
        self.assertEqual(rules['identity_timeline_conflict']['correct_rejected'], 0)
        self.assertEqual(rules['identity_timeline_conflict']['incorrect_inputs'], 7)
        self.assertEqual(rules['identity_timeline_conflict']['incorrect_caught'], 6)
        self.assertIn('false_employer_matches_wrong_record:jobs.0',
                      rules['identity_timeline_conflict']['missed_ids'])

    def test_both_wrong_swaps_null_every_wrong_job_without_realigning(self):
        for name in ('wrong_jobs_logistics', 'wrong_jobs_engineering'):
            case = self.cases[name]
            report = self.apply(case)
            self.assertEqual([entry['profile_index'] for entry in report['alignments']['jobs']], [None, None])
            self.assertEqual(report['alignments']['education'], case['input_alignment']['education'])
            for decision in report['diagnostics'][:2]:
                self.assertEqual(decision['decision'], 'IMPLAUSIBLE')
                self.assertEqual(decision['action'], 'null_conflict')
                self.assertTrue(decision['alternative_indices'])
                self.assertTrue(decision['evidence'])
            output = {collection: [dict(entry, status='AMBIGUOUS' if entry['profile_index'] is None else 'ALIGNED')
                                   for entry in entries]
                      for collection, entries in report['alignments'].items()}
            output['skills'] = [{'cv_index': index, 'profile_index': None, 'status': 'UNMATCHED',
                                 'reason': 'Skill evidence does not affect record validation.'}
                                for index in range(len(case['claims']['skills']))]
            self.assertEqual(EvidenceAlignmentValidator().validate(output, case['claims'], case['profile']), output)

    def test_wrong_education_rejected(self):
        case = self.cases['wrong_education_augmented']
        report = self.apply(case)
        self.assertIsNone(report['alignments']['education'][0]['profile_index'])
        self.assertEqual(report['alignments']['jobs'], case['input_alignment']['jobs'])

    def test_false_dates_flags_and_seniority_retain_objective_mismatch_and_score(self):
        from cv_verifier.cv_scorer import CVScorer
        required = {'start_year_contradiction': 'jobs.0.start_year',
                    'end_year_contradiction': 'jobs.1.end_year',
                    'false_current_employment': 'jobs.1.is_current',
                    'explicit_seniority_contradiction': 'jobs.0.title',
                    'graduation_year_contradiction': 'education.0.graduation_year',
                    'alias_plus_false_start': 'jobs.0.start_year',
                    'false_employer_plus_false_start': 'jobs.0.start_year'}
        verifier = DeterministicClaimVerifier()
        for name, target in required.items():
            with self.subTest(case=name):
                case = self.cases[name]
                before = verifier.verify(case['claims'], case['profile'], case['input_alignment'])
                after = verifier.verify(case['claims'], case['profile'], self.apply(case)['alignments'])
                self.assertEqual(before, after)
                self.assertEqual(next(f['status'] for f in after if f['claim_id'] == target), 'MISMATCH')
                self.assertEqual(CVScorer().score(after, identity_resolved=True), 0.1)

    def test_false_organizations_and_degree_preserved(self):
        for name in ('fabricated_employer', 'fabricated_school', 'fabricated_degree'):
            self.assertEqual(self.apply(self.cases[name])['alignments'], self.cases[name]['input_alignment'])

    def test_aliases_reordering_and_unqualified_title_preserved(self):
        for name in ('employer_alias', 'school_alias', 'reordered_employment', 'unqualified_title_extra_grade', 'bsc_alias'):
            self.assertEqual(self.apply(self.cases[name])['alignments'], self.cases[name]['input_alignment'])

    def test_all_existing_nulls_preserved_never_promoted(self):
        for case in self.cases.values():
            report = self.apply(case)
            for collection in ('jobs', 'education'):
                for original, after in zip(case['input_alignment'][collection], report['alignments'][collection]):
                    if original['profile_index'] is None:
                        self.assertEqual(after, original)

    def test_insufficient_record_evidence_is_unadjudicated_and_preserved(self):
        for name in ('unparsed_confident', 'unusable_profile_record'):
            report = self.apply(self.cases[name])
            self.assertEqual(report['alignments']['jobs'][0], self.cases[name]['input_alignment']['jobs'][0])
            self.assertEqual(report['diagnostics'][0]['decision'], 'UNADJUDICATED')
            self.assertEqual(report['diagnostics'][0]['action'], 'unadjudicated')
        case = self.cases['missing_experience_collection']
        report = self.apply(case)
        self.assertEqual(report['alignments'], case['input_alignment'])
        self.assertTrue(all(item['decision'] == 'UNADJUDICATED' for item in report['diagnostics'][:2]))

    def test_sparse_company_anchor_preserved_no_match_count_threshold(self):
        case = deepcopy(self.cases['mba_alias'])
        for key in CorrespondencePlausibilityGate.FIELDS['jobs']:
            if key != 'company':
                case['claims']['jobs'][0][key] = None
        self.assertEqual(self.apply(case)['alignments'], case['input_alignment'])

    def test_seniority_is_not_identity_veto_and_more_than_one_false_claim_survives(self):
        case = deepcopy(self.cases['fabricated_employer'])
        case['claims']['jobs'][0]['title'] = 'Junior Manager'
        self.assertEqual(self.apply(case)['alignments'], case['input_alignment'])

    def test_profile_order_independent_and_malformed_scalar_not_evidence(self):
        case = deepcopy(self.cases['wrong_jobs_logistics'])
        case['profile']['experience'].reverse()
        for entry in case['input_alignment']['jobs']:
            entry['profile_index'] = 1 - entry['profile_index']
        self.assertEqual([e['profile_index'] for e in self.apply(case)['alignments']['jobs']], [None, None])
        sparse = deepcopy(self.cases['mba_alias'])
        sparse['profile']['experience'][1]['start_year'] = True
        report = self.apply(sparse)
        self.assertEqual(report['alignments'], sparse['input_alignment'])
        evidence = report['diagnostics'][0]['evidence'][1]
        self.assertEqual(evidence['relations']['start_year'], 'unavailable')

    def test_immutable_inputs_independent_outputs_and_explicit_provenance(self):
        case = self.cases['wrong_jobs_logistics']
        before = deepcopy(case)
        report = self.apply(case)
        self.assertEqual(case, before)
        field = report['diagnostics'][0]['evidence'][1]['fields'][0]
        self.assertEqual((field['cv_path'], field['profile_path']), ('jobs.0.company', 'experience.1.company'))
        report['alignments']['jobs'].clear()
        field['cv_value'] = 'mutated'
        self.assertEqual(case, before)

    def test_suspicion_without_supported_alternative_is_unadjudicated(self):
        case = deepcopy(self.cases['wrong_jobs_logistics'])
        # Remove the competing record's role anchor, while leaving the assigned conflict.
        case['profile']['experience'][1]['title'] = None
        report = self.apply(case)
        self.assertEqual(report['alignments']['jobs'][0], case['input_alignment']['jobs'][0])
        self.assertEqual(report['diagnostics'][0]['action'], 'unadjudicated')

    async def test_unadjudicated_false_employer_match_can_still_produce_unsafe_findings(self):
        from cv_verifier.cv_scorer import CVScorer
        case = self.cases['false_employer_matches_wrong_record']
        report = self.apply(case)
        self.assertEqual(report['diagnostics'][0]['decision'], 'UNADJUDICATED')
        self.assertEqual(report['alignments']['jobs'][0], case['input_alignment']['jobs'][0])
        findings = DeterministicClaimVerifier().verify(case['claims'], case['profile'], report['alignments'])
        assessment = grounding(case, {'result': {'findings': findings, 'errors': []}})
        self.assertEqual(CVScorer().score(findings, identity_resolved=True), 0.1)
        self.assertFalse(assessment['verified_expected_discrepancy'])
        self.assertGreaterEqual(assessment['counts']['unsafe'], 2)
        payload = await self.run_cases([case])
        unadjudicated = payload['summary']['decision_outcomes']['UNADJUDICATED']['mapped']
        self.assertEqual(unadjudicated['true_discrepancies_preserved'], 0)
        self.assertGreaterEqual(unadjudicated['unsafe_definite_judgments'], 2)
        self.assertEqual(unadjudicated['unverified_expected_discrepancies'], 1)
        self.assertEqual(unadjudicated['semantic_recoveries'], 0)

    def test_same_employer_wrong_dates_and_tied_alternatives_are_not_certified(self):
        case = deepcopy(self.cases['wrong_jobs_engineering'])
        case['claims']['jobs'][0]['company'] = 'Innovate HK Ltd'
        report = self.apply(case)
        self.assertEqual(report['alignments']['jobs'][0], case['input_alignment']['jobs'][0])
        self.assertEqual(report['diagnostics'][0]['decision'], 'UNADJUDICATED')
        self.assertIn('not independently verified', report['diagnostics'][0]['reason'])
        duplicate = deepcopy(self.cases['wrong_jobs_logistics'])
        duplicate['profile']['experience'].append(deepcopy(duplicate['profile']['experience'][1]))
        report = self.apply(duplicate)
        self.assertEqual(report['diagnostics'][0]['alternative_indices'], [1, 2])
        self.assertIsNone(report['alignments']['jobs'][0]['profile_index'])

    def test_invalid_alignment_contract_rejected_without_inference(self):
        case = deepcopy(self.cases['mba_alias'])
        case['input_alignment']['jobs'][0]['profile_index'] = True
        with self.assertRaises(ValueError):
            self.apply(case)
        case = deepcopy(self.cases['mba_alias'])
        case['input_alignment']['jobs'][0]['profile_index'] = 0
        with self.assertRaises(ValueError):
            self.apply(case)
        with patch('evaluation.correspondence_gate.RecordAligner.__init__', side_effect=AssertionError('No model allowed')):
            self.apply(self.cases['mba_alias'])

    async def run_cases(self, cases, model=RunnableLambda(unknown_model), **kwargs):
        folder = Path(__file__).with_name('results')
        folder.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder) as directory:
            output = Path(directory) / 'gate.json'
            payload = await evaluate(cases, model, output, **kwargs)
            self.assertEqual(json.loads(output.read_text()), payload)
            self.assertTrue(payload['complete'])
            return payload

    async def test_exact_replay_for_unchanged_paths_no_extra_inference(self):
        case = self.cases['start_year_contradiction']
        payload = await self.run_cases([case])
        row = payload['records'][0]
        self.assertEqual(row['ungated']['result'], row['gated']['result'])
        self.assertEqual(len(row['ungated']['calls']), 1)
        self.assertEqual(row['gated']['calls'], [])
        self.assertEqual(row['gated']['isolation']['outcome'], 'passed')
        self.assertTrue(row['gated']['grounding']['verified_expected_discrepancy'])

    async def test_pair_claims_profiles_and_metadata_are_isolated(self):
        case = deepcopy(self.cases['wrong_jobs_logistics'])
        case['ground_truth_basis'] = 'SECRET GRADING DATA'
        original, seen = deepcopy(case), []
        def model(prompt):
            seen.append(json.loads(prompt.to_messages()[1].content))
            return unknown_model(prompt)
        payload = await self.run_cases([case], RunnableLambda(model))
        self.assertEqual(case, original)
        self.assertEqual(len(seen), 2)
        self.assertNotIn('SECRET GRADING DATA', json.dumps(seen))
        self.assertNotIn('null_conflict', json.dumps(seen))
        for context in seen:
            for key in ('expected_label', 'expected_claims', 'group', 'injection_type', 'targets'):
                self.assertNotIn(key, context)
        self.assertEqual(payload['summary']['gate']['incorrect_caught'], 2)
        self.assertEqual(payload['summary']['gate']['correct_rejected'], 0)
        self.assertEqual(payload['summary']['gate_model_calls'], 0)
        self.assertEqual(payload['summary']['mcp_calls'], 0)

    async def test_gate_failure_isolated_and_never_successful_abstention(self):
        base = self.gate
        class FailOne:
            def apply(self, claims, profile, alignment):
                if claims['jobs'][0]['start_year'] == 2021:
                    raise ValueError('Injected gate failure')
                return base.apply(claims, profile, alignment)
        payload = await self.run_cases([self.cases['start_year_contradiction'], self.cases['mba_alias']], gate=FailOne())
        failure, valid = payload['records']
        self.assertEqual(failure['gated']['result']['score'], 0.5)
        self.assertTrue(failure['gated']['result']['errors'])
        self.assertEqual(set(item['outcome'] for item in failure['gated']['grounding']['claims']), {'processing_failure'})
        self.assertEqual(valid['gated']['isolation']['outcome'], 'passed')
        self.assertEqual(payload['summary']['gate']['gate_error_cases'], 1)

    async def test_model_failure_remains_failure_no_retry_or_replay_inference(self):
        async def fail(_):
            raise RuntimeError('Model failure')
        payload = await self.run_cases([self.cases['mba_alias']], RunnableLambda(fail))
        row = payload['records'][0]
        self.assertEqual(len(row['ungated']['calls']), 1)
        self.assertEqual(row['gated']['calls'], [])
        self.assertEqual(row['gated']['isolation']['outcome'], 'inconclusive_control_processing_failure')

    async def test_wrong_citation_is_unsafe_even_when_status_agrees(self):
        case = self.cases['wrong_jobs_logistics']
        path = await self.run_cases([case])
        result = deepcopy(path['records'][0]['gated'])
        finding = next(f for f in result['result']['findings'] if f['claim_id'] == 'jobs.0.company')
        finding.update(status='MATCH', evidence=[{'path': 'experience.0.company', 'value': 'Carousell'}])
        target = next(item for item in grounding(case, result)['claims'] if item['claim_id'] == 'jobs.0.company')
        self.assertEqual(target['outcome'], 'unsafe')
        self.assertTrue(target['wrong_record_cited'])
        self.assertFalse(target['grounded_correct'])

    async def test_canonical_null_recovery_and_objective_immutability(self):
        case = self.cases['wrong_jobs_logistics']
        def respond(prompt):
            context = json.loads(prompt.to_messages()[1].content)
            decisions = []
            for reference in context['requested_claim_ids']:
                contract = context['evidence_path_contracts'][reference]
                parts = reference.split('.')
                if parts[0] == 'jobs':
                    anchor = case['alignment_expectations']['jobs'][int(parts[1])]
                    field = parts[2]
                    paths = [f'experience.{anchor}.{field}']
                    if field == 'title':
                        paths.append(f'experience.{anchor}.seniority')
                elif parts[0] == 'education':
                    paths = [f'education.0.{parts[2]}']
                else:
                    paths = [reference]
                status = 'MATCH' if set(paths) <= set(contract['allowed_match_evidence_paths']) else 'UNKNOWN'
                decisions.append({'claim_id': reference, 'status': status, 'evidence_paths': paths if status == 'MATCH' else [], 'reason': 'Fixed canonical test decision'})
            return AIMessage(content=json.dumps({'findings': decisions}))
        payload = await self.run_cases([case], RunnableLambda(respond))
        row = payload['records'][0]
        self.assertEqual(row['gated']['result']['score'], 0.9)
        self.assertEqual(payload['summary']['after_null_outcomes'], {'correct': 9})
        for name in ('ungated', 'gated'):
            initial = [f for f in row[name]['result']['deterministic_findings'] if f['status'] != 'UNKNOWN']
            final = {f['claim_id']: f for f in row[name]['result']['findings']}
            for item in initial:
                self.assertEqual(final[item['claim_id']], item)

    async def test_concurrency_bound_and_results_path_restriction(self):
        cases = [deepcopy(self.cases['mba_alias']) for _ in range(5)]
        for i, case in enumerate(cases):
            case['id'] = f'bound_{i}'
        active = maximum = 0
        async def respond(prompt):
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            import asyncio
            await asyncio.sleep(0.01)
            active -= 1
            return unknown_model(prompt)
        await self.run_cases(cases, RunnableLambda(respond))
        self.assertEqual(maximum, 3)
        with self.assertRaises(ValueError):
            await evaluate(cases, RunnableLambda(unknown_model), Path('/tmp/disallowed_gate.json'))


if __name__ == '__main__':
    unittest.main()
