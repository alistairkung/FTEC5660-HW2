import asyncio
import io
import json
import tempfile
import unittest
from copy import deepcopy
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from scripts import inspect_batch
from cv_verifier.deterministic_claim_verifier import DeterministicClaimVerifier
from test_claim_comparator import CLAIMS, PROFILE
from test_record_aligner import ALIGNMENTS


def record():
    profile = dict(deepcopy(PROFILE), id=2)
    baseline = DeterministicClaimVerifier().verify(CLAIMS, profile, ALIGNMENTS)
    return {'filename': 'CV.pdf', 'score': 0.9,
            'comparison_output_before_validation': baseline,
            'result': {'identity': 'resolved', 'selected_candidate_id': 2, 'identity_reason': 'Selected identity',
                       'claims': deepcopy(CLAIMS), 'profiles': [profile], 'score': 0.9,
                       'findings': deepcopy(baseline), 'errors': [], 'reflection': {'attempted': True, 'corrections': []}}}


class ShadowInspectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_components_capture_alignment_calls_and_do_not_receive_grading_labels(self):
        prompts, options = [], []
        def respond(prompt, **kwargs):
            prompts.append(prompt)
            options.append(kwargs)
            return AIMessage(content=json.dumps(ALIGNMENTS))
        with patch('langchain_deepseek.ChatDeepSeek', return_value=RunnableLambda(respond)) as factory:
            aligner, verifier = inspect_batch._build_shadow_components()
        production = record()
        production['expected'] = {'label': 1, 'reason': 'Private grading information'}
        original = deepcopy(production)
        report = await inspect_batch._inspect_shadow(production, aligner, verifier)
        self.assertEqual(production, original)
        self.assertEqual(report['alignments'], ALIGNMENTS)
        self.assertEqual(report['llm_calls'], 1)
        self.assertEqual(report['errors'], [])
        self.assertTrue(all(row['agreement'] for row in report['comparisons']))
        self.assertEqual(set(json.loads(prompts[0].to_messages()[1].content)), {'claims', 'profile'})
        self.assertNotIn('Private grading information', prompts[0].to_messages()[1].content)
        factory.assert_called_once_with(model='deepseek-v4-flash', timeout=60, max_retries=0)
        self.assertEqual(options, [{'extra_body': {'thinking': {'type': 'disabled'}},
                                   'response_format': {'type': 'json_object'}}])

    async def test_ordinary_shadow_empty_skills_remain_unknown(self):
        production = record()
        production['result']['profiles'][0]['skills'] = []
        production['comparison_output_before_validation'] = DeterministicClaimVerifier().verify(
            production['result']['claims'], production['result']['profiles'][0], ALIGNMENTS)
        original = deepcopy(production)
        aligner = Mock()
        aligner.align = AsyncMock(return_value=deepcopy(ALIGNMENTS))

        report = await inspect_batch._inspect_shadow(production, aligner, DeterministicClaimVerifier())

        skills = [finding for finding in report['findings'] if finding['claim_id'].startswith('skills.')]
        self.assertTrue(skills)
        self.assertTrue(all(finding['status'] == 'UNKNOWN' for finding in skills))
        self.assertEqual(report['errors'], [])
        self.assertEqual(production, original)

    async def test_alignment_and_verification_failures_preserve_all_production_data(self):
        for stage in ('shadow_alignment', 'shadow_verification'):
            with self.subTest(stage=stage):
                production = record()
                original = deepcopy(production)
                aligner = Mock()
                async def align(claims, profile):
                    claims['name']['value'] = 'Mutated local copy'
                    profile['name'] = 'Mutated local copy'
                    if stage == 'shadow_alignment':
                        raise RuntimeError('Alignment unavailable')
                    return deepcopy(ALIGNMENTS)
                aligner.align = AsyncMock(side_effect=align)
                verifier = Mock()
                verifier.verify.side_effect = RuntimeError('Verification unavailable')
                report = await inspect_batch._inspect_shadow(production, aligner, verifier)
                self.assertEqual(production, original)
                self.assertEqual(report['errors'][0]['stage'], stage)
                self.assertEqual(report['findings'], [])
                self.assertEqual(aligner.align.await_count, 1)

    async def test_unresolved_identity_and_unavailable_baseline_skip_shadow_calls(self):
        for reason in ('identity_unresolved', 'baseline_unavailable'):
            with self.subTest(reason=reason):
                production = record()
                if reason == 'identity_unresolved':
                    production['result']['identity'] = 'unresolved'
                else:
                    production['comparison_output_before_validation'] = None
                aligner, verifier = Mock(), Mock()
                aligner.align = AsyncMock(side_effect=AssertionError('No model call allowed'))
                report = await inspect_batch._inspect_shadow(production, aligner, verifier)
                self.assertEqual(report['skipped'], reason)
                self.assertEqual(report['llm_calls'], 0)
                aligner.align.assert_not_awaited()
                verifier.verify.assert_not_called()

    async def test_invalid_baseline_is_recorded_without_shadow_inference(self):
        production = record()
        production['comparison_output_before_validation'] = []
        aligner = Mock()
        aligner.align = AsyncMock()
        report = await inspect_batch._inspect_shadow(production, aligner, Mock())
        self.assertEqual(report['errors'][0]['stage'], 'shadow_input')
        aligner.align.assert_not_awaited()

    async def test_ordinary_batch_never_builds_shadow_components(self):
        paths = [Path('CV.pdf')]
        agent = Mock()
        agent.ainvoke = AsyncMock(return_value=record()['result'])
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / '.venv') as directory:
            with patch.object(inspect_batch, 'cv_files', return_value=paths), \
                 patch.object(inspect_batch, 'cv_text', return_value='CV'), \
                 patch.object(inspect_batch, 'load_env_file'), \
                 patch.object(inspect_batch, 'load_mcp_tools', new=AsyncMock(return_value=[])), \
                 patch.object(inspect_batch, 'build_agent', return_value=agent), \
                 patch.object(inspect_batch, 'read_ground_truth', return_value={}), \
                 patch.object(inspect_batch, '_build_shadow_components') as build, redirect_stdout(io.StringIO()):
                payload = await inspect_batch.inspect_batch(Path('public_test'), Path(directory) / 'batch.json')
            build.assert_not_called()
            self.assertNotIn('shadow', payload['records'][0])
            self.assertEqual(payload['records'][0]['result'], record()['result'])

    async def test_shadow_phase_starts_after_production_and_preserves_scores_with_limit_three(self):
        paths = [Path(f'CV_{n}.pdf') for n in range(7)]
        results = [record() for _ in paths]
        baseline = results[0]['comparison_output_before_validation']
        agent = Mock()
        agent.comparator.compare = AsyncMock(return_value=deepcopy(baseline))
        async def production(_):
            await agent.comparator.compare()
            return deepcopy(results[0]['result'])
        agent.ainvoke = AsyncMock(side_effect=production)
        active = peak = 0
        async def inspect_shadow(item, *_):
            nonlocal active, peak
            self.assertEqual(agent.ainvoke.await_count, 7)
            active += 1
            peak = max(peak, active)
            try:
                await asyncio.sleep(0.01)
                return {'comparisons': [], 'llm_calls': 0, 'elapsed_seconds': 0.01,
                        'errors': [{'stage': 'shadow_alignment', 'message': 'failed'}], 'skipped': None}
            finally:
                active -= 1
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / '.venv') as directory:
            output = Path(directory) / 'batch.json'
            with patch.object(inspect_batch, 'cv_files', return_value=paths), \
                 patch.object(inspect_batch, 'cv_text', return_value='CV'), \
                 patch.object(inspect_batch, 'load_env_file'), \
                 patch.object(inspect_batch, 'load_mcp_tools', new=AsyncMock(return_value=[])), \
                 patch.object(inspect_batch, 'build_agent', return_value=agent), \
                 patch.object(inspect_batch, 'read_ground_truth', return_value={}), \
                 patch.object(inspect_batch, '_build_shadow_components', return_value=(Mock(), Mock())), \
                 patch.object(inspect_batch, '_inspect_shadow', side_effect=inspect_shadow), \
                 patch('hw2._write_checkpoint') as checkpoint, redirect_stdout(io.StringIO()):
                payload = await inspect_batch.inspect_batch(Path('public_test'), output, shadow=True)
            self.assertEqual(peak, 3)
            self.assertEqual(json.loads(output.read_text()), payload)
            for item in payload['records']:
                self.assertEqual(item['score'], 0.9)
                self.assertEqual(item['result'], results[0]['result'])
                self.assertEqual(item['comparison_output_before_validation'], baseline)
                self.assertEqual(item['shadow']['errors'][0]['stage'], 'shadow_alignment')
            checkpoint.assert_not_called()

    async def test_setup_failure_is_separate_and_preserves_production_records(self):
        payload = {'records': [record()]}
        original = deepcopy(payload['records'])
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / '.venv') as directory:
            with patch.object(inspect_batch, '_build_shadow_components', side_effect=RuntimeError('setup failed')):
                await inspect_batch._run_shadow(payload, Path(directory) / 'batch.json', asyncio.Semaphore(3))
        self.assertEqual(payload['records'], original)
        self.assertEqual(payload['shadow_setup_errors'][0]['message'], 'setup failed')

    def test_disagreements_distinguish_alignment_semantics_missing_and_objective_evidence(self):
        kinds = [('UNKNOWN', 'Record alignment is ambiguous.', 'alignment'),
                 ('UNKNOWN', 'Missing profile evidence.', 'missing_evidence'),
                 ('UNKNOWN', 'Semantic equivalence requires review.', 'semantic_equivalence'),
                 ('MISMATCH', 'Explicit typed values contradict.', 'objective_contradiction')]
        for status, reason, expected in kinds:
            with self.subTest(expected=expected):
                findings = [{'claim_id': 'jobs.0.title', 'status': status, 'reason': reason}]
                baseline = [{'claim_id': 'jobs.0.title', 'status': 'MATCH'}]
                self.assertEqual(inspect_batch._compare_shadow_findings(baseline, findings)[0]['disagreement_type'], expected)

    def test_title_disagreement_distinguishes_missing_values_from_semantic_differences(self):
        baseline = [{'claim_id': 'jobs.0.title', 'status': 'MATCH'}]
        for title, expected in ((None, 'missing_evidence'), ('Another title', 'semantic_equivalence')):
            profile = deepcopy(PROFILE)
            profile['experience'][1]['title'] = title
            finding = next(f for f in DeterministicClaimVerifier().verify(CLAIMS, profile, ALIGNMENTS)
                           if f['claim_id'] == 'jobs.0.title')
            with self.subTest(title=title):
                self.assertEqual(inspect_batch._compare_shadow_findings(baseline, [finding])[0]['disagreement_type'], expected)
