import asyncio
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from scripts import inspect_batch


def result(score=0.9, identity='resolved', findings=None, errors=None):
    return {'score': score, 'identity': identity, 'selected_candidate_id': 2 if identity == 'resolved' else None,
            'identity_reason': 'Supporting evidence' if identity == 'resolved' else 'Insufficient evidence',
            'investigation_stop_reason': 'identity_resolved' if identity == 'resolved' else 'budget_exhausted',
            'profiles': [{'id': 2}], 'errors': [] if errors is None else errors,
            'findings': [] if findings is None else findings}


class InspectBatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_seven_cvs_limit_concurrency_preserve_results_and_save_each_completion(self):
        paths = [Path(f'CV_{n}.pdf') for n in range(1, 8)]
        active, peak = 0, 0
        async def respond(text):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            try:
                await asyncio.sleep(0.01)
                await agent.comparator.compare({'filename': text}, {})
                if text == 'CV_6.pdf':
                    return result(0.5, errors=[{'stage': 'comparison', 'message': 'Malformed JSON'}])
                return result(findings=[{'claim_id': 'name', 'status': 'MATCH'}])
            finally:
                active -= 1
        agent = Mock()
        agent.comparator.compare = AsyncMock(side_effect=lambda claims, _: [
            {'claim_id': claims['filename'], 'status': 'MATCH', 'cv_claim': 'malformed example'}])
        agent.ainvoke = AsyncMock(side_effect=respond)
        truth = {path.name: {'label': 1} for path in paths}
        completed_counts = []
        original_save = inspect_batch._save_results
        def save(output, payload):
            original_save(output, payload)
            completed_counts.append(len(json.loads(output.read_text())['records']))
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / '.venv') as directory:
            output = Path(directory) / 'diagnostics.json'
            stdout = io.StringIO()
            with patch.object(inspect_batch, 'cv_files', return_value=paths), \
                 patch.object(inspect_batch, 'cv_text', side_effect=lambda path: path.name) as loader, \
                 patch.object(inspect_batch, 'load_env_file') as env, \
                 patch.object(inspect_batch, 'load_mcp_tools', new=AsyncMock(return_value=['tool'])) as connect, \
                 patch.object(inspect_batch, 'build_agent', return_value=agent) as build, \
                 patch.object(inspect_batch, 'read_ground_truth', return_value=truth), \
                 patch.object(inspect_batch, '_save_results', side_effect=save), \
                 patch('hw2._write_checkpoint') as checkpoint, redirect_stdout(stdout):
                payload = await inspect_batch.inspect_batch(Path('public_test'), output)
            self.assertEqual(json.loads(output.read_text()), payload)
            self.assertEqual(peak, 3)
            self.assertEqual(agent.ainvoke.await_count, 7)
            self.assertEqual(loader.call_count, 7)
            env.assert_called_once_with()
            connect.assert_awaited_once_with()
            build.assert_called_once_with(['tool'])
            checkpoint.assert_not_called()
            self.assertEqual(completed_counts, list(range(8)))
            self.assertEqual([record['filename'] for record in payload['records']], [p.name for p in paths])
            missed = payload['records'][5]
            self.assertEqual(missed['fallback'], {'occurred': True, 'stage': 'comparison', 'reason': 'Malformed JSON'})
            self.assertEqual(missed['result']['errors'][0]['message'], 'Malformed JSON')
            self.assertEqual(missed['comparison_output_before_validation'][0]['claim_id'], 'CV_6.pdf')
            self.assertEqual(missed['comparison_output_before_validation'][0]['cv_claim'], 'malformed example')
            for record in payload['records']:
                self.assertEqual(record['comparison_output_before_validation'][0]['claim_id'], record['filename'])
            self.assertIn('incorrect', missed['correctness'])
            self.assertTrue(all(record['elapsed_seconds'] >= 0 for record in payload['records']))
            self.assertIn('6 correct', stdout.getvalue())

    async def test_exceptions_timeout_and_pdf_failure_are_saved_without_retry(self):
        paths = [Path('failed.pdf'), Path('slow.pdf'), Path('bad_pdf.pdf')]
        async def respond(text):
            if text == 'failed.pdf':
                raise RuntimeError('API unavailable')
            await asyncio.sleep(60)
        agent = Mock()
        agent.ainvoke = AsyncMock(side_effect=respond)
        def pdf(path):
            if path.name == 'bad_pdf.pdf':
                raise ValueError('Unreadable PDF')
            return path.name
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / '.venv') as directory:
            output = Path(directory) / 'diagnostics.json'
            with patch.object(inspect_batch, 'cv_files', return_value=paths), \
                 patch.object(inspect_batch, 'cv_text', side_effect=pdf), \
                 patch.object(inspect_batch, 'load_env_file'), \
                 patch.object(inspect_batch, 'load_mcp_tools', new=AsyncMock(return_value=[])), \
                 patch.object(inspect_batch, 'build_agent', return_value=agent), \
                 patch.object(inspect_batch, 'read_ground_truth', return_value={}), \
                 patch.object(inspect_batch, 'PER_CV_TIMEOUT_SECONDS', 0.01), redirect_stdout(io.StringIO()):
                payload = await inspect_batch.inspect_batch(Path('public_test'), output)
            self.assertEqual(json.loads(output.read_text()), payload)
            self.assertEqual([r['fallback']['stage'] for r in payload['records']],
                             ['verification', 'timeout', 'pdf_loading'])
            self.assertTrue(all(r['score'] == 0.5 for r in payload['records']))
            self.assertEqual(agent.ainvoke.await_count, 2)
            self.assertEqual(payload['records'][0]['failure']['message'], 'API unavailable')

    def test_fallback_observed_identity_comparison_unknown_and_undetermined(self):
        retrieval_failure = result(0.5, identity='unresolved', errors=[
            {'stage': 'profile', 'person_id': 2, 'message': 'Lookup failed'}])
        retrieval_failure['profiles'] = []
        cases = [
            (result(0.5, identity='unresolved'), 'identity'),
            (retrieval_failure, 'retrieval'),
            (result(0.5, errors=[{'stage': 'comparison', 'message': 'Missing claim'}]), 'comparison'),
            (result(0.5, findings=[{'claim_id': 'city', 'status': 'UNKNOWN'}]), 'comparison_unknown'),
            (result(0.5), 'undetermined'),
        ]
        for output, stage in cases:
            with self.subTest(stage=stage):
                self.assertEqual(inspect_batch._fallback_details(output, 0.5, None)['stage'], stage)
        partial_errors = result(0.9, errors=[{'stage': 'prioritization', 'message': 'Ranking failed'}])
        self.assertFalse(inspect_batch._fallback_details(partial_errors, 0.9, None)['occurred'])

    async def test_results_csv_cannot_be_overwritten(self):
        with patch.object(inspect_batch, 'cv_files', return_value=[Path('cv.pdf')]), \
             self.assertRaisesRegex(ValueError, 'must not overwrite'):
            await inspect_batch.inspect_batch(Path('public_test'), Path('results.csv'))
