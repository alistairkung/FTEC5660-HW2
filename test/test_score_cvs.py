import asyncio
import csv
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import hw2
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda


class ScoreCVsTests(unittest.IsolatedAsyncioTestCase):
    async def test_per_cv_exceptions_and_invalid_scores_do_not_crash_batch(self):
        agent = Mock()
        agent.ainvoke = AsyncMock(side_effect=[{'score': 0.9}, RuntimeError('failed'), {'score': None}, {'score': True}])
        with patch('hw2._write_checkpoint') as checkpoint:
            scores = await hw2.score_cvs(agent, {name: name for name in ('ok', 'failed', 'missing', 'invalid')})
        self.assertEqual(scores, {'ok': 0.9, 'failed': 0.5, 'missing': 0.5, 'invalid': 0.5})
        self.assertEqual(checkpoint.call_count, 4)
        self.assertEqual(list(checkpoint.call_args_list[-1].args[0]), list(scores))

    async def test_timeout_cancels_only_slow_cv(self):
        cancelled = []
        async def respond(text):
            if text == 'slow':
                try:
                    await asyncio.sleep(60)
                finally:
                    cancelled.append(text)
            return {'score': 0.9}
        agent = Mock()
        agent.ainvoke = AsyncMock(side_effect=respond)
        with patch('hw2.PER_CV_TIMEOUT_SECONDS', 0.01), patch('hw2._write_checkpoint'):
            scores = await hw2.score_cvs(agent, {'slow': 'slow', 'fast': 'fast'})
        self.assertEqual(scores, {'slow': 0.5, 'fast': 0.9})
        self.assertEqual(cancelled, ['slow'])

    async def test_at_most_three_cvs_in_flight(self):
        active, peak = 0, 0
        async def respond(_):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            try:
                await asyncio.sleep(0.01)
                return {'score': 0.9}
            finally:
                active -= 1
        agent = Mock()
        agent.ainvoke = AsyncMock(side_effect=respond)
        with patch('hw2._write_checkpoint'):
            scores = await hw2.score_cvs(agent, {str(n): 'CV' for n in range(10)})
        self.assertEqual(peak, 3)
        self.assertEqual(active, 0)
        self.assertEqual(len(scores), 10)
        self.assertTrue(all(score == 0.9 for score in scores.values()))

    async def test_checkpoint_failure_does_not_lose_scores(self):
        agent = Mock()
        agent.ainvoke = AsyncMock(return_value={'score': 0.9})
        with patch('hw2._write_checkpoint', side_effect=OSError('disk unavailable')), redirect_stderr(io.StringIO()):
            self.assertEqual(await hw2.score_cvs(agent, {'cv': 'CV'}), {'cv': 0.9})

    def test_atomic_checkpoint_keeps_old_file_on_replace_failure(self):
        root = Path(hw2.__file__).parent
        with tempfile.TemporaryDirectory(dir=root / '.venv') as directory:
            previous = Path.cwd()
            try:
                os.chdir(directory)
                Path('results.csv').write_text('old checkpoint')
                with patch('os.replace', side_effect=OSError('failed')), self.assertRaises(OSError):
                    hw2._write_checkpoint({'cv': 0.9})
                self.assertEqual(Path('results.csv').read_text(), 'old checkpoint')
                self.assertEqual(list(Path('.').glob('.results-*.csv')), [])
            finally:
                os.chdir(previous)

    async def test_runner_end_to_end_with_fake_models_and_tools_and_final_csv(self):
        claims = {'name': {'raw': 'Person', 'value': 'Person'}, 'city': {'raw': 'London', 'value': 'London'},
                  'jobs': [], 'education': [], 'skills': []}
        profiles = {1: {'id': 1, 'name': 'Person', 'city': 'London', 'experience': [], 'education': [], 'skills': []},
                    2: {'id': 2, 'name': 'Other', 'city': 'London', 'experience': [], 'education': [], 'skills': []}}
        compared_names = []
        def respond(prompt, **options):
            self.assertEqual(options, {'extra_body': {'thinking': {'type': 'disabled'}}})
            messages = prompt.to_messages()
            if messages[0].content.startswith('Extract'):
                output = claims
            elif messages[0].content.startswith('Shortlist'):
                output = [2, 1]
            elif messages[0].content.startswith('Identify'):
                output = {'candidate_id': 1, 'reason': 'Supporting evidence'}
            else:
                context = json.loads(messages[1].content)
                compared_names.append(context['profile']['name'])
                output = [{'claim_id': reference, 'cv_claim': claim, 'status': 'MATCH',
                           'evidence': [{'path': reference, 'value': context['profile'][reference]}],
                           'reason': 'Equivalent'} for reference, claim in context['claim_catalog'].items()]
            return AIMessage(content=json.dumps(output))
        def blocks(value):
            return [{'type': 'text', 'text': json.dumps(value)}]
        search, lookup = Mock(), Mock()
        search.name, lookup.name = 'search_linkedin_people', 'get_linkedin_profile'
        search.ainvoke = AsyncMock(return_value=blocks([{'id': 1}, {'id': 2}]))
        lookup.ainvoke = AsyncMock(side_effect=lambda args: blocks(profiles[args['person_id']]))
        root = Path(hw2.__file__).parent
        with tempfile.TemporaryDirectory(dir=root / '.venv') as directory:
            folder = Path(directory)
            (folder / 'ground_truth.json').write_text(json.dumps({'example.pdf': {'label': 1}}))
            previous = Path.cwd()
            checkpoint_rows = []
            original_checkpoint = hw2._write_checkpoint
            def checkpoint(scores):
                original_checkpoint(scores)
                with Path('results.csv').open() as handle:
                    checkpoint_rows.append(list(csv.reader(handle)))
            try:
                os.chdir(folder)
                with patch('hw2.cv_files', return_value=[folder / 'example.pdf']), \
                     patch('hw2.cv_text', return_value='CV'), \
                     patch('hw2.load_mcp_tools', new=AsyncMock(return_value=[search, lookup])), \
                     patch('langchain_deepseek.ChatDeepSeek', return_value=RunnableLambda(respond)), \
                     patch('hw2._write_checkpoint', side_effect=checkpoint), redirect_stdout(io.StringIO()):
                    self.assertEqual(await hw2.run(folder), 0)
                with Path('results.csv').open() as handle:
                    final = list(csv.reader(handle))
                self.assertEqual(final, [['cv', 'score', 'correctness'], ['example.pdf', '0.9000', 'correct']])
                self.assertEqual(checkpoint_rows[0][0], final[0])
                self.assertEqual(checkpoint_rows[0][1][1], '0.9000')
                self.assertIn('checkpoint', checkpoint_rows[0][1][2])
                self.assertEqual(compared_names, ['Person'])
            finally:
                os.chdir(previous)
