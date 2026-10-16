"""Submission boundaries, generated-source parity and captured model behavior."""

import ast
import contextlib
import gzip
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from evaluation.run_production import evaluate
from scripts.aggregate_source import OUTPUT, ROOT, generate, local_dependency, production_sources


class SubmissionTests(unittest.IsolatedAsyncioTestCase):
    def test_generated_source_matches_every_production_definition_and_constant(self):
        self.assertEqual(OUTPUT.read_text(), generate())
        aggregate = ast.parse(OUTPUT.read_text())
        definitions = [node for node in aggregate.body if isinstance(node, (
            ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef, ast.Assign, ast.AnnAssign))]
        expected = []
        for path in production_sources():
            for node in ast.parse((ROOT / path).read_text()).body:
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef, ast.Assign, ast.AnnAssign)):
                    expected.append(node)
        self.assertEqual([ast.dump(node) for node in definitions], [ast.dump(node) for node in expected])
        self.assertFalse(any(local_dependency(node) for node in ast.walk(aggregate)))
        classes = [node.name for node in definitions if isinstance(node, ast.ClassDef)]
        self.assertEqual(len(classes), 13)
        self.assertEqual(len(classes), len(set(classes)))

    def test_production_imports_do_not_reach_evaluation_or_experiments(self):
        source = '''
import sys
from unittest.mock import patch
from langchain_core.runnables import RunnableLambda
import hw2
with patch('langchain_deepseek.ChatDeepSeek', return_value=RunnableLambda(lambda _: None)):
    agent = hw2.build_agent([])
assert type(agent).__name__ == 'CVVerifier'
assert not any(name.startswith('evaluation') for name in sys.modules)
assert not any(any(part in name for part in ('record_aligner', 'semantic_claim_resolver',
    'deterministic_claim_verifier', 'evidence_aligner', 'hybrid_cv_verifier')) for name in sys.modules)
'''
        subprocess.run([__import__('sys').executable, '-c', source], cwd=ROOT, check=True,
                       capture_output=True, text=True)

    def test_aggregate_imports_without_modular_package(self):
        source = '''
import importlib.abc
import runpy
import sys
class BlockLocal(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('cv_verifier', 'evaluation', 'scripts'):
            raise ImportError(fullname)
sys.meta_path.insert(0, BlockLocal())
module = runpy.run_path(sys.argv[1])
assert callable(module['build_agent']) and callable(module['score_cvs'])
'''
        subprocess.run([__import__('sys').executable, '-c', source, str(OUTPUT)], cwd=ROOT,
                       check=True, capture_output=True, text=True)

    async def test_all_25_captured_responses_match_modular_and_generated_implementations(self):
        output_root = ROOT / 'evaluation/results'
        output_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output_root) as folder, contextlib.redirect_stdout(io.StringIO()):
            for capture in ('baseline', 'consolidated'):
                fixture = ROOT / f'evaluation/fixtures/production_{capture}.json.gz'
                original = json.loads(gzip.decompress(fixture.read_bytes()))
                for aggregated in (False, True):
                    report = await evaluate(Path(folder) / f'{capture}_{aggregated}.json', fixture, aggregated=aggregated)
                    self.assertTrue(report['replay_equal'])
                    self.assertEqual(len(report['records']), 25)
                    self.assertEqual(report['summary']['llm_calls'], original['summary']['llm_calls'])


if __name__ == '__main__':
    unittest.main()
