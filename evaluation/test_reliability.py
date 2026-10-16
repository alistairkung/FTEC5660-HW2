"""Deterministic checks for comparison instrumentation and input selection."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from evaluation import reliability as runner
from evaluation.reliability_report import summarize


class ReliabilityTests(unittest.IsolatedAsyncioTestCase):
    def test_binary_negative_fallback_is_not_counted_as_verified_mismatch(self):
        row = dict(result=dict(score=0.5, findings=[], errors=[{'stage': 'comparison'}]),
            grading=dict(binary_eligible=True, correct=True, prediction=0, graded_claims=2,
                claim_failures=[{}, {}], graded_targets=1, target_failures=[{}],
                off_anchor_citations=[], alignment_errors=[], alignment_abstentions=[]),
            grounding=dict(verified_expected_discrepancy=False, claims=[dict(grounded_correct=False,
                wrong_record_cited=False) for _ in range(2)]), identity_correct=True,
            errors=[dict(stage='comparison', type='ValueError')], seconds=1, trace=dict(calls=[]))
        metrics = summarize([row, dict(skipped=True)])
        self.assertEqual(metrics['correct'], 1)
        self.assertEqual(metrics['negative_half'], 1)
        self.assertEqual(metrics.get('mismatch_outcomes', 0), 0)
        self.assertEqual(metrics.get('grounded_discrepancy_outcomes', 0), 0)
        self.assertEqual(metrics['contract_failure_cases'], 1)
        self.assertEqual(metrics['skipped'], 1)

    async def test_callback_captures_real_message_usage_and_errors_in_cv_context(self):
        trace = {}
        token, stage = runner.TRACE.set(trace), runner.STAGE.set('comparison')
        try:
            handler = runner.make_handler()
            await handler.on_chat_model_start({}, [[HumanMessage(content='claims only')]], run_id='success')
            message = AIMessage(content='response', usage_metadata=dict(input_tokens=10, output_tokens=5, total_tokens=15))
            await handler.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]), run_id='success')
            await handler.on_chat_model_start({}, [[HumanMessage(content='second')]], run_id='error')
            await handler.on_llm_error(ValueError('invalid'), run_id='error')
            self.assertEqual(trace['calls'][0]['usage']['total_tokens'], 15)
            self.assertEqual(trace['calls'][0]['stage'], 'comparison')
            self.assertEqual(trace['calls'][0]['output'], 'response')
            self.assertEqual(trace['calls'][1]['error_type'], 'ValueError')
            self.assertEqual(handler.pending, {})
        finally:
            runner.TRACE.reset(token)
            runner.STAGE.reset(stage)

    async def test_instrumentation_delegates_identical_arguments_and_captures_upstream(self):
        upstream = dict(claims={'value': 'unchanged'}, identity='unresolved', profiles=[], errors=[])
        identity = SimpleNamespace(extractor=SimpleNamespace(extract=AsyncMock(return_value={'raw': 'value'})),
            prioritizer=SimpleNamespace(prioritize=AsyncMock()), retriever=SimpleNamespace(search=AsyncMock(),
            retrieve_profiles=AsyncMock()), identity_resolver=SimpleNamespace(resolve=AsyncMock()),
            ainvoke=AsyncMock(return_value=upstream))
        compare = AsyncMock(return_value=[{'finding': 'unchanged'}])
        agent = SimpleNamespace(identity_pipeline=identity, comparator=SimpleNamespace(compare=compare),
                                reflector=SimpleNamespace(reflect=AsyncMock()))
        runner.instrument(agent)
        trace = {}
        token = runner.TRACE.set(trace)
        try:
            inputs = ({'claims': 'original'}, {'profile': 'original'})
            self.assertEqual(await agent.comparator.compare(*inputs), [{'finding': 'unchanged'}])
            compare.assert_awaited_once_with(*inputs)
            self.assertIs(await identity.ainvoke('CV'), upstream)
            self.assertEqual(trace['upstream'], upstream)
            upstream['claims']['value'] = 'later mutation'
            self.assertEqual(trace['upstream']['claims']['value'], 'unchanged')
        finally:
            runner.TRACE.reset(token)

    def test_selection_is_first_observed_pair_and_retains_unresolved_without_using_scores(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(runner, 'ARTIFACTS', Path(temporary)):
            root = Path(temporary)
            order = ['production', 'hybrid', 'hybrid', 'production', 'production', 'hybrid']
            runner.save(root / 'manifest.json', dict(full_order=order))
            counters = {'production': 0, 'hybrid': 0}
            for i, name in enumerate(order):
                counters[name] += 1
                upstream = dict(identity='resolved' if i != 1 else 'unresolved',
                                selected_candidate_id=1 if i == 0 else 2 if i != 1 else None,
                                claims={'capture': i})
                runner.save(root / f'full_{name}_{counters[name]}.json', dict(records={
                    'example.pdf': dict(score=0.1 if i == 0 else 0.9, trace=dict(upstream=upstream))}))
            selected = runner.selected_inputs()['inputs']
            self.assertEqual(len(selected), 3)
            self.assertEqual([r['upstream']['selected_candidate_id'] for r in selected], [1, None, 2])
            self.assertEqual([r['upstream']['claims']['capture'] for r in selected], [0, 1, 2])
            self.assertNotIn('score', json.dumps(selected))


if __name__ == '__main__':
    unittest.main()
