import json
import unittest
from copy import deepcopy
from unittest.mock import AsyncMock, Mock, patch, call

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

import hw2

from cv_verifier.claim_comparison_validator import ClaimComparisonValidator
from cv_verifier.cv_scorer import CVScorer
from cv_verifier.deterministic_claim_verifier import DeterministicClaimVerifier
from cv_verifier.hybrid_cv_verifier import HybridCVVerifier
from cv_verifier.record_aligner import RecordAligner
from cv_verifier.semantic_claim_resolver import SemanticClaimResolver


CLAIMS = {'name': {'raw': 'Person', 'value': 'Person'},
          'city': {'raw': None, 'value': None, 'candidates': []},
          'jobs': [{'raw': '2021', 'company': None, 'title': None, 'seniority': None,
                    'dates_raw': None, 'start_year': 2021, 'end_year': None, 'is_current': None}],
          'education': [], 'skills': []}
PROFILE = {'id': 2, 'name': 'Person', 'experience': [{'start_year': 2020}], 'education': [], 'skills': []}
ALIGNMENTS = {'jobs': [{'cv_index': 0, 'profile_index': 0, 'reason': 'Fixed corresponding record.'}], 'education': []}


def identity(claims=CLAIMS, profile=PROFILE):
    return {'identity': 'resolved', 'selected_candidate_id': profile['id'], 'identity_reason': 'Supported identity',
            'claims': deepcopy(claims), 'profiles': [deepcopy(profile)], 'errors': []}


class HybridCVVerifierTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_build_agent_dependencies_options_unknown_only_tasks_and_immutable_objective_findings(self):
        prompts = []
        def respond(prompt, **options):
            self.assertEqual(options, {'extra_body': {'thinking': {'type': 'disabled'}},
                                       'response_format': {'type': 'json_object'}})
            context = json.loads(prompt.to_messages()[1].content)
            prompts.append(context)
            if prompt.to_messages()[0].content.startswith('Align CV'):
                output = ALIGNMENTS
            else:
                self.assertEqual(context['requested_claim_ids'], ['skills.0.name'])
                self.assertEqual(list(context['claim_catalog']), ['skills.0.name'])
                output = {'findings': [{'claim_id': 'skills.0.name', 'status': 'MATCH',
                           'evidence_paths': ['skills.0.name'], 'reason': 'PowerPoint alias'}]}
            return AIMessage(content=json.dumps(output))
        with patch('langchain_deepseek.ChatDeepSeek', return_value=RunnableLambda(respond)) as factory:
            verifier = hw2.build_agent([])
        self.assertEqual(factory.call_args_list, [call(model='deepseek-v4-flash', timeout=60, max_retries=0)] * 2)
        self.assertIsInstance(verifier, HybridCVVerifier)
        self.assertIsInstance(verifier.aligner, RecordAligner)
        self.assertIsInstance(verifier.deterministic_verifier, DeterministicClaimVerifier)
        self.assertIsInstance(verifier.semantic_resolver, SemanticClaimResolver)
        self.assertIsInstance(verifier.comparison_validator, ClaimComparisonValidator)
        self.assertIsInstance(verifier.scorer, CVScorer)
        self.assertFalse(hasattr(verifier, 'comparator'))
        self.assertFalse(hasattr(verifier, 'reflector'))
        claims, profile = deepcopy(CLAIMS), deepcopy(PROFILE)
        claims['skills'] = [{'raw': 'PowerPoint', 'name': 'PowerPoint'}]
        profile['skills'] = [{'name': 'Microsoft PowerPoint'}]
        saved = identity(claims, profile)
        verifier.identity_pipeline = AsyncMock()
        verifier.identity_pipeline.ainvoke.return_value = saved
        original = deepcopy(saved)
        result = await verifier.ainvoke('CV')
        self.assertEqual(result['score'], 0.1)
        self.assertTrue(result['verification_complete'])
        self.assertEqual(saved, original)
        self.assertEqual(result['alignments'], ALIGNMENTS)
        baseline = {f['claim_id']: f for f in result['deterministic_findings']}
        final = {f['claim_id']: f for f in result['findings']}
        for reference, finding in baseline.items():
            if finding['status'] != 'UNKNOWN':
                self.assertEqual(final[reference], finding)
        self.assertEqual(final['skills.0.name']['status'], 'MATCH')
        self.assertEqual(len(prompts), 2)

    async def test_stage_order_and_copied_dependencies_preserve_source_and_mapping(self):
        verifier = self.verifier()
        saved = verifier.identity_pipeline.ainvoke.return_value
        original, stages = deepcopy(saved), []
        verifier.identity_pipeline.ainvoke.side_effect = lambda _: (stages.append('identity') or saved)
        async def align(claims, profile):
            stages.append('alignment')
            claims['name']['value'] = 'Mutated injected copy'
            return deepcopy(ALIGNMENTS)
        verifier.aligner.align.side_effect = align
        actual = DeterministicClaimVerifier()
        def verify(claims, profile, alignments):
            stages.append('deterministic')
            findings = actual.verify(claims, profile, alignments)
            alignments['jobs'][0]['reason'] = 'Mutated injected copy'
            return findings
        verifier.deterministic_verifier = Mock()
        verifier.deterministic_verifier.verify.side_effect = verify
        validator = ClaimComparisonValidator()
        def validate(*args):
            stages.append('validation')
            return validator.validate(*args)
        verifier.comparison_validator = Mock()
        verifier.comparison_validator.validate.side_effect = validate
        async def resolve(claims, profile, alignment, findings):
            stages.append('semantic')
            self.assertEqual(alignment, ALIGNMENTS)
            return findings
        verifier.semantic_resolver.resolve.side_effect = resolve
        scorer = CVScorer()
        verifier.scorer = Mock()
        verifier.scorer.score.side_effect = lambda *args, **kwargs: (stages.append('score') or scorer.score(*args, **kwargs))
        result = await verifier.ainvoke('CV')
        self.assertEqual(stages, ['identity', 'alignment', 'deterministic', 'validation', 'semantic', 'validation', 'score'])
        self.assertEqual(saved, original)
        self.assertEqual(result['alignments'], ALIGNMENTS)
        self.assertTrue(result['verification_complete'])

    async def test_unresolved_invalid_and_failed_identity_skip_all_verification(self):
        for failure in ('unresolved', 'missing_id', 'duplicate_id', 'model'):
            verifier = self.verifier()
            saved = verifier.identity_pipeline.ainvoke.return_value
            if failure == 'unresolved':
                saved['identity'] = 'unresolved'
            elif failure == 'missing_id':
                saved['selected_candidate_id'] = 99
            elif failure == 'duplicate_id':
                saved['profiles'].append(deepcopy(saved['profiles'][0]))
            else:
                verifier.identity_pipeline.ainvoke.side_effect = RuntimeError('Unavailable')
            with self.subTest(failure=failure):
                result = await verifier.ainvoke('CV')
                self.assertEqual(result['score'], 0.5)
                self.assertEqual(result['identity'], 'unresolved')
                self.assertFalse(result['verification_complete'])
                verifier.aligner.align.assert_not_awaited()
                verifier.semantic_resolver.resolve.assert_not_awaited()

    async def test_actual_aligner_rejects_bad_indices_missing_coverage_invalid_json_and_model_failure(self):
        invalid = deepcopy(ALIGNMENTS)
        invalid['jobs'][0]['profile_index'] = 9
        for output in (invalid, {'jobs': [], 'education': []}, 'not JSON', RuntimeError('Unavailable')):
            verifier = self.verifier()
            def respond(_):
                if isinstance(output, Exception):
                    raise output
                return AIMessage(content=output if isinstance(output, str) else json.dumps(output))
            verifier.aligner = RecordAligner(RunnableLambda(respond))
            with self.subTest(output=output):
                result = await verifier.ainvoke('CV')
                self.assertEqual(result['score'], 0.5)
                self.assertEqual(result['errors'][-1]['stage'], 'alignment')
                self.assertFalse(result['verification_complete'])
                self.assertEqual(result['findings'], [])
                verifier.semantic_resolver.resolve.assert_not_awaited()

    def verifier(self, *, mismatch=True):
        claims, profile = deepcopy(CLAIMS), deepcopy(PROFILE)
        if not mismatch:
            claims['jobs'][0]['start_year'] = 2020
        claims['skills'] = [{'raw': 'Docker', 'name': 'Docker'}]
        profile['skills'] = [{'name': 'Python'}]
        pipeline, aligner, resolver = AsyncMock(), Mock(), Mock()
        pipeline.ainvoke.return_value = identity(claims, profile)
        aligner.align = AsyncMock(return_value=deepcopy(ALIGNMENTS))
        resolver.resolve = AsyncMock(side_effect=RuntimeError('Semantic model unavailable'))
        return HybridCVVerifier(pipeline, aligner, DeterministicClaimVerifier(), resolver,
                                ClaimComparisonValidator(), CVScorer())

    async def test_typed_discrepancy_uses_unchanged_alignment_and_skips_semantic_stage(self):
        pipeline, aligner, resolver = AsyncMock(), Mock(), Mock()
        pipeline.ainvoke.return_value = identity()
        aligner.align = AsyncMock(return_value=deepcopy(ALIGNMENTS))
        resolver.resolve = AsyncMock(side_effect=AssertionError('No UNKNOWN claims'))
        deterministic = Mock(wraps=DeterministicClaimVerifier())
        verifier = HybridCVVerifier(pipeline, aligner, deterministic, resolver,
                                    ClaimComparisonValidator(), CVScorer())
        result = await verifier.ainvoke('CV text')
        self.assertEqual(result['score'], 0.1)
        self.assertEqual(result['alignments'], ALIGNMENTS)
        self.assertEqual(deterministic.verify.call_args.args[2], ALIGNMENTS)
        self.assertEqual(result['semantic_claim_ids'], [])
        resolver.resolve.assert_not_awaited()
        self.assertEqual(next(f for f in result['findings'] if f['claim_id'] == 'jobs.0.start_year')['status'], 'MISMATCH')
        self.assertEqual(result['verification_status'], 'verified_discrepancy')

    async def test_semantic_model_failure_preserves_validated_mismatch_and_low_score(self):
        verifier = self.verifier()
        result = await verifier.ainvoke('CV')
        self.assertEqual(result['score'], 0.1)
        self.assertFalse(result['verification_complete'])
        self.assertEqual(result['verification_status'], 'discrepancy_with_semantic_failure')
        self.assertEqual(result['errors'][-1]['stage'], 'semantic')
        self.assertEqual(result['findings'], result['deterministic_findings'])

    async def test_semantic_failure_without_validated_mismatch_returns_half(self):
        result = await self.verifier(mismatch=False).ainvoke('CV')
        self.assertEqual(result['score'], 0.5)
        self.assertFalse(result['verification_complete'])
        self.assertEqual(result['verification_status'], 'processing_failure')
        self.assertFalse(any(f['status'] == 'MISMATCH' for f in result['findings']))

    async def test_unparseable_semantic_response_preserves_only_a_validated_discrepancy(self):
        for mismatch, score in ((True, 0.1), (False, 0.5)):
            verifier = self.verifier(mismatch=mismatch)
            verifier.semantic_resolver = SemanticClaimResolver(
                RunnableLambda(lambda _: AIMessage(content='not JSON')))
            with self.subTest(mismatch=mismatch):
                result = await verifier.ainvoke('CV')
                self.assertEqual(result['score'], score)
                self.assertFalse(result['verification_complete'])
                self.assertEqual(result['errors'][-1]['stage'], 'semantic')
                self.assertEqual(result['findings'], result['deterministic_findings'])

    async def test_deterministic_failure_returns_half_without_semantic_call(self):
        verifier = self.verifier()
        verifier.deterministic_verifier = Mock()
        verifier.deterministic_verifier.verify.side_effect = RuntimeError('Failed deterministic stage')
        result = await verifier.ainvoke('CV')
        self.assertEqual(result['score'], 0.5)
        self.assertEqual(result['errors'][-1]['stage'], 'deterministic')
        self.assertEqual(result['findings'], [])
        verifier.semantic_resolver.resolve.assert_not_awaited()

    async def test_canonical_evidence_failure_returns_half_even_with_prior_mismatch(self):
        for when in ('baseline', 'merged', 'inside_semantic'):
            verifier = self.verifier()
            info = verifier.identity_pipeline.ainvoke.return_value
            baseline = DeterministicClaimVerifier().verify(info['claims'], info['profiles'][0], ALIGNMENTS)
            bad = deepcopy(baseline)
            next(f for f in bad if f['claim_id'] == 'jobs.0.start_year')['evidence'][0]['value'] = 2021
            if when == 'baseline':
                verifier.deterministic_verifier = Mock()
                verifier.deterministic_verifier.verify.return_value = bad
            else:
                verifier.semantic_resolver.resolve.side_effect = (
                    ValueError('Canonical evidence validation failed') if when == 'inside_semantic' else None)
                verifier.semantic_resolver.resolve.return_value = bad
            with self.subTest(when=when):
                result = await verifier.ainvoke('CV')
                self.assertEqual(result['score'], 0.5)
                self.assertFalse(result['verification_complete'])
                self.assertEqual(result['errors'][-1]['kind'], 'contract')
                if when != 'baseline':
                    self.assertTrue(any(f['status'] == 'MISMATCH' for f in result['deterministic_findings']))

    async def test_orchestrator_rejects_semantic_override_of_definite_finding(self):
        verifier = self.verifier()
        info = verifier.identity_pipeline.ainvoke.return_value
        merged = DeterministicClaimVerifier().verify(info['claims'], info['profiles'][0], ALIGNMENTS)
        next(f for f in merged if f['claim_id'] == 'jobs.0.start_year')['status'] = 'MATCH'
        verifier.semantic_resolver.resolve.side_effect = None
        verifier.semantic_resolver.resolve.return_value = merged
        result = await verifier.ainvoke('CV')
        self.assertEqual(result['score'], 0.5)
        self.assertEqual(result['errors'][-1]['stage'], 'validation')
        self.assertEqual(next(f for f in result['findings'] if f['claim_id'] == 'jobs.0.start_year')['status'], 'MISMATCH')


if __name__ == '__main__':
    unittest.main()
