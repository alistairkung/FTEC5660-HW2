import json
import unittest
from copy import deepcopy
from unittest.mock import AsyncMock
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from cv_verifier.cv_verifier import CVVerifier
from cv_verifier.cv_scorer import CVScorer
from cv_verifier.claim_comparator import ClaimComparator
from cv_verifier.claim_comparison_validator import ClaimComparisonValidator

CLAIMS = {'name': {'raw': 'Person', 'value': 'Person'}, 'city': {'raw': None, 'value': None},
          'jobs': [], 'education': [], 'skills': []}
IDENTITY = {'identity': 'resolved', 'selected_candidate_id': 2, 'identity_reason': 'Supported identity',
            'claims': CLAIMS, 'profiles': [{'id': 1, 'name': 'Wrong Person'}, {'id': 2, 'name': 'Person'}],
            'errors': []}


class CVVerifierTests(unittest.IsolatedAsyncioTestCase):
    def verifier(self, response, identity=None):
        pipeline = AsyncMock()
        pipeline.ainvoke.return_value = deepcopy(IDENTITY if identity is None else identity)
        captured = []
        def respond(prompt):
            captured.append(json.loads(prompt.to_messages()[1].content))
            if isinstance(response, Exception):
                raise response
            return AIMessage(content=response)
        comparator = ClaimComparator(RunnableLambda(respond))
        return CVVerifier(pipeline, comparator, ClaimComparisonValidator(), CVScorer()), captured

    async def test_compares_only_validated_selected_profile_not_first_retrieved(self):
        findings = [{'claim_id': 'name', 'cv_claim': CLAIMS['name'], 'status': 'MATCH',
                     'evidence': [{'path': 'name', 'value': 'Person'}], 'reason': 'Same name'}]
        decisions = {'findings': [{'claim_id': f['claim_id'], 'status': f['status'],
                                  'evidence_paths': [e['path'] for e in f['evidence']],
                                  'reason': f['reason']} for f in findings]}
        verifier, captured = self.verifier(json.dumps(decisions))
        result = await verifier.ainvoke('CV')
        self.assertEqual(captured[0]['profile']['name'], 'Person')
        self.assertEqual(result['score'], 0.9)
        self.assertEqual(result['findings'], findings)

    async def test_unresolved_or_invalid_selection_never_compares(self):
        for selection in (None, 99, True):
            identity = deepcopy(IDENTITY)
            identity['selected_candidate_id'] = selection
            if selection is None:
                identity['identity'] = 'unresolved'
            verifier, captured = self.verifier('[]', identity)
            result = await verifier.ainvoke('CV')
            self.assertEqual(result['score'], 0.5)
            self.assertEqual(result['identity'], 'unresolved')
            self.assertEqual(captured, [])

    async def test_validated_mismatch_and_unknown_reach_scorer(self):
        for status, expected in (('MISMATCH', 0.1), ('UNKNOWN', 0.5)):
            with self.subTest(status=status):
                identity = deepcopy(IDENTITY)
                identity['profiles'][1]['name'] = 'Different Name'
                findings = [{'claim_id': 'name', 'cv_claim': CLAIMS['name'], 'status': status,
                             'evidence': [] if status == 'UNKNOWN' else [
                                 {'path': 'name', 'value': 'Different Name'}],
                             'reason': 'Contradiction' if status == 'MISMATCH' else 'Insufficient evidence'}]
                decisions = {'findings': [{'claim_id': f['claim_id'], 'status': f['status'],
                                          'evidence_paths': [e['path'] for e in f['evidence']],
                                          'reason': f['reason']} for f in findings]}
                verifier, _ = self.verifier(json.dumps(decisions), identity)
                result = await verifier.ainvoke('CV')
                self.assertEqual(result['score'], expected)
                self.assertEqual(result['findings'], findings)

    async def test_malformed_incomplete_or_failed_comparison_isolated(self):
        for response in ('not JSON', '[]', '[{"status":"MATCH"}]', RuntimeError('failed')):
            verifier, _ = self.verifier(response)
            result = await verifier.ainvoke('CV')
            self.assertEqual(result['score'], 0.5)
            self.assertEqual(result['findings'], [])
            self.assertEqual(result['errors'][-1]['stage'], 'comparison')

    async def test_identity_processing_failure_scores_fallback(self):
        verifier, captured = self.verifier('[]')
        verifier.identity_pipeline.ainvoke.side_effect = RuntimeError('failed')
        result = await verifier.ainvoke('CV')
        self.assertEqual(result['score'], 0.5)
        self.assertEqual(result['identity'], 'unresolved')
        self.assertEqual(captured, [])
