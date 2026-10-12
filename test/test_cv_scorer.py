import unittest
from cv_verifier.cv_scorer import CVScorer


def finding(reference, status):
    return {'claim_id': reference, 'cv_claim': {'value': 'Person', 'raw': 'Person'},
            'status': status, 'evidence': [] if status == 'UNKNOWN' else [{'path': 'name', 'value': 'Person'}],
            'reason': 'Supplied evidence'}


class CVScorerTests(unittest.TestCase):
    def test_all_matches_score_high(self):
        self.assertEqual(CVScorer().score([finding('name', 'MATCH')], identity_resolved=True), 0.9)

    def test_one_mismatch_outweighs_many_matches_and_unknowns(self):
        findings = [finding(f'claim.{n}', 'MATCH') for n in range(19)]
        findings += [finding('false', 'MISMATCH'), finding('missing', 'UNKNOWN')]
        self.assertEqual(CVScorer().score(findings, identity_resolved=True), 0.1)

    def test_unknown_without_mismatch_scores_fallback(self):
        self.assertEqual(CVScorer().score([finding('name', 'MATCH'), finding('city', 'UNKNOWN')],
                                        identity_resolved=True), 0.5)

    def test_unresolved_identity_scores_fallback_even_with_matches(self):
        self.assertEqual(CVScorer().score([finding('name', 'MATCH')], identity_resolved=False), 0.5)

    def test_invalid_or_empty_comparison_cannot_score_valid(self):
        for findings in (None, [], {}, [{'status': 'MATCH'}], [finding('name', 'INVALID')],
                         [finding('name', 'MATCH'), finding('name', 'MATCH')]):
            with self.subTest(findings=findings):
                self.assertEqual(CVScorer().score(findings, identity_resolved=True), 0.5)
