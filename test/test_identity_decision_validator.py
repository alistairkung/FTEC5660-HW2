import unittest
from cv_verifier.identity_decision_validator import IdentityDecisionValidator


class IdentityDecisionValidatorTests(unittest.TestCase):
    def test_valid_selection_or_null_returned_unchanged(self):
        for candidate_id in (1, None):
            decision = {'candidate_id': candidate_id, 'reason': 'Evidence supports this decision.'}
            with self.subTest(candidate_id=candidate_id):
                self.assertIs(IdentityDecisionValidator().validate(decision, [{'id': 1}]), decision)

    def test_invalid_decisions_rejected(self):
        invalid = [None, [], {}, {'candidate_id': 1}, {'reason': 'missing ID'},
                   {'candidate_id': 1, 'reason': 'ok', 'score': 0.9}]
        invalid.extend({'candidate_id': value, 'reason': 'ok'} for value in (99, True, '1', 1.0, []))
        invalid.extend({'candidate_id': 1, 'reason': value} for value in (None, 12, '', '  '))
        for decision in invalid:
            with self.subTest(decision=decision), self.assertRaises(ValueError):
                IdentityDecisionValidator().validate(decision, [{'id': 1}])

    def test_no_profiles_allows_only_null(self):
        self.assertEqual(IdentityDecisionValidator().validate(
            {'candidate_id': None, 'reason': 'No evidence'}, [])['candidate_id'], None)
        with self.assertRaises(ValueError):
            IdentityDecisionValidator().validate({'candidate_id': 1, 'reason': 'invented'}, [])
