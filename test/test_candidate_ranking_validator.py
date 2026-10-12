import unittest
from cv_verifier.candidate_ranking_validator import CandidateRankingValidator


class CandidateRankingValidatorTests(unittest.TestCase):
    def test_valid_permutation_returned_unchanged(self):
        ranking = [2, 1]
        self.assertIs(CandidateRankingValidator().validate(ranking, [{'id': 1}, {'id': 2}]), ranking)

    def test_empty_ranking_for_empty_hits(self):
        self.assertEqual(CandidateRankingValidator().validate([], []), [])

    def test_invalid_rankings_rejected_without_repair(self):
        for ranking in (None, {}, '1,2', [1], [], [1, 1], [1, 3], [1, 2, 3], [True, 2], [1.0, 2], ['1', 2]):
            with self.subTest(ranking=ranking), self.assertRaises(ValueError):
                CandidateRankingValidator().validate(ranking, [{'id': 1}, {'id': 2}])
