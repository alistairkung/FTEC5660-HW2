import unittest
from cv_verifier.candidate_ranking_validator import CandidateRankingValidator


class CandidateRankingValidatorTests(unittest.TestCase):
    def test_valid_shortlists_returned_unchanged(self):
        hits = [{'id': n} for n in range(1, 8)]
        for ranking in ([2], [2, 1], [5, 3, 2, 1, 4]):
            with self.subTest(ranking=ranking):
                self.assertIs(CandidateRankingValidator().validate(ranking, hits), ranking)

    def test_empty_ranking_for_empty_hits(self):
        self.assertEqual(CandidateRankingValidator().validate([], []), [])

    def test_invalid_rankings_rejected_without_repair(self):
        for ranking in (None, {}, '1,2', [], [1, 1], [1, 3], [1, 2, 3], [True, 2], [1.0, 2], ['1', 2]):
            with self.subTest(ranking=ranking), self.assertRaises(ValueError):
                CandidateRankingValidator().validate(ranking, [{'id': 1}, {'id': 2}])

    def test_six_known_ids_exceed_shortlist_limit(self):
        with self.assertRaisesRegex(ValueError, 'at most five'):
            CandidateRankingValidator().validate(list(range(6)), [{'id': n} for n in range(6)])
