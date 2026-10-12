import unittest
from copy import deepcopy
from cv_verifier.claim_comparison_validator import ClaimComparisonValidator

CLAIMS = {'name': {'raw': 'Person', 'value': 'Person'}, 'city': {'raw': None, 'value': None},
          'jobs': [], 'education': [], 'skills': [{'name': 'Python', 'raw': 'Python'}]}
PROFILE = {'name': 'Person', 'skills': [{'name': 'Python'}], 'headline': 'Excluded'}
FINDINGS = [
    {'claim_id': 'name', 'cv_claim': CLAIMS['name'] | {'value': 'Person'}, 'status': 'MATCH',
     'evidence': [{'path': 'name', 'value': 'Person'}], 'reason': 'Same name'},
    {'claim_id': 'skills.0.name', 'cv_claim': {'raw': 'Python', 'value': 'Python'}, 'status': 'MATCH',
     'evidence': [{'path': 'skills.0.name', 'value': 'Python'}], 'reason': 'Same skill'},
]


class ClaimComparisonValidatorTests(unittest.TestCase):
    def test_valid_findings_returned_unchanged(self):
        findings = deepcopy(FINDINGS)
        self.assertIs(ClaimComparisonValidator().validate(findings, CLAIMS, PROFILE), findings)
        self.assertEqual(findings, FINDINGS)

    def test_missing_duplicate_and_invented_references_rejected(self):
        invented = deepcopy(FINDINGS)
        invented[1]['claim_id'] = 'skills.99.name'
        for findings in ([], FINDINGS[:1], FINDINGS + [FINDINGS[0]], invented):
            with self.subTest(findings=findings), self.assertRaises(ValueError):
                ClaimComparisonValidator().validate(findings, CLAIMS, PROFILE)

    def test_malformed_findings_rejected(self):
        cases = [(None, None), ('status', 'VALID'), ('status', None), ('status', []),
                 ('claim_id', None), ('claim_id', []), ('reason', ''), ('reason', 3),
                 ('evidence', None), ('cv_claim', {}), ('cv_claim', {'raw': 'Python', 'value': 'Rust'})]
        for key, value in cases:
            findings = deepcopy(FINDINGS)
            if key is None:
                findings[0] = None
            else:
                findings[1][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                ClaimComparisonValidator().validate(findings, CLAIMS, PROFILE)
        for value in (None, {}, '[]'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ClaimComparisonValidator().validate(value, CLAIMS, PROFILE)

    def test_missing_or_extra_finding_keys_rejected(self):
        for key in FINDINGS[0]:
            findings = deepcopy(FINDINGS)
            del findings[0][key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                ClaimComparisonValidator().validate(findings, CLAIMS, PROFILE)
        findings = deepcopy(FINDINGS)
        findings[0]['score'] = 0.9
        with self.assertRaises(ValueError):
            ClaimComparisonValidator().validate(findings, CLAIMS, PROFILE)

    def test_unknown_allows_missing_evidence_but_match_and_mismatch_do_not(self):
        for status in ('MATCH', 'MISMATCH', 'UNKNOWN'):
            findings = deepcopy(FINDINGS)
            findings[1]['status'] = status
            findings[1]['evidence'] = []
            if status == 'UNKNOWN':
                self.assertEqual(ClaimComparisonValidator().validate(findings, CLAIMS, PROFILE), findings)
            else:
                with self.subTest(status=status), self.assertRaises(ValueError):
                    ClaimComparisonValidator().validate(findings, CLAIMS, PROFILE)

    def test_invented_changed_excluded_or_malformed_evidence_rejected(self):
        for citation in ({'path': 'skills.9.name', 'value': 'Python'},
                         {'path': 'headline', 'value': 'Excluded'},
                         {'path': 'skills.0.name', 'value': 'Rust'},
                         {'path': 'skills.0.name'}, {'path': [], 'value': 'Python'}, None):
            findings = deepcopy(FINDINGS)
            findings[1]['evidence'] = [citation]
            with self.subTest(citation=citation), self.assertRaises(ValueError):
                ClaimComparisonValidator().validate(findings, CLAIMS, PROFILE)

    def test_unparsed_dates_and_records_are_not_silently_omitted(self):
        claims = deepcopy(CLAIMS)
        claims['jobs'] = [{'raw': 'Unclear employer and role', 'company': None, 'title': None,
                           'seniority': None, 'start_year': None, 'end_year': None,
                           'is_current': None, 'dates_raw': None},
                          {'raw': 'Ambiguous dates', 'company': None, 'title': None,
                           'seniority': None, 'start_year': None, 'end_year': None,
                           'is_current': None, 'dates_raw': 'Early 2000s'}]
        catalog = ClaimComparisonValidator().claim_catalog(claims)
        self.assertIn('jobs.0.record', catalog)
        self.assertIn('jobs.1.dates', catalog)
        with self.assertRaisesRegex(ValueError, 'Missing comparison'):
            ClaimComparisonValidator().validate(FINDINGS, claims, PROFILE)

    def test_excluded_job_descriptions_cannot_be_cited(self):
        profile = deepcopy(PROFILE)
        profile['experience'] = [{'description': 'Expert in Python', 'title': 'Engineer'}]
        findings = deepcopy(FINDINGS)
        findings[1]['evidence'] = [{'path': 'experience.0.description', 'value': 'Expert in Python'}]
        with self.assertRaisesRegex(ValueError, 'excluded profile field'):
            ClaimComparisonValidator().validate(findings, CLAIMS, profile)

    def test_null_profile_value_cannot_support_match(self):
        profile = deepcopy(PROFILE)
        profile['name'] = None
        findings = deepcopy(FINDINGS)
        findings[0]['evidence'][0]['value'] = None
        with self.assertRaisesRegex(ValueError, 'Missing profile values'):
            ClaimComparisonValidator().validate(findings, CLAIMS, profile)
