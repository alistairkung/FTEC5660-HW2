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
DECISIONS = {'findings': [
    {'claim_id': finding['claim_id'], 'status': finding['status'],
     'evidence_paths': [citation['path'] for citation in finding['evidence']],
     'reason': finding['reason']} for finding in FINDINGS]}


class ClaimComparisonValidatorTests(unittest.TestCase):
    def test_reconstructs_canonical_claims_and_evidence_without_mutating_inputs(self):
        claims, profile, decisions = deepcopy(CLAIMS), deepcopy(PROFILE), deepcopy(DECISIONS)
        validator = ClaimComparisonValidator()
        findings = validator.reconstruct(decisions, claims, profile)
        self.assertEqual(findings, FINDINGS)
        self.assertEqual({f['claim_id']: f['cv_claim'] for f in findings}, validator.claim_catalog(claims))
        findings[0]['cv_claim']['raw'] = 'changed'
        findings[0]['evidence'][0]['value'] = 'changed'
        self.assertEqual(claims, CLAIMS)
        self.assertEqual(profile, PROFILE)
        self.assertEqual(decisions, DECISIONS)

    def test_reconstruction_preserves_numeric_boolean_and_string_types(self):
        for value in (2020, True, False, '2020', 'true'):
            with self.subTest(value=value):
                claims = deepcopy(CLAIMS)
                claims['skills'] = []
                claims['jobs'] = [{'raw': 'Original CV wording', 'company': None, 'title': None,
                                   'seniority': None, 'start_year': None, 'end_year': None,
                                   'is_current': None, 'dates_raw': None}]
                field = 'is_current' if type(value) is bool else 'start_year' if type(value) is int else 'company'
                claims['jobs'][0][field] = value
                profile = {'name': 'Person', 'experience': [{field: value}]}
                output = {'findings': [deepcopy(DECISIONS['findings'][0]),
                    {'claim_id': f'jobs.0.{field}', 'status': 'MATCH',
                     'evidence_paths': [f'experience.0.{field}'], 'reason': 'Same supplied value'}]}
                finding = ClaimComparisonValidator().reconstruct(output, claims, profile)[1]
                self.assertIs(type(finding['cv_claim']['value']), type(value))
                self.assertIs(type(finding['evidence'][0]['value']), type(value))
                self.assertEqual(finding['cv_claim']['value'], value)
                self.assertEqual(finding['evidence'][0]['value'], value)

    def test_reconstruction_rejects_missing_duplicate_or_invented_claim_ids(self):
        for references in ([], ['name'], ['name', 'name'], ['name', 'skills.99.name'], ['name', None]):
            output = deepcopy(DECISIONS)
            output['findings'] = [dict(DECISIONS['findings'][0], claim_id=ref) for ref in references]
            with self.subTest(references=references), self.assertRaises(ValueError):
                ClaimComparisonValidator().reconstruct(output, CLAIMS, PROFILE)

    def test_reconstruction_rejects_invalid_duplicate_excluded_or_null_paths(self):
        profile = deepcopy(PROFILE)
        profile.update(city=None, hometown='Excluded', experience=[{'description': 'Excluded'}])
        for paths in (None, 'name', [None], [[]], ['skills.9.name'], ['headline'],
                      ['hometown'], ['experience.0.description'], ['city'], ['name', 'name'], []):
            output = deepcopy(DECISIONS)
            output['findings'][0]['evidence_paths'] = paths
            with self.subTest(paths=paths), self.assertRaises(ValueError):
                ClaimComparisonValidator().reconstruct(output, CLAIMS, profile)

    def test_reconstruction_rejects_malformed_envelopes_and_decisions(self):
        for output in (None, [], {}, {'findings': None}, {'findings': {}},
                       {'findings': [], 'score': 0.9}, {'findings': [None]}):
            with self.subTest(output=output), self.assertRaises(ValueError):
                ClaimComparisonValidator().reconstruct(output, CLAIMS, PROFILE)
        mutations = [(key, None) for key in DECISIONS['findings'][0]]
        mutations += [('status', 'VALID'), ('status', []), ('reason', ''), ('reason', '  '),
                      ('reason', 3), ('cv_claim', {'value': '2020', 'raw': 'invented'}),
                      ('evidence', [{'path': 'name', 'value': 'invented'}])]
        for key, value in mutations:
            output = deepcopy(DECISIONS)
            output['findings'][0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                ClaimComparisonValidator().reconstruct(output, CLAIMS, PROFILE)
        for key in DECISIONS['findings'][0]:
            output = deepcopy(DECISIONS)
            del output['findings'][0][key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                ClaimComparisonValidator().reconstruct(output, CLAIMS, PROFILE)

    def test_reconstruction_unknown_without_evidence_and_empty_absence_collection(self):
        for status, paths in (('UNKNOWN', []), ('MISMATCH', ['skills'])):
            output = deepcopy(DECISIONS)
            output['findings'][1].update(status=status, evidence_paths=paths)
            profile = dict(PROFILE, skills=[])
            findings = ClaimComparisonValidator().reconstruct(output, CLAIMS, profile)
            self.assertEqual(findings[1]['status'], status)
            self.assertEqual(findings[1]['evidence'], [] if not paths else [{'path': 'skills', 'value': []}])

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
