import unittest
from copy import deepcopy

from cv_verifier.claim_comparison_validator import ClaimComparisonValidator
from evaluation.experimental.deterministic_claim_verifier import DeterministicClaimVerifier
from test_claim_comparator import CLAIMS, PROFILE
from test_record_aligner import ALIGNMENTS


class DeterministicClaimVerifierTests(unittest.TestCase):
    def verify(self, claims=None, profile=None, alignments=None, authoritative=False):
        claims = deepcopy(CLAIMS if claims is None else claims)
        profile = deepcopy(PROFILE if profile is None else profile)
        findings = DeterministicClaimVerifier().verify(claims, profile,
            deepcopy(ALIGNMENTS if alignments is None else alignments), skills_authoritative=authoritative)
        ClaimComparisonValidator().validate(findings, claims, profile)
        self.assertEqual(set(f['claim_id'] for f in findings), set(ClaimComparisonValidator().claim_catalog(claims)))
        return {finding['claim_id']: finding for finding in findings}

    def test_exact_fields_and_reordered_alignment_use_canonical_values(self):
        claims = deepcopy(CLAIMS)
        claims['education'][0]['degree'] = 'MBA'
        findings = self.verify(claims)
        self.assertTrue(all(f['status'] == 'MATCH' for f in findings.values()))
        self.assertEqual(findings['jobs.0.start_year']['evidence'], [{'path': 'experience.1.start_year', 'value': 2020}])
        self.assertEqual(findings['jobs.0.title']['cv_claim']['value'], 'Senior Manager')
        self.assertIs(type(findings['jobs.1.is_current']['evidence'][0]['value']), bool)

    def test_education_alignment_is_not_array_position(self):
        claims, profile, mappings = deepcopy(CLAIMS), deepcopy(PROFILE), deepcopy(ALIGNMENTS)
        profile['education'].insert(0, {'school': 'Other school', 'degree': 'Other degree', 'end_year': 1990})
        mappings['education'][0]['profile_index'] = 1
        findings = self.verify(claims, profile, mappings)
        self.assertEqual(findings['education.0.school']['evidence'][0]['path'], 'education.1.school')

    def test_employment_year_current_boolean_and_graduation_contradictions(self):
        for reference, source, index, field, value in (
            ('jobs.0.start_year', 'experience', 1, 'start_year', 2019),
            ('jobs.1.end_year', 'experience', 0, 'end_year', 2017),
            ('jobs.0.is_current', 'experience', 1, 'is_current', False),
            ('education.0.graduation_year', 'education', 0, 'end_year', 2011),
        ):
            with self.subTest(reference=reference):
                profile = deepcopy(PROFILE)
                profile[source][index][field] = value
                finding = self.verify(profile=profile)[reference]
                self.assertEqual(finding['status'], 'MISMATCH')
                self.assertEqual(finding['evidence'][0]['value'], value)

    def test_explicit_seniority_and_unqualified_titles(self):
        for title, level, profile_title, profile_level, expected in (
            ('Senior Analyst', None, 'Analyst', 'junior', 'MISMATCH'),
            ('Manager', None, 'Manager', 'junior', 'MATCH'),
            ('Analyst', None, 'Analyst', 'senior', 'MATCH'),
            ('Senior Manager', None, 'Manager', 'senior', 'MATCH'),
            ('Manager', 'senior', 'Manager', 'junior', 'MISMATCH'),
            ('Senior Manager', None, 'Senior Manager', None, 'MATCH'),
            ('Senior Manager', None, 'Junior Manager', 'senior', 'UNKNOWN'),
        ):
            with self.subTest(title=title, level=level, profile_level=profile_level):
                claims, profile = deepcopy(CLAIMS), deepcopy(PROFILE)
                claims['jobs'][0].update(title=title, seniority=level)
                profile['experience'][1].update(title=profile_title, seniority=profile_level)
                findings = self.verify(claims, profile)
                self.assertEqual(findings['jobs.0.title']['status'], expected)
                if level is not None:
                    self.assertEqual(findings['jobs.0.seniority']['status'], 'MISMATCH')

    def test_different_names_employers_schools_degrees_fields_and_titles_remain_unknown(self):
        profile = deepcopy(PROFILE)
        profile['name'] = 'Nickname'
        profile['experience'][1].update(company='Company Alias', title='Lead')
        profile['education'][0].update(school='School Alias', field='Related field')
        findings = self.verify(profile=profile)
        for reference in ('name', 'jobs.0.company', 'jobs.0.title', 'education.0.school',
                          'education.0.degree', 'education.0.field'):
            self.assertEqual(findings[reference]['status'], 'UNKNOWN')

    def test_missing_typed_evidence_is_unknown_without_coercion(self):
        for value in (None, '2020', True):
            profile = deepcopy(PROFILE)
            profile['experience'][1]['start_year'] = value
            self.assertEqual(self.verify(profile=profile)['jobs.0.start_year']['status'], 'UNKNOWN')
        profile = deepcopy(PROFILE)
        profile['experience'][1]['is_current'] = 1
        self.assertEqual(self.verify(profile=profile)['jobs.0.is_current']['status'], 'UNKNOWN')

    def test_null_alignment_never_declares_fabricated_records(self):
        mappings = deepcopy(ALIGNMENTS)
        for entries in mappings.values():
            for entry in entries:
                entry['profile_index'] = None
        findings = self.verify(alignments=mappings)
        self.assertTrue(all(f['status'] == 'UNKNOWN' for ref, f in findings.items()
                            if ref.startswith(('jobs.', 'education.'))))

    def test_skills_exact_alias_nonexact_and_authoritative_empty_rules(self):
        for claim, actual, authoritative, expected in (
            (' PYTHON ', 'python', False, 'MATCH'), ('UI/UX Design', 'UI/UX', True, 'UNKNOWN'),
            ('Docker', 'Kubernetes', True, 'UNKNOWN'), ('JavaScript', 'JS', True, 'UNKNOWN'),
            ('C++', 'C#', True, 'UNKNOWN'), ('B.Sc.', 'BSc', True, 'UNKNOWN'),
            ('Python', None, True, 'UNKNOWN'),
        ):
            with self.subTest(claim=claim, actual=actual):
                claims, profile = deepcopy(CLAIMS), deepcopy(PROFILE)
                claims['skills'][0]['name'] = claim
                profile['skills'] = None if actual is None else [{'name': actual}]
                self.assertEqual(self.verify(claims, profile, authoritative=authoritative)['skills.0.name']['status'], expected)
        for authoritative, expected in ((False, 'UNKNOWN'), (True, 'MISMATCH')):
            findings = self.verify(profile=dict(PROFILE, skills=[]), authoritative=authoritative)
            self.assertEqual(findings['skills.0.name']['status'], expected)
            if authoritative:
                self.assertEqual(findings['skills.0.name']['evidence'], [{'path': 'skills', 'value': []}])

    def test_city_reuses_membership_but_never_hometown_or_automatic_nonmembership(self):
        claims = deepcopy(CLAIMS)
        claims['city'] = {'raw': 'Beijing | Hong Kong', 'value': None, 'candidates': [
            {'city': 'Beijing', 'role': 'unspecified'}, {'city': 'Hong Kong', 'role': 'unspecified'}]}
        self.assertEqual(self.verify(claims, dict(PROFILE, city='Beijing'))['city']['status'], 'MATCH')
        self.assertEqual(self.verify(claims, dict(PROFILE, city='Jakarta'))['city']['status'], 'UNKNOWN')
        claims['city'] = {'raw': 'Beijing; hometown Hong Kong', 'value': 'Beijing', 'candidates': [
            {'city': 'Beijing', 'role': 'current'}, {'city': 'Hong Kong', 'role': 'hometown'}]}
        self.assertEqual(self.verify(claims, dict(PROFILE, city='Hong Kong'))['city']['status'], 'UNKNOWN')
        self.assertEqual(self.verify(claims, dict(PROFILE, city=None))['city']['status'], 'UNKNOWN')

    def test_empty_skill_mismatch_requires_explicit_boolean_authority(self):
        for unestablished in (None, 1, 'true'):
            with self.subTest(authoritative=unestablished):
                findings = self.verify(profile=dict(PROFILE, skills=[]), authoritative=unestablished)
                self.assertEqual(findings['skills.0.name']['status'], 'UNKNOWN')

    def test_input_data_is_not_mutated(self):
        claims, profile, mappings = deepcopy(CLAIMS), deepcopy(PROFILE), deepcopy(ALIGNMENTS)
        DeterministicClaimVerifier().verify(claims, profile, mappings)
        self.assertEqual((claims, profile, mappings), (CLAIMS, PROFILE, ALIGNMENTS))
