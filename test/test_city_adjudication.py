import json
import unittest
from copy import deepcopy

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from cv_verifier.claim_comparator import ClaimComparator
from cv_verifier.claim_comparison_validator import ClaimComparisonValidator


CLAIMS = {
    'name': {'raw': 'Person', 'value': 'Person'},
    'city': {'raw': 'Beijing, China | Hong Kong', 'value': None, 'candidates': [
        {'city': 'Beijing', 'role': 'unspecified'}, {'city': 'Hong Kong', 'role': 'unspecified'}]},
    'jobs': [], 'education': [], 'skills': [{'raw': 'Python', 'name': 'Python'}],
}
PROFILE = {'name': 'Person', 'city': 'Beijing', 'country': 'China', 'skills': [{'name': 'Python'}]}


class CityAdjudicationTests(unittest.TestCase):
    def findings(self, claims, profile, status='UNKNOWN'):
        catalog = ClaimComparisonValidator().claim_catalog(claims)
        return [
            {'claim_id': reference, 'cv_claim': deepcopy(claim), 'status': status if reference == 'city' else 'UNKNOWN',
             'evidence': [{'path': 'city', 'value': profile['city']}]
                         if reference == 'city' and status != 'UNKNOWN' else [],
             'reason': 'Original model judgment'} for reference, claim in catalog.items()
        ]

    def test_ambiguous_header_unknown_and_mismatch_become_canonical_match(self):
        contract = ClaimComparisonValidator()
        for status in ('UNKNOWN', 'MISMATCH', 'MATCH'):
            with self.subTest(status=status):
                claims, profile = deepcopy(CLAIMS), deepcopy(PROFILE)
                baseline = self.findings(claims, profile, status)
                original = deepcopy(baseline)
                result = contract.adjudicate_city(baseline, claims, profile)
                city = next(f for f in result if f['claim_id'] == 'city')
                self.assertEqual(city['status'], 'MATCH')
                self.assertEqual(city['cv_claim'], contract.claim_catalog(claims)['city'])
                self.assertEqual(city['evidence'], [{'path': 'city', 'value': 'Beijing'}])
                self.assertEqual([f for f in result if f['claim_id'] != 'city'],
                                 [f for f in original if f['claim_id'] != 'city'])
                self.assertEqual(baseline, original)
                self.assertEqual(claims, CLAIMS)
                self.assertEqual(profile, PROFILE)
                self.assertIs(contract.validate(result, claims, profile), result)
                if status == 'MATCH':
                    self.assertEqual(result, original)

    def test_current_role_matches_but_hometown_and_previous_roles_never_override(self):
        contract = ClaimComparisonValidator()
        for role in ('hometown', 'previous'):
            claims = deepcopy(CLAIMS)
            claims['city'] = {'raw': f'Current Beijing; {role} Hong Kong', 'value': 'Beijing', 'candidates': [
                {'city': 'Beijing', 'role': 'current'}, {'city': 'Hong Kong', 'role': role}]}
            matching = self.findings(claims, PROFILE)
            self.assertEqual(contract.adjudicate_city(matching, claims, PROFILE)[1]['status'], 'MATCH')
            for status in ('UNKNOWN', 'MISMATCH'):
                profile = dict(PROFILE, city='Hong Kong')
                baseline = self.findings(claims, profile, status)
                with self.subTest(role=role, status=status):
                    self.assertEqual(contract.adjudicate_city(baseline, claims, profile), baseline)

    def test_only_hometown_or_previous_and_incidental_locations_remain_unknown(self):
        contract = ClaimComparisonValidator()
        for candidates in ([], [{'city': 'Beijing', 'role': 'hometown'}],
                           [{'city': 'Beijing', 'role': 'previous'}]):
            claims = deepcopy(CLAIMS)
            claims['city'] = {'raw': 'Beijing mentioned', 'value': None, 'candidates': candidates}
            baseline = self.findings(claims, PROFILE)
            self.assertEqual(contract.adjudicate_city(baseline, claims, PROFILE), baseline)
            with self.assertRaisesRegex(ValueError, 'must be UNKNOWN'):
                contract.adjudicate_city(self.findings(claims, PROFILE, 'MATCH'), claims, PROFILE)

    def test_non_membership_preserves_original_unknown_and_mismatch(self):
        profile = dict(PROFILE, city='Jakarta')
        for status in ('UNKNOWN', 'MISMATCH'):
            with self.subTest(status=status):
                baseline = self.findings(CLAIMS, profile, status)
                self.assertEqual(ClaimComparisonValidator().adjudicate_city(baseline, CLAIMS, profile), baseline)

    def test_grounded_docker_mismatch_is_preserved_when_city_is_adjudicated(self):
        claims = deepcopy(CLAIMS)
        claims['skills'] = [{'raw': 'Docker', 'name': 'Docker'}]
        baseline = self.findings(claims, PROFILE)
        baseline[-1].update(status='MISMATCH', evidence=[{'path': 'skills', 'value': PROFILE['skills']}],
                            reason='Docker is absent from the supplied skills collection')
        original = deepcopy(baseline[-1])
        result = ClaimComparisonValidator().adjudicate_city(baseline, claims, PROFILE)
        self.assertEqual(result[1]['status'], 'MATCH')
        self.assertEqual(result[-1], original)

    def test_missing_null_or_empty_profile_city_never_overrides(self):
        for profile in ({'name': 'Person', 'skills': PROFILE['skills']}, dict(PROFILE, city=None),
                        dict(PROFILE, city=''), dict(PROFILE, city='   '), dict(PROFILE, city=123)):
            with self.subTest(profile=profile):
                baseline = self.findings(CLAIMS, profile)
                self.assertEqual(ClaimComparisonValidator().adjudicate_city(baseline, CLAIMS, profile), baseline)

    def test_conservative_normalization_preserves_original_evidence_values(self):
        cases = [('  BEIJING  ', 'Beijing', None), ('New-York', 'new york', None),
                 ('Hong. Kong', 'Hong_Kong', None), ('Beijing, China', 'Beijing', 'China'),
                 ('Beijing (China)', 'Beijing', ' CHINA '), ('Beijing', ' Beijing, China ', 'China'),
                 ('New York, United States', 'New York', 'United States')]
        for candidate, city, country in cases:
            with self.subTest(candidate=candidate, city=city):
                claims = deepcopy(CLAIMS)
                claims['city']['candidates'] = [{'city': candidate, 'role': 'unspecified'}]
                profile = dict(PROFILE, city=city, country=country)
                result = ClaimComparisonValidator().adjudicate_city(self.findings(claims, profile), claims, profile)
                self.assertEqual(result[1]['status'], 'MATCH')
                self.assertEqual(result[1]['evidence'], [{'path': 'city', 'value': city}])
                self.assertEqual(result[1]['cv_claim']['value']['candidates'], claims['city']['candidates'])

    def test_distinct_cities_and_uncertain_country_qualifications_are_not_merged(self):
        cases = [('New York', 'York', None), ('Beijing', 'Hong Kong', 'China'),
                 ('NYC', 'New York', 'United States'), ('London, Canada', 'London', 'United Kingdom'),
                 ('Beijing, Chaoyang, China', 'Beijing', 'China'), ('Beijing, China', 'Beijing', None),
                 ('New Mexico', 'New', 'Mexico')]
        for candidate, city, country in cases:
            with self.subTest(candidate=candidate, city=city):
                claims = deepcopy(CLAIMS)
                claims['city']['candidates'] = [{'city': candidate, 'role': 'unspecified'}]
                profile = dict(PROFILE, city=city, country=country)
                baseline = self.findings(claims, profile)
                self.assertEqual(ClaimComparisonValidator().adjudicate_city(baseline, claims, profile), baseline)

    def test_invalid_findings_are_rejected_before_positive_override(self):
        good = self.findings(CLAIMS, PROFILE)
        cases = [None, [], good[:1], good + [good[1]]]
        for field, value in (('status', 'INVALID'), ('reason', ''), ('cv_claim', {'raw': 'invented', 'value': 'Beijing'}),
                             ('evidence', [{'path': 'name', 'value': 'Person'}]),
                             ('evidence', [{'path': 'city', 'value': 'Invented city'}])):
            corrupted = deepcopy(good)
            corrupted[1][field] = value
            cases.append(corrupted)
        for findings in cases:
            with self.subTest(findings=findings), self.assertRaises(ValueError):
                ClaimComparisonValidator().adjudicate_city(findings, CLAIMS, PROFILE)

    def test_adjudicated_city_remains_ineligible_for_reflection(self):
        contract = ClaimComparisonValidator()
        baseline = contract.adjudicate_city(self.findings(CLAIMS, PROFILE), CLAIMS, PROFILE)
        original = deepcopy(baseline)
        with self.assertRaisesRegex(ValueError, 'not eligible'):
            contract.apply_corrections({'corrections': [{'claim_id': 'city', 'status': 'MISMATCH',
                'evidence_paths': ['city'], 'reason': 'Override positive city match'}]}, baseline, CLAIMS, PROFILE)
        self.assertEqual(baseline, original)


class CityComparatorTests(unittest.IsolatedAsyncioTestCase):
    def decisions(self, status='UNKNOWN', paths=None):
        return {'findings': [{'claim_id': reference, 'status': status if reference == 'city' else 'UNKNOWN',
            'evidence_paths': (paths if paths is not None else ([] if status == 'UNKNOWN' else ['city']))
                              if reference == 'city' else [], 'reason': 'Model judgment'}
            for reference in ClaimComparisonValidator().claim_catalog(CLAIMS)]}

    async def test_comparator_adjudicates_with_one_model_call(self):
        for status in ('UNKNOWN', 'MISMATCH'):
            calls = []
            def respond(prompt):
                calls.append(prompt)
                return AIMessage(content=json.dumps(self.decisions(status)))
            with self.subTest(status=status):
                result = await ClaimComparator(RunnableLambda(respond)).compare(CLAIMS, PROFILE)
                self.assertEqual(result[1]['status'], 'MATCH')
                self.assertEqual(result[1]['evidence'], [{'path': 'city', 'value': 'Beijing'}])
                self.assertEqual(len(calls), 1)

    async def test_malformed_incomplete_and_fabricated_model_output_is_never_repaired(self):
        cases = [None, {'findings': []}, self.decisions('MATCH', []),
                 self.decisions('UNKNOWN', ['name']), self.decisions('UNKNOWN', ['headline']),
                 self.decisions('UNKNOWN', ['country']), self.decisions('UNKNOWN', ['city', 'city']),
                 self.decisions('UNKNOWN', ['invented'])]
        missing = self.decisions()
        del missing['findings'][1]
        cases.extend([missing, {'findings': self.decisions()['findings'] + [self.decisions()['findings'][1]]}])
        for output in cases:
            with self.subTest(output=output), self.assertRaises(ValueError):
                await ClaimComparator(RunnableLambda(lambda _: AIMessage(content=json.dumps(output)))).compare(CLAIMS, PROFILE)

    async def test_null_city_evidence_remains_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Missing profile values'):
            await ClaimComparator(RunnableLambda(lambda _: AIMessage(content=json.dumps(
                self.decisions('MISMATCH'))))).compare(CLAIMS, dict(PROFILE, city=None))
