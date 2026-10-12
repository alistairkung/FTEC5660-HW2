import unittest
from copy import deepcopy

from cv_verifier.claim_validator import ClaimValidator


VALID_CLAIMS = {
    'name': {'raw': 'Example Person', 'value': 'Example Person'},
    'city': {'raw': None, 'value': None, 'candidates': []},
    'jobs': [{'raw': 'Engineer 2019–Present', 'company': None, 'title': 'Engineer',
              'seniority': None, 'dates_raw': '2019–Present', 'start_year': 2019,
              'end_year': None, 'is_current': True}],
    'education': [{'raw': 'BSc', 'school': None, 'degree': 'BSc', 'field': None,
                   'dates_raw': None, 'graduation_year': None}],
    'skills': [{'raw': 'Python', 'name': 'Python'}],
}


class ClaimValidatorTests(unittest.TestCase):
    def test_location_roles_and_ambiguous_candidates_preserved_unchanged(self):
        locations = [
            {'raw': 'Beijing, China | Hong Kong', 'value': None, 'candidates': [
                {'city': 'Beijing', 'role': 'unspecified'}, {'city': 'Hong Kong', 'role': 'unspecified'}]},
            {'raw': 'Lives in Beijing; hometown Hong Kong; previously London', 'value': 'Beijing',
             'candidates': [{'city': 'Beijing', 'role': 'current'}, {'city': 'Hong Kong', 'role': 'hometown'},
                            {'city': 'London', 'role': 'previous'}]},
            {'raw': 'Current cities: Beijing and Hong Kong', 'value': None, 'candidates': [
                {'city': 'Beijing', 'role': 'current'}, {'city': 'Hong Kong', 'role': 'current'}]},
            {'raw': 'Location unclear', 'value': None, 'candidates': []},
        ]
        for location in locations:
            claims = deepcopy(VALID_CLAIMS)
            claims['city'] = deepcopy(location)
            with self.subTest(location=location):
                self.assertIs(ClaimValidator().validate(claims), claims)
                self.assertEqual(claims['city'], location)

    def test_malformed_or_duplicated_location_candidates_rejected(self):
        valid = {'city': 'Beijing', 'role': 'unspecified'}
        candidates = [None, {}, 'Beijing', [None], [{}],
                      [dict(valid, role=None)], [dict(valid, role='residence')],
                      [dict(valid, role=True)], [dict(valid, city=None)],
                      [dict(valid, city='')], [dict(valid, city=42)],
                      [dict(valid, country='China')], [valid, valid],
                      [valid, {'city': ' beijing ', 'role': 'unspecified'}]]
        for value in candidates:
            claims = deepcopy(VALID_CLAIMS)
            claims['city'] = {'raw': 'Beijing', 'value': None, 'candidates': value}
            with self.subTest(candidates=value), self.assertRaises(ValueError):
                ClaimValidator().validate(claims)

    def test_contradictory_current_city_representations_rejected(self):
        for value, candidates in (
            ('Hong Kong', [{'city': 'Beijing', 'role': 'unspecified'}, {'city': 'Hong Kong', 'role': 'unspecified'}]),
            ('Hong Kong', [{'city': 'Beijing', 'role': 'current'}, {'city': 'Hong Kong', 'role': 'hometown'}]),
            ('Beijing', []), (None, [{'city': 'Beijing', 'role': 'current'}]),
            ('Beijing', [{'city': 'Beijing', 'role': 'current'}, {'city': 'Hong Kong', 'role': 'current'}]),
            ('', []),
        ):
            claims = deepcopy(VALID_CLAIMS)
            claims['city'] = {'raw': 'Beijing | Hong Kong', 'value': value, 'candidates': candidates}
            with self.subTest(value=value, candidates=candidates), self.assertRaises(ValueError):
                ClaimValidator().validate(claims)
        claims = deepcopy(VALID_CLAIMS)
        claims['city'] = {'raw': None, 'value': 'Beijing', 'candidates': [{'city': 'Beijing', 'role': 'current'}]}
        with self.assertRaises(ValueError):
            ClaimValidator().validate(claims)

    def test_same_city_with_distinct_supported_roles_is_not_a_duplicate(self):
        claims = deepcopy(VALID_CLAIMS)
        claims['city'] = {'raw': 'Lives in London; hometown London', 'value': 'London',
                          'candidates': [{'city': 'London', 'role': 'current'}, {'city': 'London', 'role': 'hometown'}]}
        self.assertIs(ClaimValidator().validate(claims), claims)

    def test_valid_claims_returned_unchanged_without_mutation(self):
        claims = deepcopy(VALID_CLAIMS)
        self.assertIs(ClaimValidator().validate(claims), claims)
        self.assertEqual(claims, VALID_CLAIMS)

    def test_missing_top_level_keys_fail(self):
        for key in VALID_CLAIMS:
            with self.subTest(missing_key=key), self.assertRaises(ValueError):
                claims = deepcopy(VALID_CLAIMS)
                del claims[key]
                ClaimValidator().validate(claims)

    def test_missing_nested_keys_fail(self):
        for field, value in VALID_CLAIMS.items():
            item = value[0] if isinstance(value, list) else value
            for key in item:
                with self.subTest(field=field, missing_key=key), self.assertRaises(ValueError):
                    claims = deepcopy(VALID_CLAIMS)
                    target = claims[field][0] if isinstance(value, list) else claims[field]
                    del target[key]
                    ClaimValidator().validate(claims)

    def test_extra_keys_fail(self):
        for path in ((), ('name',), ('jobs', 0)):
            with self.subTest(path=path), self.assertRaises(ValueError):
                claims = deepcopy(VALID_CLAIMS)
                target = claims
                for key in path:
                    target = target[key]
                target['unexpected'] = None
                ClaimValidator().validate(claims)

    def test_non_list_collections_fail(self):
        for field in ('jobs', 'education', 'skills'):
            for value in (None, {}, 'unknown'):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    claims = deepcopy(VALID_CLAIMS)
                    claims[field] = value
                    ClaimValidator().validate(claims)

    def test_non_object_claims_fail(self):
        for field in ('name', 'city', 'jobs', 'education', 'skills'):
            with self.subTest(field=field), self.assertRaises(ValueError):
                claims = deepcopy(VALID_CLAIMS)
                claims[field] = [None] if isinstance(claims[field], list) else None
                ClaimValidator().validate(claims)

    def test_incorrect_scalar_types_fail_without_coercion(self):
        cases = [(('name', 'value'), 42), (('city', 'raw'), False),
                 (('jobs', 0, 'start_year'), True), (('jobs', 0, 'end_year'), '2020'),
                 (('jobs', 0, 'is_current'), 1), (('jobs', 0, 'title'), 4),
                 (('education', 0, 'graduation_year'), 2012.0), (('skills', 0, 'name'), [])]
        for path, value in cases:
            with self.subTest(path=path, value=value), self.assertRaises(ValueError):
                claims = deepcopy(VALID_CLAIMS)
                target = claims
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                ClaimValidator().validate(claims)

    def test_missing_or_blank_name_fails(self):
        for value in (None, '', '   '):
            with self.subTest(name=value), self.assertRaisesRegex(ValueError, 'name is missing'):
                claims = deepcopy(VALID_CLAIMS)
                claims['name']['value'] = value
                ClaimValidator().validate(claims)

    def test_null_scalar_values_allowed_except_candidate_name(self):
        claims = deepcopy(VALID_CLAIMS)
        for field, value in claims.items():
            item = value[0] if isinstance(value, list) else value
            for key in item:
                if (field, key) != ('name', 'value') and not isinstance(item[key], list):
                    item[key] = None
        self.assertIs(ClaimValidator().validate(claims), claims)

    def test_empty_collections_allowed(self):
        claims = deepcopy(VALID_CLAIMS)
        for field in ('jobs', 'education', 'skills'):
            claims[field] = []
        self.assertEqual(ClaimValidator().validate(claims), claims)
