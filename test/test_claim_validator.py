import unittest
from copy import deepcopy

from cv_verifier.claim_validator import ClaimValidator


VALID_CLAIMS = {
    'name': {'raw': 'Example Person', 'value': 'Example Person'},
    'city': {'raw': None, 'value': None},
    'jobs': [{'raw': 'Engineer 2019–Present', 'company': None, 'title': 'Engineer',
              'seniority': None, 'dates_raw': '2019–Present', 'start_year': 2019,
              'end_year': None, 'is_current': True}],
    'education': [{'raw': 'BSc', 'school': None, 'degree': 'BSc', 'field': None,
                   'dates_raw': None, 'graduation_year': None}],
    'skills': [{'raw': 'Python', 'name': 'Python'}],
}


class ClaimValidatorTests(unittest.TestCase):
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
                if (field, key) != ('name', 'value'):
                    item[key] = None
        self.assertIs(ClaimValidator().validate(claims), claims)

    def test_empty_collections_allowed(self):
        claims = deepcopy(VALID_CLAIMS)
        for field in ('jobs', 'education', 'skills'):
            claims[field] = []
        self.assertEqual(ClaimValidator().validate(claims), claims)
