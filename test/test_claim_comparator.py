import json
import unittest
from copy import deepcopy
from unittest.mock import patch

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from cv_verifier.claim_comparator import ClaimComparator, COMPARISON_PROMPT
from cv_verifier.claim_comparison_validator import ClaimComparisonValidator
from hw2 import build_claim_comparator

CLAIMS = {
    'name': {'raw': 'Example Person', 'value': 'Example Person'},
    'city': {'raw': 'London', 'value': 'London',
                   'candidates': [{'city': 'London', 'role': 'current'}]},
    'jobs': [
        {'raw': 'Senior Manager, Company A, 2020–Present', 'company': 'Company A',
         'title': 'Senior Manager', 'seniority': None, 'dates_raw': '2020–Present',
         'start_year': 2020, 'end_year': None, 'is_current': True},
        {'raw': 'Senior Scientist, Company B, 2012–2018', 'company': 'Company B',
         'title': 'Senior Scientist', 'seniority': None, 'dates_raw': '2012–2018',
         'start_year': 2012, 'end_year': 2018, 'is_current': False}],
    'education': [{'raw': 'Master of Business Administration, School A, graduated 2012',
                   'school': 'School A', 'degree': 'Master of Business Administration',
                   'field': 'Logistics', 'dates_raw': 'Graduated 2012', 'graduation_year': 2012}],
    'skills': [{'raw': 'Python', 'name': 'Python'}],
}
PROFILE = {
    'name': 'Example Person', 'city': 'London',
    'experience': [
        {'company': 'Company B', 'title': 'Scientist', 'seniority': 'senior',
         'start_year': 2012, 'end_year': 2018, 'is_current': False},
        {'company': 'Company A', 'title': 'Manager', 'seniority': 'senior',
         'start_year': 2020, 'end_year': None, 'is_current': True}],
    'education': [{'school': 'School A', 'degree': 'MBA', 'field': 'Logistics', 'end_year': 2012}],
    'skills': [{'name': 'Python'}, {'name': 'SQL'}],
    'headline': 'Excluded', 'hometown': 'Excluded',
}


class ClaimComparatorTests(unittest.IsolatedAsyncioTestCase):
    async def test_location_semantics_at_model_boundary(self):
        header = {'raw': 'Beijing, China | Hong Kong', 'value': None, 'candidates': [
            {'city': 'Beijing', 'role': 'unspecified'}, {'city': 'Hong Kong', 'role': 'unspecified'}]}
        explicit = {'raw': 'Current city Beijing; hometown Hong Kong', 'value': 'Beijing', 'candidates': [
            {'city': 'Beijing', 'role': 'current'}, {'city': 'Hong Kong', 'role': 'hometown'}]}
        cases = [
            (header, 'Beijing', 'MATCH', 'Profile city is in the unlabelled personal location header'),
            (header, 'Hong Kong', 'MATCH', 'Both unspecified candidates remain possible'),
            (header, 'Jakarta', 'MISMATCH', 'Complete personal location header contains only other cities'),
            (explicit, 'Beijing', 'MATCH', 'Explicit current city agrees'),
            (explicit, 'Hong Kong', 'MISMATCH', 'Hometown is not an alternative current city'),
            ({'raw': 'Meetings in Beijing and Hong Kong', 'value': None, 'candidates': []},
             'Jakarta', 'UNKNOWN', 'Incidental mentions do not establish residence'),
            ({'raw': 'Considering Beijing or Hong Kong among other locations', 'value': None,
              'candidates': header['candidates']}, 'Jakarta', 'UNKNOWN', 'Not an exhaustive residence claim'),
            ({'raw': 'Lives in Jakarta', 'value': 'Jakarta',
              'candidates': [{'city': 'Jakarta', 'role': 'current'}]},
             'Beijing', 'MISMATCH', 'Explicit current city contradicts profile city'),
            ({'raw': 'NYC / New York, USA | HK / Hong Kong', 'value': None,
              'candidates': [{'city': 'New York', 'role': 'unspecified'}, {'city': 'Hong Kong', 'role': 'unspecified'}]},
             'New York, USA', 'MATCH', 'Legitimate city alias and country qualification'),
            ({'raw': 'Hometown Hong Kong; previously Beijing', 'value': None,
              'candidates': [{'city': 'Hong Kong', 'role': 'hometown'}, {'city': 'Beijing', 'role': 'previous'}]},
             'Hong Kong', 'UNKNOWN', 'Neither candidate asserts current residence'),
            ({'raw': 'Location unclear', 'value': None, 'candidates': []},
             'Beijing', 'UNKNOWN', 'No parsed residence candidates'),
            (explicit, None, 'UNKNOWN', 'Profile city is unavailable'),
        ]
        for location, city, status, reason in cases:
            claims, profile = deepcopy(CLAIMS), deepcopy(PROFILE)
            claims['city'], profile['city'] = deepcopy(location), city
            with self.subTest(location=location, profile_city=city):
                findings = await self.comparison(claims, profile, {'city': {
                    'status': status, 'evidence': [] if status == 'UNKNOWN' else [{'path': 'city', 'value': city}],
                    'reason': reason}})
                finding = next(f for f in findings if f['claim_id'] == 'city')
                self.assertEqual(finding['status'], status)
                self.assertEqual(finding['cv_claim'], {'value': {
                    'value': location['value'], 'candidates': location['candidates']}, 'raw': location['raw']})
        self.assertIn('complete set of plausible claimed personal locations', COMPARISON_PROMPT)
        self.assertIn('Never use hometown or previous candidates as alternative current cities', COMPARISON_PROMPT)

    async def comparison(self, claims, profile, chosen):
        contract = ClaimComparisonValidator()
        if 'city' not in chosen:
            # Non-location tests use the common London/London fixture.
            chosen = {'city': {'status': 'MATCH', 'evidence': [{'path': 'city', 'value': 'London'}],
                               'reason': 'The supplied cities agree'}, **chosen}
        findings = [{'claim_id': reference, 'cv_claim': claim, 'status': 'UNKNOWN',
                     'evidence': [], 'reason': 'Insufficient evidence'}
                    for reference, claim in contract.claim_catalog(claims).items()]
        for finding in findings:
            if finding['claim_id'] in chosen:
                finding.update(chosen[finding['claim_id']])
        captured = []
        def respond(prompt):
            captured.extend(prompt.to_messages())
            return AIMessage(content=json.dumps({'findings': [
                {'claim_id': f['claim_id'], 'status': f['status'],
                 'evidence_paths': [e['path'] for e in f['evidence']], 'reason': f['reason']}
                for f in findings]}))
        output = await ClaimComparator(RunnableLambda(respond)).compare(claims, profile)
        self.assertIs(contract.validate(output, claims, profile), output)
        self.assertEqual(output, findings)
        self.assertEqual(captured[0].content, COMPARISON_PROMPT)
        context = json.loads(captured[1].content)
        self.assertEqual(context['claims'], claims)
        self.assertEqual(context['claim_catalog'], contract.claim_catalog(claims))
        self.assertNotIn('headline', context['profile'])
        self.assertNotIn('hometown', context['profile'])
        for entry in context['profile'].get('experience') or []:
            self.assertNotIn('description', entry)
        return output

    async def test_reconstructs_numeric_and_boolean_values_from_paths_only(self):
        profile = deepcopy(PROFILE)
        profile['experience'][0]['description'] = 'Excluded job description'
        findings = await self.comparison(CLAIMS, profile, {
            'jobs.0.start_year': {'status': 'MATCH', 'evidence': [
                {'path': 'experience.1.start_year', 'value': 2020}], 'reason': 'Same start year'},
            'jobs.0.is_current': {'status': 'MATCH', 'evidence': [
                {'path': 'experience.1.is_current', 'value': True}], 'reason': 'Both current'},
            'jobs.1.is_current': {'status': 'MATCH', 'evidence': [
                {'path': 'experience.0.is_current', 'value': False}], 'reason': 'Both ended'},
        })
        for reference, expected_type in (('jobs.0.start_year', int), ('jobs.0.is_current', bool),
                                         ('jobs.1.is_current', bool)):
            finding = next(f for f in findings if f['claim_id'] == reference)
            self.assertIs(type(finding['cv_claim']['value']), expected_type)
            self.assertIs(type(finding['evidence'][0]['value']), expected_type)

    async def test_semantic_title_and_degree_equivalences_with_reordered_records(self):
        await self.comparison(CLAIMS, PROFILE, {
            'jobs.0.title': {'status': 'MATCH', 'evidence': [
                {'path': 'experience.1.title', 'value': 'Manager'},
                {'path': 'experience.1.seniority', 'value': 'senior'}], 'reason': 'Title plus seniority is equivalent'},
            'jobs.1.title': {'status': 'MATCH', 'evidence': [
                {'path': 'experience.0.title', 'value': 'Scientist'},
                {'path': 'experience.0.seniority', 'value': 'senior'}], 'reason': 'Title plus seniority is equivalent'},
            'education.0.degree': {'status': 'MATCH', 'evidence': [
                {'path': 'education.0.degree', 'value': 'MBA'}], 'reason': 'Equivalent degree wording'},
        })

    async def test_explicit_employment_year_discrepancy(self):
        profile = deepcopy(PROFILE)
        profile['experience'][1]['start_year'] = 2019
        await self.comparison(CLAIMS, profile, {
            'jobs.0.start_year': {'status': 'MISMATCH', 'evidence': [
                {'path': 'experience.1.start_year', 'value': 2019}], 'reason': '2020 contradicts 2019'},
        })

    async def test_education_discrepancy(self):
        profile = deepcopy(PROFILE)
        profile['education'][0]['degree'] = 'BSc'
        await self.comparison(CLAIMS, profile, {
            'education.0.degree': {'status': 'MISMATCH', 'evidence': [
                {'path': 'education.0.degree', 'value': 'BSc'}], 'reason': 'Bachelor differs from MBA'},
        })

    async def test_explicit_graduation_year_discrepancy(self):
        profile = deepcopy(PROFILE)
        profile['education'][0]['end_year'] = 2011
        await self.comparison(CLAIMS, profile, {
            'education.0.graduation_year': {'status': 'MISMATCH', 'evidence': [
                {'path': 'education.0.end_year', 'value': 2011}], 'reason': '2012 contradicts 2011'},
        })

    async def test_missing_evidence_is_unknown(self):
        profile = deepcopy(PROFILE)
        del profile['education'][0]['end_year']
        output = await self.comparison(CLAIMS, profile, {})
        finding = next(f for f in output if f['claim_id'] == 'education.0.graduation_year')
        self.assertEqual(finding['status'], 'UNKNOWN')
        self.assertEqual(finding['evidence'], [])

    async def test_fabricated_employer_absent_from_complete_history_is_mismatch(self):
        claims = deepcopy(CLAIMS)
        claims['jobs'][0]['company'] = 'Fabricated Employer'
        claims['jobs'][0]['raw'] = 'Senior Manager, Fabricated Employer, 2020–Present'
        history = ClaimComparisonValidator().comparison_profile(PROFILE)['experience']
        findings = await self.comparison(claims, PROFILE, {
            'jobs.0.company': {'status': 'MISMATCH', 'evidence': [
                {'path': 'experience', 'value': history}],
                'reason': 'Claimed employer absent from complete history after semantic alignment'},
        })
        finding = next(f for f in findings if f['claim_id'] == 'jobs.0.company')
        self.assertEqual(finding['status'], 'MISMATCH')
        self.assertIn('complete history after semantic record alignment is MISMATCH', COMPARISON_PROMPT)

    async def test_fabricated_school_absent_from_complete_history_is_mismatch(self):
        claims = deepcopy(CLAIMS)
        claims['education'][0]['school'] = 'Fabricated School'
        claims['education'][0]['raw'] = 'MBA, Fabricated School, graduated 2012'
        history = ClaimComparisonValidator().comparison_profile(PROFILE)['education']
        findings = await self.comparison(claims, PROFILE, {
            'education.0.school': {'status': 'MISMATCH', 'evidence': [
                {'path': 'education', 'value': history}],
                'reason': 'No equivalent school in complete education history'},
        })
        self.assertEqual(next(f for f in findings if f['claim_id'] == 'education.0.school')['status'], 'MISMATCH')

    async def test_unavailable_and_incomplete_histories_remain_unknown(self):
        for collection in ('experience', 'education'):
            for unavailable in ('absent', None, [{}]):
                with self.subTest(collection=collection, evidence=unavailable):
                    profile = deepcopy(PROFILE)
                    if unavailable == 'absent':
                        del profile[collection]
                    else:
                        profile[collection] = unavailable
                    findings = await self.comparison(CLAIMS, profile, {})
                    reference = 'jobs.0.company' if collection == 'experience' else 'education.0.school'
                    finding = next(f for f in findings if f['claim_id'] == reference)
                    self.assertEqual(finding['status'], 'UNKNOWN')
                    self.assertEqual(finding['evidence'], [])
        self.assertIn('Do not assume an incomplete history proves a claim is fabricated', COMPARISON_PROMPT)

    async def test_explicit_empty_history_is_available_contradictory_evidence(self):
        profile = deepcopy(PROFILE)
        profile['experience'] = []
        findings = await self.comparison(CLAIMS, profile, {
            'jobs.0.company': {'status': 'MISMATCH', 'evidence': [
                {'path': 'experience', 'value': []}], 'reason': 'Complete history contains no employment'},
        })
        self.assertEqual(next(f for f in findings if f['claim_id'] == 'jobs.0.company')['status'], 'MISMATCH')

    async def test_skill_discrepancy_uses_explicit_skill_list(self):
        claims = deepcopy(CLAIMS)
        claims['skills'][0] = {'name': 'Rust', 'raw': 'Rust'}
        await self.comparison(claims, PROFILE, {
            'skills.0.name': {'status': 'MISMATCH', 'evidence': [
                {'path': 'skills', 'value': [{'name': 'Python'}, {'name': 'SQL'}]}],
                'reason': 'Rust absent from supplied skill list'},
        })

    async def test_strict_json_rejects_malformed_response(self):
        for content in ('not JSON', '[{"claim_id": "name"}', '```json\n[]\n```',
                        '[]', '{"findings": []}', '{"findings": null}'):
            with self.subTest(content=content), self.assertRaises(ValueError):
                comparator = ClaimComparator(RunnableLambda(lambda _: AIMessage(content=content)))
                await comparator.compare(CLAIMS, PROFILE)

    async def test_composition_root_injects_thinking_disabled_model(self):
        options = []
        def respond(_, **kwargs):
            options.append(kwargs)
            return AIMessage(content=json.dumps({'findings': [
                {'claim_id': ref, 'status': 'UNKNOWN', 'evidence_paths': [], 'reason': 'Unavailable'}
                for ref in ClaimComparisonValidator().claim_catalog(CLAIMS)]}))
        with patch('langchain_deepseek.ChatDeepSeek', return_value=RunnableLambda(respond)) as factory:
            comparator = build_claim_comparator()
        factory.assert_called_once_with(model='deepseek-v4-flash', timeout=60, max_retries=0)
        await comparator.compare(CLAIMS, PROFILE)
        self.assertEqual(options, [{'extra_body': {'thinking': {'type': 'disabled'}},
                                    'response_format': {'type': 'json_object'}}])
