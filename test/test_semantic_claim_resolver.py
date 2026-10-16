import json
import unittest
from copy import deepcopy

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from cv_verifier.claim_comparison_validator import ClaimComparisonValidator
from evaluation.experimental.deterministic_claim_verifier import DeterministicClaimVerifier
from evaluation.experimental.semantic_claim_resolver import SemanticClaimResolver, SEMANTIC_PROMPT
from test_claim_comparator import CLAIMS, PROFILE
from test_record_aligner import ALIGNMENTS


class SemanticClaimResolverTests(unittest.IsolatedAsyncioTestCase):
    def model(self, output):
        calls = []
        def respond(prompt):
            calls.append(prompt)
            if isinstance(output, Exception):
                raise output
            return AIMessage(content=output if isinstance(output, str) else json.dumps(output))
        return SemanticClaimResolver(RunnableLambda(respond)), calls

    def decisions(self, findings, changes=None):
        changes = changes or {}
        return {'findings': [dict(claim_id=f['claim_id'], status='UNKNOWN', evidence_paths=[],
                                  reason='Evidence remains insufficient') | changes.get(f['claim_id'], {})
                            for f in findings if f['status'] == 'UNKNOWN']}

    async def resolve(self, changes, claims=None, profile=None, alignments=None, findings=None):
        claims, profile = deepcopy(CLAIMS if claims is None else claims), deepcopy(PROFILE if profile is None else profile)
        alignments = deepcopy(ALIGNMENTS if alignments is None else alignments)
        findings = DeterministicClaimVerifier().verify(claims, profile, alignments) if findings is None else findings
        original = deepcopy((claims, profile, alignments, findings))
        resolver, calls = self.model(self.decisions(findings, changes))
        result = await resolver.resolve(claims, profile, alignments, findings)
        self.assertEqual((claims, profile, alignments, findings), original)
        ClaimComparisonValidator().validate(result, claims, profile)
        for before, after in zip(findings, result):
            if before['status'] != 'UNKNOWN':
                self.assertEqual(before, after)
        return {f['claim_id']: f for f in result}, calls

    async def test_no_unknown_skips_model_entirely(self):
        claims = deepcopy(CLAIMS)
        claims['education'][0]['degree'] = 'MBA'
        findings = DeterministicClaimVerifier().verify(claims, PROFILE, ALIGNMENTS)
        resolver, calls = self.model(RuntimeError('No call permitted'))
        self.assertEqual(await resolver.resolve(claims, PROFILE, ALIGNMENTS, findings), findings)
        self.assertEqual(calls, [])

    async def test_only_requested_claims_with_relevant_aligned_context_and_no_excluded_fields(self):
        profile = deepcopy(PROFILE)
        profile['experience'][0]['description'] = 'Excluded job text'
        result, calls = await self.resolve({'education.0.degree': {
            'status': 'MATCH', 'evidence_paths': ['education.0.degree'], 'reason': 'MBA is equivalent'}}, profile=profile)
        self.assertEqual(len(calls), 1)
        context = json.loads(calls[0].to_messages()[1].content)
        self.assertEqual(set(context), {'requested_claim_ids', 'claim_catalog', 'profile_evidence', 'record_context',
                                        'evidence_path_contracts'})
        self.assertEqual(context['requested_claim_ids'], ['education.0.degree'])
        self.assertEqual(context['claim_catalog'], {'education.0.degree':
            ClaimComparisonValidator().claim_catalog(CLAIMS)['education.0.degree']})
        self.assertEqual(context['record_context'], {'education.0': {
            'cv_record': CLAIMS['education'][0], 'profile_record': PROFILE['education'][0],
            'alignment': ALIGNMENTS['education'][0]}})
        self.assertEqual(context['profile_evidence']['education.0.school'], 'School A')
        self.assertNotIn('experience', context['profile_evidence'])
        self.assertNotIn('headline', context['profile_evidence'])
        self.assertNotIn('hometown', context['profile_evidence'])
        serialized = json.dumps(context)
        for reference in ClaimComparisonValidator().claim_catalog(CLAIMS):
            if reference != 'education.0.degree':
                self.assertNotIn(reference, context['claim_catalog'])
        self.assertNotIn('Excluded job text', serialized)
        finding = result['education.0.degree']
        self.assertEqual(finding['cv_claim'], context['claim_catalog']['education.0.degree'])
        self.assertEqual(finding['evidence'], [{'path': 'education.0.degree', 'value': 'MBA'}])
        self.assertEqual(context['evidence_path_contracts'], {'education.0.degree': {
            'allowed_match_evidence_paths': ['education.0.degree'],
            'allowed_mismatch_evidence_paths': ['education.0.degree', 'education']}})
        self.assertEqual(calls[0].to_messages()[0].content, SEMANTIC_PROMPT)

    async def test_degree_skill_employer_school_field_and_title_aliases(self):
        cases = [
            ('education.0.degree', 'degree', 'Master of Business Administration', 'MBA', 'education.0.degree'),
            ('education.0.degree', 'degree', 'Bachelor of Science', 'BSc', 'education.0.degree'),
            ('skills.0.name', 'name', 'UI/UX Design', 'UI/UX', 'skills.0.name'),
            ('skills.0.name', 'name', 'Microsoft PowerPoint', 'PowerPoint', 'skills.0.name'),
            ('skills.0.name', 'name', 'JavaScript', 'JS', 'skills.0.name'),
            ('jobs.0.company', 'company', 'International Business Machines', 'IBM', 'experience.1.company'),
            ('jobs.0.title', 'title', 'Senior Mgr', 'Manager', 'experience.1.title'),
            ('education.0.school', 'school', 'Massachusetts Institute of Technology', 'MIT', 'education.0.school'),
            ('education.0.field', 'field', 'Computer Science', 'Computing', 'education.0.field'),
        ]
        for reference, field, claim, actual, path in cases:
            with self.subTest(reference=reference, claim=claim):
                claims, profile = deepcopy(CLAIMS), deepcopy(PROFILE)
                if reference.startswith('jobs.'):
                    claims['jobs'][0][field] = claim
                    profile['experience'][1][field] = actual
                elif reference.startswith('education.'):
                    claims['education'][0][field] = claim
                    profile['education'][0][field] = actual
                else:
                    claims['skills'][0][field] = claim
                    profile['skills'][0][field] = actual
                result, calls = await self.resolve({reference: {
                    'status': 'MATCH', 'evidence_paths': [path], 'reason': 'Supported semantic equivalence'}}, claims, profile)
                self.assertEqual(result[reference]['status'], 'MATCH')
                self.assertEqual(result[reference]['evidence'], [{'path': path, 'value': actual}])
                guidance = json.loads(calls[0].to_messages()[1].content)['evidence_path_contracts'][reference]
                self.assertIn(path, guidance['allowed_match_evidence_paths'])
                self.assertNotIn('education', guidance['allowed_match_evidence_paths'])
                self.assertNotIn('experience', guidance['allowed_match_evidence_paths'])
        for wording in ('UI/UX', 'PowerPoint', 'BSc', 'employer/school aliases', 'unqualified title'):
            self.assertIn(wording, SEMANTIC_PROMPT)

    async def test_docker_absent_after_semantic_review_cites_complete_collection(self):
        claims = deepcopy(CLAIMS)
        claims['skills'][0]['name'] = 'Docker'
        result, calls = await self.resolve({'skills.0.name': {
            'status': 'MISMATCH', 'evidence_paths': ['skills'],
            'reason': 'Complete supplied skills list has Python and SQL, neither equivalent to Docker'}}, claims)
        self.assertEqual(result['skills.0.name']['status'], 'MISMATCH')
        self.assertEqual(result['skills.0.name']['evidence'][0]['value'], PROFILE['skills'])
        guidance = json.loads(calls[0].to_messages()[1].content)['evidence_path_contracts']['skills.0.name']
        self.assertNotIn('skills', guidance['allowed_match_evidence_paths'])
        self.assertIn('skills', guidance['allowed_mismatch_evidence_paths'])

    async def test_city_semantics_and_hometown_exclusion(self):
        claims = deepcopy(CLAIMS)
        claims['city'] = {'raw': 'NYC | Hong Kong', 'value': None, 'candidates': [
            {'city': 'NYC', 'role': 'unspecified'}, {'city': 'Hong Kong', 'role': 'unspecified'}]}
        profile = dict(deepcopy(PROFILE), city='New York')
        result, _ = await self.resolve({'city': {
            'status': 'MATCH', 'evidence_paths': ['city'], 'reason': 'NYC is New York'}}, claims, profile)
        self.assertEqual(result['city']['status'], 'MATCH')
        claims['city'] = {'raw': 'Hometown Hong Kong', 'value': None,
                          'candidates': [{'city': 'Hong Kong', 'role': 'hometown'}]}
        result, _ = await self.resolve({}, claims, profile)
        self.assertEqual(result['city']['status'], 'UNKNOWN')
        with self.assertRaises(ValueError):
            await self.resolve({'city': {'status': 'MATCH', 'evidence_paths': ['city'], 'reason': 'Hometown match'}}, claims, profile)

    async def test_objective_year_contradiction_cannot_be_overridden(self):
        profile = deepcopy(PROFILE)
        profile['experience'][1]['start_year'] = 2019
        result, _ = await self.resolve({}, profile=profile)
        self.assertEqual(result['jobs.0.start_year']['status'], 'MISMATCH')
        findings = DeterministicClaimVerifier().verify(CLAIMS, profile, ALIGNMENTS)
        output = self.decisions(findings)
        output['findings'].append({'claim_id': 'jobs.0.start_year', 'status': 'MATCH',
                                   'evidence_paths': ['experience.1.start_year'], 'reason': 'Override contradiction'})
        resolver, _ = self.model(output)
        with self.assertRaises(ValueError):
            await resolver.resolve(CLAIMS, profile, ALIGNMENTS, findings)

    async def test_null_alignment_can_remain_unknown_or_use_defensible_collection_absence(self):
        alignments = deepcopy(ALIGNMENTS)
        alignments['jobs'][0]['profile_index'] = None
        alignments['jobs'][0]['reason'] = 'Ambiguous record correspondence'
        result, calls = await self.resolve({}, alignments=alignments)
        self.assertTrue(all(f['status'] == 'UNKNOWN' for ref, f in result.items() if ref.startswith('jobs.0.')))
        context = json.loads(calls[0].to_messages()[1].content)
        self.assertEqual(context['record_context']['jobs.0']['alignment'], alignments['jobs'][0])
        self.assertIsNone(context['record_context']['jobs.0']['profile_record'])
        self.assertIn('experience.0.title', context['profile_evidence'])
        self.assertIn('experience.1.title', context['profile_evidence'])
        claims = deepcopy(CLAIMS)
        claims['jobs'][0]['company'] = 'Fabricated employer'
        result, _ = await self.resolve({'jobs.0.company': {
            'status': 'MISMATCH', 'evidence_paths': ['experience'],
            'reason': 'Sufficient complete history establishes employer absence'}}, claims=claims, alignments=alignments)
        self.assertEqual(result['jobs.0.company']['status'], 'MISMATCH')

    async def test_anchored_record_cannot_silently_realign_but_collection_absence_is_supported(self):
        profile = deepcopy(PROFILE)
        profile['education'].append(dict(profile['education'][0], degree='Other degree'))
        for path in ('education.1.degree', 'experience.0.company'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                await self.resolve({'education.0.degree': {
                    'status': 'MISMATCH', 'evidence_paths': [path], 'reason': 'Wrong aligned evidence'}}, profile=profile)
        with self.assertRaises(ValueError):
            await self.resolve({'education.0.degree': {
                'status': 'MATCH', 'evidence_paths': ['education'], 'reason': 'Borrowing a degree from the collection'}})
        profile['education'][0]['degree'] = 'BSc'
        result, _ = await self.resolve({'education.0.degree': {
            'status': 'MISMATCH', 'evidence_paths': ['education'], 'reason': 'Complete history contradicts degree'}}, profile=profile)
        self.assertEqual(result['education.0.degree']['status'], 'MISMATCH')

    async def test_skill_tasks_exclude_unrelated_claims_and_alignment_context(self):
        claims = deepcopy(CLAIMS)
        claims['education'][0]['degree'] = 'MBA'
        claims['skills'][0]['name'] = 'UI/UX Design'
        profile = dict(deepcopy(PROFILE), skills=[{'name': 'UI/UX'}])
        result, calls = await self.resolve({'skills.0.name': {
            'status': 'MATCH', 'evidence_paths': ['skills.0.name'], 'reason': 'Skill alias'}}, claims, profile)
        context = json.loads(calls[0].to_messages()[1].content)
        self.assertEqual(context['requested_claim_ids'], ['skills.0.name'])
        self.assertEqual(set(context['claim_catalog']), {'skills.0.name'})
        self.assertEqual(context['record_context'], {})
        self.assertEqual(context['profile_evidence'], {'skills': [{'name': 'UI/UX'}], 'skills.0.name': 'UI/UX'})

    async def test_aligned_context_exposes_only_anchor_field_paths_and_full_history_for_absence(self):
        claims = deepcopy(CLAIMS)
        claims['education'][0]['degree'] = 'MBA'
        claims['jobs'][0]['company'] = 'Company A alias'
        _, calls = await self.resolve({}, claims)
        context = json.loads(calls[0].to_messages()[1].content)
        self.assertEqual(context['requested_claim_ids'], ['jobs.0.company'])
        self.assertEqual(context['record_context']['jobs.0']['cv_record'], claims['jobs'][0])
        self.assertEqual(context['record_context']['jobs.0']['profile_record'], PROFILE['experience'][1])
        self.assertEqual(context['record_context']['jobs.0']['alignment'], ALIGNMENTS['jobs'][0])
        self.assertEqual(context['profile_evidence']['experience'], PROFILE['experience'])
        self.assertIn('experience.1.title', context['profile_evidence'])
        self.assertNotIn('experience.0.title', context['profile_evidence'])
        self.assertEqual(context['evidence_path_contracts']['jobs.0.company'], {
            'allowed_match_evidence_paths': ['experience.1.company'],
            'allowed_mismatch_evidence_paths': ['experience.1.company', 'experience']})

    async def test_title_equivalence_guidance_preserves_multi_field_indexed_evidence(self):
        claims = deepcopy(CLAIMS)
        claims['jobs'][0]['title'] = 'Senior Mgr'
        result, calls = await self.resolve({'jobs.0.title': {
            'status': 'MATCH', 'evidence_paths': ['experience.1.title', 'experience.1.seniority'],
            'reason': 'Manager with senior seniority matches Senior Mgr'}}, claims)
        guidance = json.loads(calls[0].to_messages()[1].content)['evidence_path_contracts']['jobs.0.title']
        self.assertEqual(guidance['allowed_match_evidence_paths'], ['experience.1.title', 'experience.1.seniority'])
        self.assertEqual(guidance['allowed_mismatch_evidence_paths'], ['experience.1.title', 'experience.1.seniority', 'experience'])
        self.assertEqual([e['path'] for e in result['jobs.0.title']['evidence']], guidance['allowed_match_evidence_paths'])

    async def test_guidance_excludes_null_values_and_other_aligned_record_paths(self):
        profile = deepcopy(PROFILE)
        profile['education'][0]['degree'] = None
        profile['education'].append(dict(PROFILE['education'][0], degree='MBA'))
        _, calls = await self.resolve({}, profile=profile)
        guidance = json.loads(calls[0].to_messages()[1].content)['evidence_path_contracts']['education.0.degree']
        self.assertEqual(guidance['allowed_match_evidence_paths'], [])
        self.assertEqual(guidance['allowed_mismatch_evidence_paths'], ['education'])

    async def test_guidance_maps_years_and_unparsed_dates_without_inventing_evidence(self):
        claims = deepcopy(CLAIMS)
        claims['jobs'][0]['start_year'] = None
        claims['jobs'][0]['dates_raw'] = 'about 2020 to present'
        profile = deepcopy(PROFILE)
        profile['education'][0]['end_year'] = None
        _, calls = await self.resolve({}, claims, profile)
        contracts = json.loads(calls[0].to_messages()[1].content)['evidence_path_contracts']
        self.assertEqual(contracts['education.0.graduation_year']['allowed_match_evidence_paths'], [])
        self.assertEqual(contracts['jobs.0.dates']['allowed_match_evidence_paths'],
                         ['experience.1.start_year', 'experience.1.is_current'])

    async def test_employment_aliases_cannot_borrow_another_aligned_record(self):
        claims = deepcopy(CLAIMS)
        claims['jobs'][0]['company'] = 'Company A alias'
        with self.assertRaises(ValueError):
            await self.resolve({'jobs.0.company': {
                'status': 'MATCH', 'evidence_paths': ['experience.0.company'],
                'reason': 'Borrowing company from the wrong record'}}, claims)

    async def test_missing_duplicate_invented_ids_and_invalid_evidence_fail_atomically(self):
        findings = DeterministicClaimVerifier().verify(CLAIMS, PROFILE, ALIGNMENTS)
        valid = self.decisions(findings, {'education.0.degree': {
            'status': 'MATCH', 'evidence_paths': ['education.0.degree'], 'reason': 'Equivalent degree'}})
        invalid = [None, [], {}, {'findings': []}, dict(valid, extra=True)]
        for field, value in (('claim_id', 'invented'), ('claim_id', 'name'), ('claim_id', None),
                             ('status', 'VALID'), ('reason', ''), ('reason', None),
                             ('evidence_paths', None), ('evidence_paths', []),
                             ('evidence_paths', ['education.0.degree', 'education.0.degree']),
                             ('evidence_paths', [None]), ('evidence_paths', ['education.99.degree']),
                             ('evidence_paths', ['headline']), ('evidence_paths', ['hometown']),
                             ('evidence_paths', ['experience.0.description']), ('extra', True)):
            output = deepcopy(valid)
            output['findings'][0][field] = value
            invalid.append(output)
        invalid.append({'findings': valid['findings'] * 2})
        for output in invalid:
            with self.subTest(output=output):
                original = deepcopy(findings)
                resolver, calls = self.model(output)
                with self.assertRaises(ValueError):
                    await resolver.resolve(CLAIMS, PROFILE, ALIGNMENTS, findings)
                self.assertEqual(findings, original)
                self.assertEqual(len(calls), 1)

    async def test_null_evidence_malformed_json_and_model_failures(self):
        profile = deepcopy(PROFILE)
        profile['education'][0]['degree'] = None
        with self.assertRaises(ValueError):
            await self.resolve({'education.0.degree': {
                'status': 'MATCH', 'evidence_paths': ['education.0.degree'], 'reason': 'Unsupported'}}, profile=profile)
        findings = DeterministicClaimVerifier().verify(CLAIMS, PROFILE, ALIGNMENTS)
        for output in ('not JSON', '{"findings":', RuntimeError('Model unavailable')):
            resolver, calls = self.model(output)
            with self.subTest(output=output), self.assertRaises((ValueError, RuntimeError)):
                await resolver.resolve(CLAIMS, PROFILE, ALIGNMENTS, findings)
            self.assertEqual(len(calls), 1)

    async def test_empty_skills_without_completeness_can_remain_unknown(self):
        result, calls = await self.resolve({}, profile=dict(deepcopy(PROFILE), skills=[]))
        self.assertEqual(result['skills.0.name']['status'], 'UNKNOWN')
        self.assertIn('empty skills list alone does not establish exhaustiveness', SEMANTIC_PROMPT)

