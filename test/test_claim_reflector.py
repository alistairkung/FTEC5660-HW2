import json
import unittest
from copy import deepcopy

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from cv_verifier.claim_comparison_validator import ClaimComparisonValidator
from cv_verifier.claim_reflector import ClaimReflector, REFLECTION_PROMPT
from test_claim_comparator import CLAIMS, PROFILE


def evidence_value(profile, path):
    value = profile
    for part in path.split('.'):
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def baseline(claims=CLAIMS, profile=PROFILE):
    contract = ClaimComparisonValidator()
    catalog = contract.claim_catalog(claims)
    findings = []
    for reference, claim in catalog.items():
        if reference.startswith('jobs.'):
            _, index, field = reference.split('.')
            path = f'experience.{1 - int(index)}.{field}'
        elif reference.startswith('education.'):
            path = reference.replace('graduation_year', 'end_year')
        else:
            path = reference
        findings.append({'claim_id': reference, 'cv_claim': claim, 'status': 'MATCH',
                         'evidence': [{'path': path, 'value': evidence_value(profile, path)}],
                         'reason': 'Baseline judged equivalent'})
    return contract.validate(findings, claims, profile)


def correction(reference, paths, reason='Explicit contradiction'):
    return {'claim_id': reference, 'status': 'MISMATCH', 'evidence_paths': paths, 'reason': reason}


class ClaimReflectorTests(unittest.IsolatedAsyncioTestCase):
    async def reflect(self, output, claims=CLAIMS, profile=PROFILE, findings=None):
        original = baseline(claims, profile) if findings is None else findings
        saved = deepcopy((claims, profile, original))
        calls = []
        def respond(prompt):
            calls.append(prompt.to_messages())
            if isinstance(output, Exception):
                raise output
            return AIMessage(content=output if isinstance(output, str) else json.dumps(output))
        try:
            result = await ClaimReflector(RunnableLambda(respond)).reflect(claims, profile, original)
        finally:
            self.assertEqual((claims, profile, original), saved)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0].content, REFLECTION_PROMPT)
        context = json.loads(calls[0][1].content)
        self.assertEqual(context['claims'], claims)
        self.assertEqual(context['findings'], original)
        self.assertNotIn('headline', context['profile'])
        self.assertNotIn('hometown', context['profile'])
        self.assertEqual((claims, profile, original), saved)
        return result, original

    async def test_senior_title_against_junior_corrected_without_collateral_changes(self):
        claims, profile = deepcopy(CLAIMS), deepcopy(PROFILE)
        claims['jobs'][0]['title'] = 'Senior Analyst'
        profile['experience'][1].update(title='Analyst', seniority='junior', description='Excluded')
        result, original = await self.reflect({'corrections': [correction('jobs.0.title',
            ['experience.1.title', 'experience.1.seniority'])]}, claims, profile)
        changed = [f['claim_id'] for f, old in zip(result, original) if f != old]
        self.assertEqual(changed, ['jobs.0.title'])
        self.assertEqual(next(f for f in result if f['claim_id'] == 'jobs.0.title')['status'], 'MISMATCH')
        self.assertTrue(all('description' not in e for e in ClaimComparisonValidator().comparison_profile(profile)['experience']))

    async def test_title_seniority_and_mba_aliases_preserved_with_empty_corrections(self):
        result, original = await self.reflect({'corrections': []})
        self.assertEqual(result, original)
        self.assertIsNot(result, original)
        self.assertIn('Senior Manager is equivalent to Manager with seniority senior', REFLECTION_PROMPT)
        self.assertIn('MBA is equivalent to Master of Business', REFLECTION_PROMPT)

    async def test_explicit_dates_and_degree_contradictions(self):
        for reference, path, value in (
            ('jobs.0.start_year', 'experience.1.start_year', 2019),
            ('education.0.graduation_year', 'education.0.end_year', 2011),
            ('education.0.degree', 'education.0.degree', 'BSc'),
        ):
            with self.subTest(reference=reference):
                profile = deepcopy(PROFILE)
                collection, index, field = path.split('.')
                profile[collection][int(index)][field] = value
                result, _ = await self.reflect({'corrections': [correction(reference, [path])]}, profile=profile)
                finding = next(f for f in result if f['claim_id'] == reference)
                self.assertEqual(finding['status'], 'MISMATCH')
                self.assertEqual(finding['evidence'][0]['value'], value)

    async def test_authoritative_empty_collection_supports_absence(self):
        profile = dict(PROFILE, skills=[])
        findings = baseline()
        findings[-1]['evidence'] = [{'path': 'skills', 'value': []}]
        result, _ = await self.reflect({'corrections': [correction('skills.0.name', ['skills'])]},
                                       profile=profile, findings=findings)
        self.assertEqual(result[-1]['status'], 'MISMATCH')
        self.assertEqual(result[-1]['evidence'], [{'path': 'skills', 'value': []}])

    async def test_malformed_output_and_references_rejected(self):
        valid = correction('jobs.0.title', ['experience.1.title'])
        outputs = ['not JSON', '```json\n{}\n```', [], {}, {'corrections': None},
                   {'corrections': [], 'score': 0.1}, {'corrections': [None]},
                   {'corrections': [valid, valid]},
                   {'corrections': [dict(valid, claim_id='jobs.99.title')]},
                   {'corrections': [dict(valid, claim_id=None)]}]
        for key in valid:
            missing = deepcopy(valid)
            del missing[key]
            outputs.append({'corrections': [missing]})
        for key, value in (('reason', ''), ('reason', 3), ('status', 'UNKNOWN'), ('status', 'MATCH'),
                           ('status', None), ('status', []), ('evidence_paths', None),
                           ('cv_claim', {'value': 'invented', 'raw': None})):
            outputs.append({'corrections': [dict(valid, **{key: value})]})
        for output in outputs:
            with self.subTest(output=output), self.assertRaises(ValueError):
                await self.reflect(output)

    async def test_invalid_null_duplicate_excluded_or_irrelevant_paths_rejected(self):
        for paths in ([], ['headline'], ['hometown'], ['experience.0.description'],
                      ['experience.99.title'], ['experience.1.end_year'],
                      ['experience.1.title', 'experience.1.title'], [None], [[]],
                      ['skills'], ['education.0.degree'], ['experience.1.start_year']):
            with self.subTest(paths=paths), self.assertRaises(ValueError):
                await self.reflect({'corrections': [correction('jobs.0.title', paths)]})
        # Even a permitted, relevant scalar cannot support a contradiction when it agrees.
        with self.assertRaises(ValueError):
            await self.reflect({'corrections': [correction('jobs.0.start_year', ['experience.1.start_year'])]})

    async def test_non_match_findings_cannot_be_corrected(self):
        findings = baseline()
        findings[0]['status'] = 'UNKNOWN'
        with self.assertRaises(ValueError):
            await self.reflect({'corrections': [correction('name', ['name'])]}, findings=findings)

    async def test_invalid_correction_rejects_entire_batch_without_mutating_baseline(self):
        with self.assertRaises(ValueError):
            await self.reflect({'corrections': [
                correction('jobs.0.title', ['experience.1.title']),
                correction('name', ['headline']),
            ]})
