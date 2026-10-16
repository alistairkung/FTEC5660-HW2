"""Generic reproductions of saved PR25/27 contract and coverage failures."""
import json
import unittest
from copy import deepcopy
from unittest.mock import AsyncMock

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from cv_verifier.claim_comparator import ClaimComparator
from cv_verifier.claim_comparison_validator import ClaimComparisonValidator
from cv_verifier.identity_decision_validator import IdentityDecisionValidator
from test_claim_comparator import CLAIMS, PROFILE
import test_identity_investigation as investigation


def decisions(status='UNKNOWN'):
    return {'findings': [{'claim_id': ref, 'status': status, 'evidence_paths': [], 'reason': 'Unavailable'}
                         for ref in ClaimComparisonValidator().claim_catalog(CLAIMS)]}


class ComparisonRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_null_citation_retried_once_with_permitted_paths_and_full_validation(self):
        bad = decisions()
        bad['findings'][0].update(status='MATCH', evidence_paths=['experience.1.end_year'])
        good = decisions()
        inputs = []
        async def respond(prompt):
            inputs.append(json.loads(prompt.to_messages()[1].content))
            return AIMessage(content=json.dumps(bad if len(inputs) == 1 else good))
        result = await ClaimComparator(RunnableLambda(respond)).compare(CLAIMS, PROFILE)
        self.assertEqual(len(inputs), 2)
        self.assertIn('experience.1.is_current', inputs[0]['permitted_evidence_paths'])
        self.assertNotIn('experience.1.end_year', inputs[0]['permitted_evidence_paths'])
        ClaimComparisonValidator().validate(result, CLAIMS, PROFILE)

    async def test_repeated_invalid_output_cannot_become_positive(self):
        calls = []
        async def respond(_):
            calls.append(1)
            return AIMessage(content='{"findings": []}')
        with self.assertRaises(ValueError):
            await ClaimComparator(RunnableLambda(respond)).compare(CLAIMS, PROFILE)
        self.assertEqual(len(calls), 2)

    async def test_independent_mismatch_survives_unrelated_invalid_citation_and_failed_retry(self):
        bad = decisions()
        bad['findings'][0].update(status='MATCH', evidence_paths=['not.a.path'])
        target = next(d for d in bad['findings'] if d['claim_id'] == 'jobs.0.start_year')
        target.update(status='MISMATCH', evidence_paths=['experience.1.start_year'], reason='Years differ')
        profile = deepcopy(PROFILE)
        profile['experience'][1]['start_year'] = 2019
        async def respond(_):
            return AIMessage(content=json.dumps(bad))
        result = await ClaimComparator(RunnableLambda(respond)).compare(CLAIMS, profile)
        ClaimComparisonValidator().validate(result, CLAIMS, profile)
        self.assertEqual(result[0]['status'], 'UNKNOWN')
        self.assertEqual(next(f for f in result if f['claim_id'] == target['claim_id'])['status'], 'MISMATCH')

    async def test_duplicate_claim_cannot_supply_a_preserved_mismatch(self):
        bad = decisions()
        bad['findings'][0].update(status='MISMATCH', evidence_paths=['name'])
        bad['findings'].append(deepcopy(bad['findings'][0]))
        with self.assertRaises(ValueError):
            await ClaimComparator(RunnableLambda(lambda _: AIMessage(content=json.dumps(bad)))).compare(CLAIMS, PROFILE)


class UpstreamRecoveryTests(unittest.IsolatedAsyncioTestCase):
    def pipeline(self):
        return investigation.InvestigationTests().pipeline([])

    async def test_invalid_location_reextracted_once_and_validated_before_search(self):
        pipeline, retriever, _ = self.pipeline()
        valid = deepcopy(pipeline.extractor.extract.return_value)
        bad = deepcopy(valid)
        bad['city']['value'] = None  # Single current candidate contradicts null value.
        pipeline.extractor.extract.side_effect = [bad, valid]
        result = await pipeline.ainvoke('original source')
        self.assertEqual(result['claims'], valid)
        self.assertEqual(pipeline.extractor.extract.await_count, 2)
        self.assertEqual(retriever.search.await_count, 2)

    async def test_repeated_malformed_location_never_reaches_search(self):
        pipeline, retriever, _ = self.pipeline()
        pipeline.extractor.extract.return_value['city']['value'] = None
        result = await pipeline.ainvoke('source')
        self.assertIsNone(result['claims'])
        self.assertEqual(pipeline.extractor.extract.await_count, 2)
        retriever.search.assert_not_awaited()

    async def test_bad_shortlist_uses_supported_metadata_without_unknown_ids_or_budget_growth(self):
        pipeline, retriever, resolver = self.pipeline()
        claims = pipeline.extractor.extract.return_value
        claims['skills'] = [{'raw': 'Oceanography', 'name': 'Oceanography'}]
        hits = [{'id': n, 'industry': 'Retail', 'location': 'London'} for n in range(1, 12)]
        hits[-1]['industry'] = 'Oceanography'
        retriever.search.side_effect = [hits, []]
        pipeline.prioritizer.prioritize.side_effect = None
        pipeline.prioritizer.prioritize.return_value = [99999]
        result = await pipeline.ainvoke('source')
        self.assertEqual(result['attempted_candidate_ids'][0], 11)
        self.assertNotIn(99999, result['ranked_candidate_ids'])
        self.assertEqual(len(result['attempted_candidate_ids']), 8)
        self.assertEqual(set(result['ranked_candidate_ids']), set(range(1, 12)))

    async def test_unshortlisted_candidates_also_keep_supported_metadata_priority(self):
        pipeline, retriever, _ = self.pipeline()
        claims = pipeline.extractor.extract.return_value
        claims['skills'] = [{'raw': 'Oceanography', 'name': 'Oceanography'}]
        hits = [{'id': n, 'industry': 'Retail'} for n in range(1, 12)]
        hits[-1]['industry'] = 'Oceanography'
        ranking, shortlist, errors = await pipeline._prioritize(claims, hits)
        self.assertEqual(ranking[len(shortlist)], 11)


class IdentitySupportTests(unittest.IsolatedAsyncioTestCase):
    async def test_shared_name_city_skills_cannot_override_conflicting_career_and_education(self):
        profile = deepcopy(PROFILE)
        profile['id'] = 12
        for index, record in enumerate(profile['experience']):
            record.update(company=f'Unrelated Enterprise {index}', title='Architect', start_year=1990, end_year=1995)
        profile['education'] = [{'school': 'Different Institute', 'degree': 'PhD', 'field': 'Physics', 'end_year': 1990}]
        pipeline, _, resolver = investigation.InvestigationTests().pipeline([])
        resolver.resolve.return_value = {'candidate_id': 12, 'reason': 'Name city skills match'}
        decision, errors = await pipeline._resolve_identity(CLAIMS, [profile])
        self.assertIsNone(decision['candidate_id'])
        self.assertTrue(errors)

    async def test_isolated_false_employer_tolerated_when_education_identifies_person(self):
        claims = deepcopy(CLAIMS)
        claims['jobs'][0]['company'] = 'Invented Enterprise'
        profile = dict(deepcopy(PROFILE), id=12)
        decision = {'candidate_id': 12, 'reason': 'Coherent education and remaining employment'}
        pipeline, _, resolver = investigation.InvestigationTests().pipeline([])
        resolver.resolve.return_value = decision
        actual, errors = await pipeline._resolve_identity(claims, [profile])
        self.assertEqual(actual, decision)
        self.assertFalse(errors)

    async def test_rejected_broad_match_continues_to_supported_unshortlisted_identity(self):
        pipeline, retriever, resolver = investigation.InvestigationTests().pipeline([])
        pipeline.extractor.extract.return_value = deepcopy(CLAIMS)
        hits = [{"id": n, "industry": "Retail"} for n in range(1, 12)]
        hits[-1]["industry"] = "Logistics"  # Explicit CV education field, not an identity proof.
        retriever.search.side_effect = [hits, []]
        wrong = deepcopy(PROFILE)
        for record in wrong["experience"]:
            record.update(company="Unrelated Enterprise", title="Architect", start_year=1990, end_year=1995)
        wrong["education"] = [{"school": "Different Institute", "degree": "PhD", "field": "Physics", "end_year": 1990}]
        async def retrieve(batch):
            return [dict(deepcopy(PROFILE if hit["id"] == 11 else wrong), id=hit["id"]) for hit in batch], []
        retriever.retrieve_profiles.side_effect = retrieve
        resolver.resolve.side_effect = lambda claims, profiles: {
            "candidate_id": 11 if any(p["id"] == 11 for p in profiles) else 1,
            "reason": "Proposed person"}
        result = await pipeline.ainvoke("source")
        self.assertEqual(result["selected_candidate_id"], 11)
        self.assertLessEqual(len(result["attempted_candidate_ids"]), 8)
        self.assertIn(11, result["attempted_candidate_ids"])


class ExplicitGradeAdjudicationTests(unittest.IsolatedAsyncioTestCase):
    def verifier(self, correction_output, title='Senior Analyst', level='junior', paths=None):
        from test_claim_reflector import baseline
        from cv_verifier.claim_reflector import ClaimReflector
        from cv_verifier.cv_verifier import CVVerifier
        from cv_verifier.cv_scorer import CVScorer
        claims, profile = deepcopy(CLAIMS), dict(deepcopy(PROFILE), id=12)
        claims['jobs'][0]['title'] = title
        profile['experience'][1].update(title='Analyst', seniority=level)
        findings = baseline(claims, profile)
        finding = next(f for f in findings if f['claim_id'] == 'jobs.0.title')
        finding['status'] = 'UNKNOWN'
        finding['evidence'] = [{'path': 'experience.1.title', 'value': 'Analyst'},
                               {'path': 'experience.1.seniority', 'value': level}]
        if paths is not None:
            finding['evidence'] = paths
        identity, comparator = AsyncMock(), AsyncMock()
        identity.ainvoke.return_value = {'identity': 'resolved', 'claims': claims, 'profiles': [profile],
            'selected_candidate_id': 12, 'identity_reason': 'Career supports identity', 'errors': []}
        output = {'findings': [{'claim_id': f['claim_id'], 'status': f['status'],
                  'evidence_paths': [e['path'] for e in f['evidence']], 'reason': f['reason']} for f in findings]}
        comparator = ClaimComparator(RunnableLambda(lambda _: AIMessage(content=json.dumps(output))))
        calls = []
        async def respond(_):
            calls.append(1)
            if isinstance(correction_output, Exception):
                raise correction_output
            return AIMessage(content=json.dumps(correction_output))
        return CVVerifier(identity, comparator, ClaimComparisonValidator(), CVScorer(),
                          ClaimReflector(RunnableLambda(respond))), calls

    async def test_explicit_grade_conflict_survives_both_unknown_and_incorrect_match(self):
        for status in ('UNKNOWN', 'MATCH'):
            with self.subTest(status=status):
                verifier, calls = self.verifier({'corrections': []})
                def mutate(output):
                    next(f for f in output['findings'] if f['claim_id'] == 'jobs.0.title')['status'] = status
                    return output
                verifier.comparator._chain = verifier.comparator._chain | RunnableLambda(mutate)
                result = await verifier.ainvoke('source')
                self.assertEqual(result['score'], .1)
                self.assertFalse(calls)
                self.assertFalse(result['errors'])

    async def test_missing_support_unqualified_titles_and_matching_grades_stay_uncertain(self):
        for options in ({'title': 'Analyst'}, {'level': 'senior'}, {'paths': [], 'level': None}, {'paths': [
                {'path': 'experience.0.title', 'value': 'Scientist'},
                {'path': 'experience.1.seniority', 'value': 'junior'}]}):
            with self.subTest(options=options):
                verifier, calls = self.verifier({'corrections': []}, **options)
                result = await verifier.ainvoke('source')
                self.assertEqual(result['score'], .5)

    async def test_different_employer_or_dates_prevent_automatic_grade_adjudication(self):
        for field, value in (('company', 'Unrelated Enterprise'), ('start_year', 1999)):
            verifier, calls = self.verifier({'corrections': []})
            verifier.identity_pipeline.ainvoke.return_value['claims']['jobs'][0][field] = value
            # The captured comparator output uses original canonical inputs, so update
            # through its incoming context rather than fabricating canonical claims.
            result = await verifier.ainvoke('source')
            self.assertEqual(result['score'], .5)

    async def test_malformed_optional_search_metadata_does_not_break_fallback(self):
        pipeline, _, _ = investigation.InvestigationTests().pipeline([])
        hits = [{'id': 1, 'industry': ['bad'], 'location': {'bad': True}}, {'id': 2}]
        pipeline.prioritizer.prioritize.side_effect = ValueError('invalid ranking')
        ranking, shortlist, errors = await pipeline._prioritize(CLAIMS, hits)
        self.assertEqual(ranking, [1, 2])
        self.assertFalse(shortlist)

    async def test_duplicate_corresponding_records_prevent_automatic_grade_adjudication(self):
        verifier, calls = self.verifier({'corrections': []})
        profile = verifier.identity_pipeline.ainvoke.return_value['profiles'][0]
        duplicate = deepcopy(profile['experience'][1])
        duplicate['seniority'] = 'senior'
        profile['experience'].append(duplicate)
        result = await verifier.ainvoke('source')
        self.assertEqual(result['score'], .5)
        self.assertFalse(calls)


class IsolatedEmployerIdentityTests(unittest.IsolatedAsyncioTestCase):
    async def test_single_employer_without_role_dates_or_education_does_not_establish_identity(self):
        wrong = dict(deepcopy(PROFILE), id=12)
        for record in wrong['experience']:
            record.update(company='Unrelated Enterprise', title='Architect', start_year=1990, end_year=1995)
        wrong['experience'][0]['company'] = CLAIMS['jobs'][0]['company']
        wrong['education'] = [{'school': 'Different Institute', 'degree': 'PhD', 'field': 'Logistics', 'end_year': 1990}]
        pipeline, _, resolver = investigation.InvestigationTests().pipeline([])
        resolver.resolve.return_value = {'candidate_id': 12, 'reason': 'Only shared employer despite widespread disagreement'}
        decision, errors = await pipeline._resolve_identity(CLAIMS, [wrong])
        self.assertIsNone(decision['candidate_id'])
        self.assertTrue(errors)


class ExtractionFeedbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_bounded_retry_receives_validator_feedback_and_original_cv(self):
        from cv_verifier.claim_extractor import ClaimExtractor
        pipeline, retriever, _ = investigation.InvestigationTests().pipeline([])
        valid = deepcopy(pipeline.extractor.extract.return_value)
        bad = deepcopy(valid)
        bad['city']['value'] = None
        messages = []
        def respond(prompt):
            messages.append(prompt.to_messages())
            repaired = len(messages[-1]) == 3 and 'single current candidate' in messages[-1][-1].content
            return AIMessage(content=json.dumps(valid if repaired else bad))
        pipeline.extractor = ClaimExtractor(RunnableLambda(respond))
        result = await pipeline.ainvoke('original source CV')
        self.assertEqual(result['claims'], valid)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[1][1].content, 'original source CV')
        self.assertIn('ambiguity', messages[1][-1].content)
