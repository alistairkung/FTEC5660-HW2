import json
import unittest
from copy import deepcopy

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from cv_verifier.evidence_aligner import EvidenceAligner, EvidenceAlignmentValidator, EVIDENCE_ALIGNMENT_PROMPT
from test_claim_comparator import CLAIMS, PROFILE
from test_record_aligner import ALIGNMENTS


EVIDENCE_ALIGNMENTS = {collection: [dict(entry, status='ALIGNED') for entry in entries]
                       for collection, entries in ALIGNMENTS.items()}
EVIDENCE_ALIGNMENTS['skills'] = [{'cv_index': 0, 'profile_index': 0, 'status': 'ALIGNED', 'reason': 'Python'}]


class EvidenceAlignerTests(unittest.IsolatedAsyncioTestCase):
    def aligner(self, output):
        calls = []
        def respond(prompt):
            calls.append(prompt)
            if isinstance(output, Exception):
                raise output
            return AIMessage(content=json.dumps(output))
        return EvidenceAligner(RunnableLambda(respond)), calls

    async def test_full_input_preserved_single_call_and_permitted_profile(self):
        claims, profile = deepcopy(CLAIMS), deepcopy(PROFILE)
        original = deepcopy((claims, profile))
        aligner, calls = self.aligner(EVIDENCE_ALIGNMENTS)
        self.assertEqual(await aligner.align(claims, profile), EVIDENCE_ALIGNMENTS)
        context = json.loads(calls[0].to_messages()[1].content)
        self.assertEqual(context['claims'], claims)
        self.assertEqual(context['profile']['skills'], [{'name': 'Python'}, {'name': 'SQL'}])
        self.assertNotIn('headline', context['profile'])
        self.assertNotIn('hometown', context['profile'])
        self.assertEqual(calls[0].to_messages()[0].content, EVIDENCE_ALIGNMENT_PROMPT)
        self.assertEqual((claims, profile), original)
        self.assertEqual(len(calls), 1)

    async def test_record_adapter_is_exact_legacy_shape_and_skills_are_discarded(self):
        validator = EvidenceAlignmentValidator()
        output = deepcopy(EVIDENCE_ALIGNMENTS)
        output['skills'][0].update(status='UNMATCHED', profile_index=None, reason='No corresponding Python')
        original = deepcopy(output)
        records = validator.record_alignments(output, CLAIMS, PROFILE)
        self.assertEqual(records, ALIGNMENTS)
        records['jobs'][0]['reason'] = 'Changed caller copy'
        self.assertEqual(output, original)

    async def test_reordered_skills_and_education_use_original_indices(self):
        claims, profile, output = deepcopy(CLAIMS), deepcopy(PROFILE), deepcopy(EVIDENCE_ALIGNMENTS)
        claims['skills'].append({'raw': 'Microsoft PowerPoint', 'name': 'Microsoft PowerPoint'})
        profile['skills'].insert(0, {'name': 'PowerPoint'})
        output['skills'] = [{'cv_index': 0, 'profile_index': 1, 'status': 'ALIGNED', 'reason': 'Python'},
                            {'cv_index': 1, 'profile_index': 0, 'status': 'ALIGNED', 'reason': 'Recognized PowerPoint alias'}]
        profile['education'].insert(0, dict(profile['education'][0], school='Other school'))
        output['education'][0]['profile_index'] = 1
        aligner, calls = self.aligner(output)
        self.assertEqual(await aligner.align(claims, profile), output)
        context = json.loads(calls[0].to_messages()[1].content)
        self.assertEqual(context['claims']['skills'][1]['name'], 'Microsoft PowerPoint')
        self.assertEqual(context['profile']['skills'][0]['name'], 'PowerPoint')

    async def test_false_parent_field_does_not_prohibit_record_correspondence(self):
        for collection, field in (('jobs', 'company'), ('education', 'school')):
            claims = deepcopy(CLAIMS)
            claims[collection][0][field] = 'Fabricated organization'
            aligner, _ = self.aligner(EVIDENCE_ALIGNMENTS)
            self.assertEqual(await aligner.align(claims, PROFILE), EVIDENCE_ALIGNMENTS)

    async def test_unmatched_and_ambiguous_null_records_are_legacy_nulls(self):
        for status in ('UNMATCHED', 'AMBIGUOUS'):
            output = deepcopy(EVIDENCE_ALIGNMENTS)
            output['jobs'][0].update(profile_index=None, status=status, reason='No defensible corresponding record')
            aligner, _ = self.aligner(output)
            result = await aligner.align(CLAIMS, PROFILE)
            records = aligner.validator.record_alignments(result, CLAIMS, PROFILE)
            self.assertIsNone(records['jobs'][0]['profile_index'])
            self.assertNotIn('status', records['jobs'][0])

    async def test_missing_vs_empty_collections_are_distinguished_without_model(self):
        aligner, calls = self.aligner(RuntimeError('must not invoke'))
        output = await aligner.align(CLAIMS, {'experience': [], 'education': None})
        self.assertEqual([entry['status'] for entry in output['jobs']], ['UNMATCHED', 'UNMATCHED'])
        self.assertEqual(output['education'][0]['status'], 'AMBIGUOUS')
        self.assertEqual(output['skills'][0]['status'], 'AMBIGUOUS')
        self.assertEqual(calls, [])
        empty = dict(CLAIMS, jobs=[], education=[], skills=[])
        self.assertEqual(await aligner.align(empty, PROFILE), {'jobs': [], 'education': [], 'skills': []})
        self.assertEqual(calls, [])

    async def test_skills_only_still_invokes_model(self):
        claims = dict(CLAIMS, jobs=[], education=[])
        output = dict(EVIDENCE_ALIGNMENTS, jobs=[], education=[])
        aligner, calls = self.aligner(output)
        self.assertEqual(await aligner.align(claims, PROFILE), output)
        self.assertEqual(len(calls), 1)

    async def test_partial_missing_collection_requires_ambiguity(self):
        output = deepcopy(EVIDENCE_ALIGNMENTS)
        output['skills'][0].update(status='AMBIGUOUS', profile_index=None, reason='Skills evidence withheld')
        profile = dict(PROFILE, skills=None)
        aligner, calls = self.aligner(output)
        self.assertEqual(await aligner.align(CLAIMS, profile), output)
        self.assertIsNone(json.loads(calls[0].to_messages()[1].content)['profile']['skills'])
        output['skills'][0]['status'] = 'UNMATCHED'
        with self.assertRaises(ValueError):
            EvidenceAlignmentValidator().validate(output, CLAIMS, profile)

    async def test_structural_errors_are_rejected_atomically(self):
        outputs = [None, [], {}, dict(EVIDENCE_ALIGNMENTS, skills=[])]
        for field, value in (('cv_index', True), ('cv_index', -1), ('cv_index', 1), ('cv_index', 0.0),
                             ('profile_index', False), ('profile_index', -1), ('profile_index', 2),
                             ('profile_index', 0.0), ('profile_index', None), ('status', 'MATCH'),
                             ('status', None), ('status', []), ('status', 'UNMATCHED'), ('status', 'AMBIGUOUS'),
                             ('reason', ''), ('reason', ' '), ('reason', 1), ('evidence_path', 'skills.99.name'),
                             ('score', 0.9)):
            output = deepcopy(EVIDENCE_ALIGNMENTS)
            output['skills'][0][field] = value
            outputs.append(output)
        for output in outputs:
            with self.subTest(output=output), self.assertRaises(ValueError):
                EvidenceAlignmentValidator().validate(output, CLAIMS, PROFILE)
        for collection in ('jobs', 'education', 'skills'):
            output = deepcopy(EVIDENCE_ALIGNMENTS)
            output[collection].append(deepcopy(output[collection][0]))
            with self.subTest(collection=collection), self.assertRaises(ValueError):
                EvidenceAlignmentValidator().validate(output, CLAIMS, PROFILE)

    async def test_duplicate_skill_evidence_requires_abstention(self):
        claims = deepcopy(CLAIMS)
        claims['skills'].append({'raw': 'Python programming', 'name': 'Python programming'})
        output = deepcopy(EVIDENCE_ALIGNMENTS)
        output['skills'].append(dict(output['skills'][0], cv_index=1, reason='Same skill alias'))
        with self.assertRaisesRegex(ValueError, 'Duplicate aligned'):
            EvidenceAlignmentValidator().validate(output, claims, PROFILE)
        for entry in output['skills']:
            entry.update(status='AMBIGUOUS', profile_index=None, reason='Two claims compete for one profile entry')
        self.assertEqual(EvidenceAlignmentValidator().validate(output, claims, PROFILE), output)

    async def test_aligned_skill_requires_original_usable_name(self):
        for skill in ({}, {'name': None}, {'name': ''}, {'name': 1}):
            with self.subTest(skill=skill), self.assertRaisesRegex(ValueError, 'usable name'):
                EvidenceAlignmentValidator().validate(EVIDENCE_ALIGNMENTS, CLAIMS, dict(PROFILE, skills=[skill]))

    async def test_adapter_validates_skill_output_before_discarding_it(self):
        output = deepcopy(EVIDENCE_ALIGNMENTS)
        output['skills'][0]['profile_index'] = 99
        with self.assertRaises(ValueError):
            EvidenceAlignmentValidator().record_alignments(output, CLAIMS, PROFILE)

    async def test_model_failures_and_bad_json_propagate_without_retry(self):
        aligner, calls = self.aligner(RuntimeError('model unavailable'))
        with self.assertRaises(RuntimeError):
            await aligner.align(CLAIMS, PROFILE)
        self.assertEqual(len(calls), 1)
        aligner = EvidenceAligner(RunnableLambda(lambda _: AIMessage(content='not JSON')))
        with self.assertRaises(ValueError):
            await aligner.align(CLAIMS, PROFILE)

    async def test_duplicate_json_fields_are_not_silently_overwritten(self):
        raw = json.dumps(EVIDENCE_ALIGNMENTS).replace('"profile_index": 0', '"profile_index": 99, "profile_index": 0', 1)
        aligner = EvidenceAligner(RunnableLambda(lambda _: AIMessage(content=raw)))
        with self.assertRaisesRegex(ValueError, 'Duplicate JSON'):
            await aligner.align(CLAIMS, PROFILE)

    def test_prompt_separates_correspondence_from_verification_and_distinct_skills(self):
        for instruction in ('Do not classify', 'not Kubernetes', 'not C#', 'not JavaScript',
                            'never establishes a false claim', 'whole-record correspondence'):
            self.assertIn(instruction, ' '.join(EVIDENCE_ALIGNMENT_PROMPT.split()))
