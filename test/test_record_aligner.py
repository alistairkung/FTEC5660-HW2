import json
import unittest
from copy import deepcopy

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from evaluation.experimental.record_aligner import RecordAligner, ALIGNMENT_PROMPT
from test_claim_comparator import CLAIMS, PROFILE


ALIGNMENTS = {'jobs': [{'cv_index': 0, 'profile_index': 1, 'reason': 'Company A and current dates'},
                       {'cv_index': 1, 'profile_index': 0, 'reason': 'Company B and historical dates'}],
              'education': [{'cv_index': 0, 'profile_index': 0, 'reason': 'School A and graduation'}]}


class RecordAlignerTests(unittest.IsolatedAsyncioTestCase):
    def aligner(self, output):
        calls = []
        def respond(prompt):
            calls.append(prompt)
            if isinstance(output, Exception):
                raise output
            return AIMessage(content=json.dumps(output))
        return RecordAligner(RunnableLambda(respond)), calls

    async def test_reordered_records_preserve_context_and_validate_references(self):
        claims, profile = deepcopy(CLAIMS), deepcopy(PROFILE)
        claims['education'].append(dict(claims['education'][0], school='School B'))
        profile['education'].insert(0, dict(profile['education'][0], school='School B'))
        output = deepcopy(ALIGNMENTS)
        output['education'] = [{'cv_index': 0, 'profile_index': 1, 'reason': 'School A'},
                               {'cv_index': 1, 'profile_index': 0, 'reason': 'School B'}]
        aligner, calls = self.aligner(output)
        self.assertEqual(await aligner.align(claims, profile), output)
        context = json.loads(calls[0].to_messages()[1].content)
        self.assertEqual(context['claims'], claims)
        self.assertEqual(context['profile']['experience'][1]['company'], 'Company A')
        self.assertNotIn('headline', context['profile'])
        self.assertNotIn('hometown', context['profile'])
        self.assertEqual(calls[0].to_messages()[0].content, ALIGNMENT_PROMPT)
        self.assertEqual(len(calls), 1)

    async def test_one_false_field_does_not_reject_valid_alignment(self):
        for collection, field in (('jobs', 'company'), ('jobs', 'title'), ('education', 'school'), ('education', 'degree')):
            with self.subTest(collection=collection, field=field):
                claims = deepcopy(CLAIMS)
                claims[collection][0][field] = 'Different claimed value'
                aligner, calls = self.aligner(ALIGNMENTS)
                self.assertEqual(await aligner.align(claims, PROFILE), ALIGNMENTS)
                self.assertIn('isolated fabricated', calls[0].to_messages()[0].content)

    async def test_ambiguous_null_references_are_valid_and_not_repaired(self):
        output = deepcopy(ALIGNMENTS)
        for mappings in output.values():
            for mapping in mappings:
                mapping['profile_index'] = None
                mapping['reason'] = 'Ambiguous correspondence'
        aligner, _ = self.aligner(output)
        self.assertEqual(await aligner.align(CLAIMS, PROFILE), output)

    async def test_invalid_duplicate_missing_and_out_of_range_references_are_rejected(self):
        outputs = [None, [], {}, {'jobs': [], 'education': []}]
        for field, value in (('cv_index', True), ('cv_index', -1), ('cv_index', 2),
                             ('profile_index', False), ('profile_index', -1), ('profile_index', 2),
                             ('profile_index', 1.0), ('reason', ''), ('reason', None), ('extra', 1)):
            output = deepcopy(ALIGNMENTS)
            output['jobs'][0][field] = value
            outputs.append(output)
        for duplicate_field in ('cv_index', 'profile_index'):
            output = deepcopy(ALIGNMENTS)
            output['jobs'][1][duplicate_field] = output['jobs'][0][duplicate_field]
            outputs.append(output)
        output = deepcopy(ALIGNMENTS)
        del output['education'][0]['reason']
        outputs.append(output)
        for output in outputs:
            with self.subTest(output=output), self.assertRaises(ValueError):
                aligner, calls = self.aligner(output)
                await aligner.align(CLAIMS, PROFILE)
            self.assertEqual(len(calls), 1)

    async def test_absent_profile_records_and_empty_cv_records_skip_inference(self):
        aligner, calls = self.aligner(RuntimeError('must not be called'))
        output = await aligner.align(CLAIMS, {'name': 'Person'})
        self.assertTrue(all(item['profile_index'] is None for mappings in output.values() for item in mappings))
        self.assertEqual(aligner.validate(output, CLAIMS, {}), output)
        claims = dict(CLAIMS, jobs=[], education=[])
        self.assertEqual(await aligner.align(claims, PROFILE), {'jobs': [], 'education': []})
        self.assertEqual(calls, [])

    async def test_model_failures_and_malformed_json_surface_without_retry(self):
        aligner, calls = self.aligner(RuntimeError('model unavailable'))
        with self.assertRaises(RuntimeError):
            await aligner.align(CLAIMS, PROFILE)
        self.assertEqual(len(calls), 1)
        aligner = RecordAligner(RunnableLambda(lambda _: AIMessage(content='not JSON')))
        with self.assertRaises(ValueError):
            await aligner.align(CLAIMS, PROFILE)
