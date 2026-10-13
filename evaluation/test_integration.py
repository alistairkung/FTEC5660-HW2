import json
import tempfile
import unittest
from pathlib import Path

from langchain_core.runnables import RunnableLambda

from evaluation.cases import load_cases
from evaluation.run_integration import capture_reference, evaluate_integrated, grounding


class IntegrationEvaluationTests(unittest.IsolatedAsyncioTestCase):
    def test_grounding_accepts_terminal_date_corroboration_and_mixed_complete_skill_citations(self):
        case = load_cases(limit=1)[0]
        case['expected_claims'] = {'jobs.1.is_current': 'MISMATCH', 'skills.0.name': 'MISMATCH'}
        case['completeness'] = ['skills']
        result = {'score': 0.1, 'findings': [
            {'claim_id': 'jobs.1.is_current', 'status': 'MISMATCH', 'evidence': [
                {'path': 'experience.0.is_current'}, {'path': 'experience.0.end_year'}]},
            {'claim_id': 'skills.0.name', 'status': 'MISMATCH', 'evidence': [
                {'path': 'skills'}, {'path': 'skills.0.name'}]}]}
        self.assertTrue(all(c['grounded_correct'] for c in grounding(case, result)['claims']))
        result['findings'][0]['evidence'][0]['path'] = 'experience.1.is_current'
        self.assertFalse(grounding(case, result)['claims'][0]['grounded_correct'])

    async def test_reference_is_captured_first_and_production_wiring_uses_identical_saved_inputs(self):
        from evaluation.test_controlled import ControlledEvaluationTests
        model, prompts = ControlledEvaluationTests().fake_model()
        async def respond(prompt, **options):
            return await model.ainvoke(prompt)
        folder = Path(__file__).with_name('results')
        folder.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder) as directory:
            output = Path(directory) / 'reference.json'
            cases = load_cases(limit=1)
            cases[0]['explanation'] = 'SECRET EXPECTED ANSWER'
            reference = await capture_reference(cases, model, output)
            self.assertEqual(json.loads(output.read_text()), reference)
            self.assertEqual(reference['summary']['paths']['reference']['llm_calls'], 2)
            self.assertEqual(reference['records'][0]['reference']['result']['score'], 0.9)
            integrated = await evaluate_integrated(reference, RunnableLambda(respond), Path(directory) / 'integrated.json')
            record = integrated['records'][0]
            self.assertEqual(record['case'], cases[0])
            self.assertEqual(record['reference'], reference['records'][0]['reference'])
            self.assertEqual(record['integrated']['result']['score'], 0.9)
            self.assertEqual(integrated['summary']['paths']['integrated']['llm_calls'], 2)
            self.assertNotIn('SECRET EXPECTED ANSWER', json.dumps(prompts))
            self.assertTrue(all('expected_label' not in p for p in prompts))

    async def test_incomplete_reference_is_rejected_before_inference(self):
        with self.assertRaisesRegex(ValueError, 'completed reference'):
            await evaluate_integrated({'complete': False}, None, 'unused')


if __name__ == '__main__':
    unittest.main()
