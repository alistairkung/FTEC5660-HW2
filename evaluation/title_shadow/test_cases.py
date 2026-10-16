from copy import deepcopy
import unittest

from evaluation.title_shadow.cases import bind, load_cases, saved


class CorrespondenceTests(unittest.TestCase):
    def test_declared_plan_and_ground_truth_are_outside_title_inputs(self):
        cases=load_cases()
        self.assertEqual(len(cases),25)
        self.assertEqual(sum(c['repetitions'] for c in cases),45)
        self.assertEqual(sum(c['repetitions'] for c in cases if c['split']=='held_out'),18)
        self.assertTrue(all(c['binding']['status']=='REVIEWED' for c in cases))
        for case in cases:
            self.assertEqual(set(case['title_input']),{'raw_title','claimed_title','claimed_seniority','profile_title','profile_seniority'})

    def test_record_binding_uses_reviewed_facts_not_array_position(self):
        source=saved();before=bind(source['claims'],source['profile'])
        profile=deepcopy(source['profile']);profile['experience'].reverse()
        after=bind(source['claims'],profile)
        self.assertEqual(before['record_index'],1)
        self.assertEqual(after['record_index'],0)
        self.assertEqual(after['status'],'REVIEWED')

    def test_missing_duplicate_changed_or_unreviewed_facts_skip(self):
        source=saved()
        for variant in ('duplicate','missing','changed','unreviewed','null_history'):
            with self.subTest(variant=variant):
                profile=deepcopy(source['profile'])
                if variant=='duplicate':profile['experience'].append(deepcopy(profile['experience'][1]))
                if variant=='missing':profile['experience'][1].pop('end_year')
                if variant=='changed':profile['experience'][1]['company']='Different employer'
                if variant=='null_history':profile['experience']=None
                result=bind(source['claims'],profile,reviewed=variant!='unreviewed')
                self.assertEqual(result['status'],'NOT_EVALUATED')

    def test_null_grade_is_not_an_evidence_citation(self):
        case=next(c for c in load_cases() if c['id']=='director_exact')
        self.assertNotIn('experience.1.seniority',case['binding']['evidence_paths'])
