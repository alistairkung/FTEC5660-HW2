"""Saved controls plus explicitly synthetic stress cases; labels stay in grading."""

from copy import deepcopy

from cv_verifier.claim_comparison_validator import ClaimComparisonValidator
from evaluation.cases import load_cases, _refresh_wording
from evaluation.injection_cases import load_injections
from evaluation.run_controlled import neutral_record_reasons


def _records(case):
    return neutral_record_reasons({collection: [
        {'cv_index': i, 'profile_index': j if type(j) is int else None, 'reason': 'Saved fixture correspondence.'}
        for i, j in enumerate(indexes)] for collection, indexes in case['alignment_expectations'].items()})


def load_gate_cases():
    controls = load_cases()
    cases = []
    for case in controls:
        cases.append(dict(deepcopy(case), input_alignment=_records(case), original_alignment=_records(case),
                          injection_type='gate_control', group='saved_control'))
    for case in load_injections():
        if case['injection_type'] == 'skill_isolation':
            continue
        def records(alignment):
            return neutral_record_reasons({collection: [{key: deepcopy(entry[key]) for key in
                ('cv_index', 'profile_index', 'reason')} for entry in alignment[collection]]
                for collection in ('jobs', 'education')})
        cases.append(dict(deepcopy(case), input_alignment=records(case['injected_alignment']),
                          original_alignment=records(case['original_alignment']), group='saved_injection'))
    # Synthetic controls below are copies/redactions, not newly observed histories.
    base = deepcopy(cases[0])
    unparsed = deepcopy(next(case for case in cases if case['id'] == 'ambiguous_record_details'))
    unparsed.update(id='unparsed_confident', group='synthetic_uncertainty')
    unparsed['input_alignment']['jobs'][0]['profile_index'] = 1
    cases.append(unparsed)
    missing = deepcopy(base)
    missing.update(id='unusable_profile_record', group='synthetic_uncertainty', expected_label=None)
    missing['profile']['experience'][1] = {key: None for key in missing['profile']['experience'][1]}
    missing['alignment_expectations']['jobs'][0] = 'ambiguous'
    for key in missing['expected_claims']:
        if key.startswith('jobs.0.'):
            missing['expected_claims'][key] = 'UNKNOWN'
    missing['targets'] = {key: value for key, value in missing['expected_claims'].items() if key.startswith('jobs.0.')}
    cases.append(missing)
    absent = deepcopy(base)
    absent.update(id='missing_experience_collection', group='synthetic_uncertainty', expected_label=None)
    absent['profile']['experience'] = None
    absent['alignment_expectations']['jobs'] = ['ambiguous', 'ambiguous']
    absent['input_alignment'] = _records(absent)
    absent['original_alignment'] = deepcopy(absent['input_alignment'])
    for key in absent['expected_claims']:
        if key.startswith('jobs.'):
            absent['expected_claims'][key] = 'UNKNOWN'
    absent['targets'] = {key: value for key, value in absent['expected_claims'].items() if key.startswith('jobs.')}
    cases.append(absent)
    education = deepcopy(base)
    education.update(id='wrong_education_augmented', group='synthetic_wrong', injection_type='wrong_confident')
    education['profile']['education'].append(deepcopy(next(case for case in controls if case['source'] == 'CV_3.pdf')['profile']['education'][0]))
    education['input_alignment']['education'][0]['profile_index'] = 1
    education['targets'] = {key: value for key, value in education['expected_claims'].items() if key.startswith('education.0.')}
    cases.append(education)
    joint = deepcopy(next(case for case in cases if case['id'] == 'fabricated_employer'))
    joint.update(id='false_employer_plus_false_start', group='synthetic_negative', expected_label=0)
    joint['claims']['jobs'][0]['start_year'] = 2021
    _refresh_wording('jobs', joint['claims']['jobs'][0])
    joint['expected_claims']['jobs.0.start_year'] = 'MISMATCH'
    joint['targets'] = {'jobs.0.company': 'MISMATCH', 'jobs.0.start_year': 'MISMATCH'}
    cases.append(joint)
    # A real isolated date discrepancy must survive even if company wording is an alias.
    alias = deepcopy(next(case for case in cases if case['id'] == 'employer_alias'))
    alias.update(id='alias_plus_false_start', group='synthetic_negative', expected_label=0)
    alias['claims']['jobs'][0]['start_year'] = 2023
    _refresh_wording('jobs', alias['claims']['jobs'][0])
    alias['expected_claims']['jobs.0.start_year'] = 'MISMATCH'
    alias['targets'] = {'jobs.0.start_year': 'MISMATCH'}
    cases.append(alias)
    confounded = deepcopy(next(case for case in cases if case['id'] == 'wrong_jobs_engineering'))
    confounded.update(id='false_employer_matches_wrong_record', group='synthetic_negative', expected_label=0)
    confounded['claims']['jobs'][0]['company'] = 'Innovate HK Ltd'
    _refresh_wording('jobs', confounded['claims']['jobs'][0])
    confounded['expected_claims']['jobs.0.company'] = 'MISMATCH'
    confounded['targets']['jobs.0.company'] = 'MISMATCH'
    confounded['ground_truth_basis'] = (
        'Synthetic stress derived from CV_3 and the PR20 wrong_jobs_engineering swap. '
        'The CV employer alone is changed to the employer at the wrong assigned index; '
        'independent expected claims remain anchored to the original Shopee/Innovate HK correspondence.')
    cases.append(confounded)
    for case in cases:
        if case['group'].startswith('synthetic'):
            case['ground_truth_basis'] = ('Synthetic copied/redacted saved inputs; original fixture correspondence is fixed independently. '
                                         'No added profile record is claimed as real public history. '
                                         'Expected statuses are grading-only interventions, not facts inferred by the gate.')
        # Ensure mutated/redacted controls still have exactly the canonical catalog.
        assert set(case['expected_claims']) == set(ClaimComparisonValidator().claim_catalog(case['claims']))
    first = ('wrong_jobs_logistics', 'start_year_contradiction', 'wrong_jobs_engineering')
    return [next(case for case in cases if case['id'] == key) for key in first] + [case for case in cases if case['id'] not in first]


def probe_candidate_rules():
    """Offline correspondence-origin probe, not synthetic whole-CV truth inference."""
    from collections import Counter
    from evaluation.correspondence_gate import CorrespondencePlausibilityGate
    cases = [case for case in load_gate_cases() if case['group'] == 'saved_control' or case['id'] in
             ('wrong_jobs_logistics', 'wrong_jobs_engineering', 'wrong_education_augmented',
              'false_employer_matches_wrong_record')]
    sparse = deepcopy(next(case for case in cases if case['id'] == 'mba_alias'))
    sparse['id'] = 'probe_sparse_company_only'
    for field in CorrespondencePlausibilityGate.FIELDS['jobs']:
        if field != 'company':
            sparse['claims']['jobs'][0][field] = None
    duplicate = deepcopy(next(case for case in cases if case['id'] == 'fabricated_employer'))
    duplicate['id'] = 'probe_duplicate_context'
    extra = deepcopy(duplicate['profile']['experience'][1])
    extra['company'] = duplicate['claims']['jobs'][0]['company']
    duplicate['profile']['experience'].append(extra)
    # These two rows grade declared original correspondence only. The duplicate
    # cannot prove a false employer once a matching record has been added.
    probes = [(sparse, 'jobs', 0), (duplicate, 'jobs', 0)]
    for case in cases:
        for collection in ('jobs', 'education'):
            probes.extend((case, collection, entry['cv_index']) for entry in case['input_alignment'][collection]
                          if entry['profile_index'] is not None)
    gate, rows = CorrespondencePlausibilityGate(), []
    for case, collection, index in probes:
        expected = case['alignment_expectations'][collection][index]
        if type(expected) is not int:
            continue
        report = gate.apply(case['claims'], case['profile'], case['input_alignment'])
        decision = next(d for d in report['diagnostics'] if d['collection'] == collection and d['cv_index'] == index)
        assigned = decision['original_profile_index']
        relations = decision['evidence'][assigned]['relations']
        count = sum(value == 'equal' for value in relations.values())
        rows.append({'id': f"{case['id']}:{collection}.{index}", 'correct_input': assigned == expected,
                     'rejected': {'all_fields_agree': any(value == 'different' for value in relations.values()),
                                  'match_count_2': count < 2, 'match_count_3': count < 3,
                                  'higher_match_count_alternative': any(sum(value == 'equal' for value in item['relations'].values()) > count
                                                                      for item in decision['evidence']),
                                  'identity_timeline_conflict': decision['post_profile_index'] is None}})
    summary = {}
    for rule in rows[0]['rejected']:
        rejected = [row['id'] for row in rows if row['correct_input'] and row['rejected'][rule]]
        caught = [row['id'] for row in rows if not row['correct_input'] and row['rejected'][rule]]
        missed = [row['id'] for row in rows if not row['correct_input'] and not row['rejected'][rule]]
        summary[rule] = {'correct_inputs': sum(row['correct_input'] for row in rows), 'correct_rejected': len(rejected),
                         'incorrect_inputs': sum(not row['correct_input'] for row in rows), 'incorrect_caught': len(caught),
                         'rejected_ids': rejected, 'caught_ids': caught, 'missed_ids': missed}
    return {'summary': summary, 'rows': rows,
            'limitation': 'Correlated controls; synthetic duplicate-context row grades declared origin, not a verified false-CV claim.'}
