"""Hand-selected controls; labels never enter verifier inputs."""

import json
from copy import deepcopy
from pathlib import Path

from cv_verifier.claim_validator import ClaimValidator
from cv_verifier.claim_comparison_validator import ClaimComparisonValidator

SAVED_INPUTS = Path(__file__).with_name('saved_public_inputs.json')

CASES = [
    dict(id='mba_alias', category='education', source='CV_1.pdf', changes={},
         expected_label=1, targets={'education.0.degree': 'MATCH', 'education.0.graduation_year': 'MATCH'},
         control='positive', explanation='MBA and Master of Business Administration are equivalent; both graduate in 2012.', completeness=[]),
    dict(id='start_year_contradiction', category='employment', source='CV_1.pdf', changes={'jobs.0.start_year': 2021},
         expected_label=0, targets={'jobs.0.start_year': 'MISMATCH'}, control='negative',
         explanation='CV states 2021; the corresponding Deloitte record explicitly starts in 2020.', completeness=[]),
    dict(id='fabricated_employer', category='employment', source='CV_1.pdf', changes={'jobs.0.company': 'Siemens'},
         expected_label=0, targets={'jobs.0.company': 'MISMATCH'}, control='negative',
         explanation='Siemens is neither Deloitte nor Carousell and is absent from the supplied complete employment history. Remaining role/dates identify Deloitte.', completeness=['experience']),
    dict(id='employment_exact_years', category='employment', source='CV_1.pdf', changes={}, keep_jobs=[1],
         alignment_expectations={'jobs': [0]}, expected_label=1,
         targets={'jobs.0.start_year': 'MATCH', 'jobs.0.end_year': 'MATCH'}, control='positive',
         explanation='CV lists only Carousell, explicitly 2012–2018 in both inputs; omitting another profile job is not a discrepancy.', completeness=[]),
    dict(id='end_year_contradiction', category='employment', source='CV_1.pdf', changes={'jobs.1.end_year': 2019},
         expected_label=0, targets={'jobs.1.end_year': 'MISMATCH'}, control='negative',
         explanation='2019 contradicts the explicit Carousell end year 2018.', completeness=[]),
    dict(id='false_current_employment', category='employment', source='CV_1.pdf',
         changes={'jobs.1.is_current': True, 'jobs.1.end_year': None}, expected_label=0,
         targets={'jobs.1.is_current': 'MISMATCH'}, control='negative',
         explanation='Carousell is explicitly not current in the profile; CV now claims 2012–Present.', completeness=[]),
    dict(id='title_plus_seniority_equivalence', category='employment', source='CV_1.pdf',
         changes={'jobs.0.title': 'Manager', 'jobs.0.seniority': 'senior'}, expected_label=1,
         targets={'jobs.0.title': 'MATCH', 'jobs.0.seniority': 'MATCH'}, control='positive',
         explanation='Manager plus explicitly senior seniority equals the supplied Manager/senior role.', completeness=[]),
    dict(id='explicit_seniority_contradiction', category='employment', source='CV_2.pdf',
         changes={'jobs.0.title': 'Senior Manager'}, expected_label=0, targets={'jobs.0.title': 'MISMATCH'}, control='negative',
         explanation='Senior Manager explicitly claims senior grade; corresponding BCG Manager has junior seniority.', completeness=[]),
    dict(id='unqualified_title_extra_grade', category='employment', source='CV_2.pdf', changes={},
         expected_label=1, targets={'jobs.0.title': 'MATCH'}, control='positive',
         explanation='Unqualified Manager does not claim seniority; profile Manager/junior does not contradict it.', completeness=[]),
    dict(id='employer_alias', category='employment', source='CV_2.pdf', changes={'jobs.0.company': 'Boston Consulting Group'},
         expected_label=1, targets={'jobs.0.company': 'MATCH'}, control='positive',
         explanation='BCG is the recognized abbreviation for Boston Consulting Group, with unchanged role/dates.', completeness=[]),
    dict(id='reordered_employment', category='employment', source='CV_1.pdf', changes={}, reorder_jobs=[1, 0],
         alignment_expectations={'jobs': [0, 1]}, expected_label=1, targets={'jobs.0.company': 'MATCH', 'jobs.1.title': 'MATCH'},
         control='positive', explanation='Only CV array order changes; record facts are identical.', completeness=[]),
    dict(id='ambiguous_record_details', category='alignment', source='CV_1.pdf',
         changes={'jobs.0.raw': 'Employment details unavailable: employer, role and dates not specified',
                  'jobs.0.company': None, 'jobs.0.title': None, 'jobs.0.seniority': None, 'jobs.0.dates_raw': None,
                  'jobs.0.start_year': None, 'jobs.0.end_year': None, 'jobs.0.is_current': None},
         alignment_expectations={'jobs': ['ambiguous', 0]}, expected_label=None, targets={'jobs.0.record': 'UNKNOWN'},
         control='ambiguous', explanation='An unparsed record supplies no factual employer/role/date claim or defensible correspondence. No binary label is established.', completeness=[]),
    dict(id='graduation_year_contradiction', category='education', source='CV_1.pdf',
         changes={'education.0.graduation_year': 2013}, expected_label=0, targets={'education.0.graduation_year': 'MISMATCH'},
         control='negative', explanation='2013 contradicts the explicit Stanford graduation year 2012.', completeness=[]),
    dict(id='bsc_alias', category='education', source='CV_2.pdf', changes={'education.0.degree': 'Bachelor of Science'},
         expected_label=1, targets={'education.0.degree': 'MATCH'}, control='positive',
         explanation='BSc is the standard Bachelor of Science abbreviation; other education facts are unchanged.', completeness=[]),
    dict(id='school_alias', category='education', source='CV_1.pdf', changes={'education.0.school': 'Stanford University'},
         expected_label=1, targets={'education.0.school': 'MATCH'}, control='positive',
         explanation='Stanford and Stanford University refer to the same supplied school.', completeness=[]),
    dict(id='fabricated_school', category='education', source='CV_1.pdf', changes={'education.0.school': 'University of Oxford'},
         expected_label=0, targets={'education.0.school': 'MISMATCH'}, control='negative',
         explanation='Oxford is a distinct school absent from the supplied complete Stanford-only education history.', completeness=['education']),
    dict(id='fabricated_degree', category='education', source='CV_1.pdf', changes={'education.0.degree': 'PhD'},
         expected_label=0, targets={'education.0.degree': 'MISMATCH'}, control='negative',
         explanation='PhD and MBA are distinct degree levels; the corresponding record supplies only MBA.', completeness=['education']),
    dict(id='exact_skill', category='skills', source='CV_1.pdf', changes={}, keep_skills=[0], expected_label=1,
         targets={'skills.0.name': 'MATCH'}, control='positive', explanation='Only Supply Chain is claimed; it is supplied explicitly. Fewer listed CV skills are not a discrepancy.', completeness=[]),
    dict(id='skill_alias', category='skills', source='CV_3.pdf', changes={}, expected_label=1,
         targets={'skills.2.name': 'MATCH'}, control='positive', explanation='Microsoft PowerPoint and PowerPoint are the same skill.', completeness=[]),
    dict(id='fabricated_skill', category='skills', source='CV_1.pdf', changes={'skills.0.name': 'Docker', 'skills.0.raw': 'Docker'},
         expected_label=0, targets={'skills.0.name': 'MISMATCH'}, control='negative',
         explanation='Docker is absent from the complete Supply Chain/Inventory Management/Operations skills evidence and is not an alias for them.', completeness=['skills']),
    dict(id='missing_skills_evidence', category='skills', source='CV_1.pdf', changes={}, profile_redactions=['skills'],
         expected_label=None, targets={'skills.0.name': 'UNKNOWN', 'skills.1.name': 'UNKNOWN', 'skills.2.name': 'UNKNOWN'},
         control='ambiguous', explanation='All skill evidence is deliberately withheld in a copied input; absence is not evidence of a discrepancy.', completeness=[]),
    dict(id='explicit_current_city', category='location', source='CV_1.pdf',
         changes={'city': {'raw': 'Current residence: London', 'value': 'London',
                           'candidates': [{'city': 'London', 'role': 'current'}]}}, expected_label=1,
         targets={'city': 'MATCH'}, control='positive', explanation='An explicitly labelled London residence matches profile London.', completeness=[]),
    dict(id='ambiguous_city_header', category='location', source='CV_2.pdf',
         changes={'city': {'raw': 'Beijing, China | Hong Kong', 'value': None,
                           'candidates': [{'city': 'Beijing', 'role': 'unspecified'}, {'city': 'Hong Kong', 'role': 'unspecified'}]}},
         expected_label=1, targets={'city': 'MATCH'}, control='positive',
         explanation='The unlabelled personal header includes profile Beijing as a plausible current city; list order does not establish residence.', completeness=[]),
    dict(id='hometown_is_not_current_city', category='location', source='CV_1.pdf',
         changes={'city': {'raw': 'Current city: Stanford; hometown: London', 'value': 'Stanford',
                           'candidates': [{'city': 'Stanford', 'role': 'current'}, {'city': 'London', 'role': 'hometown'}]}},
         expected_label=0, targets={'city': 'MISMATCH'}, control='negative',
         explanation='Explicit Stanford current residence contradicts profile London. Matching the CV hometown cannot rescue that false current-city claim.', completeness=[]),
    dict(id='additional_fabricated_job', category='alignment', source='CV_1.pdf', changes={},
         additional_job={'raw': 'Engineer, Siemens, 2023–2024', 'company': 'Siemens', 'title': 'Engineer', 'seniority': None,
                         'dates_raw': '2023–2024', 'start_year': 2023, 'end_year': 2024, 'is_current': False},
         alignment_expectations={'jobs': [1, 0, None]}, expected_label=0,
         targets={f'jobs.2.{field}': 'MISMATCH' for field in ('company', 'title', 'start_year', 'end_year', 'is_current')},
         control='negative', explanation='An additional Siemens engineering record is absent from the complete two-record employment history; facts from unrelated jobs cannot substantiate it.', completeness=['experience']),
]


def load_cases(limit=None):
    if limit is not None and (type(limit) is not int or not 1 <= limit <= len(CASES)):
        raise ValueError('Case limit must select a positive number of available cases')
    saved = json.loads(SAVED_INPUTS.read_text())
    cases = []
    for spec in CASES[:limit]:
        original = saved['sources'][spec['source']]
        claims, profile = deepcopy(original['claims']), deepcopy(original['profile'])
        touched = set()
        for path, value in spec['changes'].items():
            parts = path.split('.')
            destination = claims
            for part in parts[:-1]:
                destination = destination[int(part)] if isinstance(destination, list) else destination[part]
            key = int(parts[-1]) if isinstance(destination, list) else parts[-1]
            destination[key] = deepcopy(value)
            if parts[0] in ('jobs', 'education') and len(parts) >= 3:
                touched.add((parts[0], int(parts[1])))
        for collection, index in touched:
            if f'{collection}.{index}.raw' not in spec['changes']:
                _refresh_wording(collection, claims[collection][index])
        if 'reorder_jobs' in spec:
            claims['jobs'] = [claims['jobs'][index] for index in spec['reorder_jobs']]
        if 'additional_job' in spec:
            claims['jobs'].append(deepcopy(spec['additional_job']))
        if 'keep_jobs' in spec:
            claims['jobs'] = [claims['jobs'][index] for index in spec['keep_jobs']]
        if 'keep_skills' in spec:
            claims['skills'] = [claims['skills'][index] for index in spec['keep_skills']]
        for field in spec.get('profile_redactions', []):
            profile[field] = None
        ClaimValidator().validate(claims)
        expected = {reference: 'MATCH' for reference in ClaimComparisonValidator().claim_catalog(claims)}
        if not set(spec['targets']) <= set(expected):
            raise ValueError('Expected references must belong to the canonical catalog')
        expected.update(spec['targets'])
        alignments = {'jobs': [1, 0] if spec['source'] != 'CV_2.pdf' else [0, 1], 'education': [0]}
        alignments.update(spec.get('alignment_expectations', {}))
        cases.append({**deepcopy(spec), 'claims': claims, 'profile': profile, 'expected_claims': expected,
                      'alignment_expectations': alignments, 'source_profile_id': profile['id'],
                      'baseline_basis': 'The source public CV is labelled valid by the supplied ground truth; '
                                        'all permitted source fields were manually checked against this saved profile. '
                                        'Untouched claims retain MATCH except explicitly declared evidence redactions.',
                      'source_artifact': saved['source_artifact'], 'source_sha256': saved['source_sha256']})
    return cases


def _refresh_wording(collection, entry):
    if collection == 'jobs':
        start = entry['start_year']
        end = 'Present' if entry['is_current'] is True else entry['end_year']
        entry['dates_raw'] = f'{start}–{end}' if start is not None and end is not None else None
        parts = [entry['title'], entry['company']]
        if entry['seniority']:
            parts.append(f"seniority: {entry['seniority']}")
        if entry['dates_raw']:
            parts.append(entry['dates_raw'])
    else:
        year = entry['graduation_year']
        entry['dates_raw'] = f'Graduated {year}' if year is not None else None
        parts = [entry['degree'], entry['field'], entry['school'], entry['dates_raw']]
    entry['raw'] = ', '.join(str(part) for part in parts if part)
