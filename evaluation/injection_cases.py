"""Thirteen fixed alignment mistakes; grading metadata never enters model inputs."""

from copy import deepcopy

from cv_verifier.evidence_aligner import EvidenceAlignmentValidator
from evaluation.cases import load_cases
from evaluation.evidence_alignment import correspondence_expectations


# First three rows are the reviewed vertical slice. Swaps change both jobs to
# preserve uniqueness; no source claim/profile facts or expected answers change.
INJECTIONS = [
    ('unmatched_job_valid', 'mba_alias', 'false_unmatched', 'jobs', [0]),
    ('ambiguous_education_valid', 'mba_alias', 'false_ambiguous', 'education', [0]),
    ('wrong_jobs_logistics', 'mba_alias', 'wrong_confident', 'jobs', [0, 1]),
    ('unmatched_job_start_year', 'start_year_contradiction', 'false_unmatched', 'jobs', [0]),
    ('unmatched_education_valid', 'mba_alias', 'false_unmatched', 'education', [0]),
    ('unmatched_education_year', 'graduation_year_contradiction', 'false_unmatched', 'education', [0]),
    ('ambiguous_job_alias', 'employer_alias', 'false_ambiguous', 'jobs', [0]),
    ('ambiguous_job_end_year', 'end_year_contradiction', 'false_ambiguous', 'jobs', [1]),
    ('ambiguous_education_degree', 'fabricated_degree', 'false_ambiguous', 'education', [0]),
    ('wrong_jobs_engineering', 'skill_alias', 'wrong_confident', 'jobs', [0, 1]),
    ('skill_powerpoint_unmatched', 'skill_alias', 'skill_isolation', 'skills', [2]),
    ('skill_uiux_ambiguous', 'bsc_alias', 'skill_isolation', 'skills', [0]),
    ('skill_docker_wrong', 'fabricated_skill', 'skill_isolation', 'skills', [0]),
]


def load_injections(limit=None):
    if limit is not None and (type(limit) is not int or not 1 <= limit <= len(INJECTIONS)):
        raise ValueError('Limit must select 1–13 fixed injections')
    sources = {case['id']: case for case in load_cases()}
    cases = []
    validator = EvidenceAlignmentValidator()
    for identifier, source_id, kind, collection, indexes in INJECTIONS[:limit]:
        case = deepcopy(sources[source_id])
        original = {name: [{'cv_index': index, 'profile_index': profile_index, 'status': status,
                            'reason': 'Correspondence references supplied evidence; factual agreement is unverified.'}
                           for index, (status, profile_index) in enumerate(entries)]
                    for name, entries in correspondence_expectations(case).items()}
        validator.validate(original, case['claims'], case['profile'])
        injected = deepcopy(original)
        if kind == 'wrong_confident':
            left, right = (injected[collection][index] for index in indexes)
            left['profile_index'], right['profile_index'] = right['profile_index'], left['profile_index']
        elif identifier == 'skill_docker_wrong':
            injected['skills'][0].update(profile_index=0, status='ALIGNED')
        else:
            status = ('UNMATCHED' if kind == 'false_unmatched' or identifier == 'skill_powerpoint_unmatched'
                      else 'AMBIGUOUS')
            for index in indexes:
                injected[collection][index].update(profile_index=None, status=status)
        # Mutate only after original validation, then require valid injected structure.
        validator.validate(injected, case['claims'], case['profile'])
        prefixes = tuple(f'{collection}.{index}.' for index in indexes)
        case.update(id=identifier, source_case=source_id, injection_type=kind,
                    target_records=[f'{collection}.{index}' for index in indexes],
                    original_alignment=original, injected_alignment=injected,
                    targets={reference: status for reference, status in case['expected_claims'].items()
                             if reference.startswith(prefixes)},
                    changes_to_alignment=[{'collection': collection, 'cv_index': index,
                                           'before': deepcopy(original[collection][index]),
                                           'after': deepcopy(injected[collection][index])} for index in indexes],
                    ground_truth_basis=case['baseline_basis'] + ' ' + case['explanation'] +
                    ' Correct record indices and skill aliases are independently declared in the PR18/19 fixtures; '
                    'only correspondence is injected, not CV or profile facts.')
        cases.append(case)
    return cases
