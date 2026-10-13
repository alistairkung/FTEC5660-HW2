"""Independent correspondence expectations and diagnostics for the same 25 controls.

These are grading metadata, never aligner/verifier inputs. Saved CV1 skills
correspond respectively to Supply Chain, Inventory Management and Operations;
CV2 to UI/UX, Prototyping and Graphic Design; CV3 to Strategy, Problem Solving
and PowerPoint. The two wording differences are accepted aliases. Docker has
no counterpart, and redacted skills permit no correspondence determination.
"""

from collections import Counter


SOURCE_SKILL_INDEXES = {'CV_1.pdf': [0, 1, 2], 'CV_2.pdf': [0, 1, 2], 'CV_3.pdf': [0, 1, 2]}
SOURCE_SKILL_ALIASES = {'CV_1.pdf': set(), 'CV_2.pdf': {0}, 'CV_3.pdf': {2}}


def correspondence_expectations(case):
    expected = {collection: [("AMBIGUOUS", None) if index == 'ambiguous' else
                             ("UNMATCHED", None) if index is None else ("ALIGNED", index)
                             for index in indexes]
                for collection, indexes in case['alignment_expectations'].items()}
    indexes = SOURCE_SKILL_INDEXES[case['source']]
    if 'keep_skills' in case:
        indexes = [indexes[index] for index in case['keep_skills']]
    expected['skills'] = [("ALIGNED", index) for index in indexes]
    if case['id'] == 'fabricated_skill':
        expected['skills'][0] = ("UNMATCHED", None)
    elif case['id'] == 'missing_skills_evidence':
        expected['skills'] = [("AMBIGUOUS", None) for _ in indexes]
    return expected


def grade_correspondence(case, result, *, broad):
    actual = result.get('evidence_alignments' if broad else 'alignments') or {}
    output = {}
    for collection, expected in correspondence_expectations(case).items():
        if collection == 'skills' and not broad:
            continue  # RecordAligner has no skill correspondence contract.
        entries = {entry['cv_index']: entry for entry in actual.get(collection, [])}
        metrics = Counter({'graded': 0, 'correct': 0, 'uncertain_expected': 0, 'uncertain_correct': 0,
                           'ambiguous': 0, 'unmatched': 0, 'aligned': 0, 'missing': 0,
                           'alias_graded': 0, 'alias_correct': 0, 'unmatched_graded': 0, 'unmatched_correct': 0})
        failures = []
        for index, (status, target) in enumerate(expected):
            entry = entries.get(index)
            if entry is None:
                metrics['missing'] += 1
            elif broad:
                metrics[entry['status'].lower()] += 1
            correct = entry is not None and entry['profile_index'] == target
            if broad:
                correct = correct and entry['status'] == status
            if status == 'AMBIGUOUS':
                metrics['uncertain_expected'] += 1
                metrics['uncertain_correct'] += correct
            else:
                metrics['graded'] += 1
                metrics['correct'] += correct
                if status == 'UNMATCHED':
                    metrics['unmatched_graded'] += 1
                    metrics['unmatched_correct'] += correct
                if collection == 'skills' and target in SOURCE_SKILL_ALIASES[case['source']]:
                    metrics['alias_graded'] += 1
                    metrics['alias_correct'] += correct
            if not correct:
                failures.append({'cv_index': index, 'expected_status': status, 'expected_index': target,
                                 'actual': entry})
        output[collection] = {**metrics, 'failures': failures}
    return output


def alignment_summary(records):
    summary = {}
    for name in ('record', 'evidence'):
        collections = {collection: Counter() for collection in
                       (('jobs', 'education', 'skills') if name == 'evidence' else ('jobs', 'education'))}
        failures = []
        invalid = 0
        errors = 0
        ambiguous_cases = 0
        for record in records:
            result = record[name]['result']
            alignment_errors = [error for error in result['errors'] if error['stage'] == 'alignment']
            errors += bool(alignment_errors)
            invalid += any(error['kind'] == 'contract' for error in alignment_errors)
            grading = grade_correspondence(record['case'], result, broad=name == 'evidence')
            ambiguous_cases += any(value['ambiguous'] for value in grading.values())
            for collection, metrics in grading.items():
                collections[collection].update({key: value for key, value in metrics.items() if key != 'failures'})
                failures.extend({'case': record['case']['id'], 'collection': collection, **failure}
                                for failure in metrics['failures'])
        summary[name] = {'collections': {collection: {**values, 'accuracy': values['correct'] / values['graded']
                                                     if values['graded'] else None}
                                         for collection, values in collections.items()},
                         'invalid_output_cases': invalid, 'alignment_error_cases': errors,
                         'invalid_output_frequency': invalid / len(records) if records else None,
                         'ambiguous_cases': ambiguous_cases if name == 'evidence' else None,
                         'ambiguous_case_frequency': ambiguous_cases / len(records) if records and name == 'evidence' else None,
                         'failures': failures}
    summary['record_mapping_differences'] = []
    summary['claim_status_differences'] = []
    for record in records:
        left, right = record['record']['result'], record['evidence']['result']
        for collection in ('jobs', 'education'):
            mappings = [{entry['cv_index']: entry['profile_index'] for entry in
                         (result.get('alignments') or {}).get(collection, [])} for result in (left, right)]
            for index in sorted(set(mappings[0]) | set(mappings[1])):
                if index not in mappings[0] or index not in mappings[1] or mappings[0][index] != mappings[1][index]:
                    summary['record_mapping_differences'].append({'case': record['case']['id'], 'collection': collection,
                                                                'cv_index': index, 'record': mappings[0].get(index),
                                                                'evidence': mappings[1].get(index),
                                                                'record_mapping_present': index in mappings[0],
                                                                'evidence_mapping_present': index in mappings[1]})
        statuses = [{finding['claim_id']: finding['status'] for finding in result['findings']} for result in (left, right)]
        for reference in sorted(set(statuses[0]) | set(statuses[1])):
            if statuses[0].get(reference) != statuses[1].get(reference):
                summary['claim_status_differences'].append({'case': record['case']['id'], 'claim_id': reference,
                                                          'record': statuses[0].get(reference), 'evidence': statuses[1].get(reference)})
    return summary
