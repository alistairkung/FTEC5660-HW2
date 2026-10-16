"""Offline common grading for reliability captures; never imported by verifiers."""

import argparse
import json
import runpy
from collections import Counter
from copy import deepcopy

from cv_verifier.claim_comparison_validator import ClaimComparisonValidator
from evaluation.cases import load_cases
from evaluation.reliability import ARTIFACTS, ROOT, digest, save
from evaluation.run_controlled import grade


def public_case(cv, upstream, audit):
    """Apply manually checked public references only after model calls have finished."""
    reference = audit[cv]
    claims = upstream['claims']
    profile = reference['profile']
    expected = {key: 'MATCH' for key in ClaimComparisonValidator().claim_catalog(claims)}
    anchors = {}
    for collection, source, field in [('jobs', 'experience', 'company'), ('education', 'education', 'school')]:
        anchors[collection] = []
        for item_index, item in enumerate(claims[collection]):
            indexes = [i for i, record in enumerate(profile[source]) if isinstance(item[field], str)
                       and record[field].casefold() == item[field].casefold()]
            reviewed = reference.get('missing_anchor_review', {}).get(f'{collection}.{item_index}')
            if not indexes and item[field] is None and reviewed is not None:
                indexes = [reviewed]
            if len(indexes) != 1:
                raise ValueError(f'Public annotation needs manual correspondence audit: {cv} {item}')
            anchors[collection].append(indexes[0])
    targets = {}
    for rule in reference['contradictions']:
        collection, field = rule['collection'], rule['field']
        for i, item in enumerate(claims[collection]):
            if item[rule['match_field']].casefold() == rule['value'].casefold():
                key = f'{collection}.{i}.{field}'
                if key in expected:
                    expected[key] = targets[key] = 'MISMATCH'
                # Explicit seniority is an additional claim when extraction supplied it.
                seniority = f'{collection}.{i}.seniority'
                if rule.get('explicit_seniority') and seniority in expected:
                    expected[seniority] = targets[seniority] = 'MISMATCH'
    return dict(id=cv, source=cv, expected_label=reference['label'], claims=claims, profile=profile,
                expected_claims=expected, targets=targets, alignment_expectations=anchors,
                completeness=['experience', 'education', 'skills'])


def summarize(records):
    metrics = Counter()
    stages, errors = Counter(), Counter()
    for row in records:
        metrics['observations'] += 1
        if row.get('skipped'):
            metrics['skipped'] += 1
            continue
        metrics['evaluated'] += 1
        result = row['result']
        grading = row['grading']
        metrics['binary_cases'] += grading['binary_eligible']
        metrics['correct'] += grading['correct'] is True
        if grading['correct'] is False:
            metrics['false_valid' if grading['prediction'] else 'false_discrepant'] += 1
        metrics['score_half'] += result['score'] == 0.5
        metrics['negative_half'] += result['score'] == 0.5 and grading['prediction'] == 0
        metrics['known_negative_half'] += result['score'] == 0.5 and grading['binary_eligible']
        metrics['uncertainty_controls'] += not grading['binary_eligible']
        metrics['clean_uncertainty_controls'] += grading.get('uncertainty_correct') is True
        metrics['omitted_reference_claims'] += len(row.get('omitted_reference_claims', []))
        metrics['mismatch_outcomes'] += result['score'] == 0.1 and any(f['status'] == 'MISMATCH' for f in result['findings'])
        metrics['grounded_discrepancy_outcomes'] += row['grounding']['verified_expected_discrepancy']
        metrics['unknown_findings'] += sum(f['status'] == 'UNKNOWN' for f in result['findings'])
        metrics['processing_failure_cases'] += bool(result.get('errors'))
        metrics['contract_failure_cases'] += any(error.get('kind') == 'contract' or error.get('type') == 'ValueError'
                                                 for error in row['errors'])
        metrics['wrong_identities'] += row.get('identity_correct') is False
        metrics['unresolved_identities'] += row.get('identity_correct') is None
        metrics['graded_claims'] += grading['graded_claims']
        metrics['correct_claims'] += grading['graded_claims'] - len(grading['claim_failures'])
        metrics['graded_targets'] += grading['graded_targets']
        metrics['correct_targets'] += grading['graded_targets'] - len(grading['target_failures'])
        grounded = row['grounding']['claims']
        metrics['grounded_claims'] += len(grounded)
        metrics['grounded_correct_claims'] += sum(c['grounded_correct'] for c in grounded)
        metrics['wrong_record_citations'] += sum(c['wrong_record_cited'] for c in grounded)
        metrics['off_anchor_citations'] += len(grading['off_anchor_citations'])
        metrics['alignment_errors'] += len(grading['alignment_errors'])
        metrics['alignment_abstentions'] += len(grading['alignment_abstentions'])
        metrics['sum_cv_seconds'] += row['seconds']
        for error in row['errors']:
            errors[error['stage']] += 1
        for call in row['trace'].get('calls', []):
            metrics['llm_calls'] += 1
            stages[call['stage']] += 1
            if call.get('usage'):
                for field in ('input_tokens', 'output_tokens', 'total_tokens'):
                    metrics[field] += call['usage'].get(field, 0)
            else:
                metrics['calls_without_usage'] += 1
    return dict(metrics, call_stages=dict(stages), error_stages=dict(errors))


def collect():
    audit = json.loads((ARTIFACTS / 'public_audit.json').read_text())
    # Use the existing stricter PR24 grounding grader, identically for both paths.
    grounding = runpy.run_path(str(ARTIFACTS / 'sources/hybrid/evaluation/run_integration.py'))['grounding']
    truth = json.loads((ROOT / 'public_test/ground_truth.json').read_text())
    fixed = {item['id']: item for item in json.loads((ARTIFACTS / 'fixed_inputs.json').read_text())['inputs']}
    extended = {case['id']: case for case in load_cases()}
    report = dict(manifest=json.loads((ARTIFACTS / 'manifest.json').read_text()), groups={}, files={})
    for mode, count in [('full', 3), ('fixed', 3), ('extended', 1)]:
        for architecture in ('production', 'hybrid'):
            all_rows, wall_seconds = [], 0
            runs = []
            for run in range(1, count + 1):
                path = ARTIFACTS / f'{mode}_{architecture}_{run}.json'
                payload = json.loads(path.read_text())
                assert payload['complete'] and payload['source_unchanged'], path
                wall_seconds += payload['wall_seconds']
                rows = []
                for identifier, record in payload['records'].items():
                    if record.get('skipped'):
                        rows.append(dict(id=identifier, cv=record['cv'], skipped=True, reason=record['reason']))
                        continue
                    upstream = (record['trace'].get('upstream', {}) if mode == 'full' else
                                fixed[identifier]['upstream'] if mode == 'fixed' else dict(
                                    claims=extended[identifier]['claims'], identity='resolved',
                                    selected_candidate_id=extended[identifier]['profile']['id']))
                    cv = identifier if mode == 'full' else record['cv']
                    resolved = upstream.get('identity') == 'resolved'
                    identity_correct = (upstream['selected_candidate_id'] == audit[cv]['profile']['id']) if resolved and mode != 'extended' else True if mode == 'extended' else None
                    result = deepcopy(record.get('result', dict(score=record.get('score', 0.5), findings=[], errors=[])))
                    if mode == 'extended':
                        case = extended[identifier]
                    elif upstream.get('claims') is not None:
                        case = public_case(cv, upstream, audit)
                    else:
                        # Missing extraction remains a scored pipeline failure; no claim denominator invented.
                        case = dict(id=cv, expected_label=truth[cv]['label'], expected_claims={}, targets={},
                                    alignment_expectations={'jobs': [], 'education': []}, completeness=[])
                    grading, grounded = grade(case, result), grounding(case, result)
                    if mode != 'extended' and identity_correct is not True:
                        for claim in grounded['claims']:
                            claim['grounded_correct'] = False
                        grounded['verified_expected_discrepancy'] = False
                    typed_errors = []
                    for error in result.get('errors', []):
                        observed = next((e for e in record['trace'].get('stage_errors', []) if e['stage'] == error['stage']), {})
                        typed_errors.append({**error, **({'type': observed['type']} if 'type' in observed else {})})
                    rows.append(dict(id=identifier, cv=cv, run=run, result=result, identity_correct=identity_correct,
                        selected_id=upstream.get('selected_candidate_id'), grading=grading, grounding=grounded,
                        seconds=record.get('seconds', 0), trace=record['trace'], errors=typed_errors,
                        upstream_sha256=digest(upstream), claims_validated=upstream.get('claims') is not None))
                    if mode == 'full' and upstream.get('claims') is not None:
                        actual = set(ClaimComparisonValidator().claim_catalog(upstream['claims']))
                        rows[-1]['omitted_reference_claims'] = sorted(set(audit[cv]['reference_claim_catalog']) - actual)
                runs.append(dict(run=run, wall_seconds=payload['wall_seconds'], summary=summarize(rows), records=rows))
                all_rows.extend(rows)
                report['files'][path.name] = dict(sha256=__import__('hashlib').sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)
            report['groups'][f'{mode}_{architecture}'] = dict(summary=summarize(all_rows), wall_seconds=wall_seconds, runs=runs)
    # Input identity and all claim/profile bytes must match across the paired workers.
    for mode, count in [('fixed', 3), ('extended', 1)]:
        for run in range(1, count + 1):
            a, b = [json.loads((ARTIFACTS / f'{mode}_{name}_{run}.json').read_text()) for name in ('production', 'hybrid')]
            assert {key: row['input_sha256'] for key, row in a['records'].items()} == {
                key: row['input_sha256'] for key, row in b['records'].items()}
    save(ARTIFACTS / 'analysis.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    report = collect()
    for name, group in report['groups'].items():
        print(name, json.dumps(group['summary'], sort_keys=True))


if __name__ == '__main__':
    main()
