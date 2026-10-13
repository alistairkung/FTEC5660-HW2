"""Bounded shadow evaluation of a deterministic correspondence veto, never production."""

import argparse
import asyncio
import json
import time
from collections import Counter
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

from evaluation.correspondence_gate import CorrespondencePlausibilityGate
from evaluation.gate_cases import load_gate_cases, probe_candidate_rules
from evaluation.run_controlled import aggregate, build_components, grade, neutral_record_reasons
from evaluation.run_injections import skill_replay, verify_records
from hw2 import MODEL_NAME, load_env_file
from scripts.inspect_batch import _save_results


def grounding(case, path):
    """Independent indexed correspondence plus status; metadata never enters verification."""
    findings = {item['claim_id']: item for item in path['result']['findings']}
    targets = []
    for reference, expected in case['expected_claims'].items():
        finding = findings.get(reference)
        status = finding['status'] if finding else None
        grounded, wrong = status == expected, False
        parts = reference.split('.')
        if finding and parts[0] in ('jobs', 'education') and status != 'UNKNOWN':
            source = 'experience' if parts[0] == 'jobs' else 'education'
            anchor = case['alignment_expectations'][parts[0]][int(parts[1])]
            indexed = [citation['path'].split('.') for citation in finding['evidence']
                       if len(citation['path'].split('.')) == 3 and citation['path'].startswith(source + '.')]
            wrong = type(anchor) is int and any(int(bits[1]) != anchor for bits in indexed)
            correct_anchor = type(anchor) is int and any(int(bits[1]) == anchor for bits in indexed)
            absence = (status == 'MISMATCH' and source in case['completeness']
                       and any(citation['path'] == source for citation in finding['evidence']))
            grounded = grounded and not wrong and (correct_anchor or absence)
        if finding is None or (status == 'UNKNOWN' and path['result']['errors']):
            outcome = 'processing_failure'
            grounded = False
        elif status != 'UNKNOWN' and not grounded:
            outcome = 'unsafe'
        elif status == 'UNKNOWN':
            outcome = 'expected_unknown' if expected == 'UNKNOWN' else 'missed_discrepancy' if expected == 'MISMATCH' else 'safe_abstention'
        else:
            outcome = 'correct'
        targets.append({'claim_id': reference, 'expected': expected, 'status': status,
                        'grounded_correct': grounded, 'wrong_record_cited': wrong, 'outcome': outcome})
    return {'claims': targets, 'counts': dict(Counter(item['outcome'] for item in targets)),
            'grounded_correct': sum(item['grounded_correct'] for item in targets),
            'verified_expected_discrepancy': any(item['expected'] == item['status'] == 'MISMATCH'
                                               and item['grounded_correct'] for item in targets)}


def summarize(rows):
    # Reuse existing aggregation under internal aliases, then expose actual path names.
    combined = aggregate([{'case': row['case'], 'production': row['ungated'], 'hybrid': row['gated']} for row in rows])
    paths = {'ungated': combined['production'], 'gated': combined['hybrid']}
    for name, metrics in paths.items():
        all_claims = [item for row in rows for item in row[name]['grounding']['claims']]
        metrics.update(grounded_correct_claims=sum(item['grounded_correct'] for item in all_claims),
                       grounded_claim_accuracy=sum(item['grounded_correct'] for item in all_claims) / len(all_claims),
                       unsafe_claim_findings=sum(item['outcome'] == 'unsafe' for item in all_claims),
                       missed_discrepancies=sum(item['outcome'] == 'missed_discrepancy' for item in all_claims),
                       unverified_expected_discrepancies=sum(item['expected'] == 'MISMATCH'
                                                             and not item['grounded_correct'] for item in all_claims),
                       verified_discrepancy_cases=sum(row[name]['grounding']['verified_expected_discrepancy'] for row in rows),
                       whole_cv_with_unknown_cases=sum(any(item['status'] == 'UNKNOWN' for item in row[name]['result']['findings']) for row in rows))
    metrics, outcomes, cases = Counter(), Counter(), []
    decision_outcomes = {}
    for row in rows:
        case, report = row['case'], row['gate']
        metrics['gate_error_cases'] += bool(report.get('error'))
        metrics['gate_seconds'] += report['seconds']
        metrics['unchanged_response_replays'] += 'isolation' in row['gated'] and row['gated']['isolation']['outcome'] == 'passed'
        metrics['unchanged_processing_failures'] += ('isolation' in row['gated'] and row['gated']['isolation']['outcome'] != 'passed')
        changed = []
        for decision in report.get('diagnostics', []):
            collection, index = decision['collection'], decision['cv_index']
            expected = case['alignment_expectations'][collection][index]
            original = decision['original_profile_index']
            nullified = original is not None and decision['post_profile_index'] is None
            metrics[decision['action']] += 1
            metrics[f"decision_{decision['decision'].lower()}"] += 1
            state = 'mapped' if original is not None else 'preexisting_null'
            bucket = decision_outcomes.setdefault(decision['decision'], {}).setdefault(state, {
                'records': 0, 'true_discrepancies_preserved': 0, 'unsafe_definite_judgments': 0,
                'semantic_recoveries': 0, 'missed_discrepancies': 0,
                'unverified_expected_discrepancies': 0, 'unknown_claims': 0,
                'processing_failure_claims': 0, 'case_ids': set()})
            bucket['records'] += 1
            bucket['case_ids'].add(case['id'])
            if original is not None and type(expected) is int:
                correct = original == expected
                metrics['correct_input_correspondences' if correct else 'incorrect_input_correspondences'] += 1
                metrics['correct_rejected' if correct else 'incorrect_caught'] += nullified
            if nullified:
                changed.append(f'{collection}.{index}.')
            prefix = f'{collection}.{index}.'
            gated_claims = [item for item in row['gated']['grounding']['claims']
                            if item['claim_id'].startswith(prefix)]
            before_findings = {item['claim_id']: item for item in row['gated']['result']['deterministic_findings']}
            for item in gated_claims:
                bucket['true_discrepancies_preserved'] += (item['expected'] == 'MISMATCH'
                    and item['status'] == 'MISMATCH' and item['grounded_correct'])
                bucket['unsafe_definite_judgments'] += item['outcome'] == 'unsafe'
                bucket['missed_discrepancies'] += item['expected'] == 'MISMATCH' and item['status'] == 'UNKNOWN'
                bucket['unverified_expected_discrepancies'] += (item['expected'] == 'MISMATCH'
                                                                 and not item['grounded_correct'])
                bucket['unknown_claims'] += item['status'] == 'UNKNOWN'
                bucket['processing_failure_claims'] += item['outcome'] == 'processing_failure'
                initial = before_findings.get(item['claim_id'])
                bucket['semantic_recoveries'] += (initial is not None and initial['status'] == 'UNKNOWN'
                                                  and item['grounded_correct'])
        recovered = [item for item in row['gated']['grounding']['claims']
                     if any(item['claim_id'].startswith(prefix) for prefix in changed)]
        outcomes.update(item['outcome'] for item in recovered)
        before = {finding['claim_id']: finding for finding in row['ungated']['result']['findings']}
        after = {finding['claim_id']: finding for finding in row['gated']['result']['findings']}
        cases.append({'id': case['id'], 'group': case['group'],
                      'ungated_score': row['ungated']['result']['score'], 'gated_score': row['gated']['result']['score'],
                      'ungated_verified_discrepancy': row['ungated']['grounding']['verified_expected_discrepancy'],
                      'gated_verified_discrepancy': row['gated']['grounding']['verified_expected_discrepancy'],
                      'claim_changes': [reference for reference in sorted(set(before) | set(after)) if before.get(reference) != after.get(reference)]})
    for decisions in decision_outcomes.values():
        for bucket in decisions.values():
            bucket['case_ids'] = sorted(bucket['case_ids'])
    return {'cases': len(rows), 'groups': dict(Counter(row['case']['group'] for row in rows)),
            'paths': paths, 'gate': dict(metrics), 'after_null_outcomes': dict(outcomes), 'case_outcomes': cases,
            'decision_outcomes': decision_outcomes,
            'alignment_model_calls': 0, 'gate_model_calls': 0, 'mcp_calls': 0,
            'limitation': 'Correlated saved identities and explicitly synthetic stresses; unchanged prompts use exact response replay.'}


async def evaluate(cases, llm, output, *, gate=None):
    output = Path(output)
    if not output.resolve().is_relative_to(Path(__file__).with_name('results').resolve()):
        raise ValueError('Gate results must remain in ignored evaluation/results')
    if not cases or len({case['id'] for case in cases}) != len(cases):
        raise ValueError('Gate cases require unique IDs and a nonempty batch')
    started = time.perf_counter()
    payload = {'model': MODEL_NAME, 'thinking': 'disabled', 'json_object': True, 'max_retries': 0,
               'max_concurrency': 3, 'complete': False, 'records': [],
               'case_sha256': sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest()}
    _save_results(output, payload)
    gate = gate or CorrespondencePlausibilityGate()
    components, semaphore, results = build_components(llm), asyncio.Semaphore(3), {}
    async def one(index, case):
        async with semaphore:
            row = {'case': deepcopy(case)}
            start = time.perf_counter()
            try:
                # Only the three runtime inputs cross the gate boundary.
                report = gate.apply(deepcopy(case['claims']), deepcopy(case['profile']), deepcopy(case['input_alignment']))
            except Exception as error:
                report = {'error': str(error) or type(error).__name__, 'diagnostics': []}
            report['seconds'] = time.perf_counter() - start
            row['gate'] = report
            ungated = neutral_record_reasons(case['input_alignment'])
            gated = neutral_record_reasons(report['alignments']) if 'alignments' in report else None
            identical = ungated == gated
            order = ('ungated', 'gated') if identical or index % 2 == 0 else ('gated', 'ungated')
            for name in order:
                if name == 'gated' and gated is None:
                    path = {'result': {'alignments': None, 'deterministic_findings': [], 'findings': [], 'score': 0.5,
                                       'errors': [{'stage': 'gate', 'kind': 'processing', 'message': report['error']}]},
                            'calls': [], 'seconds': report['seconds']}
                elif name == 'gated' and identical:
                    path = await skill_replay(case, row['ungated'], gated, components)
                else:
                    path = await verify_records(deepcopy(case['claims']), deepcopy(case['profile']),
                                                ungated if name == 'ungated' else gated, components)
                path['grading'] = grade(case, path['result'])
                path['grounding'] = grounding(case, path)
                row[name] = path
            results[index] = row
            payload['records'] = [results[i] for i in sorted(results)]
            payload['summary'] = summarize(payload['records'])
            _save_results(output, payload)
            print(f"{case['id']}: ungated={row['ungated']['result']['score']} gated={row['gated']['result']['score']}", flush=True)
    await asyncio.gather(*(one(index, case) for index, case in enumerate(cases)))
    payload.update(complete=True, seconds=time.perf_counter() - started)
    _save_results(output, payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--probe-only', action='store_true', help='Compare candidate rules offline, with no model initialization')
    parser.add_argument('--output', type=Path, default=Path('evaluation/results') / f'gate_{time.time_ns()}.json')
    args = parser.parse_args()
    if args.probe_only:
        if not args.output.resolve().is_relative_to(Path(__file__).with_name('results').resolve()):
            raise ValueError('Gate results must remain in ignored evaluation/results')
        payload = probe_candidate_rules()
        _save_results(args.output, payload)
        print(json.dumps(payload['summary'], indent=2))
        return
    cases = load_gate_cases()
    if args.limit is not None:
        if not 1 <= args.limit <= len(cases):
            raise ValueError('Limit must select a positive number of available cases')
        cases = cases[:args.limit]
    load_env_file()
    from langchain_deepseek import ChatDeepSeek
    llm = ChatDeepSeek(model=MODEL_NAME, timeout=60, max_retries=0).bind(
        extra_body={'thinking': {'type': 'disabled'}}, response_format={'type': 'json_object'})
    payload = asyncio.run(evaluate(cases, llm, args.output))
    print(json.dumps(payload['summary'], indent=2))
    print(f'RESULTS={args.output.resolve()}')


if __name__ == '__main__':
    main()
