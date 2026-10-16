"""Fixed alignment mistake experiment; no alignment inference or production writes."""

import argparse
import asyncio
import json
import time
from collections import Counter
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from cv_verifier.claim_comparison_validator import ClaimComparisonValidator
from cv_verifier.cv_scorer import CVScorer
from evaluation.experimental.evidence_aligner import EvidenceAlignmentValidator
from evaluation.experimental.semantic_claim_resolver import SemanticClaimResolver
from evaluation.injection_cases import load_injections
from evaluation.run_controlled import build_components, neutral_record_reasons, grade, _CALLS, _error_kind
from hw2 import MODEL_NAME, PER_CV_TIMEOUT_SECONDS, load_env_file
from scripts.inspect_batch import _save_results


async def verify_records(claims, profile, alignments, components):
    """The existing deterministic/UNKNOWN-only semantic pipeline with partial-state retention."""
    calls, start = [], time.perf_counter()
    result = {'alignments': deepcopy(alignments), 'deterministic_findings': [], 'findings': [], 'errors': [], 'score': 0.5}
    token = _CALLS.set(calls)
    stage = 'deterministic'
    try:
        result['findings'] = components['deterministic'].verify(claims, profile, alignments)
        result['deterministic_findings'] = deepcopy(result['findings'])
        stage = 'semantic'
        merged = await asyncio.wait_for(components['semantic'].resolve(
            deepcopy(claims), deepcopy(profile), deepcopy(alignments), deepcopy(result['findings'])),
            timeout=PER_CV_TIMEOUT_SECONDS)
        result['findings'] = ClaimComparisonValidator().validate(merged, claims, profile)
    except Exception as error:
        item = {'stage': stage, 'message': str(error) or type(error).__name__}
        item['kind'] = 'timeout' if isinstance(error, TimeoutError) else _error_kind(item, calls)
        result['errors'].append(item)
    finally:
        _CALLS.reset(token)
    final = {finding['claim_id']: finding for finding in result['findings']}
    assert all(final[finding['claim_id']] == finding for finding in result['deterministic_findings']
               if finding['status'] != 'UNKNOWN'), 'Objective findings were overwritten'
    result['score'] = CVScorer().score(result['findings'], identity_resolved=True)
    return {'result': result, 'calls': calls, 'seconds': time.perf_counter() - start}


def outcome(expected, finding, errors):
    if finding is None or (finding['status'] == 'UNKNOWN' and errors):
        return 'processing_failure'
    if finding['status'] == expected:
        return 'recovered'
    if finding['status'] == 'UNKNOWN':
        return 'safe_abstention' if expected == 'MATCH' else 'missed_discrepancy'
    return 'unsafe_error'


def grade_pair(case, control, injected):
    results = [path['result'] for path in (control, injected)]
    findings = [{finding['claim_id']: finding for finding in result['findings']} for result in results]
    deterministic = {finding['claim_id']: finding for finding in results[1]['deterministic_findings']}
    targets = []
    for reference, expected in case['targets'].items():
        before, after = (entries.get(reference) for entries in findings)
        initial = deterministic.get(reference)
        status_outcome = outcome(expected, after, results[1]['errors'])
        category = status_outcome
        control_passed = outcome(expected, before, results[0]['errors']) == 'recovered'
        parts = reference.split('.')
        correct_anchor = None
        wrong_record_cited = False
        if parts[0] in ('jobs', 'education') and after is not None:
            collection, index, _ = parts
            source = 'experience' if collection == 'jobs' else 'education'
            anchor = case['original_alignment'][collection][int(index)]['profile_index']
            correct_anchor = any(citation['path'].startswith(f'{source}.{anchor}.') for citation in after['evidence'])
            wrong_record_cited = any(len(citation['path'].split('.')) == 3
                                    and citation['path'].startswith(source + '.')
                                    and int(citation['path'].split('.')[1]) != anchor
                                    for citation in after['evidence'])
            if category == 'recovered' and wrong_record_cited:
                # A coincidentally correct label with unrelated record evidence is not recovery.
                category = 'unsafe_error'
        locked_wrong = (initial is not None and initial['status'] != 'UNKNOWN' and
                        initial['status'] != expected and after == initial)
        if case['injection_type'] == 'skill_isolation':
            category = 'isolation_control'
            status_outcome = 'status_agreement' if status_outcome == 'recovered' else 'status_disagreement'
        targets.append({'claim_id': reference, 'expected': expected,
                        'control_status': before['status'] if before else None,
                        'injected_status': after['status'] if after else None,
                        'injected_deterministic_status': initial['status'] if initial else None,
                        'control_passed': control_passed, 'control_outcome': 'correct_control' if control_passed else
                        outcome(expected, before, results[0]['errors']),
                        'outcome': category, 'status_outcome': status_outcome,
                        'wrong_record_cited': wrong_record_cited, 'correct_record_cited': correct_anchor,
                        'semantic_correspondence_recovered': category == 'recovered' and correct_anchor is True
                        and initial is not None and initial['status'] == 'UNKNOWN',
                        'incorrect_immutable_finding': locked_wrong,
                        'anchored_cross_record_recovery_prohibited': case['injection_type'] == 'wrong_confident'
                        and initial is not None and initial['status'] == 'UNKNOWN'})
    all_references = set(findings[0]) | set(findings[1])
    changes = [{'claim_id': reference, 'control': findings[0].get(reference), 'injected': findings[1].get(reference)}
               for reference in sorted(all_references)
               if findings[0].get(reference, {}).get('status') != findings[1].get(reference, {}).get('status')]
    evidence_changes = [{'claim_id': reference,
                         'control': findings[0].get(reference, {}).get('evidence'),
                         'injected': findings[1].get(reference, {}).get('evidence')}
                        for reference in sorted(all_references)
                        if findings[0].get(reference, {}).get('evidence') != findings[1].get(reference, {}).get('evidence')]
    return {'targets': targets, 'claim_status_changes': changes,
            'claim_evidence_changes': evidence_changes,
            'whole_cv': {name: {'classification_correct': grade(case, result)['correct'],
                               'prediction': grade(case, result)['prediction'],
                               'verified_expected_discrepancy': any(finding['status'] == 'MISMATCH' and
                               case['expected_claims'].get(finding['claim_id']) == 'MISMATCH'
                               for finding in result['findings'])}
                         for name, result in zip(('control', 'injected'), results)}}


async def skill_replay(case, control, records, components):
    """Assert identical tasks and reuse a response; this is isolation, not recovery."""
    original = control['result']
    baseline = components['deterministic'].verify(case['claims'], case['profile'], records)
    assert baseline == original['deterministic_findings'], 'Skills changed objective inputs'
    assert records == original['alignments'], 'Skills leaked into the record adapter'
    unknown = [finding for finding in baseline if finding['status'] == 'UNKNOWN']
    semantic_calls = control['calls']
    replayed = []
    if unknown:
        context = components['semantic']._build_context(case['claims'], case['profile'], records, unknown)
        control_context = components['semantic']._build_context(case['claims'], case['profile'], original['alignments'], unknown)
        assert context == control_context, 'Skills changed semantic inputs'
        if semantic_calls:
            assert json.loads(semantic_calls[0]['input_messages'][1]['content']) == context, 'Skills changed semantic inputs'
        if not semantic_calls or semantic_calls[0]['output'] is None:
            return {'result': deepcopy(original), 'calls': [], 'seconds': 0.0,
                    'isolation': {'inputs_identical': True, 'response_replayed': False,
                                  'outcome': 'inconclusive_control_processing_failure'}}
    async def replay(prompt):
        messages = [{'role': message.type, 'content': message.content} for message in prompt.to_messages()]
        assert messages == semantic_calls[0]['input_messages'], 'Replay prompt differs'
        replayed.append(True)
        return AIMessage(content=semantic_calls[0]['output'])
    replay_components = dict(components, semantic=SemanticClaimResolver(RunnableLambda(replay)))
    injected = await verify_records(case['claims'], case['profile'], records, replay_components)
    assert injected['result'] == original, 'Diagnostic skills changed the pipeline result'
    injected['isolation'] = {'inputs_identical': True, 'response_replayed': bool(replayed),
                             'outcome': 'passed' if not original['errors'] else 'inconclusive_control_processing_failure'}
    return injected


def summarize(records):
    summary = {'injections': len(records), 'categories': dict(Counter(row['case']['injection_type'] for row in records)),
               'alignment_model_calls': 0, 'categories_are_correlated_controls': True, 'recovery': {}, 'paths': {}}
    for kind in ('false_unmatched', 'false_ambiguous', 'wrong_confident'):
        rows = [row for row in records if row['case']['injection_type'] == kind]
        targets = [target for row in rows for target in row['grading']['targets']]
        eligible = [target for target in targets if target['control_passed']]
        counts = Counter(target['outcome'] for target in eligible)
        summary['recovery'][kind] = {'injections': len(rows), 'target_claims': len(targets),
                                     'control_failed_targets': len(targets) - len(eligible),
                                     'control_correct_targets': len(eligible), 'outcomes': dict(counts),
                                     'all_target_outcomes': dict(Counter(target['outcome'] for target in targets)),
                                     'recovery_rate': counts['recovered'] / len(eligible) if eligible else None,
                                     'unsafe_error_rate': counts['unsafe_error'] / len(eligible) if eligible else None,
                                     'incorrect_definite_statuses': sum(target['status_outcome'] == 'unsafe_error'
                                                                       for target in eligible),
                                     'incorrect_immutable_findings': sum(target['incorrect_immutable_finding'] for target in targets),
                                     'semantic_correspondence_recoveries': sum(target['semantic_correspondence_recovered']
                                                                             for target in eligible),
                                     'correct_status_wrong_record_citations': sum(target['status_outcome'] == 'recovered'
                                         and target['wrong_record_cited'] for target in eligible),
                                     'cross_record_recovery_prohibited_claims': sum(target['anchored_cross_record_recovery_prohibited']
                                                                                  for target in targets)}
    isolated = [row for row in records if row['case']['injection_type'] == 'skill_isolation']
    summary['skill_isolation'] = dict(Counter(row['injected']['isolation']['outcome'] for row in isolated))
    for name in ('control', 'injected'):
        metrics = Counter()
        for row in records:
            path, result = row[name], row[name]['result']
            metrics['cases'] += 1
            metrics['target_claims'] += len(row['case']['targets'])
            metrics['target_status_agreements'] += row[name]['grading']['graded_targets'] - len(row[name]['grading']['target_failures'])
            metrics['classification_correct'] += row['grading']['whole_cv'][name]['classification_correct'] is True
            if row['case']['injection_type'] != 'skill_isolation':
                metrics['non_isolation_cases'] += 1
                metrics['non_isolation_classification_correct'] += row['grading']['whole_cv'][name]['classification_correct'] is True
            metrics['verified_expected_discrepancy_cases'] += row['grading']['whole_cv'][name]['verified_expected_discrepancy']
            metrics['processing_error_cases'] += bool(result['errors'])
            metrics['score_half_cases'] += result['score'] == 0.5
            metrics['processing_fallback_cases'] += result['score'] == 0.5 and bool(result['errors'])
            metrics['seconds'] += path['seconds']
            metrics['contract_failure_cases'] += any(error['kind'] == 'contract' for error in result['errors'])
            metrics['timeout_cases'] += any(error['kind'] == 'timeout' for error in result['errors'])
            for call in path['calls']:
                metrics['model_call_attempts'] += 1
                metrics['missing_token_usage_calls'] += call['usage'] is None
                if call['usage'] is not None:
                    for field in ('input_tokens', 'output_tokens', 'total_tokens'):
                        metrics[field] += call['usage'].get(field, 0)
        summary['paths'][name] = dict(metrics)
    return summary


async def evaluate(cases, llm, output):
    output = Path(output)
    if not output.resolve().is_relative_to(Path(__file__).with_name('results').resolve()):
        raise ValueError('Injection results must remain in ignored evaluation/results')
    if not cases or len({case['id'] for case in cases}) != len(cases):
        raise ValueError('Injection cases must be nonempty with unique IDs')
    start = time.perf_counter()
    payload = {'model': MODEL_NAME, 'thinking': 'disabled', 'json_object': True, 'max_retries': 0,
               'max_concurrency': 3, 'alignment_inference': False, 'complete': False, 'records': [],
               'case_sha256': sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest(),
               'skill_controls': 'assert equal inputs then replay control response; never counted as recovery',
               'status_boundary': 'UNMATCHED and AMBIGUOUS labels are not consumed downstream'}
    _save_results(output, payload)
    components, validator, results, semaphore = build_components(llm), EvidenceAlignmentValidator(), {}, asyncio.Semaphore(3)
    async def one(index, case):
        async with semaphore:
            original, injected = case['original_alignment'], case['injected_alignment']
            validator.validate(original, case['claims'], case['profile'])
            validator.validate(injected, case['claims'], case['profile'])
            maps = {name: neutral_record_reasons(validator.record_alignments(alignment, case['claims'], case['profile']))
                    for name, alignment in (('control', original), ('injected', injected))}
            row = {'case': deepcopy(case)}
            order = ('control', 'injected') if index % 2 == 0 or case['injection_type'] == 'skill_isolation' else ('injected', 'control')
            for name in order:
                if name == 'injected' and case['injection_type'] == 'skill_isolation':
                    row[name] = await skill_replay(case, row['control'], maps[name], components)
                else:
                    row[name] = await verify_records(deepcopy(case['claims']), deepcopy(case['profile']), maps[name], components)
                row[name]['grading'] = grade(case, row[name]['result'])
            row['grading'] = grade_pair(case, row['control'], row['injected'])
            results[index] = row
            payload['records'] = [results[i] for i in sorted(results)]
            payload['summary'] = summarize(payload['records'])
            _save_results(output, payload)
            print(f"{case['id']}: control={row['control']['result']['score']} injected={row['injected']['result']['score']}", flush=True)
    await asyncio.gather(*(one(index, case) for index, case in enumerate(cases)))
    payload.update(complete=True, seconds=time.perf_counter() - start)
    _save_results(output, payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--output', type=Path, default=Path('evaluation/results') / f'injections_{time.time_ns()}.json')
    args = parser.parse_args()
    load_env_file()
    from langchain_deepseek import ChatDeepSeek
    llm = ChatDeepSeek(model=MODEL_NAME, timeout=60, max_retries=0).bind(
        extra_body={'thinking': {'type': 'disabled'}}, response_format={'type': 'json_object'})
    payload = asyncio.run(evaluate(load_injections(args.limit), llm, args.output))
    print(json.dumps(payload['summary'], indent=2))
    print(f'RESULTS={args.output.resolve()}')


if __name__ == '__main__':
    main()
