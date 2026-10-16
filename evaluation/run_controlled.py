"""Development-only paired evaluation; no extraction, discovery, MCP or CSV writes."""

import argparse
import asyncio
import json
import time
from hashlib import sha256
from collections import Counter
from contextvars import ContextVar
from copy import deepcopy
from pathlib import Path

from langchain_core.runnables import RunnableLambda
from cv_verifier.claim_comparator import ClaimComparator
from cv_verifier.claim_comparison_validator import ClaimComparisonValidator
from cv_verifier.claim_reflector import ClaimReflector
from cv_verifier.cv_scorer import CVScorer
from cv_verifier.cv_verifier import CVVerifier
from evaluation.experimental.record_aligner import RecordAligner
from evaluation.experimental.evidence_aligner import EvidenceAligner
from evaluation.experimental.deterministic_claim_verifier import DeterministicClaimVerifier
from evaluation.experimental.semantic_claim_resolver import SemanticClaimResolver
from hw2 import MODEL_NAME, PER_CV_TIMEOUT_SECONDS, THRESHOLD, load_env_file
from scripts.inspect_batch import _save_results
from evaluation.cases import load_cases
from evaluation.evidence_alignment import alignment_summary

_CALLS = ContextVar('controlled_evaluation_calls', default=None)


def build_components(llm, *, broad_alignment=False):
    def measured(stage):
        async def invoke(prompt):
            call = {'stage': stage, 'seconds': 0.0, 'usage': None, 'output': None}
            call['input_messages'] = [{'role': message.type, 'content': message.content}
                                      for message in prompt.to_messages()]
            calls = _CALLS.get()
            if calls is not None:
                calls.append(call)
            start = time.perf_counter()
            try:
                response = await llm.ainvoke(prompt)
                call['usage'] = response.usage_metadata
                call['output'] = response.content
                return response
            except Exception as error:
                call['error_type'] = type(error).__name__
                call['error'] = str(error)
                raise
            finally:
                call['seconds'] = time.perf_counter() - start
        return RunnableLambda(invoke)
    aligner = EvidenceAligner(measured('alignment')) if broad_alignment else RecordAligner(measured('alignment'))
    components = {'comparator': ClaimComparator(measured('comparison')), 'reflector': ClaimReflector(measured('reflection')),
                  'aligner': aligner, 'deterministic': DeterministicClaimVerifier(),
                  'semantic': SemanticClaimResolver(measured('semantic'))}
    if broad_alignment:
        components['record_adapter'] = aligner.validator.record_alignments
    return components


async def production(claims, profile, components):
    def saved_identity(_):
        return {'claims': deepcopy(claims), 'profiles': [deepcopy(profile)], 'identity': 'resolved',
                'selected_candidate_id': profile['id'], 'identity_reason': 'Previously selected saved identity', 'errors': []}
    verifier = CVVerifier(RunnableLambda(saved_identity), components['comparator'], ClaimComparisonValidator(),
                          CVScorer(), components['reflector'])
    return await verifier.ainvoke('')


def neutral_record_reasons(alignments):
    """Shared experiment adapter: retain indices, keep generative prose diagnostic."""
    return {collection: [dict(deepcopy(entry), reason=(
        'Record correspondence is unresolved; alignment establishes neither support nor contradiction.'
        if entry['profile_index'] is None else
        'Record correspondence references the supplied profile index; factual agreement remains unverified.'))
        for entry in entries] for collection, entries in alignments.items()}


async def hybrid(claims, profile, components):
    report = {'findings': [], 'deterministic_findings': [], 'alignments': None, 'errors': [],
              'diagnostic_errors': [], 'score': 0.5}
    stage = 'alignment'
    try:
        if 'record_adapter' in components:
            alignment = await components['aligner'].align_with_diagnostics(deepcopy(claims), deepcopy(profile))
            report['evidence_alignments'] = alignment['output']
            report['alignment_validation'] = alignment['validation']
            report['diagnostic_errors'] = [{'stage': 'alignment_diagnostic', 'kind': 'contract', **error}
                                           for error in alignment['validation']['errors'] if error['collection'] == 'skills']
            report['alignments'] = components['record_adapter'](report['evidence_alignments'], claims, profile)
        else:
            report['alignments'] = await components['aligner'].align(deepcopy(claims), deepcopy(profile))
        if components.get('neutral_reasons'):
            report['original_record_alignments'] = deepcopy(report['alignments'])
            report['alignments'] = neutral_record_reasons(report['alignments'])
        stage = 'deterministic'
        findings = components['deterministic'].verify(claims, profile, report['alignments'])
        report['findings'] = findings
        report['deterministic_findings'] = deepcopy(findings)
        stage = 'semantic'
        merged = await components['semantic'].resolve(deepcopy(claims), deepcopy(profile), deepcopy(report['alignments']), deepcopy(findings))
        report['findings'] = ClaimComparisonValidator().validate(merged, claims, profile)
    except Exception as error:
        report['errors'].append({'stage': stage, 'message': str(error) or type(error).__name__})
    report['score'] = CVScorer().score(report['findings'], identity_resolved=True)
    return report


def grade(case, result):
    statuses = {finding['claim_id']: finding['status'] for finding in result.get('findings', [])}
    expected = case.get('expected_label')
    prediction = int(result['score'] > THRESHOLD)
    failures = [{'claim_id': reference, 'expected': status, 'actual': statuses.get(reference)}
                for reference, status in case['expected_claims'].items()
                if status is not None and statuses.get(reference) != status]
    target_failures = [{'claim_id': reference, 'expected': status, 'actual': statuses.get(reference)}
                       for reference, status in case['targets'].items()
                       if status is not None and statuses.get(reference) != status]
    alignment_errors, alignment_abstentions = [], []
    if result.get('alignments') is not None:
        for collection, expected_indexes in case['alignment_expectations'].items():
            for alignment in result['alignments'][collection]:
                target = expected_indexes[alignment['cv_index']]
                actual = alignment['profile_index']
                if target == 'ambiguous':
                    continue
                if actual is None and target is not None:
                    alignment_abstentions.append({'collection': collection, 'cv_index': alignment['cv_index'], 'expected': target})
                elif actual != target:
                    alignment_errors.append({'collection': collection, 'cv_index': alignment['cv_index'], 'expected': target, 'actual': actual})
    citation_errors = []
    for finding in result.get('findings', []):
        parts = finding['claim_id'].split('.')
        if parts[0] not in ('jobs', 'education'):
            continue
        target = case['alignment_expectations'][parts[0]][int(parts[1])]
        if target == 'ambiguous':
            continue
        source = 'experience' if parts[0] == 'jobs' else 'education'
        for citation in finding.get('evidence', []):
            path = citation['path'].split('.')
            if len(path) == 3 and path[0] == source and int(path[1]) != target:
                citation_errors.append({'claim_id': finding['claim_id'], 'path': citation['path'], 'expected_index': target})
    return {'prediction': prediction, 'binary_eligible': expected in (0, 1),
            'correct': prediction == expected if expected in (0, 1) else None,
            'claim_failures': failures, 'graded_claims': sum(v is not None for v in case['expected_claims'].values()),
            'target_failures': target_failures, 'graded_targets': sum(v is not None for v in case['targets'].values()),
            'alignment_errors': alignment_errors, 'alignment_abstentions': alignment_abstentions,
            'off_anchor_citations': citation_errors,
            'uncertainty_expected': expected is None,
            'uncertainty_correct': (result['score'] == 0.5 and not result['errors'] and not failures) if expected is None else None}


def aggregate(records, *, compare_aligners=False):
    summary = {'cases': len(records), 'categories': dict(Counter(r['case']['category'] for r in records)),
               'identity': 'fixed saved identities; discovery and identity resolution not evaluated'}
    names = ('record', 'evidence') if compare_aligners else ('production', 'hybrid')
    for name in names:
        metrics = Counter({key: 0 for key in ('false_valid', 'false_discrepant', 'classification_errors_with_processing_error',
                                             'llm_calls', 'input_tokens', 'output_tokens', 'total_tokens', 'missing_token_usage_calls')})
        errors, kinds, stages = Counter(), Counter(), Counter()
        for record in records:
            result, grading = record[name]['result'], record[name]['grading']
            metrics['binary_cases'] += grading['binary_eligible']
            metrics['correct'] += grading['correct'] is True
            metrics['ambiguous_cases'] += not grading['binary_eligible']
            if grading['correct'] is False:
                metrics['false_valid' if grading['prediction'] else 'false_discrepant'] += 1
                metrics['classification_errors_with_processing_error'] += bool(result['errors'])
            metrics['graded_claims'] += grading['graded_claims']
            metrics['correct_claims'] += grading['graded_claims'] - len(grading['claim_failures'])
            metrics['graded_targets'] += grading['graded_targets']
            metrics['correct_targets'] += grading['graded_targets'] - len(grading['target_failures'])
            metrics['unknown_claims'] += sum(f['status'] == 'UNKNOWN' for f in result['findings'])
            metrics['unknown_cases'] += any(f['status'] == 'UNKNOWN' for f in result['findings'])
            metrics['score_half_cases'] += result['score'] == 0.5
            metrics['fallback_cases'] += result['score'] == 0.5 and bool(result['errors'])
            metrics['processing_error_cases'] += bool(result['errors'])
            metrics['diagnostic_error_cases'] += bool(result.get('diagnostic_errors'))
            metrics['alignment_errors'] += len(grading['alignment_errors'])
            metrics['alignment_abstentions'] += len(grading['alignment_abstentions'])
            metrics['off_anchor_citations'] += len(grading['off_anchor_citations'])
            metrics['uncertainty_correct_cases'] += grading['uncertainty_correct'] is True
            metrics['reflection_corrections'] += len(result.get('reflection', {}).get('corrections', []))
            metrics['seconds'] += record[name]['seconds']
            errors.update(error['stage'] for error in result['errors'])
            kinds.update(error['kind'] for error in result['errors'])
            metrics['semantic_contract_failures'] += sum(error['stage'] == 'semantic' and error['kind'] == 'contract'
                                                        for error in result['errors'])
            for call in record[name]['calls']:
                stages[call['stage']] += 1
                metrics['llm_calls'] += 1
                usage = call['usage']
                if usage is None:
                    metrics['missing_token_usage_calls'] += 1
                else:
                    for field in ('input_tokens', 'output_tokens', 'total_tokens'):
                        metrics[field] += usage.get(field, 0)
        summary[name] = {**dict(metrics), 'accuracy': metrics['correct'] / metrics['binary_cases'] if metrics['binary_cases'] else None,
                         'claim_accuracy': metrics['correct_claims'] / metrics['graded_claims'] if metrics['graded_claims'] else None,
                         'target_accuracy': metrics['correct_targets'] / metrics['graded_targets'] if metrics['graded_targets'] else None,
                         'error_stages': dict(errors), 'error_kinds': dict(kinds), 'call_stages': dict(stages)}
    if compare_aligners:
        summary['alignment'] = alignment_summary(records)
        return summary
    summary['hybrid_fixes'] = [r['case']['id'] for r in records if r['production']['grading']['correct'] is False and r['hybrid']['grading']['correct'] is True]
    summary['hybrid_regressions'] = [r['case']['id'] for r in records if r['production']['grading']['correct'] is True and r['hybrid']['grading']['correct'] is False]
    summary['hybrid_claim_fixes'] = [r['case']['id'] for r in records if len(r['hybrid']['grading']['claim_failures']) < len(r['production']['grading']['claim_failures'])]
    summary['hybrid_claim_regressions'] = [r['case']['id'] for r in records if len(r['hybrid']['grading']['claim_failures']) > len(r['production']['grading']['claim_failures'])]
    summary['hybrid_target_fixes'] = [r['case']['id'] for r in records if len(r['hybrid']['grading']['target_failures']) < len(r['production']['grading']['target_failures'])]
    summary['hybrid_target_regressions'] = [r['case']['id'] for r in records if len(r['hybrid']['grading']['target_failures']) > len(r['production']['grading']['target_failures'])]
    return summary


def _error_kind(error, calls):
    if error['stage'] == 'timeout':
        return 'timeout'
    failed = next((call for call in reversed(calls) if call['stage'] == error['stage'] and 'error_type' in call), None)
    if failed:
        return 'timeout' if 'Timeout' in failed['error_type'] else 'model'
    return 'contract' if error['stage'] in ('comparison', 'reflection', 'alignment', 'semantic') else 'processing'


async def evaluate(cases, components, output, *, evidence_components=None):
    output = Path(output)
    if not output.resolve().is_relative_to(Path(__file__).with_name('results').resolve()):
        raise ValueError('Evaluation results must remain in the ignored evaluation/results directory')
    if len({case['id'] for case in cases}) != len(cases):
        raise ValueError('Evaluation case IDs must be unique')
    start = time.perf_counter()
    payload = {'model': MODEL_NAME, 'max_concurrency': 3, 'records': [], 'complete': False,
               'thinking': 'disabled', 'json_object': True, 'max_retries': 0,
               'case_sha256': sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest()}
    compare_aligners = evidence_components is not None
    if compare_aligners:
        payload['experiment'] = 'record_vs_evidence_alignment; skills diagnostic; shared neutral reasons; downstream unchanged'
        components = dict(components, neutral_reasons=True)
        evidence_components = dict(evidence_components, neutral_reasons=True)
    _save_results(output, payload)
    results = {}
    semaphore = asyncio.Semaphore(3)
    async def evaluate_one(index, case):
        async with semaphore:
            record = {'case': deepcopy(case)}
            # Alternate order to avoid always giving one path the earlier cache position.
            paths = ([('record', hybrid, components), ('evidence', hybrid, evidence_components)] if compare_aligners
                     else [('production', production, components), ('hybrid', hybrid, components)])
            if index % 2:
                paths.reverse()
            for name, method, path_components in paths:
                calls = []
                token = _CALLS.set(calls)
                path_start = time.perf_counter()
                try:
                    result = await asyncio.wait_for(method(deepcopy(case['claims']), deepcopy(case['profile']), path_components),
                                                    timeout=PER_CV_TIMEOUT_SECONDS)
                except Exception as error:
                    result = {'score': 0.5, 'findings': [], 'errors': [
                        {'stage': 'timeout' if isinstance(error, TimeoutError) else 'evaluation',
                         'message': str(error) or type(error).__name__}]}
                finally:
                    _CALLS.reset(token)
                for error in result['errors']:
                    error['kind'] = _error_kind(error, calls)
                record[name] = {'result': result, 'calls': calls, 'seconds': time.perf_counter() - path_start,
                                'grading': grade(case, result)}
            results[index] = record
            payload['records'] = [results[i] for i in sorted(results)]
            payload['summary'] = aggregate(payload['records'], compare_aligners=compare_aligners)
            _save_results(output, payload)
            print(f"{case['id']}: " + ' '.join(f"{name}={record[name]['result']['score']}" for name, _, _ in paths), flush=True)
    await asyncio.gather(*(evaluate_one(index, case) for index, case in enumerate(cases)))
    payload['complete'] = True
    payload['seconds'] = time.perf_counter() - start
    _save_results(output, payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--compare-aligners', action='store_true', help='Record vs broad alignment; skill mappings are diagnostic only')
    parser.add_argument('--output', type=Path, default=Path('evaluation/results') / f'paired_{time.time_ns()}.json')
    args = parser.parse_args()
    load_env_file()
    from langchain_deepseek import ChatDeepSeek
    llm = ChatDeepSeek(model=MODEL_NAME, timeout=60, max_retries=0).bind(
        extra_body={'thinking': {'type': 'disabled'}}, response_format={'type': 'json_object'})
    evidence_components = build_components(llm, broad_alignment=True) if args.compare_aligners else None
    payload = asyncio.run(evaluate(load_cases(args.limit), build_components(llm), args.output,
                                  evidence_components=evidence_components))
    print(json.dumps(payload['summary'], indent=2))
    print(f"RESULTS={args.output.resolve()}")


if __name__ == '__main__':
    main()
