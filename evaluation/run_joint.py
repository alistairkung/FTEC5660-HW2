"""Bounded production-vs-joint experiment on saved inputs; no discovery or MCP calls."""

import argparse
import asyncio
import json
import time
from collections import Counter
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

from langchain_core.runnables import RunnableLambda

from cv_verifier.cv_scorer import CVScorer
from evaluation.cases import load_cases
from evaluation.joint_comparison import JointRecordComparator
from evaluation.run_controlled import build_components, production, grade, aggregate, _CALLS, _error_kind
from hw2 import MODEL_NAME, PER_CV_TIMEOUT_SECONDS, load_env_file
from scripts.inspect_batch import _save_results


def grounding(case, result):
    """Grading-only status, field relevance and independent record-origin checks."""
    findings = {f['claim_id']: f for f in result['findings']}
    rows = []
    for reference, expected in case['expected_claims'].items():
        finding = findings.get(reference)
        status = finding['status'] if finding else None
        correct = status == expected
        wrong_record = False
        basis = 'status'
        if finding and status != 'UNKNOWN':
            paths = {c['path'] for c in finding['evidence']}
            parts = reference.split('.')
            if parts[0] in ('jobs', 'education'):
                collection, index, field = parts
                source = 'experience' if collection == 'jobs' else 'education'
                anchor = case['alignment_expectations'][collection][int(index)]
                fields = (('title', 'seniority') if field in ('title', 'seniority') else
                          ('end_year',) if field == 'graduation_year' else
                          ('start_year', 'end_year', 'is_current') if field == 'dates' else (field,))
                relevant = {p for p in paths if len(p.split('.')) == 3 and p.startswith(source + '.')
                            and p.split('.')[2] in fields}
                wrong_record = type(anchor) is int and any(int(p.split('.')[1]) != anchor for p in relevant)
                supported_record = type(anchor) is int and bool(relevant) and not wrong_record and relevant == paths
                entries = case['profile'].get(source)
                exhaustive = {f'{source}.{i}.{f}' for i, entry in enumerate(entries or []) for f in fields
                              if entry.get(f) is not None}
                absence = (expected == status == 'MISMATCH' and source in case['completeness']
                           and (paths == {source} or bool(exhaustive) and paths == exhaustive))
                correct = correct and (supported_record or absence)
                basis = 'declared_complete_collection_absence' if absence else 'corresponding_record'
            elif parts[0] == 'skills':
                relevant = paths == {'skills'} or all(p.startswith('skills.') and p.endswith('.name') for p in paths)
                names = {f'skills.{i}.name' for i, entry in enumerate(case['profile'].get('skills') or [])
                         if entry.get('name') is not None}
                if status == 'MISMATCH':
                    relevant = 'skills' in case['completeness'] and (paths == {'skills'} or bool(names) and paths == names)
                correct = correct and relevant
                basis = 'skill_evidence'
            else:
                correct = correct and paths == {reference}
        if not finding:
            outcome = 'processing_failure'
        elif status == 'UNKNOWN':
            outcome = 'correct_abstention' if correct else 'missed_discrepancy' if expected == 'MISMATCH' else 'abstention'
        else:
            outcome = 'grounded_definite' if correct else 'unsafe_definite'
        rows.append({'claim_id': reference, 'expected': expected, 'status': status,
                     'grounded_correct': correct, 'wrong_record_cited': wrong_record,
                     'basis': basis, 'outcome': outcome})
    return {'claims': rows, 'counts': dict(Counter(r['outcome'] for r in rows)),
            'verified_expected_discrepancy': any(r['expected'] == r['status'] == 'MISMATCH'
                                               and r['grounded_correct'] for r in rows)}


def correspondence(case, path):
    """Inspect raw proposals even when a later finding fails; never feed labels back."""
    calls = [c for c in path['calls'] if c['stage'] == 'joint']
    try:
        records = json.loads(calls[0]['output'])['records']
    except (IndexError, KeyError, TypeError, ValueError):
        return {'rows': [], 'unreadable': True}
    rows = []
    for collection, indexes in case['alignment_expectations'].items():
        entries = records.get(collection) if isinstance(records, dict) else None
        if not isinstance(entries, list):
            return {'rows': rows, 'unreadable': True}
        for index, expected in enumerate(indexes):
            found = [e for e in entries if isinstance(e, dict) and type(e.get('cv_index')) is int and e['cv_index'] == index]
            if len(found) != 1:
                rows.append({'collection': collection, 'cv_index': index, 'outcome': 'malformed'})
                continue
            entry, source = found[0], 'experience' if collection == 'jobs' else 'education'
            assigned, state = entry.get('profile_index'), entry.get('state')
            available = case['profile'].get(source)
            valid = ((state == 'ALIGNED' and type(assigned) is int and isinstance(available, list)
                      and 0 <= assigned < len(available)) or
                     (state in ('ABSENT', 'UNRESOLVED') and assigned is None))
            outcome = ('malformed' if not valid else
                       'correct_ambiguity' if expected == 'ambiguous' and state == 'UNRESOLVED' else
                       'unsupported_confidence' if expected == 'ambiguous' else
                       'correct' if assigned == expected and (state == 'ALIGNED' if type(expected) is int else state == 'ABSENT') else
                       'unresolved' if state == 'UNRESOLVED' else 'incorrect')
            rows.append({'collection': collection, 'cv_index': index, 'expected': expected,
                         'profile_index': assigned, 'state': state, 'outcome': outcome,
                         'finding_contract_accepted': not path['result']['errors']})
    return {'rows': rows, 'unreadable': False}


def summarize(records):
    base = aggregate([{'case': r['case'], 'production': r['production'], 'hybrid': r['joint']} for r in records])
    paths = {'production': base['production'], 'joint': base['hybrid']}
    for name, metrics in paths.items():
        claims = [c for r in records for c in r[name]['grounding']['claims']]
        targets = [c for r in records for c in r[name]['grounding']['claims'] if c['claim_id'] in r['case']['targets']]
        metrics.update(grounded_correct_claims=sum(c['grounded_correct'] for c in claims),
                       grounded_claim_accuracy=sum(c['grounded_correct'] for c in claims) / len(claims),
                       grounded_correct_targets=sum(c['grounded_correct'] for c in targets),
                       grounded_target_accuracy=sum(c['grounded_correct'] for c in targets) / len(targets),
                       grounded_definite_claims=sum(c['outcome'] == 'grounded_definite' for c in claims),
                       correct_abstentions=sum(c['outcome'] == 'correct_abstention' for c in claims),
                       unsafe_definite_claims=sum(c['outcome'] == 'unsafe_definite' for c in claims),
                       verified_discrepancy_cases=sum(r[name]['grounding']['verified_expected_discrepancy'] for r in records),
                       unverified_expected_discrepancies=sum(c['expected'] == 'MISMATCH' and not c['grounded_correct'] for c in claims),
                       contract_failure_cases=sum(any(e['kind'] == 'contract' for e in r[name]['result']['errors']) for r in records))
    changes = {'target_fixes': [], 'target_regressions': [], 'binary_fixes': [], 'binary_regressions': []}
    for row in records:
        before = {c['claim_id']: c for c in row['production']['grounding']['claims']}
        after = {c['claim_id']: c for c in row['joint']['grounding']['claims']}
        for reference in row['case']['targets']:
            a, b = before[reference]['grounded_correct'], after[reference]['grounded_correct']
            if a != b:
                changes['target_fixes' if b else 'target_regressions'].append({'id': row['case']['id'], 'claim_id': reference})
        a, b = row['production']['grading']['correct'], row['joint']['grading']['correct']
        if a is not None and a != b:
            changes['binary_fixes' if b else 'binary_regressions'].append(row['case']['id'])
    proposals = [e for row in records for e in row['joint']['correspondence']['rows']]
    return {'cases': len(records), 'paths': paths, 'changes': changes,
            'joint_correspondence': {'records': len(proposals), 'outcomes': dict(Counter(e['outcome'] for e in proposals)),
                                    'unreadable_cases': sum(r['joint']['correspondence']['unreadable'] for r in records),
                                    'accepted_outcomes': dict(Counter(e['outcome'] for e in proposals if e.get('finding_contract_accepted')))},
            'mcp_calls': 0, 'extraction_calls': 0, 'identity_calls': 0, 'alignment_calls': 0,
            'limitation': '25 correlated saved controls from three identities; correspondence and completeness labels are grading-only.'}


async def evaluate(cases, llm, output):
    output = Path(output)
    if not output.resolve().is_relative_to(Path(__file__).with_name('results').resolve()):
        raise ValueError('Joint results must remain in ignored evaluation/results')
    if not cases or len({c['id'] for c in cases}) != len(cases):
        raise ValueError('Cases must be nonempty with unique IDs')
    components = build_components(llm)
    async def measured(prompt):
        call = {'stage': 'joint', 'seconds': 0.0, 'usage': None, 'output': None,
                'input_messages': [{'role': m.type, 'content': m.content} for m in prompt.to_messages()]}
        _CALLS.get().append(call)
        start = time.perf_counter()
        try:
            response = await llm.ainvoke(prompt)
            call.update(usage=response.usage_metadata, output=response.content)
            return response
        except Exception as error:
            call.update(error_type=type(error).__name__, error=str(error))
            raise
        finally:
            call['seconds'] = time.perf_counter() - start
    comparator = JointRecordComparator(RunnableLambda(measured))
    async def joint(claims, profile):
        accepted = await comparator.compare(claims, profile)
        return {**accepted, 'alignments': accepted['records'], 'errors': [],
                'score': CVScorer().score(accepted['findings'], identity_resolved=True)}
    started = time.perf_counter()
    payload = {'model': MODEL_NAME, 'thinking': 'disabled', 'json_object': True, 'max_retries': 0,
               'max_concurrency': 3, 'complete': False, 'records': [],
               'case_sha256': sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest()}
    _save_results(output, payload)
    semaphore, results = asyncio.Semaphore(3), {}
    async def one(index, case):
        async with semaphore:
            row = {'case': deepcopy(case)}
            order = ('production', 'joint') if index % 2 == 0 else ('joint', 'production')
            for name in order:
                calls, start = [], time.perf_counter()
                token = _CALLS.set(calls)
                try:
                    method = production(deepcopy(case['claims']), deepcopy(case['profile']), components) if name == 'production' else joint(
                        deepcopy(case['claims']), deepcopy(case['profile']))
                    result = await asyncio.wait_for(method, timeout=PER_CV_TIMEOUT_SECONDS)
                except Exception as error:
                    stage = 'timeout' if isinstance(error, TimeoutError) else 'joint' if name == 'joint' else 'evaluation'
                    result = {'score': 0.5, 'findings': [], 'errors': [{'stage': stage, 'message': str(error) or type(error).__name__}]}
                finally:
                    _CALLS.reset(token)
                for error in result['errors']:
                    failed_call = next((c for c in reversed(calls) if c['stage'] == error['stage'] and 'error_type' in c), None)
                    error['kind'] = ('model' if failed_call else 'contract') if error['stage'] == 'joint' else _error_kind(error, calls)
                row[name] = {'result': result, 'calls': calls, 'seconds': time.perf_counter() - start,
                             'grading': grade(case, result), 'grounding': grounding(case, result)}
                if name == 'joint':
                    row[name]['correspondence'] = correspondence(case, row[name])
            results[index] = row
            payload['records'] = [results[i] for i in sorted(results)]
            payload['summary'] = summarize(payload['records'])
            _save_results(output, payload)
            print(f"{case['id']}: production={row['production']['result']['score']} joint={row['joint']['result']['score']}", flush=True)
    await asyncio.gather(*(one(i, case) for i, case in enumerate(cases)))
    payload.update(complete=True, seconds=time.perf_counter() - started)
    _save_results(output, payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--output', type=Path, default=Path('evaluation/results') / f'joint_{time.time_ns()}.json')
    args = parser.parse_args()
    cases = load_cases(args.limit)
    load_env_file()
    from langchain_deepseek import ChatDeepSeek
    llm = ChatDeepSeek(model=MODEL_NAME, timeout=60, max_retries=0).bind(
        extra_body={'thinking': {'type': 'disabled'}}, response_format={'type': 'json_object'})
    payload = asyncio.run(evaluate(cases, llm, args.output))
    print(json.dumps(payload['summary'], indent=2))
    print(f'RESULTS={args.output.resolve()}')


if __name__ == '__main__':
    main()
