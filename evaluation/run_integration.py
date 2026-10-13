"""Reference-first validation of actual production hybrid wiring on saved PR18 inputs."""

import argparse
import asyncio
import json
import time
from contextvars import ContextVar
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

from langchain_core.runnables import RunnableLambda

import hw2
from evaluation.cases import load_cases
from evaluation.run_controlled import build_components, production, grade, aggregate, _CALLS, _error_kind
from scripts.inspect_batch import _save_results

_SAVED_CASE = ContextVar('integration_saved_case')


def grounding(case, result):
    """Independent grading of source-record origin, relevance and declared absence."""
    findings = {f['claim_id']: f for f in result['findings']}
    rows = []
    for reference, expected in case['expected_claims'].items():
        finding = findings.get(reference)
        status = finding['status'] if finding else None
        correct, wrong = status == expected, False
        if finding and status != 'UNKNOWN':
            paths = {c['path'] for c in finding['evidence']}
            parts = reference.split('.')
            if parts[0] in ('jobs', 'education'):
                collection, index, field = parts
                source = 'experience' if collection == 'jobs' else 'education'
                anchor = case['alignment_expectations'][collection][int(index)]
                fields = (('title', 'seniority') if field in ('title', 'seniority') else
                          ('is_current', 'end_year') if field == 'is_current' else
                          ('end_year',) if field == 'graduation_year' else (field,))
                relevant = {p for p in paths if len(p.split('.')) == 3 and p.startswith(source + '.') and p.split('.')[2] in fields}
                wrong = type(anchor) is int and any(int(p.split('.')[1]) != anchor for p in relevant)
                supported = type(anchor) is int and bool(relevant) and not wrong and relevant == paths
                exhaustive = {f'{source}.{i}.{f}' for i, entry in enumerate(case['profile'].get(source) or []) for f in fields if entry.get(f) is not None}
                absence = (expected == status == 'MISMATCH' and source in case['completeness']
                           and (paths == {source} or bool(exhaustive) and paths == exhaustive))
                correct = correct and (supported or absence)
            elif parts[0] == 'skills':
                names = {f'skills.{i}.name' for i, entry in enumerate(case['profile'].get('skills') or []) if entry.get('name') is not None}
                relevant = bool(paths) and paths <= names | {'skills'}
                if status == 'MISMATCH':
                    relevant = (relevant and 'skills' in case['completeness'] and
                                ('skills' in paths or bool(names) and names <= paths))
                correct = correct and relevant
            else:
                correct = correct and paths == {reference}
        outcome = ('processing_failure' if finding is None else 'correct_unknown' if status == 'UNKNOWN' and correct else
                   'unknown' if status == 'UNKNOWN' else 'grounded_definite' if correct else 'unsafe_definite')
        rows.append({'claim_id': reference, 'expected': expected, 'status': status,
                     'grounded_correct': correct, 'wrong_record_cited': wrong, 'outcome': outcome})
    return {'claims': rows, 'verified_expected_discrepancy': result['score'] == 0.1 and any(
        r['expected'] == r['status'] == 'MISMATCH' and r['grounded_correct'] for r in rows)}


def summarize(records, phase):
    if phase == 'reference':
        base = aggregate([{'case': r['case'], 'production': r['reference'], 'hybrid': r['reference']} for r in records])
        names = {'reference': base['production']}
    else:
        base = aggregate([{'case': r['case'], 'production': r['reference'], 'hybrid': r['integrated']} for r in records])
        names = {'reference': base['production'], 'integrated': base['hybrid']}
    for name, metrics in names.items():
        claims = [c for r in records for c in r[name]['grounding']['claims']]
        targets = [c for r in records for c in r[name]['grounding']['claims'] if c['claim_id'] in r['case']['targets']]
        metrics.update(grounded_correct_claims=sum(c['grounded_correct'] for c in claims),
                       grounded_correct_targets=sum(c['grounded_correct'] for c in targets),
                       unsafe_definite_claims=sum(c['outcome'] == 'unsafe_definite' for c in claims),
                       verified_discrepancy_cases=sum(r[name]['grounding']['verified_expected_discrepancy'] for r in records),
                       incomplete_verification_cases=sum(r[name]['result'].get('verification_complete') is False for r in records),
                       contract_failure_cases=sum(any(e['kind'] == 'contract' for e in r[name]['result']['errors']) for r in records))
    summary = {'cases': len(records), 'paths': names,
               'limitations': 'Saved identities and extracted claims; correlated PR18 controls, not an unseen-set evaluation.'}
    if phase == 'integrated':
        summary['regressions'] = {name: base[key] for name, key in
            (('binary', 'hybrid_regressions'), ('claims', 'hybrid_claim_regressions'),
             ('targets', 'hybrid_target_regressions'))}
    return summary


async def _collect(cases, method, output, phase, reference=None):
    output = Path(output)
    if not output.resolve().is_relative_to(Path(__file__).with_name('results').resolve()):
        raise ValueError('Integration diagnostics must remain in ignored evaluation/results')
    if not cases or len({c['id'] for c in cases}) != len(cases):
        raise ValueError('Cases require nonempty unique IDs')
    payload = {'phase': phase, 'model': hw2.MODEL_NAME, 'thinking': 'disabled', 'json_object': True,
               'max_retries': 0, 'max_concurrency': 3, 'complete': False, 'records': [],
               'case_sha256': sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest()}
    start, semaphore, results = time.perf_counter(), asyncio.Semaphore(3), {}
    _save_results(output, payload)
    async def one(index, case):
        async with semaphore:
            row = deepcopy(reference[index]) if reference else {'case': deepcopy(case)}
            calls, started = [], time.perf_counter()
            call_token, case_token = _CALLS.set(calls), _SAVED_CASE.set(case)
            try:
                result = await asyncio.wait_for(method(case), timeout=hw2.PER_CV_TIMEOUT_SECONDS)
            except Exception as error:
                result = {'score': 0.5, 'findings': [], 'errors': [{'stage': 'timeout' if isinstance(error, TimeoutError) else 'evaluation', 'message': str(error) or type(error).__name__}]}
            finally:
                _CALLS.reset(call_token)
                _SAVED_CASE.reset(case_token)
            for error in result['errors']:
                if error.get('kind') not in ('contract', 'format'):
                    error['kind'] = _error_kind(error, calls)
            row[phase] = {'result': result, 'calls': calls, 'seconds': time.perf_counter() - started,
                          'grading': grade(case, result), 'grounding': grounding(case, result)}
            results[index] = row
            payload['records'] = [results[i] for i in sorted(results)]
            payload['summary'] = summarize(payload['records'], phase)
            _save_results(output, payload)
            print(f"{case['id']}: {phase}={result['score']} errors={len(result['errors'])}", flush=True)
    await asyncio.gather(*(one(i, case) for i, case in enumerate(cases)))
    payload.update(complete=True, seconds=time.perf_counter() - start)
    _save_results(output, payload)
    return payload


async def capture_reference(cases, llm, output):
    components = build_components(llm)
    async def invoke(case):
        return await production(deepcopy(case['claims']), deepcopy(case['profile']), components)
    return await _collect(cases, invoke, output, 'reference')


async def evaluate_integrated(reference, llm, output):
    if not reference.get('complete') or reference.get('phase') != 'reference':
        raise ValueError('Integrated validation requires a completed reference first')
    cases = [r['case'] for r in reference['records']]
    if sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest() != reference['case_sha256']:
        raise ValueError('Reference input hash differs')
    async def measured(prompt, **options):
        system = prompt.to_messages()[0].content
        stage = 'alignment' if system.startswith('Align CV') else 'semantic' if system.startswith('Resolve only') else None
        if stage is None:
            raise AssertionError('Saved-input validation must not call extraction, identity or comparison models')
        call = {'stage': stage, 'seconds': 0.0, 'usage': None, 'output': None,
                'input_messages': [{'role': m.type, 'content': m.content} for m in prompt.to_messages()]}
        _CALLS.get().append(call)
        started = time.perf_counter()
        try:
            response = await llm.ainvoke(prompt, **options)
            call.update(usage=response.usage_metadata, output=response.content)
            return response
        except Exception as error:
            call.update(error_type=type(error).__name__, error=str(error))
            raise
        finally:
            call['seconds'] = time.perf_counter() - started
    # Instantiate the actual production construction path; only replace the
    # out-of-scope identity dependency with the same saved identity as reference.
    with patch('langchain_deepseek.ChatDeepSeek', return_value=RunnableLambda(measured)):
        agent = hw2.build_agent([])
    def saved_identity(_):
        case = _SAVED_CASE.get()
        return {'claims': deepcopy(case['claims']), 'profiles': [deepcopy(case['profile'])],
                'identity': 'resolved', 'selected_candidate_id': case['profile']['id'],
                'identity_reason': 'Previously selected saved identity', 'errors': []}
    agent.identity_pipeline = RunnableLambda(saved_identity)
    async def invoke(_):
        return await agent.ainvoke('')
    return await _collect(cases, invoke, output, 'integrated', reference['records'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', action='store_true')
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.reference == (args.baseline is not None):
        parser.error('Choose --reference or --baseline, not both')
    hw2.load_env_file()
    from langchain_deepseek import ChatDeepSeek
    llm = ChatDeepSeek(model=hw2.MODEL_NAME, timeout=60, max_retries=0).bind(
        extra_body={'thinking': {'type': 'disabled'}}, response_format={'type': 'json_object'})
    payload = asyncio.run(capture_reference(load_cases(), llm, args.output) if args.reference else
                          evaluate_integrated(json.loads(args.baseline.read_text()), llm, args.output))
    print(json.dumps(payload['summary'], indent=2))


if __name__ == '__main__':
    main()
