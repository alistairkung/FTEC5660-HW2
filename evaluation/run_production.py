"""Production-only 25-case evaluation and strict captured-response replay.

Run from the repository root with ``python -m evaluation.run_production``.
Expected answers are used only by grading, after verification has completed.
"""

import argparse
import asyncio
import gzip
import json
import runpy
import subprocess
import time
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from evaluation.cases import load_cases
from evaluation.run_controlled import _CALLS, _error_kind, aggregate, build_components, grade, production
from evaluation.run_gate import grounding
from hw2 import MODEL_NAME, PER_CV_TIMEOUT_SECONDS, load_env_file
from scripts.inspect_batch import _save_results


def summarize(records):
    rows = [dict(case=row['case'], production=row['production'], hybrid=row['production']) for row in records]
    metrics = aggregate(rows)['production']
    grounded = [item for row in records for item in row['production']['grounding']['claims']]
    metrics.update(grounded_correct_claims=sum(item['grounded_correct'] for item in grounded),
                   grounded_claims=len(grounded),
                   grounded_claim_accuracy=sum(item['grounded_correct'] for item in grounded) / len(grounded),
                   verified_discrepancy_cases=sum(row['production']['grounding']['verified_expected_discrepancy']
                                                  for row in records),
                   mismatch_cases=sum(any(f['status'] == 'MISMATCH' for f in row['production']['result']['findings'])
                                      for row in records))
    return metrics


async def evaluate(output, replay=None, *, aggregated=False):
    cases = load_cases()
    reference = None
    if replay:
        data = gzip.decompress(replay.read_bytes()) if replay.suffix == '.gz' else replay.read_bytes()
        reference = json.loads(data)
    if aggregated and not reference:
        raise ValueError('Aggregate verification uses captured replay only')
    namespace = runpy.run_path('submission/production_source.py') if aggregated else None
    case_hash = sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest()
    if reference and (not reference['complete'] or reference['case_sha256'] != case_hash):
        raise ValueError('Replay requires a complete run with identical cases')
    if reference and [r['case']['id'] for r in reference['records']] != [c['id'] for c in cases]:
        raise ValueError('Replay case IDs/order differ')
    if not output.resolve().is_relative_to(Path(__file__).with_name('results').resolve()):
        raise ValueError('Raw evaluation output belongs in ignored evaluation/results/')
    if replay and output.resolve() == replay.resolve():
        raise ValueError('Replay must preserve its reference file')
    llm = None
    if not reference:
        load_env_file()
        from langchain_deepseek import ChatDeepSeek
        llm = ChatDeepSeek(model=MODEL_NAME, timeout=60, max_retries=0).bind(
            extra_body={'thinking': {'type': 'disabled'}}, response_format={'type': 'json_object'})
    started = time.perf_counter()
    payload = dict(model=MODEL_NAME, commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                   case_sha256=case_hash, max_concurrency=3, per_cv_timeout_seconds=PER_CV_TIMEOUT_SECONDS,
                   replay_of=str(replay) if replay else None, complete=False, records=[])
    payload['implementation'] = 'aggregate' if aggregated else 'modular'
    semaphore = asyncio.Semaphore(3)
    completed = {}

    async def one(index, case):
        async with semaphore:
            calls = []
            token = _CALLS.set(calls)
            position = 0
            original = reference['records'][index]['production'] if reference else None
            def replay_response(prompt):
                nonlocal position
                expected = original['calls'][position]
                position += 1
                messages = [{'role': message.type, 'content': message.content} for message in prompt.to_messages()]
                if messages != expected['input_messages']:
                    raise AssertionError(f"Replay prompt differs: {case['id']} call {position}")
                if 'error_type' in expected:
                    raise type(expected['error_type'], (Exception,), {})(expected['error'])
                return AIMessage(content=expected['output'], usage_metadata=expected['usage'])
            components = build_components(RunnableLambda(replay_response) if reference else llm)
            async def verify():
                if namespace is None:
                    return await production(deepcopy(case['claims']), deepcopy(case['profile']), components)
                # Exercise aggregate classes with precisely the same model boundary.
                model = components['comparator']._chain.steps[1]
                identity = RunnableLambda(lambda _: dict(claims=deepcopy(case['claims']),
                    profiles=[deepcopy(case['profile'])], identity='resolved',
                    selected_candidate_id=case['profile']['id'],
                    identity_reason='Previously selected saved identity', errors=[]))
                verifier = namespace['CVVerifier'](identity, namespace['ClaimComparator'](model),
                    namespace['ClaimComparisonValidator'](), namespace['CVScorer'](),
                    namespace['ClaimReflector'](components['reflector']._chain.steps[1]))
                return await verifier.ainvoke('')
            start = time.perf_counter()
            try:
                result = await asyncio.wait_for(verify(), timeout=PER_CV_TIMEOUT_SECONDS)
            except Exception as error:
                result = dict(score=0.5, findings=[], errors=[dict(
                    stage='timeout' if isinstance(error, TimeoutError) else 'evaluation',
                    message=str(error) or type(error).__name__)])
            finally:
                _CALLS.reset(token)
            for error in result['errors']:
                error['kind'] = _error_kind(error, calls)
            report = dict(result=result, calls=calls, seconds=time.perf_counter() - start, grading=grade(case, result))
            report['grounding'] = grounding(case, report)
            if reference:
                report['replay_equal'] = result == original['result'] and position == len(original['calls'])
                # A prompt mismatch is fatal even if both implementations fell back to 0.5.
                report['replay_equal'] &= all('Replay prompt differs' not in c.get('error', '') for c in calls)
            completed[index] = dict(case=case, production=report)
            payload['records'] = [completed[i] for i in sorted(completed)]
            payload['summary'] = summarize(payload['records'])
            _save_results(output, payload)
            print(f"{case['id']}: score={result['score']} errors={len(result['errors'])}", flush=True)

    _save_results(output, payload)
    await asyncio.gather(*(one(i, case) for i, case in enumerate(cases)))
    payload.update(complete=True, seconds=time.perf_counter() - started)
    if reference:
        payload['replay_equal'] = all(row['production']['replay_equal'] for row in payload['records'])
    _save_results(output, payload)
    print(json.dumps(dict(summary=payload['summary'], seconds=payload['seconds'],
                         replay_equal=payload.get('replay_equal')), indent=2))
    if reference and not payload['replay_equal']:
        raise AssertionError('Captured-response replay differs')
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replay', type=Path)
    parser.add_argument('--aggregate', action='store_true', help='Replay against the generated aggregate')
    args = parser.parse_args()
    asyncio.run(evaluate(args.output, args.replay, aggregated=args.aggregate))


if __name__ == '__main__':
    main()
