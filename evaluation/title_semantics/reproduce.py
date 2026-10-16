"""Development-only captured-response replay and bounded live title probe.

Exit 1 demonstrates the reported false-valid decision; exit 2 is inconclusive.
Expected labels and captured outcomes never enter live model inputs.
"""

import argparse
import asyncio
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from cv_verifier.claim_validator import ClaimValidator
from evaluation.run_controlled import _CALLS, _save_results, build_components, production
from hw2 import MODEL_NAME, PER_CV_TIMEOUT_SECONDS, load_env_file

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = Path(__file__).with_name('evidence.json')
SAVED = ROOT / 'evaluation/saved_public_inputs.json'


def make_case(saved, identifier, title='Director', profile_title='Manager', grade='senior'):
    claims, profile = deepcopy(saved['claims']), deepcopy(saved['profile'])
    claims['jobs'][0].update(title=title, seniority=None,
                            raw=f'{title}, Deloitte    2020 – Present')
    profile['experience'][1].update(title=profile_title, seniority=grade)
    ClaimValidator().validate(claims)
    return {'id': identifier, 'claims': claims, 'profile': profile}


def replay_model(recorded):
    pending = iter(deepcopy(recorded))

    def respond(prompt):
        stage = 'reflection' if prompt.to_messages()[0].content.startswith('Audit') else 'comparison'
        response = next(pending, None)
        if response is None or response['stage'] != stage:
            raise ValueError('Recorded model responses do not match the invoked stage')
        return AIMessage(content=json.dumps(response['output'], ensure_ascii=False))

    return RunnableLambda(respond)


async def run(args):
    evidence = json.loads(EVIDENCE.read_text())
    if hashlib.sha256(SAVED.read_bytes()).hexdigest() != evidence['saved_inputs_sha256']:
        raise ValueError('Saved inputs differ from the investigated fixture')
    for path, expected in evidence['production_sha256'].items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Production source differs from the investigated version: {path}')
    saved = json.loads(SAVED.read_text())['sources']['CV_1.pdf']
    primary = make_case(saved, 'director_manager_1')
    input_hash = hashlib.sha256(json.dumps(
        {k: primary[k] for k in ('claims', 'profile')}, sort_keys=True).encode()).hexdigest()
    if input_hash != evidence['primary_input_sha256']:
        raise ValueError('Reproduction inputs differ from the recorded model inputs')
    cases = [make_case(saved, f'director_manager_{i + 1}') for i in range(args.repeats)]
    if args.controls:
        cases.extend(make_case(saved, identifier, title, supplied, grade) for identifier, title, supplied, grade in [
            ('manager_manager', 'Manager', 'Manager', 'senior'),
            ('senior_manager_manager', 'Senior Manager', 'Manager', 'senior'),
            ('manager_director', 'Manager', 'Director', 'senior'),
            ('director_director', 'Director', 'Director', 'senior'),
            ('manager_mgr', 'Manager', 'Mgr.', 'senior'),
            ('senior_analyst_junior', 'Senior Analyst', 'Analyst', 'junior'),
            ('org_specific_associate_director', 'Associate Director', 'Senior Manager', 'senior'),
        ])
    if args.mode == 'live':
        from langchain_deepseek import ChatDeepSeek

        load_env_file()
        llm = ChatDeepSeek(model=MODEL_NAME, timeout=60, max_retries=0).bind(
            extra_body={'thinking': {'type': 'disabled'}}, response_format={'type': 'json_object'})
    else:
        llm = replay_model(evidence['replay_decisions'])
    components = build_components(llm)
    output = ROOT / 'evaluation/results' / f'title_semantics_{args.mode}_{time.time_ns()}.json'
    payload = {'mode': args.mode, 'recorded_source_commit': evidence['source_commit'],
               'records': [], 'complete': False}
    for case in cases:
        calls = []
        token = _CALLS.set(calls)
        started = time.perf_counter()
        try:
            result = await asyncio.wait_for(production(
                deepcopy(case['claims']), deepcopy(case['profile']), components), PER_CV_TIMEOUT_SECONDS)
        except Exception as error:
            result = {'score': .5, 'findings': [], 'errors': [{'stage': 'probe', 'message': str(error)}]}
        finally:
            _CALLS.reset(token)
        target = next((f for f in result['findings'] if f['claim_id'] == 'jobs.0.title'), None)
        record = {'case': case, 'result': result, 'calls': calls,
                  'seconds': time.perf_counter() - started}
        payload['records'].append(record)
        _save_results(output, payload)
        print(json.dumps({'case': case['id'], 'score': result['score'], 'title_finding': target,
                          'reflection': result.get('reflection'), 'errors': result['errors']}), flush=True)
    primary_records = [r for r in payload['records'] if r['case']['id'].startswith('director_manager_')]
    reproduced = any(r['result']['score'] == .9 and any(
        f['claim_id'] == 'jobs.0.title' and f['status'] == 'MATCH' for f in r['result']['findings'])
        for r in primary_records)
    inconclusive = any(r['result']['score'] == .5 or r['result']['errors'] for r in payload['records'])
    payload.update(complete=True, reproduced=reproduced, inconclusive=inconclusive)
    _save_results(output, payload)
    print(f'RESULTS={output}')
    if reproduced:
        print('REPRODUCED: Director accepted as MATCH against the corresponding Manager record (0.9).')
        return 1
    if inconclusive:
        print('INCONCLUSIVE: processing/contract errors or uncertainty; not a verified discrepancy.')
        return 2
    print('The reported false-valid decision was not observed in this bounded sample.')
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('replay', 'live'), default='replay')
    parser.add_argument('--repeats', type=int, choices=range(1, 6), default=1)
    parser.add_argument('--controls', action='store_true', help='Run the seven title controls once (live only)')
    args = parser.parse_args()
    if args.mode == 'replay' and (args.repeats != 1 or args.controls):
        parser.error('Recorded replay is one case; repeats and controls require --mode live')
    return asyncio.run(run(args))


if __name__ == '__main__':
    raise SystemExit(main())
