"""Evaluate frozen PR25/PR24 sources without changing either submission.

Raw outputs remain in ignored evaluation/results/pr26. Run prepare, full, select,
fixed and extended in that order. Each worker is a fresh process with one frozen
source directory first on sys.path and the same Python environment.
"""

import argparse
import asyncio
import hashlib
import importlib.metadata
import io
import json
import os
import runpy
import subprocess
import sys
import tarfile
import time
from contextvars import ContextVar
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / 'evaluation/results/pr26'
REVISIONS = {'production': '8f26dd98702393bea8c650e764e16d1a7e861211',
             'hybrid': '82ccb4bac6611deddc14abb1e68d20fb46ff0815'}
TRACE = ContextVar('reliability_trace', default=None)
STAGE = ContextVar('reliability_stage', default='unknown')


def save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def source_manifest(folder):
    paths = [folder / 'hw2.py', folder / 'requirements.txt', *sorted((folder / 'cv_verifier').glob('*.py')),
             *sorted((folder / 'public_test').glob('*'))]
    return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}


def prepare():
    manifest = {'revisions': REVISIONS, 'sources': {}, 'environment': {
        'python': sys.version, 'executable': sys.executable,
        'packages': dict(sorted((d.metadata['Name'], d.version) for d in importlib.metadata.distributions()))},
        'full_order': ['production', 'hybrid', 'hybrid', 'production', 'production', 'hybrid'],
        'fairness_exception': 'Extraction SDK max_retries: production=2; hybrid=0. Preserved.'}
    for name, revision in REVISIONS.items():
        folder = ARTIFACTS / 'sources' / name
        if folder.exists():
            raise ValueError(f'Source snapshot already exists: {folder}')
        archive = subprocess.check_output(['git', 'archive', revision], cwd=ROOT)
        with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
            for member in bundle.getmembers():
                if member.isfile():
                    target = folder / member.name
                    if not target.resolve().is_relative_to(folder.resolve()):
                        raise ValueError('Unsafe archive member')
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(bundle.extractfile(member).read())
        manifest['sources'][name] = source_manifest(folder)
    upstream = ['claim_extractor', 'claim_validator', 'candidate_pipeline', 'candidate_prioritizer',
                'candidate_ranking_validator', 'profile_retriever', 'identity_resolver', 'identity_decision_validator']
    manifest['identical_upstream_modules'] = {m: manifest['sources']['production'][f'cv_verifier/{m}.py'] ==
        manifest['sources']['hybrid'][f'cv_verifier/{m}.py'] for m in upstream}
    assert all(manifest['identical_upstream_modules'].values())
    for path, fingerprint in manifest['sources']['production'].items():
        if path.startswith('public_test/') or path == 'requirements.txt':
            assert manifest['sources']['hybrid'][path] == fingerprint
    save(ARTIFACTS / 'manifest.json', manifest)


def instrument(agent):
    identity = agent.identity_pipeline
    methods = [(identity.extractor, 'extract', 'extraction'),
               (identity.prioritizer, 'prioritize', 'prioritization'),
               (identity.retriever, 'search', 'search'),
               (identity.retriever, 'retrieve_profiles', 'retrieval'),
               (identity.identity_resolver, 'resolve', 'identity')]
    if hasattr(agent, 'comparator'):
        methods += [(agent.comparator, 'compare', 'comparison'), (agent.reflector, 'reflect', 'reflection')]
    else:
        methods += [(agent.aligner, 'align', 'alignment'), (agent.semantic_resolver, 'resolve', 'semantic')]
    for component, name, stage in methods:
        original = getattr(component, name)
        async def measured(*args, _original=original, _stage=stage, **kwargs):
            trace, start = TRACE.get(), time.perf_counter()
            token = STAGE.set(_stage)
            try:
                value = await _original(*args, **kwargs)
                if trace is not None and _stage == 'extraction':
                    trace['extracted_before_validation'] = deepcopy(value)
                return value
            except Exception as error:
                if trace is not None:
                    trace.setdefault('stage_errors', []).append(dict(stage=_stage, type=type(error).__name__, message=str(error)))
                raise
            finally:
                if trace is not None:
                    trace.setdefault('stage_timings', []).append(dict(stage=_stage, seconds=time.perf_counter() - start))
                STAGE.reset(token)
        setattr(component, name, measured)
    original_identity = identity.ainvoke
    async def capture_identity(text):
        value = await original_identity(text)
        trace = TRACE.get()
        if trace is not None:
            trace['upstream'] = deepcopy(value)
            trace['claims_validated'] = value.get('claims') is not None
        return value
    identity.ainvoke = capture_identity


def make_handler():
    from langchain_core.callbacks import AsyncCallbackHandler
    class Handler(AsyncCallbackHandler):
        run_inline = True
        def __init__(self):
            self.pending = {}
        async def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
            trace = TRACE.get()
            if trace is None:
                return
            call = dict(stage=STAGE.get(), started=time.perf_counter(), usage=None,
                        input_messages=[{'role': m.type, 'content': m.content} for m in messages[0]])
            trace.setdefault('calls', []).append(call)
            self.pending[run_id] = call
        async def on_llm_end(self, response, *, run_id, **kwargs):
            call = self.pending.pop(run_id, None)
            if call is not None:
                message = response.generations[0][0].message
                call.update(seconds=time.perf_counter() - call.pop('started'), output=message.content,
                            usage=message.usage_metadata)
        async def on_llm_error(self, error, *, run_id, **kwargs):
            call = self.pending.pop(run_id, None)
            if call is not None:
                call.update(seconds=time.perf_counter() - call.pop('started'),
                            error_type=type(error).__name__, error=str(error))
    return Handler()


def selected_inputs():
    """First chronological capture of each observed CV/profile pair; no labels/scores."""
    manifest = json.loads((ARTIFACTS / 'manifest.json').read_text())
    selected = {}
    counters = {'production': 0, 'hybrid': 0}
    for architecture in manifest['full_order']:
        counters[architecture] += 1
        run = counters[architecture]
        payload = json.loads((ARTIFACTS / f'full_{architecture}_{run}.json').read_text())
        for cv, record in payload['records'].items():
            upstream = record.get('trace', {}).get('upstream', {})
            profile_id = upstream.get('selected_candidate_id') if upstream.get('identity') == 'resolved' else None
            key = (cv, profile_id)
            if key not in selected:
                selected[key] = dict(id=f'{Path(cv).stem}_{profile_id or "unresolved"}', cv=cv,
                    source=dict(architecture=architecture, run=run), upstream=deepcopy(upstream),
                    input_sha256=digest(upstream))
    payload = dict(selection='First chronological observation per (CV, resolved ID or unresolved); no labels/scores used',
                   inputs=list(selected.values()))
    save(ARTIFACTS / 'fixed_inputs.json', payload)
    return payload


def worker(architecture, mode, run):
    source = ARTIFACTS / 'sources' / architecture
    expected = json.loads((ARTIFACTS / 'manifest.json').read_text())['sources'][architecture]
    assert source_manifest(source) == expected
    os.chdir(source)
    sys.path.insert(0, str(source))
    import hw2
    from langchain_deepseek import ChatDeepSeek
    handler = make_handler()
    constructors = []
    def construct(*args, **kwargs):
        constructors.append(deepcopy(kwargs))
        model = ChatDeepSeek(*args, **kwargs)
        model.callbacks = [handler]
        return model
    output = ARTIFACTS / f'{mode}_{architecture}_{run}.json'
    if output.exists():
        raise ValueError(f'Refusing to overwrite a completed/failed run: {output}')
    started = time.perf_counter()
    payload = dict(architecture=architecture, revision=REVISIONS[architecture], mode=mode,
                   run=run, complete=False, records={}, constructors=constructors)
    save(output, payload)
    try:
        with patch('langchain_deepseek.ChatDeepSeek', side_effect=construct):
            if mode == 'full':
                # Reuse exactly the PR25 diagnostics against both frozen hw2 modules.
                public = runpy.run_path(str(ROOT / 'scripts/evaluate_public.py'))
                original_build, original_text = hw2.build_agent, hw2.cv_text
                names, traces = {}, {}
                def load(path):
                    text = original_text(path)
                    names[text] = path.name
                    return text
                def build(tools):
                    agent = original_build(tools)
                    instrument(agent)
                    original = agent.ainvoke
                    async def invoke(text):
                        trace = traces.setdefault(names[text], {})
                        token = TRACE.set(trace)
                        try:
                            return await original(text)
                        finally:
                            TRACE.reset(token)
                    agent.ainvoke = invoke
                    return agent
                destination = source / 'evaluation/results/observed_public.json'
                with patch.object(hw2, 'cv_text', load), patch.object(hw2, 'build_agent', build):
                    result = public['evaluate'](destination)
                for cv, record in result['records'].items():
                    record['trace'] = traces[cv]
                payload.update(result)
                (ARTIFACTS / f'full_{architecture}_{run}.csv').write_bytes((source / 'results.csv').read_bytes())
            else:
                from langchain_core.runnables import RunnableLambda
                from cv_verifier.claim_validator import ClaimValidator
                from cv_verifier.identity_decision_validator import IdentityDecisionValidator
                from evaluation.cases import load_cases
                cases = (json.loads((ARTIFACTS / 'fixed_inputs.json').read_text())['inputs'] if mode == 'fixed' else
                         [dict(id=c['id'], cv=c['source'], case=c, upstream=dict(claims=c['claims'],
                          profiles=[c['profile']], identity='resolved', selected_candidate_id=c['profile']['id'],
                          identity_reason='Previously selected saved identity', errors=[])) for c in load_cases()])
                agent = hw2.build_agent([])
                instrument(agent)
                context = ContextVar('fixed_upstream')
                agent.identity_pipeline = RunnableLambda(lambda _: deepcopy(context.get()))
                semaphore = asyncio.Semaphore(3)
                async def one(item):
                    async with semaphore:
                        upstream = deepcopy(item['upstream'])
                        fingerprint = digest(upstream)
                        record = dict(cv=item['cv'], input_sha256=fingerprint, trace={}, skipped=False)
                        payload['records'][item['id']] = record
                        profiles = [p for p in upstream.get('profiles', []) if p['id'] == upstream.get('selected_candidate_id')]
                        if upstream.get('identity') != 'resolved' or len(profiles) != 1 or upstream.get('claims') is None:
                            record.update(skipped=True, reason='No validated resolved upstream; no profile fabricated')
                            save(output, payload)
                            return
                        ClaimValidator().validate(upstream['claims'])
                        IdentityDecisionValidator().validate(dict(candidate_id=upstream['selected_candidate_id'],
                            reason=upstream['identity_reason']), upstream['profiles'])
                        trace_token, input_token = TRACE.set(record['trace']), context.set(upstream)
                        start = time.perf_counter()
                        try:
                            record['result'] = await asyncio.wait_for(agent.ainvoke(''), timeout=hw2.PER_CV_TIMEOUT_SECONDS)
                        except Exception as error:
                            record['result'] = dict(score=0.5, findings=[], errors=[dict(stage='timeout' if
                                isinstance(error, TimeoutError) else 'evaluation', message=str(error) or type(error).__name__)])
                        finally:
                            record['seconds'] = time.perf_counter() - start
                            TRACE.reset(trace_token)
                            context.reset(input_token)
                        assert digest(upstream) == fingerprint, 'Verifier mutated frozen inputs'
                        save(output, payload)
                        print(f"{item['id']}: {architecture}={record['result']['score']}", flush=True)
                async def collect():
                    await asyncio.gather(*(one(item) for item in cases))
                asyncio.run(collect())
        payload['complete'] = True
    except Exception as error:
        payload['fatal_error'] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        payload['wall_seconds'] = time.perf_counter() - started
        payload['source_unchanged'] = source_manifest(source) == expected
        save(output, payload)
        assert payload['source_unchanged']


def dispatch(mode):
    import hw2
    hw2.load_env_file(ROOT / '.env')
    failures = []
    order = [('production', 1), ('hybrid', 1), ('hybrid', 2), ('production', 2), ('production', 3), ('hybrid', 3)]
    if mode == 'extended':
        order = [('production', 1), ('hybrid', 1)]
    for architecture, run in order:
        output = ARTIFACTS / f'{mode}_{architecture}_{run}.json'
        if output.exists():
            raise ValueError(f'Existing run would be overwritten/selected away: {output}')
        with (ARTIFACTS / f'{mode}_{architecture}_{run}.log').open('w') as log:
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), 'worker', '--architecture',
                architecture, '--mode', mode, '--run', str(run)], cwd=ROOT, env=os.environ.copy(),
                stdout=log, stderr=subprocess.STDOUT, timeout=1800)
        if result.returncode:
            failures.append(output.name)
        print(f'{mode} {architecture} run {run}: exit={result.returncode}', flush=True)
    if failures:
        raise RuntimeError(f'Failed runs retained: {failures}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'full', 'select', 'fixed', 'extended', 'worker'])
    parser.add_argument('--architecture', choices=list(REVISIONS))
    parser.add_argument('--mode', choices=['full', 'fixed', 'extended'])
    parser.add_argument('--run', type=int)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    elif args.action == 'select':
        selected_inputs()
    elif args.action == 'worker':
        worker(args.architecture, args.mode, args.run)
    else:
        dispatch(args.action)


if __name__ == '__main__':
    main()
