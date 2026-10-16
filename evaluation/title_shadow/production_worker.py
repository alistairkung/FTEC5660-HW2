"""Run actual source-checkout components, live or paired-response replay."""
import argparse
import asyncio
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import time

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from evaluation.run_controlled import _CALLS, _save_results, build_components, production
from cv_verifier.claim_validator import ClaimValidator
from hw2 import MODEL_NAME


async def run(args):
    rows=json.loads(args.inputs.read_text())
    replay={r['sample_id']:r for r in json.loads(args.replay.read_text())['records']} if args.replay else None
    if replay is None:
        from langchain_deepseek import ChatDeepSeek
        live=ChatDeepSeek(model=MODEL_NAME,timeout=60,max_retries=0).bind(
            extra_body={'thinking':{'type':'disabled'}},response_format={'type':'json_object'})
    payload={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
             'production_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest()
                                  for f in [Path('hw2.py'),*sorted(Path('cv_verifier').glob('*.py'))]},
             'records':[],'complete':False,'replay':replay is not None}
    results={};semaphore=asyncio.Semaphore(3);started=time.perf_counter()
    async def one(index,row):
        async with semaphore:
            calls=[];consumed=[];pending=list(replay[row['sample_id']]['calls']) if replay else []
            def recorded(prompt):
                messages=[{'role':m.type,'content':m.content} for m in prompt.to_messages()]
                if not pending or pending[0]['input_messages']!=messages:
                    raise ValueError('Paired replay prompt mismatch; not comparable')
                original=pending.pop(0);consumed.append(original)
                if original.get('error_type'):
                    raise RuntimeError('Recorded provider failure is not replayable: '+original['error_type'])
                return AIMessage(content=original['output'])
            before=time.perf_counter();token=_CALLS.set(calls)
            record={'sample_id':row['sample_id'],'case_id':row['case_id']}
            try:
                ClaimValidator().validate(row['claims'])
                model=RunnableLambda(recorded) if replay else live
                result=await asyncio.wait_for(production(deepcopy(row['claims']),deepcopy(row['profile']),
                                                         build_components(model)),180)
                if replay and any(c['stage']!='reflection' for c in pending):
                    raise ValueError('Unconsumed comparison response; not comparable')
                record['result']=result
            except Exception as error:
                record['worker_error']={'type':type(error).__name__,'message':str(error)}
            finally:
                _CALLS.reset(token)
            record.update(calls=calls,seconds=time.perf_counter()-before,
                          replayed_token_equivalents=sum((c.get('usage') or {}).get('total_tokens',0) for c in consumed),
                          skipped_recorded_stages=[c['stage'] for c in pending])
            results[index]=record;payload['records']=[results[i] for i in sorted(results)]
            _save_results(args.output,payload)
            print(row['sample_id'],record.get('result',{}).get('score'),flush=True)
    await asyncio.gather(*(one(i,row) for i,row in enumerate(rows)))
    payload.update(complete=True,seconds=time.perf_counter()-started)
    _save_results(args.output,payload)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--replay',type=Path)
    asyncio.run(run(parser.parse_args()))
