"""One predeclared shadow study; never writes grading scores or changes production."""
import argparse
import asyncio
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from langchain_core.runnables import RunnableLambda
from evaluation.run_controlled import _save_results
from evaluation.title_shadow.cases import bind, load_cases
from evaluation.title_shadow.reviewer import PROMPT, PROMPT_VERSION, TitleEquivalenceReviewer, TitleInput
from hw2 import MODEL_NAME, load_env_file

ROOT=Path(__file__).resolve().parents[2]
MAIN='d50b42f2ecb4604c099eb46a208d1a0aa7645097'
PR33='5ef43c88371e96edf58aca3ea01231969e690ca4'


async def review_one(row,llm):
    record={'sample_id':row['sample_id'],'case_id':row['case_id'],'calls':[]}
    if row['binding']['status']!='REVIEWED':
        return {**record,'status':'NOT_EVALUATED','seconds':0.0}
    async def measured(prompt):
        call={'input_messages':[{'role':m.type,'content':m.content} for m in prompt.to_messages()]}
        record['calls'].append(call);start=time.perf_counter()
        try:
            response=await llm.ainvoke(prompt)
            call.update(output=response.content,usage=response.usage_metadata)
            return response
        except Exception as error:
            call.update(error_type=type(error).__name__,error=str(error));raise
        finally:call['seconds']=time.perf_counter()-start
    start=time.perf_counter()
    try:
        source=TitleInput(**row['title_input'])
        record.update(await asyncio.wait_for(TitleEquivalenceReviewer(RunnableLambda(measured)).review(source),65))
    except Exception as error:
        record.update(status='ERROR',error={'type':type(error).__name__,'message':str(error)})
    record.update(seconds=time.perf_counter()-start,binding=deepcopy(row['binding']),
                  input=asdict(TitleInput(**row['title_input'])))
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    if not out.is_relative_to((ROOT/'evaluation/results').resolve()):raise ValueError('Raw results must stay ignored')
    if (out/'plan.json').exists():raise ValueError('Existing study: use its captures, do not overwrite/restart')
    out.mkdir(parents=True,exist_ok=True);load_env_file()
    cases=load_cases();rows=[]
    for case in cases:
        for repeat in range(1,case['repetitions']+1):
            row={k:deepcopy(case[k]) for k in ('claims','profile','binding','title_input')}
            row.update(sample_id=f"{case['id']}__{repeat}",case_id=case['id'])
            rows.append(row)
    assert len(cases)==25 and len(rows)==45
    files=[Path(__file__),Path(__file__).with_name('reviewer.py'),Path(__file__).with_name('cases.py'),Path(__file__).with_name('production_worker.py')]
    plan={'prompt_version':PROMPT_VERSION,'prompt_sha256':hashlib.sha256(PROMPT.encode()).hexdigest(),
          'files_sha256':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
          'cases_sha256':hashlib.sha256(json.dumps(cases,sort_keys=True).encode()).hexdigest(),
          'cases':cases,'samples':[{'sample_id':r['sample_id'],'input_sha256':hashlib.sha256(json.dumps(r,sort_keys=True).encode()).hexdigest()} for r in rows],
          'main':MAIN,'pr33':PR33,'study_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
          'model':MODEL_NAME,'thinking':'disabled','SDK_retries':0,'reviewer_calls_per_sample':1,
          'concurrency':3,'max_samples':45,'pip_freeze':subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)}
    _save_results(out/'plan.json',plan);_save_results(out/'inputs.json',rows)
    snapshot=out/'main-source'
    subprocess.run(['git','clone','--no-hardlinks','--no-checkout',str(ROOT),str(snapshot)],check=True)
    subprocess.run(['git','-C',str(snapshot),'checkout','--detach',MAIN],check=True)
    worker=Path(__file__).with_name('production_worker.py')
    for name,cwd,replay in [('A-main',snapshot,None),('B-pr33',ROOT,out/'A-main.json')]:
        env={**os.environ,'PYTHONPATH':str(cwd)}
        command=[sys.executable,str(worker),'--inputs',str(out/'inputs.json'),'--output',str(out/f'{name}.json')]
        if replay:command+=['--replay',str(replay)]
        with (out/f'{name}.log').open('w') as log:
            subprocess.run(command,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    async def review_all():
        from langchain_deepseek import ChatDeepSeek
        llm=ChatDeepSeek(model=MODEL_NAME,timeout=60,max_retries=0).bind(
            extra_body={'thinking':{'type':'disabled'}},response_format={'type':'json_object'})
        payload={'records':[],'complete':False};semaphore=asyncio.Semaphore(3);results={};start=time.perf_counter()
        async def one(index,row):
            async with semaphore:
                if bind(row['claims'],row['profile'])!=row['binding']:raise ValueError('Correspondence drift')
                results[index]=await review_one(row,llm)
                payload['records']=[results[i] for i in sorted(results)]
                _save_results(out/'C-reviewer.json',payload)
        await asyncio.gather(*(one(i,row) for i,row in enumerate(rows)))
        payload.update(complete=True,seconds=time.perf_counter()-start);_save_results(out/'C-reviewer.json',payload)
    asyncio.run(review_all())
    for file in files:
        if hashlib.sha256(file.read_bytes()).hexdigest()!=plan['files_sha256'][str(file.relative_to(ROOT))]:raise ValueError('Study source drift')
    print('COMPLETE: one declared study, all samples retained, no production integration')


if __name__=='__main__':main()
