"""Grade the frozen plan and publish only curated results, never raw model traces."""
from collections import Counter
import json
from pathlib import Path
import sys

from cv_verifier.claim_comparison_validator import ClaimComparisonValidator


def report(folder):
    read=lambda name:json.loads((folder/name).read_text())
    plan=read('plan.json');cases={c['id']:c for c in plan['cases']}
    paths={k:read(name) for k,name in [('A','A-main.json'),('B','B-pr33.json'),('C','C-reviewer.json')]}
    assert all(d['complete'] and len(d['records'])==45 for d in paths.values())
    maps={k:{r['sample_id']:r for r in d['records']} for k,d in paths.items()}
    rows=[];paired=0
    for sample in plan['samples']:
        sid=sample['sample_id'];case=cases[maps['A'][sid]['case_id']]
        row={'sample_id':sid,'case_id':case['id'],'split':case['split'],'expected':case['expected'],
             'input':case['title_input'],'binding':case['binding']}
        for path in ('A','B','C'):
            source=maps[path][sid];calls=source['calls']
            if path in ('A','B'):
                result=source.get('result',{});finding=next((f for f in result.get('findings',[]) if f['claim_id']=='jobs.0.title'),None)
                status='ERROR' if source.get('worker_error') or finding is None else finding['status']
                wrong=[e['path'] for e in (finding or {}).get('evidence',[]) if e['path'].startswith('experience.')
                       and len(e['path'].split('.'))==3 and int(e['path'].split('.')[1])!=case['binding']['record_index']]
                canonical=False
                try:
                    ClaimComparisonValidator().validate(result['findings'],case['claims'],case['profile'])
                    canonical=finding is not None
                except (ValueError,TypeError,KeyError):pass
                relevant=any(e['path'] in case['binding']['evidence_paths'] for e in (finding or {}).get('evidence',[]))
                decision={'status':status,'score':result.get('score'),'errors':result.get('errors',[]),
                          'worker_error':source.get('worker_error'),'evidence':(finding or {}).get('evidence',[]),
                          'wrong_record_citations':wrong,'canonical_valid':canonical,
                          'grounded_correct':status==case['expected'] and canonical and not wrong and (status=='UNKNOWN' or relevant)}
            else:
                status=source['status'];binding=source.get('binding')==case['binding'] and source.get('input')==case['title_input']
                decision={'status':status,'reason_principle':source.get('reason'),'error':source.get('error'),
                          'binding_valid':binding,'grounded_correct':status==case['expected'] and binding
                          and source.get('reason') in case['grounded_reasons']}
            decision.update(correct=status==case['expected'],logical_calls=len(calls),
                            new_calls=0 if path=='B' else len(calls),seconds=source['seconds'],
                            input_tokens=sum((c.get('usage') or {}).get('input_tokens',0) for c in calls),
                            output_tokens=sum((c.get('usage') or {}).get('output_tokens',0) for c in calls),
                            new_tokens=sum((c.get('usage') or {}).get('total_tokens',0) for c in calls),
                            replayed_token_equivalents=source.get('replayed_token_equivalents',0))
            row[path]=decision
        raw=[]
        for call in maps['A'][sid]['calls']:
            if call['stage']!='comparison':continue
            try:raw.append(next(f['status'] for f in json.loads(call['output'])['findings'] if f['claim_id']=='jobs.0.title'))
            except (ValueError,TypeError,KeyError,StopIteration):raw.append('INVALID')
        row['raw_comparator_statuses']=raw
        a,b=maps['A'][sid]['calls'],maps['B'][sid]['calls']
        assert len(b)<=len(a)
        for original,replayed in zip(a,b):
            assert original['input_messages']==replayed['input_messages'] and original['output']==replayed['output']
            assert not replayed.get('error_type');paired+=1
        rows.append(row)
    summary={}
    for path in ('A','B','C'):
        summary[path]={}
        for split in ('all','prompt_example','held_out'):
            subset=[r for r in rows if split=='all' or r['split']==split]
            counts={'samples':len(subset),'correct':sum(r[path]['correct'] for r in subset),
                    'grounded_correct':sum(r[path]['grounded_correct'] for r in subset)}
            for name,expected,actual in [('false_match','MISMATCH','MATCH'),('false_mismatch','MATCH','MISMATCH'),
                                         ('missed_discrepancy_unknown','MISMATCH','UNKNOWN'),('equivalent_unknown','MATCH','UNKNOWN')]:
                counts[name]=sum(r['expected']==expected and r[path]['status']==actual for r in subset)
            counts.update(ambiguity_certainty=sum(r['expected']=='UNKNOWN' and r[path]['status'] in ('MATCH','MISMATCH') for r in subset),
                          errors=sum(r[path]['status']=='ERROR' for r in subset),
                          decisions=dict(Counter(r[path]['status'] for r in subset)))
            summary[path][split]=counts
        summary[path]['cost']={key:sum(r[path][key] for r in rows) for key in
                              ('logical_calls','new_calls','new_tokens','input_tokens','output_tokens','replayed_token_equivalents','seconds')}
        summary[path]['cost']['wall_seconds']=paths[path]['seconds']
        summary[path]['variation']={case_id:dict(Counter(r[path]['status'] for r in rows if r['case_id']==case_id))
                                    for case_id,c in cases.items() if c['repetitions']>1}
    result={'main':plan['main'],'pr33':plan['pr33'],'study_commit':plan['study_commit'],
            'prompt_version':plan['prompt_version'],'prompt_sha256':plan['prompt_sha256'],
            'paired_requests':paired,'summary':summary,'records':rows,
            'not_evaluated_controls':read('correspondence-skips.json')}
    destination=Path(__file__).with_name('results.json')
    header=json.dumps({k:v for k,v in result.items() if k!='records'},indent=2)
    compact_rows=',\n'.join('    '+json.dumps(row,ensure_ascii=False,sort_keys=True) for row in rows)
    destination.write_text(header[:-2]+',\n  "records": [\n'+compact_rows+'\n  ]\n}\n')
    render(result,plan,destination.with_name('REPORT.md'))
    print(json.dumps(summary,indent=2))


def render(data,plan,destination):
    lines=['# Title-equivalence shadow results','',
           '**Recommendation: retain the existing baseline for delivery. The reviewer merits further shadow testing, not an automatic veto.**',
           '',f"Frozen study `{data['study_commit']}`, prompt `{data['prompt_version']}`. Main `{data['main']}`, PR33 `{data['pr33']}` (still unmerged when tested). Production source unchanged; no public campaign or integration was run.",
           '', '| Approach | Teaching-example status | Held-out status | All status | False MATCH on discrepancies | False MISMATCH on equivalents | Incorrect certainty on ambiguity | No final title / contract failure |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for k,name in [('A','Main comparator/reflection'),('B','PR33 paired replay'),('C','Independent reviewer')]:
        s=data['summary'][k];a=s['all'];p=s['prompt_example'];h=s['held_out']
        lines.append(f"| {name} | {p['correct']}/{p['samples']} | {h['correct']}/{h['samples']} | {a['correct']}/{a['samples']} | {a['false_match']} | {a['false_mismatch']} | {a['ambiguity_certainty']} | {a['errors']} |")
    lines += ['', 'Errors remain in denominators. Teaching examples are not independent generalization. These 45 decisions cover 25 constructed cases in one shared saved profile context; held-out means absent from this prompt, including close paraphrases and previously observed reverse combinations, not unseen CVs.',
              '', '## Grounding and failure analysis','',
              'Existing structured controls `ambiguous_record_details.jobs.0.title` and `additional_fabricated_job.jobs.2.title` are independently unsupported/ambiguous correspondences. Both produce NOT_EVALUATED with zero model calls, separately from the 45 title decisions. No selected record is fabricated and these skips are not counted as UNKNOWN accuracy.',
              'Main and PR33 have zero wrong-record title citations among returned findings. Three samples fail comparison contracts after bounded recovery because the model cites null seniority: Associate Director sample2, Head of Sales sample2, and the missing-grade case. Main also rejects one reflection correction on Director sample1 and preserves its validated baseline; PR33 skips that audit after its deterministic negative. No schema/evidence constraint was weakened.',
              '', 'The reviewer has no output-contract/provider failures. Its bounded reason vocabulary prevents novel employer/duty assertions, but does not prove semantic correctness. Predeclared principle-basis grounding is 43/45 overall and 16/18 held-out: missing-grade status is wrong, and the retail alias selects the qualifier principle instead of the applicable alias principle. The latter is a rationale-basis mismatch, not a false discrepancy or invented employer fact. Baseline canonical/correspondence checks are a different grounding test; do not treat these as identical rationale audits.',
              '', '- **Missing grade:** Senior Scientist versus Scientist, profile seniority null. Expected UNKNOWN. Reviewer MATCH with the same-base qualifier principle is unsupported: absence of grade is neither confirmation nor contradiction. Main/PR33 instead fail the evidence contract on this input. This is a substantive reviewer miss, retained without prompt revision.',
              '- **Known substitutions:** main raw comparator returns MATCH in all ten primary samples. Reflection catches Senior Analyst once (1/5), never Director. PR33 corrects Director5/5 and leaves Senior Analyst1/5; reviewer corrects both5/5. Both pairs are explicit teaching examples, so this tests compliance, not generalization.',
              '- **Held-out gain:** every approach gets all 13 known-label held-out samples right (nine equivalent, four discrepant). Reviewer improves four ambiguity samples (Managing Director and three Head of Operations); its fifth held-out ambiguity sample, missing grade, is wrong. There is no demonstrated false-MISMATCH advantage on unseen equivalents: all three have zero here.',
              '- **Uncertainty:** UNKNOWN is a title abstention, not a CV score or a fabricated discrepancy. The reviewer is never used to turn a validated positive into a 0.5 negative.',
              '', '## Calls, tokens and latency','',
              '| Approach | Logical invocations | New API calls | New input/output tokens | New total tokens | Replayed token equivalents | Wall seconds | Summed case seconds |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for k in ('A','B','C'):
        c=data['summary'][k]['cost']
        lines.append(f"| {k} | {c['logical_calls']} | {c['new_calls']} | {c['input_tokens']:,} / {c['output_tokens']:,} | {c['new_tokens']:,} | {c['replayed_token_equivalents']:,} | {c['wall_seconds']:.2f} | {c['seconds']:.2f} |")
    lines += ['',f"All {data['paired_requests']} consumed B message/response pairs match A byte-for-byte; five recorded reflection calls are skipped by the guard. B is an actual component replay, not an independent live draw. Its CPU timing is not network performance and its reused token equivalents are not new billing. Actual API usage is A+C; tokens are a cost proxy, not a dollar estimate. C's compact one-call timing does not mean adding it would speed up the current pipeline.",
              '', '## All repeated outcomes','', '| Case | Samples | Main | PR33 | Reviewer |','|---|---:|---|---|---|']
    for case in plan['cases']:
        if case['repetitions']>1:
            distributions=[str(data['summary'][k]['variation'][case['id']]) for k in ('A','B','C')]
            lines.append(f"| {case['id']} | {case['repetitions']} | {' | '.join(distributions)} |")
    lines += ['', 'Captured initial comparator and reviewer messages are byte-identical across repeats for every case. Production reflection naturally depends on its previous model response. Reviewer statuses are stable on all eight repeated inputs. That consistency covers only the declared few-shot/near-shot examples; a one-sample error cannot establish its frequency.',
              '', '## Every sample','', '| Sample | Split | Expected | Raw comparator attempts | Main final | PR33 final | Reviewer | Reviewer basis correct |','|---|---|---|---|---|---|---|---|']
    for row in data['records']:
        lines.append(f"| {row['sample_id']} | {row['split']} | {row['expected']} | {','.join(row['raw_comparator_statuses'])} | {row['A']['status']} | {row['B']['status']} | {row['C']['status']} | {row['C']['grounded_correct']} |")
    lines += ['', 'See `results.json` for each actual baseline score, canonical title evidence, errors, reviewer principle, binding, token use and case runtime. Full prompts/responses are ignored under the original study directory.',
              '', '## Integration design only','',
              '| Option | Trigger/budget | Disagreement, UNKNOWN and failure policy | Evidence and authority |','|---|---|---|---|',
              '| Shadow diagnostics (recommended now) | Only independently reviewed title records; fixed evaluation budget | Log every result; never change score | Useful for more diverse blind data and missing-grade analysis; no production authority |',
              '| Optional audit of questionable MATCH | Design cap one title call/CV, only uniquely corroborated records; triggers require a separately reviewed policy | MATCH leaves baseline; UNKNOWN/errors/timeouts preserve baseline; MISMATCH is a flagged proposal | Same model/evidence is not an independent fact. Automatic veto is not justified by this small set |',
              '| Existing positive-only reflection seam | If eventually approved, replace/restrict the title portion of an existing audit rather than stack unbounded votes; explicit timeout and one-call bound | Only a supported MATCH-to-MISMATCH proposal; no UNKNOWN downgrade or retries/fallback agents; preserve baseline on failure | Python reconstructs canonical selected-record fields, full original validators run; no record realignment or invented evidence. New held-out false-positive evidence is needed first |',
              '', 'A possible future trigger is a positive baseline with a title MATCH on a uniquely corroborated employer/period, where normalized base roles differ or an explicit qualifier lacks matching metadata. That is only a review trigger, never a string-based discrepancy rule. Candidate policy and ordering across multiple jobs need separate review; cap at one call per CV and skip when the remaining per-CV deadline cannot accommodate the bounded call. Do not review unsupported correspondence.',
              '', 'Replacing existing reflection outright would lose its non-title audits; preserving them while adding a specialist increases calls. A safe integration study must measure that trade-off rather than silently remove coverage. UNKNOWN, timeout, malformed output and disagreement preserve the baseline. A specialist MISMATCH is only a proposal, not sufficient override authority from this experiment: canonical reconstruction proves supplied provenance, not semantic truth.',
              '', 'One false discrepancy can reject a valid CV. Both reviewers use the same DeepSeek model and the same supplied facts; hiding the comparator output prevents anchoring but does not create independent knowledge. Agreement is not a majority-vote proof. Do not auto-promote a fluent MISMATCH over a validated MATCH. Any integration needs separate authorization, correspondence confidence, a measured latency/call budget, missing-evidence controls and broader independently labeled aliases.',
              '', '## Limits and reproducibility','',
              'One prompt version, one declared run, no resampling or optimization. Labels and acceptable reason principles were frozen before execution. Cases are policy judgments, not a universal title taxonomy; ambiguous mappings are UNKNOWN. The retail alias label uses a narrow conventional lexical interpretation supported by the O*NET reference in README, not automatic equivalence of all titles in an occupation. No independent expert panel or private-set evaluation was conducted.',
              '', 'The class/tests, dataset/plan and actual production-worker source are committed under `evaluation/` only. Raw plan, worker outputs, full requests/responses, runtime logs, frozen source checkout and hashes are ignored. Run the study command in README in a new directory to create a transparently separate run; an existing plan cannot be overwritten. Rerun this report against saved captures without new API calls. Stop for review; do not merge or integrate into scoring.']
    destination.write_text('\n'.join(lines)+'\n')


if __name__=='__main__':report(Path(sys.argv[1]).resolve())
