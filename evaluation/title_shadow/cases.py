"""Predeclared labels stay outside inference. Controlled edits of one saved context."""
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

from .reviewer import REASONS, TitleInput

SAVED = Path(__file__).resolve().parents[1]/'saved_public_inputs.json'
SAVED_SHA256 = '354c704933523ee238db5bb3ebc1932fa94a8465a9b30a06be226ac4cd49f252'
# id, raw CV title, LinkedIn title, metadata grade, expected status, justification
SPECS = [
 ('manager_exact','Manager','Manager',None,'MATCH','Exact role, no conflicting qualifier.'),
 ('manager_abbreviation','Manager','Mgr.',None,'MATCH','Conventional Manager abbreviation.'),
 ('senior_manager_grade','Senior Manager','Manager','senior','MATCH','Explicit same-base grade supports the qualifier.'),
 ('manager_extra_grade','Manager','Manager','senior','MATCH','Unqualified claim does not contradict extra profile grade.'),
 ('senior_manager_junior','Senior Manager','Manager','junior','MISMATCH','Explicit senior versus junior contradiction.'),
 ('director_manager','Director','Manager','senior','MISMATCH','Approved material role substitution; no equivalence evidence.'),
 ('senior_analyst_manager','Senior Analyst','Manager','senior','MISMATCH','Distinct base roles; senior metadata does not supply Analyst.'),
 ('senior_analyst_junior','Senior Analyst','Analyst','junior','MISMATCH','Explicit senior versus junior contradiction.'),
 ('director_exact','Director','Director',None,'MATCH','Exact unqualified role.'),
 ('associate_director_mapping','Associate Director','Senior Manager',None,'UNKNOWN','No supplied organizational mapping establishes equivalence or contradiction.'),
 ('head_sales_mapping','Head of Sales','Sales Manager',None,'UNKNOWN','Head/Manager mapping depends on unprovided organizational context.'),
 ('scientist_qualifier','Sr. Scientist','Scientist','senior','MATCH','Sr. is explicit senior on the same Scientist role.'),
 ('analyst_extra_junior','Analyst','Analyst','junior','MATCH','No explicit grade claim to contradict junior metadata.'),
 ('engineer_inflation','Junior Engineer','Engineer','senior','MISMATCH','Explicit junior versus senior contradiction, including split extraction.'),
 ('manager_director','Manager','Director','senior','MISMATCH','Reverse distinct-role substitution; not an inferred universal hierarchy.'),
 ('cfo_alias','Chief Financial Officer','CFO',None,'MATCH','Conventional expansion of the occupational acronym CFO.'),
 ('retail_alias','Retail Sales Associate','Retail Salesperson',None,'MATCH','Conventional retail worker title variants; O*NET 41-2031 alternate titles, not managerial-role equivalence.'),
 ('nurse_alias','Registered Nurse','RN',None,'MATCH','Conventional registered-nurse occupational acronym.'),
 ('different_occupations','Database Administrator','Graphic Designer',None,'MISMATCH','Distinct database administration and visual-design occupations.'),
 ('managing_director_mapping','Managing Director','Director','senior','UNKNOWN','Organization-specific qualifier is not established by broad senior metadata.'),
 ('head_operations_mapping','Head of Operations','Operations Manager','senior','UNKNOWN','No organizational evidence resolves Head/Manager equivalence.'),
 ('associate_director_exact','Associate Director','Associate Director',None,'MATCH','Exact qualified title.'),
 ('data_analyst_qualifier','Senior Data Analyst','Data Analyst','senior','MATCH','Supported qualifier on the same multiword base role.'),
 ('different_it_roles','Chief Technology Officer','IT Support Specialist','senior','MISMATCH','Explicit executive technology role differs from technical support; broad domain overlap is insufficient.'),
 ('missing_grade','Senior Scientist','Scientist',None,'UNKNOWN','Missing grade does not prove explicit senior qualification true or false.'),
]
REPEATED_THREE = {'manager_abbreviation','senior_manager_grade','associate_director_mapping',
                  'head_sales_mapping','cfo_alias','head_operations_mapping'}
REPEATED_FIVE = {'director_manager','senior_analyst_manager'}
INDEPENDENT = ('company','start_year','end_year','is_current')


def saved():
    if sha256(SAVED.read_bytes()).hexdigest()!=SAVED_SHA256:
        raise ValueError('Saved upstream source changed')
    return json.loads(SAVED.read_text())['sources']['CV_1.pdf']


def correspondence_digest(claims, profile):
    job={k:claims['jobs'][0][k] for k in INDEPENDENT if k in claims['jobs'][0]}
    history=sorted((json.dumps({k:r[k] for k in INDEPENDENT if k in r},sort_keys=True)
                    for r in profile.get('experience',[])))
    return sha256(json.dumps([profile.get('id'),job,history],sort_keys=True).encode()).hexdigest()


def bind(claims, profile, reviewed=True):
    """Validate a frozen PR31 review, not a general-purpose alignment resolver."""
    original=saved()
    if (not reviewed or not isinstance(profile.get('experience'),list)
            or not all(isinstance(r,dict) for r in profile['experience'])
            or correspondence_digest(claims,profile)!=correspondence_digest(original['claims'],original['profile'])):
        return {'status':'NOT_EVALUATED','reason':'No valid independent correspondence review for these facts'}
    job=claims['jobs'][0]
    matches=[i for i,r in enumerate(profile['experience']) if all(k in r and r[k]==job[k] for k in INDEPENDENT)]
    if len(matches)!=1:
        return {'status':'NOT_EVALUATED','reason':'Reviewed employment record is not unique'}
    index=matches[0]
    evidence=[{'path':f'experience.{index}.{field}','value':profile['experience'][index][field]}
              for field in ('title','seniority') if profile['experience'][index].get(field) is not None]
    return {'status':'REVIEWED','claim_id':'jobs.0.title','profile_id':profile['id'],
            'record_index':index,'evidence_paths':[e['path'] for e in evidence],'evidence':evidence,
            'basis':'PR31 reviewed employer/start/end/current correspondence, unchanged by title-only edits'}


def load_cases():
    cases=[]
    for index,(identifier,raw,title,grade,expected,why) in enumerate(SPECS):
        base=saved();claims,profile=deepcopy(base['claims']),deepcopy(base['profile'])
        anchor=bind(claims,profile);record=profile['experience'][anchor['record_index']]
        claimed,claimed_grade=raw,None
        if identifier=='scientist_qualifier':claimed,claimed_grade='Scientist','senior'
        if identifier=='engineer_inflation':claimed,claimed_grade='Engineer','junior'
        claims['jobs'][0].update(title=claimed,seniority=claimed_grade,
                                 raw=f'{raw}, Deloitte    2020 – Present')
        record.update(title=title,seniority=grade)
        source=TitleInput(raw,claimed,claimed_grade,title,grade)
        if expected=='UNKNOWN':
            principles=REASONS['UNKNOWN'][1:] if identifier=='missing_grade' else REASONS['UNKNOWN']
        elif identifier in {'senior_manager_junior','senior_analyst_junior','engineer_inflation'}:
            principles=REASONS['MISMATCH'][1:]
        elif expected=='MISMATCH':principles=REASONS['MISMATCH'][:1]
        elif identifier in {'manager_abbreviation','cfo_alias','retail_alias','nurse_alias'}:
            principles=REASONS['MATCH'][1:2]
        elif identifier in {'senior_manager_grade','scientist_qualifier','data_analyst_qualifier'}:
            principles=REASONS['MATCH'][2:]
        else:principles=(REASONS['MATCH'][0],REASONS['MATCH'][2])
        cases.append({'id':identifier,'split':'prompt_example' if index<11 else 'held_out',
                      'expected':expected,'justification':why,'grounded_reasons':list(principles),
                      'repetitions':5 if identifier in REPEATED_FIVE else 3 if identifier in REPEATED_THREE else 1,
                      'claims':claims,'profile':profile,'title_input':asdict(source),
                      'binding':bind(claims,profile)})
    return cases
