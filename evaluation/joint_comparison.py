"""Development-only joint correspondence and verification; never imported by production."""

import json
from copy import deepcopy

from cv_verifier.claim_comparator import COMPARISON_PROMPT
from cv_verifier.claim_comparison_validator import ClaimComparisonValidator


JOINT_PROMPT = COMPARISON_PROMPT.partition('Return ONLY a JSON object')[0] + """
Assess correspondence and claim judgments jointly in this ONE response, using the
whole record context. Correspondence is a semantic hypothesis, not independent proof.
For EVERY CV job and education record return cv_index, profile_index, state and reason:
- ALIGNED: a supported whole-record correspondence; profile_index is an existing integer.
  Remaining attributes may identify a record even if one employer, school, degree,
  title or date claim is false. Do not require every attribute to match.
- UNRESOLVED: ambiguous or insufficient correspondence; profile_index=null and EVERY
  finding for this record must be UNKNOWN. Never pick arbitrary records for definitive findings.
- ABSENT: an identifiable claimed record genuinely absent after semantic inspection
  of the available authoritative full collection; profile_index=null. Its discrepancy
  findings cite the whole experience/education collection, not unrelated indexed fields.
  Missing/null collections do not establish absence. Missing exact names, unknown aliases,
  or one false attribute alone do not establish an absent whole record.
Do not require one-to-one assignments: consolidated or overlapping roles may share an
index when genuinely supported. Uncertainty must remain explicit, not forced uniqueness.

ALIGNED job/education findings cite ONLY their corresponding indexed record and fields
relevant to the claim. A company claim cites company; school cites school; degree cites
degree; field cites field; graduation_year cites end_year. Title/seniority may cite
title and seniority of the SAME record. Explicit title qualifiers need cited grade
evidence: Senior Manager vs Manager/junior is MISMATCH; unqualified Manager remains
compatible with Manager/junior. A qualified profile title can supply explicit grade.
Definite typed year/currentness judgments must agree with the cited typed scalar.
Missing or null scalar evidence requires UNKNOWN, never proof of a discrepancy.
UNKNOWN is permitted when evidence or meaning is insufficient; Python does not
promote uncertain findings or assign alternative records. Skills absence may cite
the supplied full skills list; a missing/null list requires UNKNOWN.
For name cite name, for city cite city. Python supplies every claim and evidence value.

Return ONLY JSON with exactly records and findings:
{"records": {"jobs": [{"cv_index": 0, "profile_index": 1, "state": "ALIGNED",
"reason": "The remaining role and timeline identify this record."}], "education": []},
"findings": [{"claim_id": "jobs.0.company", "status": "MATCH",
"evidence_paths": ["experience.1.company"], "reason": "The corresponding employer agrees."}]}
Account for every catalog claim exactly once; findings have exactly claim_id, status,
evidence_paths and reason. Never return scores, copied evidence values, extra keys or Markdown.
"""


class JointRecordComparator:
    """One inference, followed by independently testable consistency validation."""

    def __init__(self, llm):
        from langchain_core.prompts import ChatPromptTemplate
        self._chain = ChatPromptTemplate.from_messages([
            ('system', JOINT_PROMPT.replace('{', '{{').replace('}', '}}')),
            ('human', '{context}'),
        ]) | llm
        self.validator = JointComparisonValidator()

    async def compare(self, claims, profile):
        contract = ClaimComparisonValidator()
        context = json.dumps({'claims': claims, 'claim_catalog': contract.claim_catalog(claims),
                              'profile': contract.comparison_profile(profile)}, ensure_ascii=False)
        response = await self._chain.ainvoke({'context': context})
        return self.validator.validate(json.loads(response.content), claims, profile)


class JointComparisonValidator:
    """Check joint-output consistency, without certifying semantic correspondence."""

    COLLECTIONS = {'jobs': 'experience', 'education': 'education'}

    def validate(self, output, claims, profile):
        if not isinstance(output, dict) or set(output) != {'records', 'findings'}:
            raise ValueError('Joint output requires records and findings')
        contract = ClaimComparisonValidator()
        findings = contract.reconstruct({'findings': output['findings']}, claims, profile)
        records = self._records(output['records'], claims, profile)
        for finding in findings:
            self._paths(finding, records, profile)
            self._typed(finding)
            self._seniority(finding, records, claims, profile, contract)
        return {'records': deepcopy(output['records']), 'findings': findings}

    def _records(self, records, claims, profile):
        if not isinstance(records, dict) or set(records) != set(self.COLLECTIONS):
            raise ValueError('Joint records require jobs and education')
        indexed = {}
        for collection, source in self.COLLECTIONS.items():
            if not isinstance(records[collection], list):
                raise ValueError('Record correspondence must be an array')
            for entry in records[collection]:
                if not isinstance(entry, dict) or set(entry) != {'cv_index', 'profile_index', 'state', 'reason'}:
                    raise ValueError('Correspondence requires cv_index, profile_index, state and reason')
                index, assigned = entry['cv_index'], entry['profile_index']
                if type(index) is not int or not 0 <= index < len(claims[collection]) or (collection, index) in indexed:
                    raise ValueError('CV record index is invalid or duplicated')
                if entry['state'] not in ('ALIGNED', 'UNRESOLVED', 'ABSENT'):
                    raise ValueError('Unsupported correspondence state')
                if not isinstance(entry['reason'], str) or not entry['reason'].strip():
                    raise ValueError('Correspondence reason must be nonempty')
                evidence = profile.get(source)
                if entry['state'] == 'ALIGNED':
                    if type(assigned) is not int or not isinstance(evidence, list) or not 0 <= assigned < len(evidence):
                        raise ValueError('Aligned profile record index does not exist')
                elif assigned is not None:
                    raise ValueError('Unresolved/absent correspondence requires null profile_index')
                if entry['state'] == 'ABSENT' and not isinstance(evidence, list):
                    raise ValueError('Absence requires an available authoritative collection')
                indexed[collection, index] = entry
            if len(records[collection]) != len(claims[collection]):
                raise ValueError('Every CV record requires correspondence')
        return indexed

    def _paths(self, finding, records, profile):
        reference, status = finding['claim_id'], finding['status']
        parts = reference.split('.')
        if parts[0] in self.COLLECTIONS:
            collection, index, field = parts
            source = self.COLLECTIONS[collection]
            record = records[collection, int(index)]
            if record['state'] == 'UNRESOLVED' and status != 'UNKNOWN':
                raise ValueError('Unresolved correspondence permits only UNKNOWN findings')
            if record['state'] == 'ABSENT' and status not in ('MISMATCH', 'UNKNOWN'):
                raise ValueError('Absent records cannot justify MATCH findings')
            fields = self._fields(collection, field)
            for citation in finding['evidence']:
                path = citation['path']
                if path == source:
                    if record['state'] == 'ALIGNED' or status == 'MATCH':
                        raise ValueError('Aligned findings must cite their corresponding record')
                    continue
                bits = path.split('.')
                if len(bits) != 3 or bits[0] != source or bits[2] not in fields:
                    raise ValueError('Evidence path is not relevant to the claim')
                if record['state'] == 'ABSENT' or (record['state'] == 'ALIGNED' and int(bits[1]) != record['profile_index']):
                    raise ValueError('Evidence must cite the corresponding record')
            if record['state'] == 'ABSENT' and status == 'MISMATCH' and not any(c['path'] == source for c in finding['evidence']):
                raise ValueError('Absent record discrepancy requires collection evidence')
        else:
            for citation in finding['evidence']:
                path = citation['path']
                relevant = (path == reference) if parts[0] != 'skills' else (
                    path == 'skills' or
                    (len(path.split('.')) == 3 and path.startswith('skills.') and path.endswith('.name')))
                if not relevant:
                    raise ValueError('Evidence path is not relevant to the claim')

    def _fields(self, collection, field):
        if field in ('title', 'seniority'):
            return ('title', 'seniority')
        if field == 'graduation_year':
            return ('end_year',)
        if field == 'dates':
            return ('start_year', 'end_year', 'is_current') if collection == 'jobs' else ('end_year',)
        if field == 'record':
            return ClaimComparisonValidator.JOB_FIELDS if collection == 'jobs' else ('school', 'degree', 'field', 'end_year')
        return (field,)

    def _typed(self, finding):
        parts = finding['claim_id'].split('.')
        if parts[0] not in self.COLLECTIONS or finding['status'] == 'UNKNOWN':
            return
        field = parts[2]
        if field not in ('start_year', 'end_year', 'graduation_year', 'is_current'):
            return
        # Collection-level absence has no corresponding scalar to compare.
        scalars = [c['value'] for c in finding['evidence'] if len(c['path'].split('.')) == 3]
        if not scalars:
            return
        expected_type = bool if field == 'is_current' else int
        value = finding['cv_claim']['value']
        if type(value) is not expected_type or any(type(actual) is not expected_type for actual in scalars):
            raise ValueError('Definite typed judgment requires valid typed evidence')
        expected = 'MATCH' if all(value == actual for actual in scalars) else 'MISMATCH'
        if finding['status'] != expected:
            raise ValueError('Judgment contradicts the cited typed values')

    def _seniority(self, finding, records, claims, profile, contract):
        parts = finding['claim_id'].split('.')
        if parts[0] != 'jobs' or parts[2] not in ('title', 'seniority') or finding['status'] == 'UNKNOWN':
            return
        index, field = int(parts[1]), parts[2]
        alignment = records['jobs', index]
        if alignment['state'] != 'ALIGNED':
            return
        cv, actual = claims['jobs'][index], profile['experience'][alignment['profile_index']]
        cv_base, qualifier = contract._title_parts(cv['title'])
        profile_base, profile_qualifier = contract._title_parts(actual.get('title'))
        # A separate seniority claim is checked by its own finding. An unqualified
        # title does not acquire that claim or require a duplicated grade citation.
        claimed = contract._seniority(cv['seniority']) if field == 'seniority' else qualifier
        explicit = contract._seniority(actual.get('seniority'))
        levels = {'junior', 'mid', 'senior'}
        if claimed in levels and explicit in levels and profile_qualifier in levels and explicit != profile_qualifier:
            raise ValueError('Conflicting profile seniority requires UNKNOWN')
        observed = explicit or profile_qualifier
        paths = {c['path'] for c in finding['evidence']}
        title_path = f"experience.{alignment['profile_index']}.title"
        if field == 'title' and finding['status'] == 'MATCH' and title_path not in paths:
            raise ValueError('Title MATCH requires cited role evidence')
        if claimed in levels and observed not in levels and (field == 'seniority' or finding['status'] == 'MATCH' or cv_base == profile_base):
            raise ValueError('Missing explicit grade evidence requires UNKNOWN')
        if field == 'title' and claimed is None and cv_base and cv_base == profile_base and finding['status'] == 'MISMATCH':
            raise ValueError('Extra profile grade cannot contradict an unqualified matching title')
        if claimed in levels and observed in levels:
            if claimed != observed and finding['status'] == 'MATCH':
                raise ValueError('MATCH contradicts explicit seniority evidence')
            if claimed == observed and finding['status'] == 'MISMATCH' and (field == 'seniority' or cv_base == profile_base):
                raise ValueError('MISMATCH contradicts compatible title/seniority evidence')
            prefix = f"experience.{alignment['profile_index']}"
            grade_paths = ([f'{prefix}.seniority'] if explicit in levels else []) + (
                [f'{prefix}.title'] if profile_qualifier in levels else [])
            if (field == 'seniority' or finding['status'] == 'MATCH' or claimed != observed) and not any(
                    citation['path'] in grade_paths for citation in finding['evidence']):
                raise ValueError('Explicit seniority judgment requires cited grade evidence')
