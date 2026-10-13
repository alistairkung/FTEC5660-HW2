"""Development shadow gate; no claim decisions, realignment or inference."""

from copy import deepcopy

from cv_verifier.claim_validator import ClaimValidator
from cv_verifier.deterministic_claim_verifier import DeterministicClaimVerifier
from cv_verifier.record_aligner import RecordAligner


class CorrespondencePlausibilityGate:
    """Reject narrow cross-record conflicts; preservation does not certify correctness.

    Only normalized exact text is recognized. Aliases and ambiguous text remain
    downstream semantic tasks. Seniority is deliberately not a correspondence
    veto. Alternatives are diagnostic evidence, never replacement assignments.
    """

    COLLECTIONS = RecordAligner.COLLECTIONS
    FIELDS = {'jobs': ('company', 'title', 'start_year', 'end_year', 'is_current'),
              'education': ('school', 'degree', 'field', 'graduation_year')}

    def apply(self, claims, profile, alignments):
        ClaimValidator().validate(claims)
        self._validate(alignments, claims, profile)
        output, diagnostics = deepcopy(alignments), []
        for collection, source in self.COLLECTIONS.items():
            records = profile.get(source)
            for entry in output[collection]:
                index, assigned = entry['cv_index'], entry['profile_index']
                item = {'collection': collection, 'cv_index': index,
                        'original_profile_index': assigned, 'decision': 'UNADJUDICATED',
                        'action': 'preserved_null',
                        'reason': 'Existing unresolved correspondence is not promoted.', 'evidence': []}
                if assigned is not None:
                    cv = claims[collection][index]
                    item['evidence'] = [self._compare(cv, record, collection, index, j)
                                        for j, record in enumerate(records)]
                    observed = item['evidence'][assigned]['relations']
                    competitors = [e['profile_index'] for e in item['evidence']
                                   if e['profile_index'] != assigned and self._supports_alternative(e['relations'], collection)]
                    item['alternative_indices'] = competitors
                    if self._conflicts(observed, collection) and competitors:
                        item.update(decision='IMPLAUSIBLE', action='null_conflict',
                                    reason='Identity and timeline conflict with this mapping, while an alternative record has corroborating evidence.')
                    else:
                        if self._supports_assigned(observed, collection):
                            item.update(decision='PLAUSIBLE', action='retained',
                                        reason='Multiple record attributes support this correspondence; the gate does not verify claim truth.')
                        else:
                            item.update(decision='UNADJUDICATED', action='unadjudicated',
                                        reason='Usable evidence does not establish or reject this correspondence; it is not independently verified.')
                    if item['decision'] == 'IMPLAUSIBLE':
                        entry['profile_index'] = None
                        entry['reason'] = item['reason']
                item['post_profile_index'] = entry['profile_index']
                diagnostics.append(item)
        self._validate(output, claims, profile)
        return {'alignments': output, 'diagnostics': diagnostics}

    def _validate(self, alignments, claims, profile):
        if not isinstance(alignments, dict) or set(alignments) != set(self.COLLECTIONS):
            raise ValueError('Gate requires validated jobs/education record alignments')
        for collection, source in self.COLLECTIONS.items():
            records = profile.get(source)
            RecordAligner._validate_collection(alignments[collection], len(claims[collection]),
                                              len(records) if isinstance(records, list) else 0)

    def _compare(self, cv, record, collection, cv_index, profile_index):
        record = record if isinstance(record, dict) else {}
        verifier, relations, fields = DeterministicClaimVerifier(), {}, []
        for field in self.FIELDS[collection]:
            target = 'end_year' if field == 'graduation_year' else field
            left, right = cv.get(field), record.get(target)
            if field in ('start_year', 'end_year', 'graduation_year', 'is_current'):
                expected_type = bool if field == 'is_current' else int
                available = type(left) is expected_type and type(right) is expected_type
                equal = left == right
            else:
                a, b = (verifier._title_parts(value)[0] for value in (left, right)) if field == 'title' else (
                    verifier._text(left), verifier._text(right))
                available, equal = bool(a and b), a == b
            relations[field] = ('equal' if equal else 'different') if available else 'unavailable'
            fields.append({'cv_path': f'{collection}.{cv_index}.{field}', 'cv_value': deepcopy(left),
                           'profile_path': f'{self.COLLECTIONS[collection]}.{profile_index}.{target}',
                           'profile_value': deepcopy(right), 'relation': relations[field],
                           'basis': 'base_title' if field == 'title' else
                           'typed_equality' if field in ('start_year', 'end_year', 'graduation_year', 'is_current')
                           else 'normalized_exact_text'})
        return {'profile_index': profile_index, 'relations': relations, 'fields': fields}

    def _conflicts(self, relations, collection):
        if collection == 'jobs':
            # Start and terminal facts must both conflict. End/current facts are
            # alternatives describing the terminal state, not independent votes.
            return (relations['company'] == 'different' and relations['start_year'] == 'different'
                    and (relations['end_year'] == 'different' or relations['is_current'] == 'different'))
        return (relations['school'] == 'different' and relations['graduation_year'] == 'different'
                and relations['degree'] != 'equal' and relations['field'] != 'equal')

    def _supports_alternative(self, relations, collection):
        if collection == 'jobs':
            return (relations['company'] == relations['title'] == relations['start_year'] == 'equal'
                    and (relations['end_year'] == 'equal' or relations['is_current'] == 'equal')
                    and all(relations[field] != 'different' for field in ('start_year', 'end_year', 'is_current')))
        return (relations['school'] == relations['graduation_year'] == 'equal'
                and (relations['degree'] == 'equal' or relations['field'] == 'equal'))

    def _supports_assigned(self, relations, collection):
        if collection == 'jobs':
            terminal_equal = relations['end_year'] == 'equal' or relations['is_current'] == 'equal'
            return ((relations['company'] == relations['title'] == 'equal'
                     and (relations['start_year'] == 'equal' or terminal_equal))
                    or (relations['company'] == relations['start_year'] == 'equal' and terminal_equal)
                    or (relations['title'] == relations['start_year'] == 'equal' and terminal_equal))
        return ((relations['school'] == 'equal'
                 and any(relations[field] == 'equal' for field in ('degree', 'field', 'graduation_year')))
                or (relations['degree'] == relations['field'] == relations['graduation_year'] == 'equal'))
