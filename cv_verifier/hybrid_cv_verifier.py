"""Direct production composition of the existing hybrid verification stages."""

from copy import deepcopy
from json import JSONDecodeError

from .identity_decision_validator import IdentityDecisionValidator


class HybridCVVerifier:
    def __init__(self, identity_pipeline, aligner, deterministic_verifier, semantic_resolver,
                 comparison_validator, scorer):
        self.identity_pipeline = identity_pipeline
        self.aligner = aligner
        self.deterministic_verifier = deterministic_verifier
        self.semantic_resolver = semantic_resolver
        self.comparison_validator = comparison_validator
        self.scorer = scorer

    async def ainvoke(self, cv):
        """Verify one selected identity; isolate failures and retain validated findings."""
        try:
            result = deepcopy(await self.identity_pipeline.ainvoke(cv))
        except Exception as error:
            return {'identity': 'unresolved', 'findings': [], 'score': 0.5,
                    'verification_status': 'processing_failure', 'verification_complete': False,
                    'errors': [{'stage': 'identity', 'kind': 'processing',
                                'message': str(error) or type(error).__name__}]}
        result.update(alignments=None, deterministic_findings=[], semantic_claim_ids=[], findings=[], score=0.5,
                      verification_status='identity_unresolved', verification_complete=False)
        stage = 'identity'
        try:
            profile = self._selected_profile(result)
            if profile is None:
                return result
            claims = result['claims']
            stage = 'alignment'
            result['alignments'] = await self.aligner.align(deepcopy(claims), deepcopy(profile))
            stage = 'deterministic'
            baseline = self.deterministic_verifier.verify(
                deepcopy(claims), deepcopy(profile), deepcopy(result['alignments']))
            stage = 'validation'
            baseline = self.comparison_validator.validate(baseline, claims, profile)
            result['deterministic_findings'] = deepcopy(baseline)
            result['findings'] = deepcopy(baseline)
            result['semantic_claim_ids'] = [f['claim_id'] for f in baseline if f['status'] == 'UNKNOWN']
            stage = 'semantic'
            merged = baseline
            if result['semantic_claim_ids']:
                merged = await self.semantic_resolver.resolve(
                    deepcopy(claims), deepcopy(profile), deepcopy(result['alignments']), deepcopy(baseline))
            stage = 'validation'
            final = self.comparison_validator.validate(merged, claims, profile)
            indexed = {f['claim_id']: f for f in final}
            if any(indexed[f['claim_id']] != f for f in baseline if f['status'] != 'UNKNOWN'):
                raise ValueError('Semantic resolution changed a definite deterministic finding')
            result['findings'] = final
            result['score'] = self.scorer.score(result['findings'], identity_resolved=True)
            result['verification_complete'] = True
            result['verification_status'] = (
                'verified_discrepancy' if result['score'] == 0.1 else
                'consistent' if result['score'] == 0.9 else 'unresolved_evidence')
        except Exception as error:
            if stage == 'identity':
                result.update(identity='unresolved', selected_candidate_id=None,
                              identity_reason='Identity selection could not be validated.')
            contract_failure = isinstance(error, ValueError) and not isinstance(error, JSONDecodeError)
            kind = 'contract' if contract_failure else 'format' if isinstance(error, JSONDecodeError) else 'processing'
            result['errors'].append({'stage': stage,
                                     'kind': kind,
                                     'message': str(error) or type(error).__name__})
            result['verification_status'] = 'processing_failure'
            # Service/JSON decoding failures cannot erase a validated discrepancy.
            # Parsed contract/evidence validation failures still fall back.
            if stage == 'semantic' and not contract_failure and any(
                    f['status'] == 'MISMATCH' for f in result['deterministic_findings']):
                result['score'] = self.scorer.score(result['deterministic_findings'], identity_resolved=True)
                result['verification_status'] = 'discrepancy_with_semantic_failure'
        return result

    def _selected_profile(self, result):
        if result['identity'] != 'resolved':
            return None
        decision = {'candidate_id': result['selected_candidate_id'], 'reason': result['identity_reason']}
        IdentityDecisionValidator().validate(decision, result['profiles'])
        profiles = [p for p in result['profiles'] if p['id'] == decision['candidate_id']]
        if decision['candidate_id'] is None or len(profiles) != 1:
            raise ValueError('Resolved identity must select exactly one retrieved profile')
        return profiles[0]
