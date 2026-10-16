# Title-equivalence shadow results

**Recommendation: retain the existing baseline for delivery. The reviewer merits further shadow testing, not an automatic veto.**

Frozen study `4222c5d5cf480f36d406d8bc3c08db633454cd53`, prompt `title-equivalence-v1`. Main `d50b42f2ecb4604c099eb46a208d1a0aa7645097`, PR33 `5ef43c88371e96edf58aca3ea01231969e690ca4` (still unmerged when tested). Production source unchanged; no public campaign or integration was run.

| Approach | Teaching-example status | Held-out status | All status | False MATCH on discrepancies | False MISMATCH on equivalents | Incorrect certainty on ambiguity | No final title / contract failure |
|---|---:|---:|---:|---:|---:|---:|---:|
| Main comparator/reflection | 12/27 | 13/18 | 25/45 | 9 | 0 | 8 | 3 |
| PR33 paired replay | 17/27 | 13/18 | 30/45 | 4 | 0 | 8 | 3 |
| Independent reviewer | 27/27 | 17/18 | 44/45 | 0 | 0 | 1 | 0 |

Errors remain in denominators. Teaching examples are not independent generalization. These 45 decisions cover 25 constructed cases in one shared saved profile context; held-out means absent from this prompt, including close paraphrases and previously observed reverse combinations, not unseen CVs.

## Grounding and failure analysis

Existing structured controls `ambiguous_record_details.jobs.0.title` and `additional_fabricated_job.jobs.2.title` are independently unsupported/ambiguous correspondences. Both produce NOT_EVALUATED with zero model calls, separately from the 45 title decisions. No selected record is fabricated and these skips are not counted as UNKNOWN accuracy.
Main and PR33 have zero wrong-record title citations among returned findings. Three samples fail comparison contracts after bounded recovery because the model cites null seniority: Associate Director sample2, Head of Sales sample2, and the missing-grade case. Main also rejects one reflection correction on Director sample1 and preserves its validated baseline; PR33 skips that audit after its deterministic negative. No schema/evidence constraint was weakened.

The reviewer has no output-contract/provider failures. Its bounded reason vocabulary prevents novel employer/duty assertions, but does not prove semantic correctness. Predeclared principle-basis grounding is 43/45 overall and 16/18 held-out: missing-grade status is wrong, and the retail alias selects the qualifier principle instead of the applicable alias principle. The latter is a rationale-basis mismatch, not a false discrepancy or invented employer fact. Baseline canonical/correspondence checks are a different grounding test; do not treat these as identical rationale audits.

- **Missing grade:** Senior Scientist versus Scientist, profile seniority null. Expected UNKNOWN. Reviewer MATCH with the same-base qualifier principle is unsupported: absence of grade is neither confirmation nor contradiction. Main/PR33 instead fail the evidence contract on this input. This is a substantive reviewer miss, retained without prompt revision.
- **Known substitutions:** main raw comparator returns MATCH in all ten primary samples. Reflection catches Senior Analyst once (1/5), never Director. PR33 corrects Director5/5 and leaves Senior Analyst1/5; reviewer corrects both5/5. Both pairs are explicit teaching examples, so this tests compliance, not generalization.
- **Held-out gain:** every approach gets all 13 known-label held-out samples right (nine equivalent, four discrepant). Reviewer improves four ambiguity samples (Managing Director and three Head of Operations); its fifth held-out ambiguity sample, missing grade, is wrong. There is no demonstrated false-MISMATCH advantage on unseen equivalents: all three have zero here.
- **Uncertainty:** UNKNOWN is a title abstention, not a CV score or a fabricated discrepancy. The reviewer is never used to turn a validated positive into a 0.5 negative.

## Calls, tokens and latency

| Approach | Logical invocations | New API calls | New input/output tokens | New total tokens | Replayed token equivalents | Wall seconds | Summed case seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 84 | 84 | 226,562 / 41,850 | 268,412 | 0 | 71.09 | 202.38 |
| B | 79 | 0 | 0 / 0 | 0 | 251,829 | 0.48 | 0.97 |
| C | 45 | 45 | 48,295 / 1,139 | 49,434 | 0 | 13.57 | 39.94 |

All 79 consumed B message/response pairs match A byte-for-byte; five recorded reflection calls are skipped by the guard. B is an actual component replay, not an independent live draw. Its CPU timing is not network performance and its reused token equivalents are not new billing. Actual API usage is A+C; tokens are a cost proxy, not a dollar estimate. C's compact one-call timing does not mean adding it would speed up the current pipeline.

## All repeated outcomes

| Case | Samples | Main | PR33 | Reviewer |
|---|---:|---|---|---|
| manager_abbreviation | 3 | {'MATCH': 3} | {'MATCH': 3} | {'MATCH': 3} |
| senior_manager_grade | 3 | {'MATCH': 3} | {'MATCH': 3} | {'MATCH': 3} |
| director_manager | 5 | {'MATCH': 5} | {'MISMATCH': 5} | {'MISMATCH': 5} |
| senior_analyst_manager | 5 | {'MATCH': 4, 'MISMATCH': 1} | {'MATCH': 4, 'MISMATCH': 1} | {'MISMATCH': 5} |
| associate_director_mapping | 3 | {'MISMATCH': 1, 'ERROR': 1, 'MATCH': 1} | {'MISMATCH': 1, 'ERROR': 1, 'MATCH': 1} | {'UNKNOWN': 3} |
| head_sales_mapping | 3 | {'MATCH': 1, 'ERROR': 1, 'MISMATCH': 1} | {'MATCH': 1, 'ERROR': 1, 'MISMATCH': 1} | {'UNKNOWN': 3} |
| cfo_alias | 3 | {'MATCH': 3} | {'MATCH': 3} | {'MATCH': 3} |
| head_operations_mapping | 3 | {'MATCH': 3} | {'MATCH': 3} | {'UNKNOWN': 3} |

Captured initial comparator and reviewer messages are byte-identical across repeats for every case. Production reflection naturally depends on its previous model response. Reviewer statuses are stable on all eight repeated inputs. That consistency covers only the declared few-shot/near-shot examples; a one-sample error cannot establish its frequency.

## Every sample

| Sample | Split | Expected | Raw comparator attempts | Main final | PR33 final | Reviewer | Reviewer basis correct |
|---|---|---|---|---|---|---|---|
| manager_exact__1 | prompt_example | MATCH | MATCH | MATCH | MATCH | MATCH | True |
| manager_abbreviation__1 | prompt_example | MATCH | MATCH | MATCH | MATCH | MATCH | True |
| manager_abbreviation__2 | prompt_example | MATCH | MATCH | MATCH | MATCH | MATCH | True |
| manager_abbreviation__3 | prompt_example | MATCH | MATCH | MATCH | MATCH | MATCH | True |
| senior_manager_grade__1 | prompt_example | MATCH | MATCH | MATCH | MATCH | MATCH | True |
| senior_manager_grade__2 | prompt_example | MATCH | MATCH | MATCH | MATCH | MATCH | True |
| senior_manager_grade__3 | prompt_example | MATCH | MATCH | MATCH | MATCH | MATCH | True |
| manager_extra_grade__1 | prompt_example | MATCH | MATCH | MATCH | MATCH | MATCH | True |
| senior_manager_junior__1 | prompt_example | MISMATCH | MATCH | MISMATCH | MISMATCH | MISMATCH | True |
| director_manager__1 | prompt_example | MISMATCH | MATCH | MATCH | MISMATCH | MISMATCH | True |
| director_manager__2 | prompt_example | MISMATCH | MATCH | MATCH | MISMATCH | MISMATCH | True |
| director_manager__3 | prompt_example | MISMATCH | MATCH | MATCH | MISMATCH | MISMATCH | True |
| director_manager__4 | prompt_example | MISMATCH | MATCH | MATCH | MISMATCH | MISMATCH | True |
| director_manager__5 | prompt_example | MISMATCH | MATCH | MATCH | MISMATCH | MISMATCH | True |
| senior_analyst_manager__1 | prompt_example | MISMATCH | MATCH | MATCH | MATCH | MISMATCH | True |
| senior_analyst_manager__2 | prompt_example | MISMATCH | MATCH | MISMATCH | MISMATCH | MISMATCH | True |
| senior_analyst_manager__3 | prompt_example | MISMATCH | MATCH | MATCH | MATCH | MISMATCH | True |
| senior_analyst_manager__4 | prompt_example | MISMATCH | MATCH | MATCH | MATCH | MISMATCH | True |
| senior_analyst_manager__5 | prompt_example | MISMATCH | MATCH | MATCH | MATCH | MISMATCH | True |
| senior_analyst_junior__1 | prompt_example | MISMATCH | MISMATCH | MISMATCH | MISMATCH | MISMATCH | True |
| director_exact__1 | prompt_example | MATCH | MATCH | MATCH | MATCH | MATCH | True |
| associate_director_mapping__1 | prompt_example | UNKNOWN | MATCH,MATCH | MISMATCH | MISMATCH | UNKNOWN | True |
| associate_director_mapping__2 | prompt_example | UNKNOWN | MATCH,MATCH | ERROR | ERROR | UNKNOWN | True |
| associate_director_mapping__3 | prompt_example | UNKNOWN | MATCH | MATCH | MATCH | UNKNOWN | True |
| head_sales_mapping__1 | prompt_example | UNKNOWN | MATCH | MATCH | MATCH | UNKNOWN | True |
| head_sales_mapping__2 | prompt_example | UNKNOWN | MATCH,MATCH | ERROR | ERROR | UNKNOWN | True |
| head_sales_mapping__3 | prompt_example | UNKNOWN | MISMATCH | MISMATCH | MISMATCH | UNKNOWN | True |
| scientist_qualifier__1 | held_out | MATCH | MATCH | MATCH | MATCH | MATCH | True |
| analyst_extra_junior__1 | held_out | MATCH | MATCH | MATCH | MATCH | MATCH | True |
| engineer_inflation__1 | held_out | MISMATCH | MATCH | MISMATCH | MISMATCH | MISMATCH | True |
| manager_director__1 | held_out | MISMATCH | MISMATCH | MISMATCH | MISMATCH | MISMATCH | True |
| cfo_alias__1 | held_out | MATCH | MATCH | MATCH | MATCH | MATCH | True |
| cfo_alias__2 | held_out | MATCH | MATCH | MATCH | MATCH | MATCH | True |
| cfo_alias__3 | held_out | MATCH | MATCH | MATCH | MATCH | MATCH | True |
| retail_alias__1 | held_out | MATCH | MATCH | MATCH | MATCH | MATCH | False |
| nurse_alias__1 | held_out | MATCH | MATCH | MATCH | MATCH | MATCH | True |
| different_occupations__1 | held_out | MISMATCH | MISMATCH | MISMATCH | MISMATCH | MISMATCH | True |
| managing_director_mapping__1 | held_out | UNKNOWN | MATCH | MATCH | MATCH | UNKNOWN | True |
| head_operations_mapping__1 | held_out | UNKNOWN | MATCH | MATCH | MATCH | UNKNOWN | True |
| head_operations_mapping__2 | held_out | UNKNOWN | MATCH | MATCH | MATCH | UNKNOWN | True |
| head_operations_mapping__3 | held_out | UNKNOWN | MATCH | MATCH | MATCH | UNKNOWN | True |
| associate_director_exact__1 | held_out | MATCH | MATCH | MATCH | MATCH | MATCH | True |
| data_analyst_qualifier__1 | held_out | MATCH | MATCH | MATCH | MATCH | MATCH | True |
| different_it_roles__1 | held_out | MISMATCH | MISMATCH | MISMATCH | MISMATCH | MISMATCH | True |
| missing_grade__1 | held_out | UNKNOWN | MATCH,MATCH | ERROR | ERROR | MATCH | False |

See `results.json` for each actual baseline score, canonical title evidence, errors, reviewer principle, binding, token use and case runtime. Full prompts/responses are ignored under the original study directory.

## Integration design only

| Option | Trigger/budget | Disagreement, UNKNOWN and failure policy | Evidence and authority |
|---|---|---|---|
| Shadow diagnostics (recommended now) | Only independently reviewed title records; fixed evaluation budget | Log every result; never change score | Useful for more diverse blind data and missing-grade analysis; no production authority |
| Optional audit of questionable MATCH | Design cap one title call/CV, only uniquely corroborated records; triggers require a separately reviewed policy | MATCH leaves baseline; UNKNOWN/errors/timeouts preserve baseline; MISMATCH is a flagged proposal | Same model/evidence is not an independent fact. Automatic veto is not justified by this small set |
| Existing positive-only reflection seam | If eventually approved, replace/restrict the title portion of an existing audit rather than stack unbounded votes; explicit timeout and one-call bound | Only a supported MATCH-to-MISMATCH proposal; no UNKNOWN downgrade or retries/fallback agents; preserve baseline on failure | Python reconstructs canonical selected-record fields, full original validators run; no record realignment or invented evidence. New held-out false-positive evidence is needed first |

A possible future trigger is a positive baseline with a title MATCH on a uniquely corroborated employer/period, where normalized base roles differ or an explicit qualifier lacks matching metadata. That is only a review trigger, never a string-based discrepancy rule. Candidate policy and ordering across multiple jobs need separate review; cap at one call per CV and skip when the remaining per-CV deadline cannot accommodate the bounded call. Do not review unsupported correspondence.

Replacing existing reflection outright would lose its non-title audits; preserving them while adding a specialist increases calls. A safe integration study must measure that trade-off rather than silently remove coverage. UNKNOWN, timeout, malformed output and disagreement preserve the baseline. A specialist MISMATCH is only a proposal, not sufficient override authority from this experiment: canonical reconstruction proves supplied provenance, not semantic truth.

One false discrepancy can reject a valid CV. Both reviewers use the same DeepSeek model and the same supplied facts; hiding the comparator output prevents anchoring but does not create independent knowledge. Agreement is not a majority-vote proof. Do not auto-promote a fluent MISMATCH over a validated MATCH. Any integration needs separate authorization, correspondence confidence, a measured latency/call budget, missing-evidence controls and broader independently labeled aliases.

## Limits and reproducibility

One prompt version, one declared run, no resampling or optimization. Labels and acceptable reason principles were frozen before execution. Cases are policy judgments, not a universal title taxonomy; ambiguous mappings are UNKNOWN. The retail alias label uses a narrow conventional lexical interpretation supported by the O*NET reference in README, not automatic equivalence of all titles in an occupation. No independent expert panel or private-set evaluation was conducted.

The class/tests, dataset/plan and actual production-worker source are committed under `evaluation/` only. Raw plan, worker outputs, full requests/responses, runtime logs, frozen source checkout and hashes are ignored. Run the study command in README in a new directory to create a transparently separate run; an existing plan cannot be overwritten. Rerun this report against saved captures without new API calls. Stop for review; do not merge or integrate into scoring.
