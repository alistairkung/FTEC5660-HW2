# Final skill equivalence experiment

Recommendation: retain the narrowly scoped specialist for review. Public campaign: 69/70, with the one failure independently reproduced on frozen main. Do not merge automatically.

## Baseline and scope

- Fetched main: `060dc1580f198d9910f12afbe5444cd3363d8c2d`; PR #33 and PR #35 confirmed merged. Production freeze: `65f0ecb`.
- Experimental branch: `skill-equivalence-experiment`. Local main ref remains unchanged; no PR #32 changes imported. Task 2, comparator prompt, title reviewer, reflection implementation, runner and existing global budgets are unchanged.
- Original source archive, tracked-file hashes and pre-existing acceptance snapshots preserved under ignored `evaluation/results/skill-equivalence/baseline/`. Known-good PR #35 campaign had 70/70 labels and 30 grounded negatives. The later merged-main snapshot captured eight runs; its first CV7 result was 0.5, which is a correct negative label but not a verified discrepancy. No baseline result is overwritten.

## Evidence policy and correction boundary

A supplied non-null LinkedIn skills list is authoritative for explicitly claimed CV skills. Exact skills and conventional aliases match; absent explicit skills mismatch. Null/missing collections are unknown; empty lists support absence. Ambiguous claims/relationships remain unknown. Extra profile skills never contradict omissions from a CV. No skills are inferred from occupations, descriptions or employers.

The reviewer receives only raw claim, structured claim and the complete original skills collection. Each batch decision has exactly `status`, `matched_skill_index`, `reason`; Python checks structure, index and permitted evidence-only reason. Canonical MATCH evidence is `skills.<index>.name`; absence cites the complete permitted `skills` collection, projected to names by the existing validator. A valid index guarantees source membership, not semantic equivalence. The LLM owns the latter judgment. Exact supplied names cannot support an absence decision.

The initial validated comparison is followed by at most one optional skill batch, then existing scoring/reflection/title handling. Only UNKNOWN and non-exact MATCH skills are eligible; existing MISMATCH findings are preserved without exception. UNKNOWN, invalid output, provider failures and cancellation preserve the entire previous validated batch. Corrections have no direct score authority. The call is capped at 15 seconds, with 121 seconds reserved for existing reflection (60), title audit (60), and finalization (1); a late CV skips it. No global retry, identity or discovery budget changed.

## Frozen shadow study

Prompt `skill-equivalence-v1`, SHA256 `81cedbaf8b69988944ff8184de682f96f2cd4896f1bbc21ae8548c7ce6d3933a`. Cases, runner and prompt were hashed before inference. Three repetitions of 25 cases, grouped into batches of five, concurrency three, no model retries; `deepseek-v4-flash`, thinking disabled. Inputs contain no comparator verdict, rationale, score or label.

A protocol setup mistake omitted the literal word JSON required by provider JSON mode: all 15 initial batches were rejected with HTTP400 before inference. Every rejection is retained. The only prompt repair added that word; there was no semantic tuning or resampling of model judgments. A prior local fixture assertion was also corrected to reference the actual saved Docker UNKNOWN; that failed before any network calls.

| Split | Fixtures | Judgments | Correct |
| --- | ---: | ---: | ---: |
| Exact teaching examples | 9 | 27 | 27 |
| Saved cases with taught relationships | 2 | 6 | 6 |
| Held-out combinations | 14 | 42 | 42 |

One held-out combination repeats the taught JavaScript/JS equivalence with a nonzero index and extra skills; it tests evidence selection, not a new alias. Excluding it leaves 39/39 other held-out judgments. Saved Docker is taken from the actual PR35 comparator UNKNOWN and selected MCP profile. Saved UI/UX uses the real CV2 claim/profile; the paired structured suite below also reproduces an actual UI/UX UNKNOWN.

| Case | Split | Expected | Correct / samples |
| --- | --- | --- | ---: |
| teaching_0 | teaching | MATCH | 3/3 |
| teaching_1 | teaching | MATCH | 3/3 |
| teaching_2 | teaching | MATCH | 3/3 |
| teaching_3 | teaching | MATCH | 3/3 |
| teaching_4 | teaching | MISMATCH | 3/3 |
| teaching_5 | teaching | MISMATCH | 3/3 |
| teaching_6 | teaching | MATCH | 3/3 |
| teaching_7 | teaching | UNKNOWN | 3/3 |
| teaching_8 | teaching | MISMATCH | 3/3 |
| saved_uiux | seen_relation | MATCH | 3/3 |
| saved_docker | seen_relation | MISMATCH | 3/3 |
| js_nonzero_extra | held_out | MATCH | 3/3 |
| k8s_alias | held_out | MATCH | 3/3 |
| node_alias | held_out | MATCH | 3/3 |
| excel_vendor | held_out | MATCH | 3/3 |
| python_wording | held_out | MATCH | 3/3 |
| java_vs_js | held_out | MISMATCH | 3/3 |
| react_vs_native | held_out | MISMATCH | 3/3 |
| mysql_vs_postgres | held_out | MISMATCH | 3/3 |
| docker_vs_terraform | held_out | MISMATCH | 3/3 |
| ambiguous_pm | held_out | UNKNOWN | 3/3 |
| ambiguous_generic | held_out | UNKNOWN | 3/3 |
| missing_python | held_out | UNKNOWN | 3/3 |
| empty_git | held_out | MISMATCH | 3/3 |
| exact_extra | held_out | MATCH | 3/3 |

Correct statuses: 36 MATCH, 27 MISMATCH, 12 UNKNOWN. False MISMATCH on aliases: 0; false MATCH on distinct technologies: 0; certainty on ambiguous cases: 0; contract failures after protocol repair: 0. All 75 judgments and all batch input/output/usage metadata are retained. The frozen gate passed without further prompt tuning.

Measured shadow cost: 15 inference calls; 19,191 input + 2,325 output = 21,516 tokens; median batch latency 1.759s, maximum 2.071s, cumulative call time 26.187s, wall time 9.157s. Token usage for provider-rejected protocol requests was not returned; no zero billing claim is made.

## Integration and controlled regression

Offline tests: baseline 197 production + 12 harness tests; integrated 214 production + 12 harness tests. The 17 added tests cover canonical corrections, alias/technology distinctions, exact-name contradiction rejection, null/empty collections, minimal model context, malformed indices/keys/counts, protected discrepancies, unchanged unrelated claims, one atomic batch, service errors, time reservation and actual cancellation. Seven evaluation-only model-contract checks also passed. Standards and spec reviews independently found no material issues in the production commit.

The full CVVerifier runs use the same captured initial comparator responses for frozen main and integration. Exact downstream prompts are replayed; changed downstream prompts make fresh calls and are reported separately. Fixed saved identities mean discovery/identity accuracy is not evaluated by these controls. The provided suite and grading code remain unchanged.

| Structured metric | Frozen main | Integrated |
| --- | ---: | ---: |
| Known binary labels | 22/23 | 23/23 |
| Target findings | 35/35 | 35/35 |
| Catalog findings | 442/446 | 443/446 |
| UNKNOWN findings | 6 | 5 |
| Uncertainty controls | 2/2 | 2/2 |
| Off-anchor citations | 0 | 0 |

The only status/score change is `bsc_alias` / `skills.0.name`: UI/UX Design versus UI/UX changes UNKNOWN to MATCH, 0.5 to 0.9. No existing MISMATCH is lost; no employment, education, city or identity finding changes status. Docker is already MISMATCH in this particular captured main comparison, so its live repair evidence comes from the shadow saved UNKNOWN and deterministic integration tests rather than an invented paired improvement.

The 25-case integration made six skill calls (6,901 input + 228 output = 7,129 tokens, cumulative 7.928s, maximum 1.742s) and five fresh downstream reflection calls because their input findings changed. One fresh reflection on `ambiguous_city_header` proposed an invalid unrelated title correction; canonical validation rejected it, preserving the corrected findings and 0.9. This is a newly observed downstream recovery, not a clean-error-free result and not an unsafe skill correction.

All 45 title samples are unchanged against main: 34/45 overall and 13/18 held-out. No skill calls were eligible there; all upstream/downstream responses replayed identically. Three inherited comparison-contract failures (`associate_director_mapping__2`, `head_sales_mapping__2`, `missing_grade__1`) produce 0.5 with no findings. These are unresolved inherited failures, not verified discrepancies or new invalid citations. The remaining non-correct title judgments also remain unresolved. No claim of perfect title adjudication is made.

## Public acceptance

One smoke and exactly ten consecutive runs executed `python3 hw2.py --cv-folder public_test` on frozen production commit `65f0ecb`, with the existing virtual environment on PATH. Expected scores: `0.9, 0.9, 0.9, 0.1, 0.1, 0.9, 0.1`. No source, prompt, dependency or input changes occurred during the campaign; every tracked-file hash and commit was checked after each run. There were no replacement runs.

| Run | Correct labels | Grounded negatives | Half scores | Skill calls / canonical replacements | Seconds | Total tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Smoke | 7/7 | 3 | 0 | 2 / 2 | 51.20 | 84,368 |
| 1 | 7/7 | 3 | 0 | 2 / 2 | 52.57 | 82,544 |
| 2 | 7/7 | 3 | 0 | 2 / 2 | 47.07 | 83,916 |
| 3 | 7/7 | 3 | 0 | 2 / 2 | 45.26 | 81,673 |
| 4 | 7/7 | 3 | 0 | 3 / 3 | 45.15 | 85,886 |
| 5 | 7/7 | 3 | 0 | 2 / 2 | 44.50 | 84,483 |
| 6 | 7/7 | 3 | 0 | 2 / 2 | 42.75 | 85,246 |
| 7 | 7/7 | 3 | 0 | 2 / 2 | 46.32 | 82,802 |
| 8 | 7/7 | 3 | 0 | 2 / 2 | 43.08 | 81,287 |
| 9 | 7/7 | 3 | 0 | 2 / 2 | 43.56 | 85,457 |
| 10 | 6/7 | 3 | 1 | 2 / 2 | 42.74 | 78,905 |

Campaign total: **69/70 correct labels**, **30/30 grounded negative decisions**, **9/10 exact-score runs**, one unexpected 0.5, zero wrong final identities, zero unsupported citations, zero skill-review errors and zero unsafe specialist corrections. Smoke is reported separately: 7/7 with three grounded negatives. Skill review made 21 calls/canonical replacements during the campaign, plus two during smoke. Replacements include reaffirmed MATCH decisions and must not all be described as repaired errors. Run4 also establishes the CV7 Docker MISMATCH from a previously eligible UNKNOWN/MATCH finding, with canonical full-list absence evidence; the literal CLI does not retain the pre-correction status.

Complete warnings/recoveries:

- Smoke and run4: one comparator contract retry warning each (`Comparison output requires exactly a findings envelope`), followed by successful recovery.
- Run2 CV5: recovered identity JSON parse error; final identity and score correct.
- Runs5 and7 CV2: reflection proposed a title correction rejected by the existing validator (`Reflection must contradict the claimed title or explicit seniority`); correct findings/score preserved.
- Run8 CV3: recovered prioritization response with unknown candidate IDs; final identity and score correct.
- Run10 CV1: comparator returns `jobs.1.end_year = UNKNOWN`, falsely stating the selected Carousell record lacks an end year even though the original profile supplies 2018. Final score is 0.5, a false rejection, with no model exception. Skill review has no eligible claims and is not invoked; reflection/title review are not invoked either.

Offline replay of the captured run10 CV1 **validated findings** through actual frozen-main source and integrated source reproduces the identical findings and 0.5 with zero optional calls. This establishes that the scoring failure is outside the specialist path; it does not replay the original raw provider response, which the unchanged CLI does not expose. The false rejection is retained and not counted as a verified discrepancy. No extra model calls, retries, prompt tuning or unrelated fix were used to rescue it. Thus public acceptance is not an unqualified 10/10 success, despite no attributable specialist regression.

## Costs and attribution

The ten-run public campaign used 832,199 total model tokens; median 83,359/run, median wall time 44.825s, maximum 52.566s. Smoke used another 84,368 tokens and 51.203s. These are aggregate CLI model-usage totals, including extraction, discovery-related model work, comparison and reflection; specialist-only public token/latency metadata is not exposed by the unchanged CLI. Direct specialist measurements are the shadow and fixed-input costs above. The only added production model stage is one batched call per eligible CV, no retries, capped at 15s.

For context, the preserved PR35 ten-run campaign had median 79,626.5 tokens and 41.691s; the eight-run merged-main snapshot had median 79,485.5 tokens and 46.361s. The new median token total is about 4.7% above PR35 and 4.9% above that main snapshot; wall time is about 7.5% above PR35 and 3.3% below that main snapshot. These end-to-end runs do not share upstream responses, so the differences are observations, not isolated causal overhead estimates. No timeout/budget exhaustion occurred. The six measured controlled skill calls total 7,129 tokens and max 1.742s each; five downstream reflection prompts changed, with their separate usage retained in raw results.

There is one newly observed downstream reflection recovery in the structured check and two in the public campaign. Skill corrections change the reflector's input, so those fresh outcomes are not represented as identical replay. The same guard failure is present in the saved merged-main public snapshot (CV3 run3), but these samples cannot establish an unchanged failure rate. All were contained by existing validation without score/finding corruption. The final CV1 failure has no specialist invocation at all. No attributable false alias rejection, fabricated-skill acceptance, evidence invention, discrepancy erasure, or material runtime regression was observed.

**Final recommendation: retain the small specialist for review.** The experiment supports the stated skill policy and produces actual UI/UX and Docker improvements, while preserving the known baseline under attributable checks. Keep the review-only PR unmerged until the user decides. The absolute public result remains 69/70; inherited comparator/title limitations remain unresolved. No further architecture experiments or Task1 tuning are proposed.

## Artifact locations and limits

All research drivers, frozen inputs, raw responses, protocol errors, manifests, logs and CSVs remain ignored in `evaluation/results/skill-equivalence/`. Source, tests and this concise summary are the reviewable deliverable. All persisted experiment files remain under HW2; shared planning is in the authorized workspace .scratch directory.

Public fixtures are a regression gate. Small repeated held-out controls support the stated skill policy, not general reliability on arbitrary CVs. Existing MISMATCH skills are deliberately protected, so this slice does not repair pre-existing false MISMATCH judgments. No further architecture experiments are proposed. Task 1 should freeze after the review decision; Task 2 is outside this change.
