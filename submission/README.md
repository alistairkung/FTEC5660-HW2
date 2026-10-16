# Task 1 submission and consolidation evidence

The executable submission is **`hw2.py` plus `cv_verifier/`**. The generated
`production_source.py` is a lecturer record, not the authoritative implementation.
The assignment PDF does not prescribe an aggregate filename/format; modular code
is permitted per the lecturer confirmation supplied with this task.

## Production architecture and dependency inventory

```mermaid
flowchart TD
    PDF[CV PDF / supplied loader] --> E[ClaimExtractor]
    E --> V[ClaimValidator]
    V --> P[CandidatePipeline]
    P --> R[ProfileRetriever / SocialGraph LinkedIn]
    R --> Q[CandidatePrioritizer / CandidateRankingValidator]
    Q --> I[IdentityResolver / IdentityDecisionValidator]
    I --> C[ClaimComparator]
    C --> D[ClaimComparisonValidator]
    D --> S[CVScorer]
    S -->|0.9 only| F[ClaimReflector / validated bounded audit]
    S -->|0.1 or 0.5| O[Final score]
    F --> O
```

`CVVerifier.ainvoke(cv)` is the primary orchestration method. Dependencies are
injected explicitly by `build_agent(tools)`; `score_cvs` runs at most three CVs
concurrently, applies the existing 180-second per-CV timeout, isolates failures
at 0.5, and writes atomic incremental checkpoints. The supplied runner loads
PDFs, grades and writes the final `results.csv`.

| Production module / class | Responsibility / injected collaborators |
|---|---|
| `claim_extractor` / `ClaimExtractor` | CV text → structured claims; extraction LLM |
| `claim_validator` / `ClaimValidator` | Validate extracted schema and locations |
| `profile_retriever` / `ProfileRetriever` | LinkedIn MCP search/profile retrieval; supplied tools |
| `candidate_prioritizer` / `CandidatePrioritizer` | Rank candidate investigation; LLM |
| `candidate_ranking_validator` / `CandidateRankingValidator` | Validate shortlist against search hits |
| `identity_resolver` / `IdentityResolver` | Resolve among retrieved profiles; LLM |
| `identity_decision_validator` / `IdentityDecisionValidator` | Validate candidate decision and profile IDs |
| `candidate_pipeline` / `CandidatePipeline` | Extract/validate, discover, rank, retrieve, resolve; seven collaborators above |
| `claim_comparator` / `ClaimComparator` | Semantic correspondence/comparison; LLM, canonical comparison contract |
| `claim_comparison_validator` / `ClaimComparisonValidator` | Reconstruct canonical claims/evidence, validate coverage/citations, city adjudication and reflection corrections |
| `cv_scorer` / `CVScorer` | 0.1 if any established mismatch; 0.5 if unresolved/unknown; otherwise 0.9 |
| `claim_reflector` / `ClaimReflector` | One positive-result audit; LLM, existing correction contract |
| `cv_verifier` / `CVVerifier` | Identity → comparison → validation/scoring → conditional reflection |

There are **13 production classes in 13 implementation modules**, plus empty
`__init__.py` and the `hw2.py` entry point. Production source totals **1,522 lines**
including `hw2.py`, comments and blank lines. The aggregate has **1,565 lines**.
Runtime imports from these files reach no evaluation/experimental implementation.
External libraries remain those in the unchanged `requirements.txt`.

All retained production files, including `hw2.py`, are byte-identical to baseline
`27267d7de72815e1c6a39c1f5cb2efc86ad2c60e` (PR #22, base
`task1-scoring-integration`). This preserves prompts, thinking-disabled
`deepseek-v4-flash`, existing extraction SDK retries (2), other model retries (0),
60-second request timeouts, search/retrieval limits, identity policy, validation,
reflection and scoring. No new verification stage or retry was added.

## Cleanup and recoverability

Four implementations moved from `cv_verifier/` to `evaluation/experimental/`:
`RecordAligner`, `EvidenceAligner` (with its validator),
`DeterministicClaimVerifier`, and `SemanticClaimResolver`. Only their comparison
contract import paths changed. Historical evaluation/shadow imports and tests
were updated; their behavior, fixtures and findings remain available.
`CorrespondencePlausibilityGate` stays in evaluation only. Historical adapters
remain in evaluation/development scripts, outside submission imports.

`HybridCVVerifier` and `JointRecordComparator` were never in the merged baseline:
PRs #24 and #23 were confirmed closed/unmerged. Their branches/history remain
recoverable; their code was not copied into this branch. No Facebook integration,
hybrid promotion, new agent stage, prompt redesign or Task 2 change was made.

## Extended evaluation: identical 25-case PR #18 catalog

| Metric | Baseline | Consolidated |
|---|---:|---:|
| Known-label binary accuracy | 23/23 (100%) | 23/23 (100%) |
| Target-claim accuracy | 33/35 (94.29%) | 32/35 (91.43%) |
| Full-catalog claim accuracy | 443/446 (99.33%) | 423/446 (94.84%) |
| Grounded claim accuracy | 442/446 (99.10%) | 422/446 (94.62%) |
| False-valid predictions | 0 | 0 |
| False-discrepant predictions | 0 | 0 |
| UNKNOWN findings | 6 | 8 |
| Contract / model errors | 0 / 0 | 1 / 0 |
| LLM calls | 38 | 39 |
| Input / output tokens | 94,622 / 21,030 | 97,772 / 21,135 |
| Total tokens | 115,652 | 118,907 |
| Wall runtime | 34.99s | 34.93s |
| Sum of per-case runtimes | 97.27s | 96.96s |
| Genuine uncertainty controls preserved | 2/2 | 2/2 |
| Cases with a MISMATCH finding | 9 | 9 |
| Independently grounded expected discrepancy | 8 | 8 |

Expected labels, target answers, completeness assumptions and correspondence
anchors are used only after model verification for grading. Full-catalog accuracy
uses all 446 catalog claims, including unaffected claims; target accuracy uses
35 explicitly designated claims. Grounded accuracy additionally checks indexed
record correspondence or explicitly justified collection absence using the
existing evaluation grader. Neither metric establishes semantic proof in general.

The consolidated `end_year_contradiction` response cited unavailable profile
evidence and failed the unchanged contract with “Missing profile values are not
supporting or contradictory evidence”. Its 0.5 fallback still meets the negative
binary label but **does not verify a discrepancy**. Other differences include
ambiguous fabricated-employer fields, a borrowed `is_current` citation on the
fabricated extra job, and unresolved fabricated-skill evidence. The baseline
also had an off-anchor employer citation and UNKNOWN fabricated-degree/skill
findings. These are retained model weaknesses, not tuned away.

Both live captures replay **exactly across all 25 cases** through the consolidated
modular classes and generated classes, including prompts, canonical evidence,
findings, scores and errors (100 case replays across two captures × two forms).
Production source hashes are unchanged. The independent-run differences therefore
come from model responses, not consolidation behavior. Captures are stored as
deterministic gzip JSON under `evaluation/fixtures/`; each includes prompts,
responses, usage, inputs, case hashes and grading for offline reproduction.

## Original public PDFs

Only `public_test/CV_1.pdf` through `CV_7.pdf` were used. All seven match the
original starter commit `7e9cf75` byte-for-byte; their SHA256 hashes are recorded
in [the machine-readable results](../evaluation/consolidation_results.json).
No synthetic/mutated/generated PDF was included. The wrapper calls the actual
`hw2.main()` CLI parser and supplied `run()` and delegates to the unchanged
`build_agent`, `score_cvs` and checkpoint writer. It records results without
passing labels, filenames or expected answers into verification.

### Public run 1

Accuracy **6/7**, wall runtime **33.87s**, seven checkpoints and exactly seven final CSV data rows.

| CV | Selected LinkedIn ID | Score | Binary | Negative basis | Per-CV seconds | Errors |
|---|---:|---:|---:|---|---:|---|
| CV_1.pdf | 6337 | 0.9 | 1 | — | 13.03 | none |
| CV_2.pdf | 47 | 0.9 | 1 | — | 13.98 | reflection |
| CV_3.pdf | 6541 | 0.9 | 1 | — | 13.25 | none |
| CV_4.pdf | 8235 | 0.1 | 0 | MISMATCH | 15.73 | none |
| CV_5.pdf | unresolved | 0.5 | 0 | failure_or_unresolved | 1.92 | extraction |
| CV_6.pdf | 8715 | 0.1 | 0 | MISMATCH | 17.01 | none |
| CV_7.pdf | 9590 | 0.5 | 0 | failure_or_unresolved | 16.99 | comparison |

CV_5 failed extraction location validation. CV_7 failed comparison evidence validation. CV_2 proposed an unsupported title/seniority reflection correction; the validator rejected it and preserved 0.9. CV_6 selected profile 8715 with different employers and scored 0.1 despite its valid public label; the other runs selected 6561 and returned 0.9.

### Public run 2

Accuracy **7/7**, wall runtime **48.69s**, seven checkpoints and exactly seven final CSV data rows.

| CV | Selected LinkedIn ID | Score | Binary | Negative basis | Per-CV seconds | Errors |
|---|---:|---:|---:|---|---:|---|
| CV_1.pdf | 6337 | 0.9 | 1 | — | 18.61 | none |
| CV_2.pdf | 47 | 0.9 | 1 | — | 18.64 | none |
| CV_3.pdf | 6541 | 0.9 | 1 | — | 17.67 | none |
| CV_4.pdf | 8235 | 0.1 | 0 | MISMATCH | 14.59 | none |
| CV_5.pdf | 1297 | 0.1 | 0 | MISMATCH | 16.23 | none |
| CV_6.pdf | 6561 | 0.9 | 1 | — | 15.89 | none |
| CV_7.pdf | unresolved | 0.5 | 0 | failure_or_unresolved | 14.96 | none |

CV_7 exhausted the existing eight-profile investigation budget without resolving identity; 0.5 is a fallback, not a verified discrepancy. No processing error was recorded.

### Public run 3 (fresh clone)

Accuracy **6/7**, wall runtime **49.27s**, seven checkpoints and exactly seven final CSV data rows.

| CV | Selected LinkedIn ID | Score | Binary | Negative basis | Per-CV seconds | Errors |
|---|---:|---:|---:|---|---:|---|
| CV_1.pdf | 6337 | 0.9 | 1 | — | 13.25 | none |
| CV_2.pdf | 47 | 0.9 | 1 | — | 13.63 | none |
| CV_3.pdf | unresolved | 0.5 | 0 | failure_or_unresolved | 15.45 | prioritization |
| CV_4.pdf | 8235 | 0.1 | 0 | MISMATCH | 12.63 | none |
| CV_5.pdf | 1297 | 0.1 | 0 | MISMATCH | 13.95 | none |
| CV_6.pdf | 6561 | 0.9 | 1 | — | 15.28 | none |
| CV_7.pdf | 9590 | 0.1 | 0 | MISMATCH | 14.76 | none |

CV_3's prioritizer returned an unknown candidate ID, which the ranking validator rejected. The existing fallback order then exhausted the eight-profile investigation budget before reaching profile 6541 and returned 0.5, misclassifying a valid CV. This was one recorded prioritization contract error. CV_7 selected 9590 and established MISMATCH findings rather than receiving a fallback.

Across these three runs, mean binary accuracy was **19/21 (90.48%)**. All per-CV verification times were below 180 seconds and all batches below 30 minutes. `MISMATCH` above means a validated pipeline finding against its selected profile; run 1 CV_6 shows that this alone does not guarantee correct identity. `failure_or_unresolved` denotes a 0.5 negative classification with no verified finding. There were no public UNKNOWN-only negative decisions in these runs.

Raw public diagnostics, including profiles, findings/errors, reflection and final CSV rows, are preserved under `evaluation/fixtures/public_run_*.json.gz`. [The machine-readable report](../evaluation/consolidation_results.json) contains all per-CV finding counts, errors, reflection outcomes and runtimes. Ground-truth labels remain outside verification prompts.

### Fresh clone

A fresh `git clone --no-hardlinks` of implementation commit `5dc0c7eecce909c5ffe4425a287eefaf6c3e1dda` was installed into an empty Python 3.14 virtual environment using the unchanged requirements. All 79 installed packages passed `uv pip check`; exact observed versions are retained in [the installation record](../evaluation/clean_clone_requirements.txt). The clone passed 246 implementation tests, 62 evaluation tests, aggregate freshness and CLI import/help checks, then executed public run 3 through the supplied runner. Its worktree stayed clean. The existing key was passed only in the subprocess environment; no `.env` was copied. Subsequent delivery changes contain report data/documentation only.

## Reproduce and submit

Use Python 3.10+ and an API key supplied through the environment or ignored `.env`:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python hw2.py --cv-folder public_test
.venv/bin/python -m unittest discover -s test -q
.venv/bin/python -m unittest discover -s evaluation -q
.venv/bin/python -m scripts.aggregate_source --check
# Regenerate only after changing authoritative modular source:
.venv/bin/python -m scripts.aggregate_source
# Fresh 25-case live run:
.venv/bin/python -m evaluation.run_production --output evaluation/results/production.json
# Offline captured-response parity (add --aggregate to check generated classes):
.venv/bin/python -m evaluation.run_production --replay evaluation/fixtures/production_baseline.json.gz --output evaluation/results/replay.json
# Observe the original seven-PDF supplied runner:
.venv/bin/python -m scripts.evaluate_public --output evaluation/results/public.json
```

Keep `hw2.py`, the whole `cv_verifier/` package, `requirements.txt` and report
in the submitted repository. Do not replace modular execution with the aggregate.
No key is included in artifacts. Baseline commit/branches remain recoverable;
the official base is unchanged until review and merge.

## Checks and remaining risks

- 246 implementation/submission tests and 62 historical evaluation tests pass.
  These exercise class boundaries, strict evidence handling, scoring, three-CV
  concurrency, timeout cancellation, failure isolation, incremental checkpoints,
  atomic replacement failure, final CSV and supplied runner integrity.
- Generated output is deterministic, compiles, imports without the modular
  package, contains no unresolved local import/duplicate class, matches all
  production definitions/constants by AST and embeds each source SHA256.
- `git diff --check` passes. No GitHub Actions workflow is configured in this
  repository (API workflow inventory empty), so there are no remote CI jobs to run.
- Identity selection and model contract failures can produce wrong results or
  0.5 fallbacks; binary accuracy must not be read as verified discrepancy coverage.
  Three public runs cannot establish unseen-input accuracy. No production policy
  was changed to improve these results.
- This delivery covers Task 1 consolidation only. Existing Task 2/Task 3 submission
  work remains outside this PR's scope.
