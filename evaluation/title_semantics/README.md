# Director versus Manager: investigation only

**Finding:** the unchanged production verifier accepted an unsupported title equivalence. The comparator labeled **Director = Manager/senior**, the evidence validator accepted the structurally grounded MATCH, and the positive-result reflector returned no corrections. This reproduced in **5/5 fixed-input runs**, plus one separate PDF-to-live-identity run. It is a semantic verification failure in this controlled context, not an extraction or identity failure.

Source: `eea632e14952f7ce7ce91e026d10c11952a584d2`. The original failing PDF/log was not supplied or found locally, so its exact historical failure cannot be certified. The reproduction uses the retained CV_1 saved claims/profile. Production modules, prompts, existing tests and the protected runner remain byte-identical to that source and the current `main` baseline (`d50b42f`). This investigation-only PR adds this report, a small reproduction tool and curated evidence. Full runtime captures remain ignored. No correction is implemented.

**Relevant input and complete trace.** A separate PDF generated from CV_1 text replaced “Senior Manager” with “Director” and normalized layout/dashes. It stayed outside `public_test`. The trace uses the real PDF loader, `build_agent` and live MCP pipeline; fixed-upstream comparisons instead use immutable saved claims/profile copies.

| Stage | Captured evidence |
|---|---|
| PDF loading | Loaded text contains `Director, Deloitte 2020 - Present`. |
| Structured extraction | `jobs.0.title="Director"`, `seniority=null`, `company="Deloitte"`, `start_year=2020`, `end_year=null`, `is_current=true`; original wording preserved. |
| Discovery / identity | Shortlist and attempted IDs: `[6337, 8283, 23, 1494, 6478]`; selected **6337**. Resolver cites matching London, Carousell dates and Stanford MBA, explicitly allowing title-level discrepancies while identifying the person. |
| Correspondence | `experience.1`: Deloitte, **Manager**, seniority **senior**, 2020–Present. Correct employer, dates and current status; no wrong-record citation. |
| Raw comparator | `jobs.0.title`: **MATCH**, paths `experience.1.title`, `experience.1.seniority`; reason: “Director aligns with Manager at senior seniority.” |
| Reconstruction / adjudication | Canonical cited values remain **Manager** and **senior**. Title remains MATCH; no evidence-contract error. |
| Reflection | Actually invoked; returned exactly `{"corrections": []}`. |
| Scoring | All findings MATCH, resolved identity → **0.9**; reflection leaves it unchanged. No processing errors. |

The [curated evidence](evidence.json) includes the PDF trace's relevant claims, selected record, comparator decision and reflection response, plus the recorded comparison/reflection responses for offline replay. Full original inputs come from the existing saved fixture; raw prompts, logs and generated PDF remain local under ignored `evaluation/results/title-semantics/`. The PDF was a controlled fixture, not the original reported artifact.

**What is confirmed versus inferred.** `_title_parts` distinguishes `Director → (director, null)` from `Manager → (manager, null)`; it does not equate them. Extraction also does not collapse their seniority. The comparator itself makes the incorrect equivalence judgment. `_validate_finding` validates references, values, coverage and allowed evidence, not general title semantics: accurate citations do not guarantee the status is correct.

`adjudicate_explicit_grades` requires the same normalized base role and an explicit senior/junior/mid conflict. Director/Manager fails the base-role condition and has no explicit claimed grade, so that safeguard cannot intervene. In contrast, the Senior Analyst/junior control produced an incorrect raw MATCH that this safeguard correctly changed to MISMATCH.

The reflector omitted a correction; it was **not blocked from making one**. A separate deterministic capability probe fed the existing validator a hypothetical `jobs.0.title` MISMATCH citing `experience.1.title`: it accepted the correction and the existing scorer returned 0.1. This is not a model result or implemented fix. The deterministic capability check is described here; its raw output remains in the ignored investigation artifacts.

Over-generalization from the legitimate Senior Manager/Manager-senior prompt example, correspondence/title conflation and reflection anchoring are plausible mechanisms, not proven internal causes. Correct correspondence and the junior-grade probe rule out wrong-record selection and senior metadata alone as necessary explanations.

**Bounded controlled results.** All primary repetitions held the full saved claims, profile, Deloitte employer, dates and correspondence constant; only the original claimed title/raw wording changed. Expectations were used only offline. Five repeats were declared before sampling; none was discarded.

| Repeat | Comparator → validated title | Reflection correction | Score | Calls | Tokens | Seconds |
|---|---|---|---:|---:|---:|---:|
| 1 | MATCH → MATCH | none | 0.9 | 2 | 6,433 | 4.14 |
| 2 | MATCH → MATCH | none | 0.9 | 2 | 6,397 | 4.22 |
| 3 | MATCH → MATCH | none | 0.9 | 2 | 6,481 | 3.66 |
| 4 | MATCH → MATCH | none | 0.9 | 2 | 6,391 | 3.48 |
| 5 | MATCH → MATCH | none | 0.9 | 2 | 6,405 | 3.79 |

The remaining seven controls ran once each. Counterfactual profile titles/grades below are changes to isolated copies, not claims about the real profile.

| CV title | Supplied profile title / grade | Raw comparison → final title | Final score | Interpretation |
|---|---|---|---:|---|
| Manager | Manager / senior | MATCH → MATCH | 0.9 | Exact base role preserved. |
| Senior Manager | Manager / senior | MATCH → MATCH | 0.9 | Legitimate explicit-grade equivalence preserved. |
| Manager | Director / senior | MISMATCH → MISMATCH | 0.1 | Reverse substitution was not assumed equivalent. |
| Director | Director / senior | MATCH → MATCH | 0.9 | Exact title preserved. |
| Manager | Mgr. / senior | MATCH → MATCH | 0.9 | Abbreviation preserved. |
| Senior Analyst | Analyst / junior | MATCH → MISMATCH | 0.1 | Deterministic explicit-grade safeguard corrected the model. |
| Associate Director | Senior Manager / senior | MATCH → MISMATCH after reflection | 0.1 | Organization-specific mapping is unspecified; UNKNOWN may be appropriate. No universal binary oracle asserted. |

The last case needed one existing comparison contract retry for a missing-value citation. Its reflector later justified MISMATCH by the different titles alone, illustrating the false-positive risk of an overly literal correction policy. We cannot establish whether these organization-specific titles are equivalent from the supplied data; this case is not counted as proven correct or incorrect.

Three predeclared probes, once each: **Manager/junior** still accepted Director at 0.9; **null profile grade** exhausted both existing comparison attempts by citing null seniority and finished at **0.5 with a comparison error**, not a verified discrepancy; a **reduced name-plus-one-job** input correctly rejected Director at **0.1**. Five full-context failures do not establish a universal error rate. The single reduced sample cannot separate context effects from stochastic variation; minimization did not yield a smaller failing fixture.

Scratch tooling also had an import setup error before model calls and a display error after saving the null-grade failure. Only unexecuted cases were resumed; all logs and failed outcomes are retained.

**Regression evidence on unchanged production.**

| Check | Observed result |
|---|---|
| Original seven PDFs, literal CLI once | **7/7**, expected `0.9,0.9,0.9,0.1,0.1,0.9,0.1`; correct identities; three grounded MISMATCH outcomes; zero errors/UNKNOWN/fallbacks. |
| Existing 25 controls | **23/23 binary**, **34/35 targets**, **444/446 full-catalog grounded claim judgments**; zero wrong-record citations or processing failures; both genuine uncertainty controls retained correctly. |
| Known-label negative fallback | `fabricated_degree` returned UNKNOWN/0.5 instead of verified MISMATCH: binary correct does not establish successful verification. |
| Other catalog error | `fabricated_employer` also marked its otherwise valid title MISMATCH; expected company discrepancy remained correctly negative. |
| Retained unit suites | **167 production + 12 evaluation tests passed**, including title aliases, unclaimed seniority, actual role differences, explicit-grade conflicts and correspondence safeguards. No existing Director/Manager model-semantic control was present. |

The prior two public acceptance campaigns remain historical public-set evidence; these seven PDFs do not test this role-substitution gap. No proposed correction has been implemented or evaluated, so these results cannot be described as post-fix regression success.

| Experiment | Application model calls | Tokens | Wall seconds |
|---|---:|---:|---:|
| Twelve fixed-upstream title cases | 23 | 73,454 | 47.98, sequential |
| Three additional probes | 5 | 14,370 | 13.31, sequential |
| One controlled PDF/live identity trace | 5 | 12,100 | 13.61 |
| Existing 25 controls | 37 | 116,521 | 32.55, concurrency 3 |
| Original public CLI | Not exposed | 79,996 | 37, rounded |

Calls include existing application contract retries; SDK transport attempts are not counted. The 25-case summed per-case runtime was 88.98 seconds. Environment/model stayed the prior clean grading environment: DeepSeek v4 flash, thinking disabled, existing timeouts/retries; no evidence beyond supplied SocialGraph data/copies. The [curated evidence](evidence.json) contains per-case outcomes, relevant model decisions/citations and the aggregate baseline metrics. Full raw captures remain local and excluded from this PR.

**Recommendation, proposal only.** Reopen a narrowly scoped Task 1 semantic correction for review; this is a reproducible uncovered false-valid pattern that could affect unseen title-inflation cases. It does not justify a new architecture or any claimed private-set improvement. Keep the accepted implementation as the rollback baseline.

Start with a small **comparator instruction clarification**: matching employer/dates identifies the record, not the truth of its title; a shared managerial role or seniority category does not establish two different professional titles as aliases. Compare the specific role title separately, preserve established abbreviations and explicit-grade equivalents, flag clear non-equivalent substitutions, and use UNKNOWN when an organization-specific mapping is genuinely unresolved. This is the only existing stage that can appropriately originate such UNKNOWN findings; positive-only reflection accepts concrete MISMATCH corrections and cannot downgrade to uncertainty.

| Option | Assessment / false-positive risk |
|---|---|
| Comparator clarification | Recommended first, tested separately before adoption; risk of over-rejecting legitimate professional synonyms or internal/external titles. Preserve Manager/Mgr., Senior Manager/Manager-senior, and ambiguity controls. |
| Reflector clarification | Possible follow-up to audit concrete role substitutions independently of baseline reasoning; same semantic/anchoring risks, and no UNKNOWN downgrade under its existing contract. |
| Deterministic Director > Manager rule | Not recommended from current evidence. Employer/dates can establish correspondence but not a universal hierarchy or company-specific title mapping; no authoritative mapping is supplied. |
| Blanket different-title mismatch | Reject: string inequality is not proof of different work or rank. |

Any separately authorized fix needs these title controls, the 25 controls, seven originals, existing tests and new frozen acceptance. No post-fix accuracy or regression claim is currently justified.

For **Task 2**, this provides a concrete local attack hypothesis—role-title substitution can exploit our semantic-equivalence judgments. Five undisclosed verifiers may normalize titles, prompts and evidence differently; no transfer or success rate is established. Their actual outcomes must be measured separately.

**Reproduce from the repository root**, using the lecturer dependencies in `requirements.txt`:

```bash
# No API calls or credentials: replay recorded model decisions through the real
# production comparator, reflector, validators and scorer.
python3 -m evaluation.title_semantics.reproduce

# Optional new measurement: unchanged live production comparison/reflection.
# Loads the existing .env. Fixed upstream identity; no extraction/MCP stage.
python3 -m evaluation.title_semantics.reproduce --mode live --repeats 5 --controls
```

The default replay deterministically exits **1** on the recorded false-valid 0.9/MATCH; this is the intended demonstration, not a failed setup. Replay proves the recorded responses pass the actual contracts and scoring; it is **not a fresh semantic-reliability measurement**. Live mode is bounded to 1–5 primary samples plus seven optional controls, with no added verifier retries. It exits 1 if the false-valid outcome is observed, 2 for inconclusive errors/uncertainty when no false-valid outcome is observed, otherwise 0. Every outcome is saved to a fresh ignored `evaluation/results/title_semantics_*.json`; outputs never overwrite the captured evidence. Live behavior can vary.

Fixture/profile/source hashes are checked before execution. Grading expectations, historical results and other metadata never enter live model inputs. This tool is not imported by `hw2.py`, production components or retained tests. No production fix or automatic merge is included.
