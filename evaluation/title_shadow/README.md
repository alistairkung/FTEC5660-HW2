# Title equivalence shadow study

Development-only. Nothing here is imported by `hw2.py` or `cv_verifier/` and no reviewer decision changes a production score.

## Frozen plan v1

- A: actual main `d50b42f2ecb4604c099eb46a208d1a0aa7645097` comparator, validation and conditional reflection in an isolated checkout.
- B: actual PR33 production components, byte-identical to `5ef43c88371e96edf58aca3ea01231969e690ca4`, with A's recorded responses. Consumed model messages must match exactly. Conditional reflection can be skipped; no paid call or independent LLM vote is added in B.
- C: independent `TitleEquivalenceReviewer`, `deepseek-v4-flash`, thinking disabled, JSON mode, SDK retries0, model timeout60s, one call/sample. The five-field input has no comparator judgment, rationale, score, label, case ID, or unrelated claims.

There are 25 cases: 11 exact teaching examples and 14 held out from this prompt. The two observed substitutions receive five samples each. Manager/Mgr., Senior Manager/Manager-senior, Associate Director/Senior Manager, Head of Sales/Sales Manager, held-out CFO and held-out Head of Operations receive three each. The remaining cases receive one: **45 samples total**, 27 prompt-example and 18 held-out. Repeated inference data is unchanged; actual production reflection naturally also depends on its prior model output. No prompt optimization or public campaign is planned.

Titles are controlled edits of one saved PR31 context, not 25 independent people or unseen CVs. Correspondence reuses PR31's investigated employer/period identity and checks a frozen digest of all non-title facts; record order is irrelevant. Changed/duplicate/unreviewed correspondence is `NOT_EVALUATED`, not a title discrepancy. No generalized alignment system is introduced.

Labels and their justifications are in `cases.py` and never passed to inference. Exact teaching examples are not generalization evidence. Held-out cases include close paraphrases, reverse substitutions and novel lexical combinations; they are not a blind external benchmark. The conventional retail worker alias is supported by [O*NET alternate titles for Retail Salespersons](https://www.onetonline.org/find/score/41-2031.00?s=Retail+Sales+Representative). This terminology reference justifies an offline label; it is not new candidate evidence or a rule that every title in an occupation is equivalent.

The reason vocabulary prevents invented employer/responsibility prose by construction. Selecting an inapplicable principle or an incorrect status remains a semantic error; JSON validity does not prove judgment truth. Predeclared applicable principles permit a separate rationale-basis check. Fixed explanations trade free-form diagnostic detail for an enforceable evidence boundary.

## Reproduction

```bash
python3 -m unittest discover -s evaluation/title_shadow -p 'test_*.py' -v
python3 -m evaluation.title_shadow.run --output evaluation/results/title-shadow/study-v1
```

An existing plan cannot be overwritten. The runner freezes hashes before calls, retains every record atomically, and uses the existing environment/keys without logging credentials. All full prompts/responses and runtime artifacts stay ignored. Production-worker subprocesses import each checkout's actual components; no production class is monkeypatched or replaced. B replay timing is not network latency and its token equivalents are not newly billed usage.

Do not run another public acceptance campaign for this shadow experiment. Results and integration proposals will be recorded here after the single declared run; production integration is outside scope.
