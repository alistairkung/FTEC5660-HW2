"""Development-only batch diagnostics; never writes results.csv or retries CVs."""

import argparse
import asyncio
import json
import os
import tempfile
import time
from copy import deepcopy
from collections import Counter
from contextvars import ContextVar
from pathlib import Path

from hw2 import (
    MODEL_NAME, PER_CV_TIMEOUT_SECONDS, build_agent, correctness_text, cv_files, cv_text,
    load_env_file, load_mcp_tools, parse_score, read_ground_truth,
)

DEFAULT_OUTPUT_DIR = Path(".venv/diagnostics")
MAX_CONCURRENT_CVS = 3
_COMPARISON_CAPTURE = ContextVar("batch_comparison_capture", default=None)
_SHADOW_CAPTURE = ContextVar("batch_shadow_capture", default=None)


def _build_shadow_components():
    from langchain_deepseek import ChatDeepSeek
    from langchain_core.runnables import RunnableLambda
    from evaluation.experimental.record_aligner import RecordAligner
    from evaluation.experimental.deterministic_claim_verifier import DeterministicClaimVerifier
    from evaluation.experimental.semantic_claim_resolver import SemanticClaimResolver

    model = ChatDeepSeek(model=MODEL_NAME, timeout=60, max_retries=0).bind(
        extra_body={"thinking": {"type": "disabled"}}, response_format={"type": "json_object"},
    )
    def measured(stage):
        async def invoke(prompt):
            capture = _SHADOW_CAPTURE.get()
            if capture is not None:
                capture["llm_calls"] += 1
                capture[stage + "_llm_calls"] += 1
            return await model.ainvoke(prompt)
        return RunnableLambda(invoke)
    return (RecordAligner(measured("alignment")), DeterministicClaimVerifier(),
            SemanticClaimResolver(measured("semantic")))


def _compare_shadow_findings(baseline, findings, *, status_field="deterministic_status"):
    originals = {finding["claim_id"]: finding for finding in baseline}
    comparisons = []
    for finding in findings:
        reference = finding["claim_id"]
        original = originals[reference]
        agreement = original["status"] == finding["status"]
        cause = None
        if not agreement:
            if finding["status"] == "UNKNOWN":
                reason = finding["reason"]
                cause = "alignment" if reason.startswith("Record alignment") else (
                    "missing_evidence" if reason.startswith("Missing") else "semantic_equivalence")
            else:
                cause = "objective_contradiction" if "MISMATCH" in (finding["status"], original["status"]) else "objective_match"
        comparisons.append({"claim_id": reference, "category": reference.split(".")[0],
                            "llm_status": original["status"], status_field: finding["status"],
                            "agreement": agreement, "disagreement_type": cause, "reason": finding["reason"]})
    return comparisons


async def _inspect_shadow(record, aligner, verifier, resolver):
    from cv_verifier.claim_comparison_validator import ClaimComparisonValidator
    from cv_verifier.identity_decision_validator import IdentityDecisionValidator
    from cv_verifier.cv_scorer import CVScorer

    baseline = record.get("comparison_output_before_validation")
    report = {"attempted": False, "skipped": None, "llm_findings": deepcopy(baseline),
              "production_findings": deepcopy(record["result"].get("findings", [])),
              "production_score": record["score"], "baseline_score": None,
              "alignments": None, "deterministic_findings": [], "semantic_claim_ids": [],
              "semantic_findings": [], "findings": [], "shadow_score": None,
              "comparisons": [], "hybrid_comparisons": [], "final_production_comparisons": [],
              "alignment_disagreements": [], "errors": [], "llm_calls": 0,
              "alignment_llm_calls": 0, "semantic_llm_calls": 0,
              "alignment_seconds": 0.0, "verification_seconds": 0.0,
              "semantic_seconds": 0.0, "elapsed_seconds": 0.0}
    start = time.perf_counter()
    token = _SHADOW_CAPTURE.set(report)
    stage = "shadow_input"
    try:
        result = record["result"]
        if result.get("identity") != "resolved":
            report["skipped"] = "identity_unresolved"
            return report
        if baseline is None:
            report["skipped"] = "baseline_unavailable"
            return report
        IdentityDecisionValidator().validate({"candidate_id": result["selected_candidate_id"],
            "reason": result["identity_reason"]}, result["profiles"])
        selected = [profile for profile in result["profiles"] if profile["id"] == result["selected_candidate_id"]]
        if len(selected) != 1:
            raise ValueError("Shadow requires exactly one selected profile")
        claims, profile = deepcopy(result["claims"]), deepcopy(selected[0])
        ClaimComparisonValidator().validate(baseline, claims, profile)
        report["baseline_score"] = CVScorer().score(baseline, identity_resolved=True)
        report["attempted"] = True
        stage = "shadow_alignment"
        alignment_start = time.perf_counter()
        try:
            report["alignments"] = await asyncio.wait_for(aligner.align(deepcopy(claims), deepcopy(profile)),
                                                         timeout=PER_CV_TIMEOUT_SECONDS)
        finally:
            report["alignment_seconds"] = time.perf_counter() - alignment_start
        stage = "shadow_verification"
        verification_start = time.perf_counter()
        try:
            report["findings"] = verifier.verify(claims, profile, report["alignments"])
            report["deterministic_findings"] = deepcopy(report["findings"])
            report["comparisons"] = _compare_shadow_findings(baseline, report["findings"])
        finally:
            report["verification_seconds"] = time.perf_counter() - verification_start
        report["semantic_claim_ids"] = [finding["claim_id"] for finding in report["findings"]
                                        if finding["status"] == "UNKNOWN"]
        if report["semantic_claim_ids"]:
            stage = "shadow_semantic"
            semantic_start = time.perf_counter()
            try:
                merged = await asyncio.wait_for(resolver.resolve(deepcopy(claims), deepcopy(profile),
                    deepcopy(report["alignments"]), deepcopy(report["findings"])), timeout=PER_CV_TIMEOUT_SECONDS)
                ClaimComparisonValidator().validate(merged, claims, profile)
                report["findings"] = merged
                report["semantic_findings"] = [finding for finding in merged
                                                if finding["claim_id"] in report["semantic_claim_ids"]]
            finally:
                report["semantic_seconds"] = time.perf_counter() - semantic_start
    except Exception as error:
        report["errors"].append({"stage": stage, "message": str(error) or type(error).__name__})
    finally:
        if report["findings"]:
            report["shadow_score"] = CVScorer().score(report["findings"], identity_resolved=True)
            report["hybrid_comparisons"] = _compare_shadow_findings(baseline, report["findings"], status_field="shadow_status")
            report["final_production_comparisons"] = _compare_shadow_findings(
                report["production_findings"], report["findings"], status_field="shadow_status")
            report["alignment_disagreements"] = _alignment_disagreements(report)
        report["elapsed_seconds"] = time.perf_counter() - start
        _SHADOW_CAPTURE.reset(token)
    return report


def _alignment_disagreements(report):
    """Expose record disagreements for inspection, without diagnosing alignment by index alone."""
    concerns = []
    for finding in report["findings"]:
        parts = finding["claim_id"].split(".")
        if parts[0] not in ("jobs", "education"):
            continue
        alignment = next(entry for entry in report["alignments"][parts[0]] if entry["cv_index"] == int(parts[1]))
        disagreement = any(not row["agreement"] and row["claim_id"] == finding["claim_id"]
                           for row in report["hybrid_comparisons"])
        if disagreement or (finding["status"] == "UNKNOWN" and alignment["profile_index"] is None):
            concerns.append({"claim_id": finding["claim_id"], "alignment": deepcopy(alignment),
                             "reason": finding["reason"], "status": finding["status"]})
    return concerns


async def _run_shadow(payload, output, semaphore):
    start = time.perf_counter()
    payload["shadow_enabled"] = True
    try:
        aligner, verifier, resolver = _build_shadow_components()
    except Exception as error:
        payload["shadow_setup_errors"] = [{"stage": "shadow_setup", "message": str(error)}]
        _save_results(output, payload)
        return
    async def inspect_one(record):
        async with semaphore:
            record["shadow"] = await _inspect_shadow(record, aligner, verifier, resolver)
            _save_results(output, payload)
            shadow = record["shadow"]
            disagreements = sum(not row["agreement"] for row in shadow["hybrid_comparisons"])
            print(f"{record['filename']} shadow: disagreements={disagreements}; "
                  f"score={shadow['shadow_score']}; semantic claims={len(shadow['semantic_claim_ids'])}; "
                  f"calls={shadow['llm_calls']}; {shadow['elapsed_seconds']:.2f}s; "
                  f"errors={len(shadow['errors'])}; skipped={shadow['skipped']}", flush=True)
    await asyncio.gather(*(inspect_one(record) for record in payload["records"]))
    payload["shadow_elapsed_seconds"] = time.perf_counter() - start
    _save_results(output, payload)


def _capture_comparisons(agent):
    """Retain comparison output and time comparison/reflection without changing them."""
    def measured(method, stage):
        async def invoke(*args, **kwargs):
            start = time.perf_counter()
            capture = _COMPARISON_CAPTURE.get()
            try:
                output = await method(*args, **kwargs)
                if capture is not None and stage == "comparison":
                    capture["output"] = output
                return output
            finally:
                if capture is not None:
                    timing = capture.setdefault("timings", {}).setdefault(stage, {"seconds": 0.0, "calls": 0})
                    timing["seconds"] += time.perf_counter() - start
                    timing["calls"] += 1
        return invoke

    agent.comparator.compare = measured(agent.comparator.compare, "comparison")
    agent.reflector.reflect = measured(agent.reflector.reflect, "reflection")


def _fallback_details(result, score, failure):
    """Describe the observed blocking stage; do not infer causes from earlier errors."""
    if score != 0.5:
        return {"occurred": False, "stage": None, "reason": None}
    if failure:
        return {"occurred": True, "stage": failure["stage"], "reason": failure["message"]}
    errors = result.get("errors", [])
    for error in errors:
        if error.get("stage") == "extraction":
            return {"occurred": True, "stage": "extraction", "reason": error.get("message")}
    if result.get("identity") == "unresolved":
        if not result.get("profiles"):
            for stage, origin in (("profile", "retrieval"), ("search", "search")):
                observed = [error.get("message", "") for error in errors if error.get("stage") == stage]
                if observed:
                    return {"occurred": True, "stage": origin, "reason": "; ".join(observed)}
        reason = result.get("identity_reason") or "Identity remained unresolved"
        stop = result.get("investigation_stop_reason")
        return {"occurred": True, "stage": "identity", "reason": f"{reason}; stop={stop}"}
    for error in errors:
        if error.get("stage") == "comparison":
            return {"occurred": True, "stage": "comparison", "reason": error.get("message")}
    unknown = [finding.get("claim_id") for finding in result.get("findings", [])
               if finding.get("status") == "UNKNOWN"]
    if unknown:
        return {"occurred": True, "stage": "comparison_unknown",
                "reason": f"Validated UNKNOWN findings: {', '.join(unknown)}"}
    return {"occurred": True, "stage": "undetermined",
            "reason": "Score 0.5 without an observed blocking error or UNKNOWN finding"}


def _save_results(output, payload):
    """Replace the development JSON atomically after each completed CV."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                         prefix=".batch-", suffix=".json", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _print_record(record):
    result = record["result"]
    counts = Counter(finding["status"] for finding in result.get("findings", []))
    findings = "/".join(f"{status}={counts[status]}" for status in ("MATCH", "MISMATCH", "UNKNOWN"))
    fallback = record["fallback"]
    origin = fallback["stage"] if fallback["occurred"] else "none"
    corrections = len(result.get("reflection", {}).get("corrections", []))
    reflection_errors = sum(error.get("stage") == "reflection" for error in result.get("errors", []))
    print(f"{record['filename']}: score={record['score']:.1f}; {record['correctness']}; "
          f"identity={result.get('identity', 'undetermined')}/{result.get('selected_candidate_id')}; "
          f"{findings}; {record['elapsed_seconds']:.2f}s; fallback={origin}; "
          f"reflection corrections={corrections}/errors={reflection_errors}", flush=True)


async def inspect_batch(folder: Path, output: Path | None = None, *, shadow: bool = False):
    if output is None:
        output = DEFAULT_OUTPUT_DIR / f"public_batch_{time.time_ns()}.json"
    paths = cv_files(folder)
    if not paths:
        raise ValueError(f"No CV PDFs found in {folder}")
    if output.resolve() == Path("results.csv").resolve():
        raise ValueError("Diagnostic JSON must not overwrite results.csv")
    truth = read_ground_truth(folder)
    records = {}
    payload = {"cv_folder": str(folder), "max_concurrency": MAX_CONCURRENT_CVS,
               "per_cv_timeout_seconds": PER_CV_TIMEOUT_SECONDS, "records": []}
    _save_results(output, payload)
    load_env_file()
    agent = build_agent(await load_mcp_tools())
    _capture_comparisons(agent)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_CVS)

    async def inspect_one(path):
        async with semaphore:
            start = time.perf_counter()
            result, failure = {}, None
            comparison_capture = {}
            token = _COMPARISON_CAPTURE.set(comparison_capture)
            stage = "pdf_loading"
            try:
                text = cv_text(path)
                stage = "verification"
                result = await asyncio.wait_for(agent.ainvoke(text), timeout=PER_CV_TIMEOUT_SECONDS)
                if not isinstance(result, dict):
                    raise ValueError("Pipeline returned a non-object result")
                score = parse_score(result.get("score"))
                if score is None:
                    raise ValueError("Pipeline returned a missing or invalid score")
            except Exception as error:
                score = 0.5
                timed_out = isinstance(error, TimeoutError)
                message = str(error) or ("Per-CV timeout elapsed" if timed_out else type(error).__name__)
                failure = {"stage": "timeout" if timed_out else stage,
                           "type": type(error).__name__, "message": message}
                if not isinstance(result, dict):
                    result = {}
            finally:
                _COMPARISON_CAPTURE.reset(token)
            record = {
                "filename": path.name, "score": score,
                "elapsed_seconds": time.perf_counter() - start,
                "correctness": correctness_text(score, truth.get(path.name)),
                "expected": truth.get(path.name), "failure": failure,
                "fallback": _fallback_details(result, score, failure), "result": result,
                "comparison_output_before_validation": comparison_capture.get("output"),
                "stage_timings": comparison_capture.get("timings", {}),
            }
            records[path.name] = record
            payload["records"] = [records[p.name] for p in paths if p.name in records]
            _save_results(output, payload)
            _print_record(record)

    await asyncio.gather(*(inspect_one(path) for path in paths))
    # Complete/persist the ordinary production batch before issuing any shadow inference.
    if shadow:
        await _run_shadow(payload, output, semaphore)
    correct = sum(record["correctness"] == "correct" for record in payload["records"])
    print(f"Completed {len(records)} CVs; {correct} correct; details: {output}")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-folder", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="Detailed JSON path; default creates a unique development file")
    parser.add_argument("--shadow", action="store_true", help="Evaluate hybrid findings after production completes")
    args = parser.parse_args()
    asyncio.run(inspect_batch(args.cv_folder, args.output, shadow=args.shadow))


if __name__ == "__main__":
    main()
