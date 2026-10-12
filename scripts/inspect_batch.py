"""Development-only batch diagnostics; never writes results.csv or retries CVs."""

import argparse
import asyncio
import json
import os
import tempfile
import time
from collections import Counter
from contextvars import ContextVar
from pathlib import Path

from hw2 import (
    PER_CV_TIMEOUT_SECONDS, build_agent, correctness_text, cv_files, cv_text,
    load_env_file, load_mcp_tools, parse_score, read_ground_truth,
)

DEFAULT_OUTPUT_DIR = Path(".venv/diagnostics")
MAX_CONCURRENT_CVS = 3
_COMPARISON_CAPTURE = ContextVar("batch_comparison_capture", default=None)


def _capture_comparisons(agent):
    """Retain returned findings even when production validation rejects them."""
    compare = agent.comparator.compare

    async def observed_compare(*args, **kwargs):
        proposed = await compare(*args, **kwargs)
        capture = _COMPARISON_CAPTURE.get()
        if capture is not None:
            capture["output"] = proposed
        return proposed

    agent.comparator.compare = observed_compare


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
    print(f"{record['filename']}: score={record['score']:.1f}; {record['correctness']}; "
          f"identity={result.get('identity', 'undetermined')}/{result.get('selected_candidate_id')}; "
          f"{findings}; {record['elapsed_seconds']:.2f}s; fallback={origin}", flush=True)


async def inspect_batch(folder: Path, output: Path | None = None):
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
            }
            records[path.name] = record
            payload["records"] = [records[p.name] for p in paths if p.name in records]
            _save_results(output, payload)
            _print_record(record)

    await asyncio.gather(*(inspect_one(path) for path in paths))
    correct = sum(record["correctness"] == "correct" for record in payload["records"])
    print(f"Completed {len(records)} CVs; {correct} correct; details: {output}")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-folder", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="Detailed JSON path; default creates a unique development file")
    args = parser.parse_args()
    asyncio.run(inspect_batch(args.cv_folder, args.output))


if __name__ == "__main__":
    main()
