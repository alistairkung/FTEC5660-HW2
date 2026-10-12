"""Print one CV's intermediate pipeline results for development inspection."""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from hw2 import build_agent, cv_text, load_env_file, load_mcp_tools


class _CountedInvocation:
    """Delegate an injected tool or chain while counting invocation attempts."""

    def __init__(self, target, counters, counter):
        self.target = target
        self.counters = counters
        self.counter = counter

    def __getattr__(self, name):
        return getattr(self.target, name)

    async def ainvoke(self, *args, **kwargs):
        self.counters[self.counter] += 1
        if self.counter == "mcp_calls" and self.target.name == "get_linkedin_profile":
            self.counters["profile_lookup_attempts"] += 1
        return await self.target.ainvoke(*args, **kwargs)


def _timed_async(method, timings, stage, counters=None, counter=None):
    async def measured(*args, **kwargs):
        start = time.perf_counter()
        if counter is not None:
            counters[counter] += 1
        try:
            return await method(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            timings[stage]["seconds"] += elapsed
            timings[stage]["calls"] += 1
    return measured


def _instrument_agent(agent, timings, counters):
    for component, method_name, stage in (
        (agent.extractor, "extract", "CV extraction"),
        (agent.retriever, "search", "LinkedIn search"),
        (agent.prioritizer, "prioritize", "Prioritization"),
        (agent.retriever, "retrieve_profiles", "Profile retrieval"),
        (agent.identity_resolver, "resolve", "Identity resolution"),
    ):
        setattr(component, method_name, _timed_async(
            getattr(component, method_name), timings, stage, counters,
            "identity_resolution_attempts" if stage == "Identity resolution" else None,
        ))
    # Each chain invokes one LLM. Count at this boundary so empty-input shortcuts
    # do not count as model calls. SDK transport retries are outside this count.
    for component in (agent.extractor, agent.prioritizer, agent.identity_resolver):
        component._chain = _CountedInvocation(component._chain, counters, "llm_calls")


def _print_performance(timings, counters, total):
    print("Performance (stage totals include failed calls):", file=sys.stderr)
    for stage, timing in timings.items():
        print(f"  {stage}: {timing['seconds']:.3f}s / {timing['calls']} calls", file=sys.stderr)
    print(f"  End-to-end: {total:.3f}s", file=sys.stderr)
    print(f"  LLM invocations: {counters['llm_calls']} (SDK retries excluded); "
          f"MCP tool calls: {counters['mcp_calls']} (discovery excluded)", file=sys.stderr)
    print(f"  Profile lookup attempts: {counters['profile_lookup_attempts']}; "
          f"identity-resolution attempts: {counters['identity_resolution_attempts']}", file=sys.stderr)


async def inspect_cv(path: Path) -> None:
    timings = {stage: {"seconds": 0.0, "calls": 0} for stage in (
        "PDF load", "CV extraction", "LinkedIn search", "Prioritization",
        "Profile retrieval", "Identity resolution",
    )}
    counters = dict.fromkeys(("llm_calls", "mcp_calls", "profile_lookup_attempts",
                              "identity_resolution_attempts"), 0)
    start = time.perf_counter()
    try:
        load_env_file()
        pdf_start = time.perf_counter()
        try:
            text = cv_text(path)
        finally:
            timings["PDF load"]["seconds"] += time.perf_counter() - pdf_start
            timings["PDF load"]["calls"] += 1
        tools = await load_mcp_tools()
        measured_tools = [_CountedInvocation(tool, counters, "mcp_calls") for tool in tools]
        agent = build_agent(measured_tools)
        _instrument_agent(agent, timings, counters)
        result = await agent.ainvoke(text)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    finally:
        _print_performance(timings, counters, time.perf_counter() - start)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv", required=True, type=Path, help="CV PDF to inspect")
    args = parser.parse_args()
    if not args.cv.is_file():
        parser.error(f"CV file does not exist: {args.cv}")
    asyncio.run(inspect_cv(args.cv))


if __name__ == "__main__":
    main()
