import io
import json
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
from itertools import count

from scripts import inspect_cv


class InspectCVTests(unittest.IsolatedAsyncioTestCase):
    async def test_reuses_dependencies_and_prints_complete_pipeline_result(self):
        path = Path("public_test/CV 1.pdf")
        tools = [object()]
        result = {
            "claims": {"name": {"raw": "陳 Example", "value": "陳 Example"}},
            "search_hits": [{"id": 1}, {"id": 2}],
            "ranked_candidate_ids": [2, 1],
            "profiles": [{"id": 1}], "identity": "unresolved",
            "selected_candidate_id": None, "identity_reason": "Insufficient evidence",
            "attempted_candidate_ids": [1, 2], "broader_search_performed": False,
            "investigation_stop_reason": "candidates_exhausted",
            "retrieval_sampled": False, "retrieval_complete": False,
            "errors": [{"stage": "profile", "person_id": 2, "message": "failed"}],
        }
        agent = Mock()
        agent.ainvoke = AsyncMock(return_value=result)
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(inspect_cv, "load_env_file") as load_env, \
             patch.object(inspect_cv, "cv_text", return_value="CV text") as load_pdf, \
             patch.object(inspect_cv, "load_mcp_tools", new=AsyncMock(return_value=tools)) as connect, \
             patch.object(inspect_cv, "build_agent", return_value=agent) as build, \
             redirect_stdout(stdout), redirect_stderr(stderr):
            await inspect_cv.inspect_cv(path)
        load_env.assert_called_once_with()
        load_pdf.assert_called_once_with(path)
        connect.assert_awaited_once_with()
        build.assert_called_once()
        self.assertEqual([tool.target for tool in build.call_args.args[0]], tools)
        self.assertIn("End-to-end:", stderr.getvalue())
        agent.ainvoke.assert_awaited_once_with("CV text")
        self.assertEqual(json.loads(stdout.getvalue()), result)
        self.assertEqual(stdout.getvalue(), json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    async def test_timing_aggregates_repeated_calls_and_includes_failures(self):
        timings = {"search": {"seconds": 0.0, "calls": 0}}
        method = AsyncMock(side_effect=["hits", RuntimeError("failed")])
        measured = inspect_cv._timed_async(method, timings, "search")
        with patch.object(inspect_cv.time, "perf_counter", side_effect=[10, 12, 20, 23]):
            self.assertEqual(await measured(), "hits")
            with self.assertRaises(RuntimeError):
                await measured()
        self.assertEqual(timings["search"], {"seconds": 5.0, "calls": 2})

    async def test_failed_tool_invocations_count_as_lookup_attempts(self):
        counters = {"mcp_calls": 0, "profile_lookup_attempts": 0}
        tool = Mock(name="tool")
        tool.name = "get_linkedin_profile"
        tool.ainvoke = AsyncMock(side_effect=RuntimeError("failed"))
        measured = inspect_cv._CountedInvocation(tool, counters, "mcp_calls")
        with self.assertRaises(RuntimeError):
            await measured.ainvoke({"person_id": 1})
        self.assertEqual(counters, {"mcp_calls": 1, "profile_lookup_attempts": 1})

    async def test_inspection_counts_real_component_workflow_without_changing_stdout(self):
        from langchain_core.messages import AIMessage
        from langchain_core.runnables import RunnableLambda
        from hw2 import build_agent

        claims = {"name": {"raw": "Person", "value": "Person"},
                  "city": {"raw": "London", "value": "London"},
                  "jobs": [], "education": [], "skills": []}
        def model_response(prompt, **_options):
            system = prompt.to_messages()[0].content
            if system.startswith("Extract"):
                result = claims
            elif system.startswith("Shortlist"):
                result = [3, 1, 2]
            elif system.startswith("Compare"):
                context = json.loads(prompt.to_messages()[1].content)
                result = [{"claim_id": reference, "cv_claim": claim, "status": "UNKNOWN",
                           "evidence": [], "reason": "Unavailable evidence"}
                          for reference, claim in context["claim_catalog"].items()]
            else:
                result = {"candidate_id": 3, "reason": "Supporting evidence"}
            return AIMessage(content=json.dumps(result))
        search, lookup = Mock(), Mock()
        search.name, lookup.name = "search_linkedin_people", "get_linkedin_profile"
        def blocks(value):
            return [{"type": "text", "text": json.dumps(value)}]
        search.ainvoke = AsyncMock(side_effect=[blocks([{"id": 1}, {"id": 2}]),
                                              blocks([{"id": 2}, {"id": 3}])])
        lookup.ainvoke = AsyncMock(side_effect=lambda args: blocks({
            "id": args["person_id"], "name": "Person", "experience": [], "education": [], "skills": [],
        }))
        stdout, stderr = io.StringIO(), io.StringIO()
        ticks = count()
        with patch.object(inspect_cv, "load_env_file"), \
             patch.object(inspect_cv, "cv_text", return_value="CV"), \
             patch.object(inspect_cv, "load_mcp_tools", new=AsyncMock(return_value=[search, lookup])), \
             patch.object(inspect_cv, "build_agent", side_effect=build_agent), \
             patch("langchain_deepseek.ChatDeepSeek", return_value=RunnableLambda(model_response)), \
             patch.object(inspect_cv.time, "perf_counter", side_effect=lambda: next(ticks)), \
             redirect_stdout(stdout), redirect_stderr(stderr):
            await inspect_cv.inspect_cv(Path("example.pdf"))
        result = json.loads(stdout.getvalue())
        self.assertEqual(result["selected_candidate_id"], 3)
        self.assertNotIn("performance", result)
        summary = stderr.getvalue()
        self.assertIn("LinkedIn search: 2.000s / 2 calls", summary)
        self.assertIn("LLM invocations: 4", summary)
        self.assertIn("MCP tool calls: 4", summary)
        self.assertIn("Profile lookup attempts: 2; identity-resolution attempts: 1", summary)
