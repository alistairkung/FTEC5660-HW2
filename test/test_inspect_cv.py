import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

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
        stdout = io.StringIO()
        with patch.object(inspect_cv, "load_env_file") as load_env, \
             patch.object(inspect_cv, "cv_text", return_value="CV text") as load_pdf, \
             patch.object(inspect_cv, "load_mcp_tools", new=AsyncMock(return_value=tools)) as connect, \
             patch.object(inspect_cv, "build_agent", return_value=agent) as build, \
             redirect_stdout(stdout):
            await inspect_cv.inspect_cv(path)
        load_env.assert_called_once_with()
        load_pdf.assert_called_once_with(path)
        connect.assert_awaited_once_with()
        build.assert_called_once_with(tools)
        agent.ainvoke.assert_awaited_once_with("CV text")
        self.assertEqual(json.loads(stdout.getvalue()), result)
        self.assertEqual(stdout.getvalue(), json.dumps(result, indent=2, ensure_ascii=False) + "\n")
