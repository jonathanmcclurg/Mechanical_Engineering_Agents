"""Regression coverage for audit remediation changes."""

from __future__ import annotations

import asyncio
import unittest
from pathlib import Path

from src.agents.base_agent import AgentOutput
from src.agents.report_agent import ReportAgent
from src.orchestrator.crew import RCACrew
from src.schemas.evidence import Citation
from src.tools.markdown_trace import generate_markdown_trace
import apps.api.main as api_main


class _StubAgent:
    def __init__(self, output: AgentOutput):
        self._output = output

    def execute(self, _context):
        return self._output

    def get_execution_log(self):
        return []

    def reset_execution_log(self):
        return None


class AuditRemediationRegressionTests(unittest.TestCase):
    def test_workflow_log_is_reset_per_run(self):
        crew = RCACrew(verbose=False)
        crew._workflow_log = [{"message": "old-log-entry"}]

        intake = AgentOutput(
            agent_name="intake",
            success=True,
            data={"case": {"case_id": "CASE-1", "failure_type": "leak_test_fail"}},
        )
        product_guide = AgentOutput(agent_name="product_guide", success=True, data={})
        research = AgentOutput(agent_name="research", success=True, data={})
        hypothesis = AgentOutput(
            agent_name="hypothesis",
            success=True,
            data={"hypotheses": [{"hypothesis_id": "H1", "title": "Cause A", "prior_confidence": 0.6}]},
        )
        test_plan = AgentOutput(agent_name="test_plan", success=True, data={"pull_plan": []})
        stats = AgentOutput(
            agent_name="stats",
            success=True,
            data={"evidence": [], "hypothesis_results": [], "summary": {"supported": 0, "refuted": 0, "inconclusive": 0}},
        )
        critic = AgentOutput(agent_name="critic", success=True, data={"needs_more_evidence": False, "issues_found": []})
        report = AgentOutput(agent_name="report", success=True, data={"report": {"title": "ok"}, "report_id": "RPT-1"})

        crew.intake_agent = _StubAgent(intake)
        crew.product_guide_agent = _StubAgent(product_guide)
        crew.research_agent = _StubAgent(research)
        crew.hypothesis_agent = _StubAgent(hypothesis)
        crew.test_plan_agent = _StubAgent(test_plan)
        crew.stats_agent = _StubAgent(stats)
        crew.critic_agent = _StubAgent(critic)
        crew.report_agent = _StubAgent(report)

        result = crew.run({"case_id": "CASE-1"})
        self.assertTrue(result["success"])
        self.assertTrue(result["workflow_log"])
        self.assertNotIn("old-log-entry", [entry.get("message") for entry in result["workflow_log"]])

    def test_report_citation_normalization_handles_sql_and_unknown_types(self):
        sql_citation = ReportAgent._normalize_citation_dict({"source_type": "sql_query"})
        self.assertEqual(sql_citation["source_type"], "historical_case")

        unknown_citation = ReportAgent._normalize_citation_dict({"source_type": "not_a_real_type"})
        self.assertEqual(unknown_citation["source_type"], "test_data")

    def test_citation_schema_normalizes_legacy_source_types(self):
        sql_citation = Citation(
            source_type="sql_query",
            source_id="Q-1",
            source_name="legacy-sql",
            excerpt="legacy query output",
        )
        self.assertEqual(sql_citation.source_type.value, "historical_case")

        api_citation = Citation(
            source_type="internal_data_api",
            source_id="DF-1",
            source_name="internal-api",
            excerpt="fetched manufacturing records",
        )
        self.assertEqual(api_citation.source_type.value, "test_data")

    def test_markdown_trace_includes_explicit_truncation_markers(self):
        result = {
            "success": True,
            "outputs": {
                "agent_a": {
                    "agent_name": "AgentA",
                    "success": True,
                    "data": {
                        "big_dict": {f"k{i}": i for i in range(65)},
                        "big_list": list(range(70)),
                    },
                }
            },
            "workflow_log": [],
            "agent_logs": {},
        }
        trace = generate_markdown_trace(case_id="CASE-X", case_data={"a": 1}, result=result)
        self.assertIn("__truncated_keys__", trace)
        self.assertIn("__truncated_items__", trace)

    def test_pdf_regeneration_uses_stored_workflow_context(self):
        report_id = "RPT-REGEN-1"
        called: dict = {}
        original_reports = dict(api_main._reports)
        original_generator = api_main.generate_pdf_report

        def fake_generate_pdf_report(**kwargs):
            called.update(kwargs)
            output = Path(kwargs["output_path"])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"%PDF-1.4\n%fake\n")
            return str(output)

        try:
            api_main._reports.clear()
            api_main._reports[report_id] = {
                "report_id": report_id,
                "case_id": "CASE-1",
                "report": {"report_id": report_id, "case_id": "CASE-1", "processing_time_seconds": 1.2},
                "workflow_outputs": {"stats": {"data": {"summary": {}}}},
                "workflow_log": [{"message": "step"}],
                "created_at": "2026-01-01T00:00:00",
            }
            api_main.generate_pdf_report = fake_generate_pdf_report

            response = asyncio.run(api_main.get_report_pdf(report_id))
            self.assertEqual(response.media_type, "application/pdf")
            self.assertEqual(called.get("workflow_outputs"), api_main._reports[report_id]["workflow_outputs"])
            self.assertEqual(called.get("workflow_log"), api_main._reports[report_id]["workflow_log"])
        finally:
            api_main.generate_pdf_report = original_generator
            api_main._reports.clear()
            api_main._reports.update(original_reports)


if __name__ == "__main__":
    unittest.main()
