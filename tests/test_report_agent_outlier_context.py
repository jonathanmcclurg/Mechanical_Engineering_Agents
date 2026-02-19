"""ReportAgent regression tests for outlier relevance reporting."""

from __future__ import annotations

import unittest

from src.agents.report_agent import ReportAgent


class ReportAgentOutlierContextTests(unittest.TestCase):
    def test_report_includes_outlier_relevance_and_hypothesis_balance(self):
        agent = ReportAgent(llm=None, verbose=False)

        context = {
            "case": {
                "case_id": "CASE-REPORT-001",
                "failure_type": "leak_test_fail",
                "failure_description": "Unit failed helium leak test.",
                "failure_datetime": "2026-02-01T10:00:00Z",
                "part_number": "HYD-VALVE-200",
                "lot_number": "LOT-XYZ",
            },
            "product_guide_output": {
                "data": {"sections_found": {"Sealing System": ["chunk-1"]}},
                "citations_used": [],
            },
            "research_output": {
                "data": {
                    "data_sources_queried": ["similar_cases", "analysis_data", "top_test_outliers"],
                    "total_records": 120,
                    "recipe_mode": "advisory",
                    "data_retrieved": {
                        "top_test_outliers": {
                            "data": [
                                {"test_id": "LEAK_RATE", "stddev_from_mean": 4.8},
                                {"test_id": "PRESSURE_DROP", "stddev_from_mean": 1.6},
                            ]
                        },
                        "outlier_relevance": {
                            "summary": {
                                "total": 2,
                                "likely_relevant": 1,
                                "inconclusive": 0,
                                "likely_non_causal": 1,
                            },
                            "evaluations": [
                                {
                                    "test_id": "LEAK_RATE",
                                    "classification": "likely_relevant",
                                    "scores": {"relevance_score": 0.72},
                                },
                                {
                                    "test_id": "PRESSURE_DROP",
                                    "classification": "likely_non_causal",
                                    "scores": {"relevance_score": 0.31},
                                },
                            ],
                        },
                    },
                    "summary": {"data_gaps": []},
                },
                "citations_used": [],
            },
            "hypothesis_output": {
                "data": {
                    "hypotheses": [
                        {
                            "hypothesis_id": "HYP-1",
                            "case_id": "CASE-REPORT-001",
                            "title": "Seal compression drift",
                            "description": "Compression drift caused leak",
                            "mechanism": "seal deformation",
                            "expected_signatures": ["LEAK_RATE spike"],
                            "recommended_tests": [
                                {
                                    "test_type": "xbar_s_chart",
                                    "target_variable": "leak_rate",
                                    "grouping_key": "lot_number",
                                    "expected_signal": "special cause",
                                }
                            ],
                            "required_data_sources": ["leak_test_history"],
                            "origin": "unit_specific_anomaly",
                            "prior_confidence": 0.6,
                        }
                    ],
                    "outlier_hypotheses_count": 1,
                    "non_outlier_hypotheses_count": 1,
                },
                "citations_used": [],
            },
            "stats_output": {
                "data": {
                    "hypothesis_results": [
                        {
                            "hypothesis_id": "HYP-1",
                            "hypothesis_title": "Seal compression drift",
                            "overall_support": "supported",
                            "tests_run": [{"test_type": "xbar_s_chart"}],
                        }
                    ],
                    "summary": {"supported": 1, "refuted": 0, "inconclusive": 0},
                    "evidence": [],
                    "total_tests_run": 1,
                },
                "citations_used": [],
            },
            "critic_output": {
                "data": {"critical_issues": 0, "issues_found": []},
                "citations_used": [],
            },
        }

        output = agent.execute(context)
        self.assertTrue(output.success)

        report = output.data["report"]
        sections = {section["title"]: section["content"] for section in report["sections"]}

        self.assertIn("Outlier Signal Relevance", sections)
        self.assertIn("Likely relevant", sections["Outlier Signal Relevance"])
        self.assertIn("Likely non-causal", sections["Outlier Signal Relevance"])

        self.assertIn("Investigation Scope", sections)
        self.assertIn("Recipe Mode", sections["Investigation Scope"])
        self.assertIn("Top Outlier Test IDs Retrieved", sections["Investigation Scope"])

        self.assertIn("Evidence Summary", sections)
        self.assertIn("Hypothesis balance", sections["Evidence Summary"])


if __name__ == "__main__":
    unittest.main()
