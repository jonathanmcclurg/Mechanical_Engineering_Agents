"""Tests for top-outlier test-id context integration in agent flow."""

from __future__ import annotations

import unittest

from src.agents.hypothesis_agent import HypothesisAgent
from src.agents.research_agent import PrivateDataResearchAgent
from src.tools.data_catalog import DataCatalog
from src.tools.data_fetch_tool import DataFetchTool


class OutlierIntegrationTests(unittest.TestCase):
    def test_research_agent_pulls_and_uses_top_outliers(self):
        catalog = DataCatalog(catalog_dir="./data/catalog", auto_load=True)
        data_fetch = DataFetchTool(mock_mode=True, data_catalog=catalog)
        agent = PrivateDataResearchAgent(
            llm=None,
            verbose=False,
            data_catalog=catalog,
            data_fetch_tool=data_fetch,
        )

        output = agent.execute(
            {
                "case": {
                    "case_id": "CASE-OUTLIER-001",
                    "failure_type": "leak_test_fail",
                    "failure_description": "Unit failed leak with strong process drift indications.",
                    "part_number": "HYD-VALVE-200",
                    "lot_number": "LOT-2026-W05-001",
                    "serial_number": "SN-789456",
                },
                "product_guide_output": {"data": {}, "citations_used": []},
                "recipe": None,
            }
        )

        self.assertTrue(output.success)
        data_retrieved = output.data["data_retrieved"]
        self.assertIn("top_test_outliers", data_retrieved)

        outliers = data_retrieved["top_test_outliers"]["data"]
        self.assertTrue(outliers)
        self.assertTrue(all("test_id" in item for item in outliers))

        analysis_data = data_retrieved["analysis_data"]
        self.assertIn("top_outlier_tests_considered", analysis_data)
        self.assertTrue(analysis_data["top_outlier_tests_considered"])
        self.assertIn("outlier_relevance", data_retrieved)
        self.assertIn("summary", data_retrieved["outlier_relevance"])

    def test_hypothesis_prompt_includes_outlier_context(self):
        agent = HypothesisAgent(llm=None, verbose=False)
        prompt = agent._build_hypothesis_prompt(
            case={
                "failure_type": "leak_test_fail",
                "failure_description": "Leak detected",
                "part_number": "HYD-VALVE-200",
                "lot_number": "LOT-1",
                "test_value": 2.3e-6,
                "spec_upper": 1.0e-6,
            },
            recipe=None,
            similar_cases=[],
            critical_features=[],
            expected_signatures=[],
            product_guide_citations=[],
            catalog_context={
                "test_ids": ["LEAK_RATE"],
                "roa_parameters": [],
                "operator_buyoffs": [],
                "process_parameters": [],
                "catalog_candidates_considered": [],
                "outlier_test_ids": ["PRESSURE_DROP"],
            },
            unit_data_summary={"warnings": [], "top_test_outliers": []},
            top_test_outliers=[
                {
                    "test_id": "PRESSURE_DROP",
                    "stddev_from_mean": 4.2,
                    "direction": "high",
                    "description": "Pressure decay outlier",
                }
            ],
        )

        self.assertIn("Top outlier test IDs", prompt)
        self.assertIn("PRESSURE_DROP", prompt)
        self.assertIn("Prioritize hypotheses", prompt)


if __name__ == "__main__":
    unittest.main()
