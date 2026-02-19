"""Tests for balancing outlier-driven and non-outlier hypotheses."""

from __future__ import annotations

import unittest

from src.agents.hypothesis_agent import HypothesisAgent
from src.schemas.hypothesis import Hypothesis, HypothesisOrigin, StatisticalTest


class HypothesisOutlierBalancingTests(unittest.TestCase):
    def test_adds_non_outlier_pathway_when_all_hypotheses_are_outlier_driven(self):
        agent = HypothesisAgent(llm=None, verbose=False)
        hypotheses = [
            Hypothesis(
                hypothesis_id="H1",
                case_id="CASE-1",
                title="LEAK_RATE drift from seal compression",
                description="Outlier LEAK_RATE explains failure",
                mechanism="Leak pathway",
                expected_signatures=["LEAK_RATE high"],
                recommended_tests=[
                    StatisticalTest(
                        test_type="xbar_s_chart",
                        target_variable="leak_rate",
                        grouping_key="lot_number",
                        expected_signal="Special cause variation",
                    )
                ],
                required_data_sources=["leak_test_history"],
                origin=HypothesisOrigin.UNIT_SPECIFIC_ANOMALY,
                prior_confidence=0.65,
            )
        ]

        result = agent._ensure_competing_pathways(
            hypotheses=hypotheses,
            case_id="CASE-1",
            failure_type="leak_test_fail",
            outlier_relevance={"evaluations": []},
            top_test_outliers=[{"test_id": "LEAK_RATE"}],
        )

        self.assertGreaterEqual(len(result), 2)
        self.assertTrue(any(h.title == "Non-outlier causal pathway" for h in result))

    def test_penalizes_non_causal_outlier_and_keeps_non_outlier_competitive(self):
        agent = HypothesisAgent(llm=None, verbose=False)
        outlier_h = Hypothesis(
            hypothesis_id="H1",
            case_id="CASE-2",
            title="LEAK_RATE explains root cause",
            description="Outlier LEAK_RATE is causal",
            mechanism="Leak pathway",
            expected_signatures=["LEAK_RATE spike"],
            recommended_tests=[
                StatisticalTest(
                    test_type="xbar_s_chart",
                    target_variable="leak_rate",
                    grouping_key="lot_number",
                    expected_signal="Special cause variation",
                )
            ],
            required_data_sources=["leak_test_history"],
            origin=HypothesisOrigin.UNIT_SPECIFIC_ANOMALY,
            prior_confidence=0.70,
        )
        non_outlier_h = Hypothesis(
            hypothesis_id="H2",
            case_id="CASE-2",
            title="Fixture wear pathway",
            description="Independent fixture drift",
            mechanism="Alignment wear",
            expected_signatures=["Machine-dependent failures"],
            recommended_tests=[
                StatisticalTest(
                    test_type="one_way_anova",
                    target_variable="measured_value",
                    grouping_key="fixture_id",
                    expected_signal="Fixture group effect",
                )
            ],
            required_data_sources=["assembly_parameters"],
            origin=HypothesisOrigin.ENGINEERING_REASONING,
            prior_confidence=0.30,
        )

        result = agent._ensure_competing_pathways(
            hypotheses=[outlier_h, non_outlier_h],
            case_id="CASE-2",
            failure_type="leak_test_fail",
            outlier_relevance={
                "evaluations": [
                    {"test_id": "LEAK_RATE", "classification": "likely_non_causal"}
                ]
            },
            top_test_outliers=[{"test_id": "LEAK_RATE"}],
        )

        updated_outlier = next(h for h in result if h.hypothesis_id == "H1")
        updated_non = next(h for h in result if h.hypothesis_id == "H2")

        self.assertLess(updated_outlier.prior_confidence, 0.70)
        self.assertGreater(updated_non.prior_confidence, 0.30)
        self.assertGreaterEqual(updated_non.prior_confidence, updated_outlier.prior_confidence - 0.08)


if __name__ == "__main__":
    unittest.main()
