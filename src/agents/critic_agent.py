"""Critic Evidence Agent - validates evidence and challenges weak claims."""

from typing import Any

from .base_agent import BaseRCAAgent, AgentOutput


class CriticEvidenceAgent(BaseRCAAgent):
    """Agent that critiques evidence and ensures citation quality."""
    
    name = "CriticEvidenceAgent"
    description = "Enforces citations, challenges confounders, ensures statistical claims match outputs"
    
    system_prompt = """You are a critical reviewer for manufacturing root cause analysis.
Your job is to:
1. Challenge weak evidence and identify potential confounders
2. Verify that all claims have proper citations
3. Check that statistical conclusions match the test outputs
4. Identify gaps in the evidence that need additional investigation
5. Ensure alternative explanations have been considered

Be skeptical but constructive. The goal is to strengthen the analysis, not reject it."""

    def execute(self, context: dict[str, Any]) -> AgentOutput:
        """Review and critique the evidence.
        
        Args:
            context: Should contain:
                - 'hypotheses': List of hypothesis results
                - 'stats_results': Statistical analysis output
                - 'evidence': List of evidence items
                - 'product_guide_output': Product guide citations
                
        Returns:
            AgentOutput with critique and recommendations
        """
        self.log("Starting evidence review")
        
        hypotheses = context.get("hypotheses", [])
        stats_results = context.get("stats_results", {})
        evidence = context.get("evidence", [])
        product_guide = context.get("product_guide_output", {})
        
        issues = []
        recommendations = []
        needs_more_evidence = False
        
        # Check citation coverage
        citation_issues = self._check_citation_coverage(evidence, hypotheses)
        issues.extend(citation_issues)
        
        # Check statistical validity
        stats_issues = self._check_statistical_validity(stats_results)
        issues.extend(stats_issues)
        
        # Check for confounders
        confounder_issues = self._check_for_confounders(hypotheses, stats_results)
        issues.extend(confounder_issues)
        
        # Check hypothesis coverage
        coverage_issues = self._check_hypothesis_coverage(hypotheses, stats_results)
        issues.extend(coverage_issues)
        
        # Determine if more evidence is needed
        critical_issues = [i for i in issues if i.get("severity") == "critical"]
        if critical_issues:
            needs_more_evidence = True
            recommendations.append({
                "action": "additional_research",
                "reason": f"Found {len(critical_issues)} critical issues",
                "details": [i["issue"] for i in critical_issues],
            })
        
        # Generate recommendations
        for issue in issues:
            if issue.get("recommendation"):
                recommendations.append(issue["recommendation"])
        
        # Calculate overall confidence adjustment
        confidence_penalty = min(0.3, len(issues) * 0.05)
        adjusted_confidence = max(0.5, 1.0 - confidence_penalty)
        deterministic_reasoning = (
            f"Reviewed evidence and found {len(issues)} issues ({len(critical_issues)} critical)"
        )
        llm_reasoning, used_llm = self._llm_analysis_or_fallback(
            objective="Summarize quality risks, confounders, and confidence adjustments.",
            context_payload={
                "issue_count": len(issues),
                "critical_issue_count": len(critical_issues),
                "needs_more_evidence": needs_more_evidence,
                "recommendations": recommendations[:10],
                "stats_summary": stats_results.get("summary", {}),
            },
            fallback_reasoning=deterministic_reasoning,
        )
        
        self.log(f"Review complete. Found {len(issues)} issues, {len(critical_issues)} critical")
        
        return AgentOutput(
            agent_name=self.name,
            success=True,
            data={
                "issues_found": issues,
                "critical_issues": len(critical_issues),
                "recommendations": recommendations,
                "needs_more_evidence": needs_more_evidence,
                "confidence_adjustment": -confidence_penalty,
                "approved_for_report": not needs_more_evidence,
                "llm_analysis_used": used_llm,
                "llm_analysis": llm_reasoning,
            },
            confidence=adjusted_confidence,
            reasoning=llm_reasoning
        )
    
    def _check_citation_coverage(
        self,
        evidence: list[dict],
        hypotheses: list[dict],
    ) -> list[dict]:
        """Check that all claims have proper citations."""
        issues = []
        
        for ev in evidence:
            citations = ev.get("citations", [])
            if not citations:
                issues.append({
                    "type": "missing_citation",
                    "severity": "high",
                    "issue": f"Evidence '{ev.get('claim', 'Unknown')}' has no citations",
                    "evidence_id": ev.get("evidence_id"),
                    "recommendation": {
                        "action": "add_citation",
                        "details": "Add source citation for this claim"
                    }
                })
            
            # Check citation quality
            for citation in citations:
                if not citation.get("excerpt"):
                    issues.append({
                        "type": "incomplete_citation",
                        "severity": "medium",
                        "issue": f"Citation {citation.get('source_id')} missing excerpt",
                        "citation_id": citation.get("source_id"),
                    })
        
        return issues
    
    def _check_statistical_validity(self, stats_results: dict) -> list[dict]:
        """Check statistical test validity."""
        issues = []
        
        hypothesis_results = stats_results.get("hypothesis_results", [])
        
        for hyp_result in hypothesis_results:
            for test in hyp_result.get("tests_run", []):
                # Check for assumption violations
                if not test.get("assumptions_satisfied", True):
                    issues.append({
                        "type": "assumption_violation",
                        "severity": "medium",
                        "issue": f"Test {test.get('test_name')} has assumption violations",
                        "hypothesis": hyp_result.get("hypothesis_title"),
                        "warnings": test.get("warnings", []),
                        "recommendation": {
                            "action": "review_test_choice",
                            "details": "Consider using nonparametric alternative"
                        }
                    })
                
                # Check for low sample size
                n_total = test.get("n_total", 0)
                if n_total < 30:
                    issues.append({
                        "type": "low_sample_size",
                        "severity": "high" if n_total < 10 else "medium",
                        "issue": f"Test {test.get('test_name')} has only {n_total} samples",
                        "hypothesis": hyp_result.get("hypothesis_title"),
                        "recommendation": {
                            "action": "get_more_data",
                            "details": "Expand time window or relax filters"
                        }
                    })
                
                # Check for p-hacking risk (many tests without correction)
                total_tests = sum(
                    len(h.get("tests_run", [])) 
                    for h in hypothesis_results
                )
                if total_tests > 10 and test.get("p_value", 1) < 0.05:
                    issues.append({
                        "type": "multiple_comparison",
                        "severity": "low",
                        "issue": f"Running {total_tests} tests increases false positive risk",
                        "recommendation": {
                            "action": "note_in_report",
                            "details": "Consider Bonferroni correction or note multiple comparisons"
                        }
                    })
        
        return issues
    
    def _check_for_confounders(
        self,
        hypotheses: list[dict],
        stats_results: dict,
    ) -> list[dict]:
        """Check for potential confounding variables."""
        issues = []
        
        # Common confounders in manufacturing
        confounders = {
            "time": ["shift", "day_of_week", "operator_id"],
            "equipment": ["machine_id", "fixture_id", "station_id"],
            "environment": ["ambient_temp", "ambient_humidity"],
            "material": ["supplier_id", "material_lot"],
        }
        
        # Check if any hypothesis has strong support but potential confounders
        hypothesis_results = stats_results.get("hypothesis_results", [])
        
        for hyp_result in hypothesis_results:
            if hyp_result.get("overall_support") == "supported":
                # Check what confounders weren't controlled for
                controlled_vars = set()
                for test in hyp_result.get("tests_run", []):
                    if test.get("method") in ["two_sample_ttest", "one_way_anova"]:
                        # The grouping variable was controlled
                        pass
                
                issues.append({
                    "type": "confounder_warning",
                    "severity": "low",
                    "issue": f"Hypothesis '{hyp_result.get('hypothesis_title')}' may have confounders",
                    "hypothesis": hyp_result.get("hypothesis_title"),
                    "potential_confounders": [
                        c for category in confounders.values() for c in category
                    ][:5],
                    "recommendation": {
                        "action": "note_in_report",
                        "details": "Note potential confounding variables in conclusions"
                    }
                })
        
        return issues
    
    def _check_hypothesis_coverage(
        self,
        hypotheses: list[dict],
        stats_results: dict,
    ) -> list[dict]:
        """Check that all hypotheses were adequately tested."""
        issues = []
        
        hypothesis_results = stats_results.get("hypothesis_results", [])
        tested_ids = {h.get("hypothesis_id") for h in hypothesis_results}
        
        for hyp in hypotheses:
            hyp_id = hyp.get("hypothesis_id") if isinstance(hyp, dict) else hyp.hypothesis_id
            if hyp_id not in tested_ids:
                issues.append({
                    "type": "untested_hypothesis",
                    "severity": "critical",
                    "issue": f"Hypothesis {hyp_id} was not tested",
                    "hypothesis_id": hyp_id,
                    "recommendation": {
                        "action": "run_tests",
                        "details": "Execute statistical tests for this hypothesis"
                    }
                })
        
        # Check for hypotheses with no conclusive results
        for hyp_result in hypothesis_results:
            if hyp_result.get("overall_support") == "inconclusive":
                tests_run = len(hyp_result.get("tests_run", []))
                if tests_run < 2:
                    issues.append({
                        "type": "insufficient_testing",
                        "severity": "medium",
                        "issue": f"Hypothesis '{hyp_result.get('hypothesis_title')}' has inconclusive results with only {tests_run} test(s)",
                        "hypothesis": hyp_result.get("hypothesis_title"),
                        "recommendation": {
                            "action": "additional_tests",
                            "details": "Run additional statistical tests"
                        }
                    })
        
        return issues
