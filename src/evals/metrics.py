"""Metrics for evaluating RCA system performance."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from enum import Enum


class MetricCategory(str, Enum):
    """Categories of evaluation metrics."""
    
    ACCURACY = "accuracy"
    STATISTICAL = "statistical"
    CITATION = "citation"
    EFFICIENCY = "efficiency"
    FEEDBACK = "feedback"


@dataclass
class MetricResult:
    """Result of computing a metric."""
    
    metric_name: str
    category: MetricCategory
    value: float
    threshold: Optional[float] = None
    passed: Optional[bool] = None
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        if self.threshold is not None:
            self.passed = self.value >= self.threshold


class RCAMetrics:
    """Compute evaluation metrics for RCA system."""
    
    # Default thresholds
    DEFAULT_THRESHOLDS = {
        "top_3_hit_rate": 0.7,
        "citation_coverage": 0.9,
        "assumption_check_rate": 0.95,
        "correct_method_rate": 0.85,
        "engineer_satisfaction": 0.8,
        "time_to_report": 300,  # seconds
    }
    
    def __init__(self, thresholds: Optional[dict[str, float]] = None):
        self.thresholds = {**self.DEFAULT_THRESHOLDS, **(thresholds or {})}
    
    def compute_all(
        self,
        cases: list[dict],
        reports: list[dict],
        feedback: list[dict],
    ) -> list[MetricResult]:
        """Compute all metrics for a set of cases.
        
        Args:
            cases: List of case data dicts
            reports: List of report data dicts
            feedback: List of feedback data dicts
            
        Returns:
            List of MetricResult objects
        """
        results = []
        
        # Accuracy metrics
        results.append(self.top_n_hit_rate(feedback, n=1))
        results.append(self.top_n_hit_rate(feedback, n=3))
        
        # Statistical metrics
        results.append(self.assumption_check_rate(reports))
        results.append(self.correct_method_rate(feedback))
        results.append(self.practical_significance_usage(reports))
        
        # Citation metrics
        results.append(self.citation_coverage(reports))
        results.append(self.product_guide_citation_rate(reports))
        
        # Efficiency metrics
        results.append(self.average_time_to_report(reports))
        
        # Feedback metrics
        results.append(self.engineer_satisfaction_rate(feedback))
        results.append(self.time_saved_estimate(feedback))
        
        return results
    
    def top_n_hit_rate(
        self,
        feedback: list[dict],
        n: int = 3,
    ) -> MetricResult:
        """Calculate rate at which correct root cause was in top N hypotheses.
        
        Args:
            feedback: List of feedback records
            n: Number of top hypotheses to consider
            
        Returns:
            MetricResult with hit rate
        """
        if not feedback:
            return MetricResult(
                metric_name=f"top_{n}_hit_rate",
                category=MetricCategory.ACCURACY,
                value=0.0,
                threshold=self.thresholds.get(f"top_{n}_hit_rate", 0.7),
                details={"feedback_count": 0, "note": "No feedback available"}
            )
        
        hits = 0
        total = 0
        
        for fb in feedback:
            data = fb.get("data", fb)
            
            # Check if root cause was in top N
            if n == 3:
                if data.get("root_cause_was_in_top_3"):
                    hits += 1
            else:
                # For n=1, check if correct hypothesis was #1
                correct_id = data.get("correct_hypothesis_id")
                if correct_id:
                    # This would need the report to check rank
                    # For now, use a heuristic
                    pass
            
            total += 1
        
        rate = hits / total if total > 0 else 0.0
        
        return MetricResult(
            metric_name=f"top_{n}_hit_rate",
            category=MetricCategory.ACCURACY,
            value=rate,
            threshold=self.thresholds.get(f"top_{n}_hit_rate", 0.7),
            details={
                "hits": hits,
                "total": total,
            }
        )
    
    def assumption_check_rate(self, reports: list[dict]) -> MetricResult:
        """Calculate rate at which statistical assumptions were checked.
        
        Args:
            reports: List of report records
            
        Returns:
            MetricResult with assumption check rate
        """
        if not reports:
            return MetricResult(
                metric_name="assumption_check_rate",
                category=MetricCategory.STATISTICAL,
                value=0.0,
                threshold=self.thresholds.get("assumption_check_rate", 0.95),
                details={"note": "No reports available"}
            )
        
        tests_with_checks = 0
        total_tests = 0
        
        for report in reports:
            report_data = report.get("report", report)
            
            # Navigate to hypothesis results
            for hyp in report_data.get("ranked_hypotheses", []):
                evidence = hyp.get("evidence_for", []) + hyp.get("evidence_against", [])
                
                for ev in evidence:
                    stats_result = ev.get("statistical_result", {})
                    if stats_result:
                        total_tests += 1
                        
                        assumptions = stats_result.get("assumptions_checked", [])
                        if assumptions:
                            tests_with_checks += 1
        
        rate = tests_with_checks / total_tests if total_tests > 0 else 1.0
        
        return MetricResult(
            metric_name="assumption_check_rate",
            category=MetricCategory.STATISTICAL,
            value=rate,
            threshold=self.thresholds.get("assumption_check_rate", 0.95),
            details={
                "tests_with_checks": tests_with_checks,
                "total_tests": total_tests,
            }
        )
    
    def correct_method_rate(self, feedback: list[dict]) -> MetricResult:
        """Calculate rate at which appropriate statistical methods were used.
        
        Based on engineer feedback on method appropriateness.
        
        Args:
            feedback: List of feedback records with stats_feedback
            
        Returns:
            MetricResult with correct method rate
        """
        appropriate = 0
        total = 0
        
        for fb in feedback:
            data = fb.get("data", fb)
            stats_feedback = data.get("stats_feedback", [])
            
            for sf in stats_feedback:
                total += 1
                if sf.get("was_appropriate"):
                    appropriate += 1
        
        rate = appropriate / total if total > 0 else 1.0
        
        return MetricResult(
            metric_name="correct_method_rate",
            category=MetricCategory.STATISTICAL,
            value=rate,
            threshold=self.thresholds.get("correct_method_rate", 0.85),
            details={
                "appropriate": appropriate,
                "total": total,
            }
        )
    
    def practical_significance_usage(self, reports: list[dict]) -> MetricResult:
        """Calculate rate at which practical significance was considered.
        
        Args:
            reports: List of report records
            
        Returns:
            MetricResult with practical significance usage rate
        """
        with_practical = 0
        total_tests = 0
        
        for report in reports:
            report_data = report.get("report", report)
            
            for hyp in report_data.get("ranked_hypotheses", []):
                evidence = hyp.get("evidence_for", []) + hyp.get("evidence_against", [])
                
                for ev in evidence:
                    stats_result = ev.get("statistical_result", {})
                    if stats_result.get("p_value") is not None:
                        total_tests += 1
                        
                        if stats_result.get("effect_size") is not None:
                            with_practical += 1
        
        rate = with_practical / total_tests if total_tests > 0 else 1.0
        
        return MetricResult(
            metric_name="practical_significance_usage",
            category=MetricCategory.STATISTICAL,
            value=rate,
            threshold=0.9,
            details={
                "with_practical": with_practical,
                "total_tests": total_tests,
            }
        )
    
    def citation_coverage(self, reports: list[dict]) -> MetricResult:
        """Calculate rate of claims with proper citations.
        
        Args:
            reports: List of report records
            
        Returns:
            MetricResult with citation coverage rate
        """
        with_citations = 0
        total_claims = 0
        
        for report in reports:
            report_data = report.get("report", report)
            
            for hyp in report_data.get("ranked_hypotheses", []):
                # Each hypothesis claim should have citations
                evidence = hyp.get("evidence_for", []) + hyp.get("evidence_against", [])
                
                for ev in evidence:
                    total_claims += 1
                    if ev.get("citations"):
                        with_citations += 1
        
        rate = with_citations / total_claims if total_claims > 0 else 1.0
        
        return MetricResult(
            metric_name="citation_coverage",
            category=MetricCategory.CITATION,
            value=rate,
            threshold=self.thresholds.get("citation_coverage", 0.9),
            details={
                "with_citations": with_citations,
                "total_claims": total_claims,
            }
        )
    
    def product_guide_citation_rate(self, reports: list[dict]) -> MetricResult:
        """Calculate rate of reports that cite the product guide.
        
        Args:
            reports: List of report records
            
        Returns:
            MetricResult with product guide citation rate
        """
        with_guide = 0
        total = len(reports)
        
        for report in reports:
            report_data = report.get("report", report)
            all_citations = report_data.get("all_citations", [])
            
            for citation in all_citations:
                source_type = citation.get("source_type", "")
                if "product_guide" in source_type.lower():
                    with_guide += 1
                    break
        
        rate = with_guide / total if total > 0 else 0.0
        
        return MetricResult(
            metric_name="product_guide_citation_rate",
            category=MetricCategory.CITATION,
            value=rate,
            threshold=0.8,
            details={
                "with_guide": with_guide,
                "total": total,
            }
        )
    
    def average_time_to_report(self, reports: list[dict]) -> MetricResult:
        """Calculate average time to generate a report.
        
        Args:
            reports: List of report records
            
        Returns:
            MetricResult with average time in seconds
        """
        times = []
        
        for report in reports:
            report_data = report.get("report", report)
            processing_time = report_data.get("processing_time_seconds")
            if processing_time:
                times.append(processing_time)
        
        avg_time = sum(times) / len(times) if times else 0.0
        
        return MetricResult(
            metric_name="average_time_to_report",
            category=MetricCategory.EFFICIENCY,
            value=avg_time,
            threshold=self.thresholds.get("time_to_report", 300),
            passed=avg_time <= self.thresholds.get("time_to_report", 300) if times else None,
            details={
                "times": times,
                "min": min(times) if times else None,
                "max": max(times) if times else None,
            }
        )
    
    def engineer_satisfaction_rate(self, feedback: list[dict]) -> MetricResult:
        """Calculate rate of reports marked as useful by engineers.
        
        Args:
            feedback: List of feedback records
            
        Returns:
            MetricResult with satisfaction rate
        """
        useful = 0
        total = len(feedback)
        
        for fb in feedback:
            data = fb.get("data", fb)
            if data.get("report_useful"):
                useful += 1
        
        rate = useful / total if total > 0 else 0.0
        
        return MetricResult(
            metric_name="engineer_satisfaction_rate",
            category=MetricCategory.FEEDBACK,
            value=rate,
            threshold=self.thresholds.get("engineer_satisfaction", 0.8),
            details={
                "useful": useful,
                "total": total,
            }
        )
    
    def time_saved_estimate(self, feedback: list[dict]) -> MetricResult:
        """Calculate average time saved as reported by engineers.
        
        Args:
            feedback: List of feedback records
            
        Returns:
            MetricResult with average time saved in minutes
        """
        times = []
        
        for fb in feedback:
            data = fb.get("data", fb)
            time_saved = data.get("time_saved_minutes")
            if time_saved is not None:
                times.append(time_saved)
        
        avg_time = sum(times) / len(times) if times else 0.0
        
        return MetricResult(
            metric_name="average_time_saved_minutes",
            category=MetricCategory.FEEDBACK,
            value=avg_time,
            threshold=30,  # At least 30 minutes saved
            details={
                "times": times,
                "total_saved": sum(times),
            }
        )
