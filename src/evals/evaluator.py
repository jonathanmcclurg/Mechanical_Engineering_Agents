"""Evaluator for running evaluations and gating changes."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional
import json
from pathlib import Path

from .metrics import RCAMetrics, MetricResult, MetricCategory


@dataclass
class EvaluationResult:
    """Result of a complete evaluation run."""
    
    evaluation_id: str
    timestamp: datetime
    
    # Metrics
    metrics: list[MetricResult]
    
    # Summary
    total_metrics: int
    passed_metrics: int
    failed_metrics: int
    
    # Gating decision
    approved: bool
    blocking_failures: list[str]
    
    # Context
    case_count: int
    feedback_count: int
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "evaluation_id": self.evaluation_id,
            "timestamp": self.timestamp.isoformat(),
            "total_metrics": self.total_metrics,
            "passed_metrics": self.passed_metrics,
            "failed_metrics": self.failed_metrics,
            "approved": self.approved,
            "blocking_failures": self.blocking_failures,
            "case_count": self.case_count,
            "feedback_count": self.feedback_count,
            "metrics": [
                {
                    "name": m.metric_name,
                    "category": m.category.value,
                    "value": m.value,
                    "threshold": m.threshold,
                    "passed": m.passed,
                }
                for m in self.metrics
            ],
        }


class RCAEvaluator:
    """Evaluator for the RCA system with gating capabilities.
    
    This evaluator runs metrics on a set of cases and feedback,
    and determines whether changes should be approved based on
    metric thresholds.
    """
    
    # Metrics that must pass for changes to be approved
    BLOCKING_METRICS = [
        "top_3_hit_rate",
        "citation_coverage",
        "assumption_check_rate",
    ]
    
    def __init__(
        self,
        metrics: Optional[RCAMetrics] = None,
        blocking_metrics: Optional[list[str]] = None,
        results_dir: str = "./data/evals",
    ):
        self.metrics = metrics or RCAMetrics()
        self.blocking_metrics = blocking_metrics or self.BLOCKING_METRICS
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate(
        self,
        cases: list[dict],
        reports: list[dict],
        feedback: list[dict],
        save_results: bool = True,
    ) -> EvaluationResult:
        """Run a complete evaluation.
        
        Args:
            cases: List of case data
            reports: List of report data
            feedback: List of feedback data
            save_results: Whether to save results to disk
            
        Returns:
            EvaluationResult with all metrics and gating decision
        """
        timestamp = datetime.utcnow()
        evaluation_id = f"EVAL-{timestamp.strftime('%Y%m%d%H%M%S')}"
        
        # Compute all metrics
        metric_results = self.metrics.compute_all(cases, reports, feedback)
        
        # Count passed/failed
        passed = sum(1 for m in metric_results if m.passed is True)
        failed = sum(1 for m in metric_results if m.passed is False)
        
        # Check blocking metrics
        blocking_failures = []
        for m in metric_results:
            if m.metric_name in self.blocking_metrics and m.passed is False:
                blocking_failures.append(
                    f"{m.metric_name}: {m.value:.2%} < {m.threshold:.2%}"
                )
        
        approved = len(blocking_failures) == 0
        
        result = EvaluationResult(
            evaluation_id=evaluation_id,
            timestamp=timestamp,
            metrics=metric_results,
            total_metrics=len(metric_results),
            passed_metrics=passed,
            failed_metrics=failed,
            approved=approved,
            blocking_failures=blocking_failures,
            case_count=len(cases),
            feedback_count=len(feedback),
        )
        
        if save_results:
            self._save_results(result)
        
        return result
    
    def _save_results(self, result: EvaluationResult) -> None:
        """Save evaluation results to disk."""
        output_path = self.results_dir / f"{result.evaluation_id}.json"
        
        with open(output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
    
    def load_results(self, evaluation_id: str) -> Optional[dict]:
        """Load saved evaluation results."""
        result_path = self.results_dir / f"{evaluation_id}.json"
        
        if not result_path.exists():
            return None
        
        with open(result_path) as f:
            return json.load(f)
    
    def compare_evaluations(
        self,
        baseline_id: str,
        candidate_id: str,
    ) -> dict:
        """Compare two evaluation runs.
        
        Args:
            baseline_id: ID of baseline evaluation
            candidate_id: ID of candidate evaluation
            
        Returns:
            Dict with comparison results
        """
        baseline = self.load_results(baseline_id)
        candidate = self.load_results(candidate_id)
        
        if not baseline or not candidate:
            return {"error": "Could not load one or both evaluations"}
        
        # Build metric lookup
        baseline_metrics = {m["name"]: m for m in baseline["metrics"]}
        candidate_metrics = {m["name"]: m for m in candidate["metrics"]}
        
        # Compare each metric
        comparisons = []
        for name, baseline_m in baseline_metrics.items():
            candidate_m = candidate_metrics.get(name)
            if candidate_m:
                delta = candidate_m["value"] - baseline_m["value"]
                comparisons.append({
                    "metric": name,
                    "baseline": baseline_m["value"],
                    "candidate": candidate_m["value"],
                    "delta": delta,
                    "improved": delta > 0 if baseline_m["threshold"] else None,
                })
        
        # Determine if candidate is better
        improvements = sum(1 for c in comparisons if c.get("improved"))
        regressions = sum(1 for c in comparisons if c.get("improved") is False)
        
        return {
            "baseline_id": baseline_id,
            "candidate_id": candidate_id,
            "baseline_approved": baseline["approved"],
            "candidate_approved": candidate["approved"],
            "comparisons": comparisons,
            "improvements": improvements,
            "regressions": regressions,
            "recommendation": "approve" if candidate["approved"] and improvements >= regressions else "reject",
        }
    
    def gate_change(
        self,
        change_description: str,
        cases: list[dict],
        reports: list[dict],
        feedback: list[dict],
        baseline_id: Optional[str] = None,
    ) -> dict:
        """Gate a proposed change based on evaluation results.
        
        Args:
            change_description: Description of the proposed change
            cases: Test cases to evaluate
            reports: Reports from the test run
            feedback: Feedback data
            baseline_id: Optional baseline evaluation to compare against
            
        Returns:
            Dict with gating decision and reasoning
        """
        # Run evaluation
        result = self.evaluate(cases, reports, feedback)
        
        decision = {
            "change_description": change_description,
            "evaluation_id": result.evaluation_id,
            "approved": result.approved,
            "metrics_passed": result.passed_metrics,
            "metrics_failed": result.failed_metrics,
            "blocking_failures": result.blocking_failures,
        }
        
        # Compare to baseline if provided
        if baseline_id:
            comparison = self.compare_evaluations(baseline_id, result.evaluation_id)
            decision["comparison"] = comparison
            
            # Override approval if regression
            if comparison.get("recommendation") == "reject":
                decision["approved"] = False
                decision["rejection_reason"] = "Regression from baseline"
        
        return decision
