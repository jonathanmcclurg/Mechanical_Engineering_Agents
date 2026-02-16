"""Feedback analyzer for the learning loop.

Analyzes engineer feedback to identify patterns and suggest
improvements to prompts, retrieval, and analysis recipes.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional
import json
from pathlib import Path


@dataclass
class ImprovementSuggestion:
    """A suggested improvement based on feedback analysis."""
    
    suggestion_id: str
    category: str  # prompt, retrieval, recipe, tool
    priority: str  # high, medium, low
    description: str
    evidence: list[str]
    affected_components: list[str]
    estimated_impact: str


class FeedbackAnalyzer:
    """Analyzes feedback to drive system improvements."""
    
    def __init__(self, feedback_dir: str = "./data/feedback"):
        self.feedback_dir = Path(feedback_dir)
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
    
    def analyze(self, feedback: list[dict]) -> dict:
        """Analyze feedback and generate improvement suggestions.
        
        Args:
            feedback: List of feedback records
            
        Returns:
            Dict with analysis results and suggestions
        """
        if not feedback:
            return {
                "feedback_count": 0,
                "analysis": {},
                "suggestions": [],
            }
        
        analysis = {
            "feedback_count": len(feedback),
            "hypothesis_accuracy": self._analyze_hypothesis_accuracy(feedback),
            "stats_appropriateness": self._analyze_stats_methods(feedback),
            "citation_quality": self._analyze_citations(feedback),
            "missing_data": self._analyze_missing_data(feedback),
            "common_root_causes": self._analyze_root_causes(feedback),
        }
        
        suggestions = self._generate_suggestions(analysis, feedback)
        
        return {
            "analysis": analysis,
            "suggestions": [s.__dict__ for s in suggestions],
            "timestamp": datetime.utcnow().isoformat(),
        }
    
    def _analyze_hypothesis_accuracy(self, feedback: list[dict]) -> dict:
        """Analyze hypothesis accuracy patterns."""
        correct_hypotheses = []
        incorrect_hypotheses = []
        missed_root_causes = []
        
        for fb in feedback:
            data = fb.get("data", fb)
            
            hyp_feedback = data.get("hypothesis_feedback", [])
            for hf in hyp_feedback:
                if hf.get("outcome") == "correct":
                    correct_hypotheses.append(hf)
                elif hf.get("outcome") == "incorrect":
                    incorrect_hypotheses.append(hf)
            
            actual = data.get("actual_root_cause")
            if actual and not data.get("root_cause_was_in_top_3"):
                missed_root_causes.append(actual)
        
        return {
            "correct_count": len(correct_hypotheses),
            "incorrect_count": len(incorrect_hypotheses),
            "missed_root_causes": missed_root_causes,
            "top_3_miss_rate": len(missed_root_causes) / len(feedback) if feedback else 0,
        }
    
    def _analyze_stats_methods(self, feedback: list[dict]) -> dict:
        """Analyze statistical method appropriateness."""
        appropriate = 0
        inappropriate = 0
        better_alternatives = Counter()
        
        for fb in feedback:
            data = fb.get("data", fb)
            
            for sf in data.get("stats_feedback", []):
                if sf.get("was_appropriate"):
                    appropriate += 1
                else:
                    inappropriate += 1
                    alt = sf.get("better_alternative")
                    if alt:
                        better_alternatives[alt] += 1
        
        return {
            "appropriate_count": appropriate,
            "inappropriate_count": inappropriate,
            "appropriateness_rate": appropriate / (appropriate + inappropriate) if (appropriate + inappropriate) > 0 else 1.0,
            "suggested_alternatives": dict(better_alternatives.most_common(5)),
        }
    
    def _analyze_citations(self, feedback: list[dict]) -> dict:
        """Analyze citation quality and gaps."""
        helpful_sections = Counter()
        missing_sections = Counter()
        
        for fb in feedback:
            data = fb.get("data", fb)
            
            for section in data.get("product_guide_sections_helpful", []):
                helpful_sections[section] += 1
            
            for section in data.get("product_guide_sections_missing", []):
                missing_sections[section] += 1
        
        return {
            "most_helpful_sections": dict(helpful_sections.most_common(10)),
            "commonly_missing_sections": dict(missing_sections.most_common(10)),
        }
    
    def _analyze_missing_data(self, feedback: list[dict]) -> dict:
        """Analyze what evidence was commonly missing."""
        missing_evidence = Counter()
        
        for fb in feedback:
            data = fb.get("data", fb)
            
            missing = data.get("important_evidence_missing")
            if missing:
                # Extract key terms
                for term in missing.lower().split():
                    if len(term) > 3:
                        missing_evidence[term] += 1
        
        return {
            "commonly_missing": dict(missing_evidence.most_common(10)),
        }
    
    def _analyze_root_causes(self, feedback: list[dict]) -> dict:
        """Analyze actual root causes to improve hypothesis generation."""
        root_causes = Counter()
        
        for fb in feedback:
            data = fb.get("data", fb)
            
            actual = data.get("actual_root_cause")
            if actual:
                root_causes[actual] += 1
            
            # Also count confirmed hypotheses
            for hf in data.get("hypothesis_feedback", []):
                if hf.get("was_actual_root_cause"):
                    # Would need hypothesis title here
                    pass
        
        return {
            "common_root_causes": dict(root_causes.most_common(10)),
        }
    
    def _generate_suggestions(
        self,
        analysis: dict,
        feedback: list[dict],
    ) -> list[ImprovementSuggestion]:
        """Generate improvement suggestions based on analysis."""
        suggestions = []
        suggestion_id = 0
        
        # Check for high miss rate
        hyp_analysis = analysis.get("hypothesis_accuracy", {})
        if hyp_analysis.get("top_3_miss_rate", 0) > 0.3:
            missed = hyp_analysis.get("missed_root_causes", [])
            suggestion_id += 1
            suggestions.append(ImprovementSuggestion(
                suggestion_id=f"SUG-{suggestion_id:03d}",
                category="recipe",
                priority="high",
                description="Add commonly missed root causes to analysis recipes",
                evidence=missed[:5],
                affected_components=["analysis_recipes", "hypothesis_agent"],
                estimated_impact="Could improve top-3 hit rate by 10-20%",
            ))
        
        # Check for inappropriate stats methods
        stats_analysis = analysis.get("stats_appropriateness", {})
        if stats_analysis.get("appropriateness_rate", 1.0) < 0.85:
            alternatives = stats_analysis.get("suggested_alternatives", {})
            suggestion_id += 1
            suggestions.append(ImprovementSuggestion(
                suggestion_id=f"SUG-{suggestion_id:03d}",
                category="tool",
                priority="high",
                description="Update statistical method selection logic",
                evidence=[f"Engineers suggested {k} ({v} times)" for k, v in list(alternatives.items())[:3]],
                affected_components=["stats_agent", "analysis_recipes"],
                estimated_impact="Could improve method appropriateness by 10-15%",
            ))
        
        # Check for missing product guide sections
        citation_analysis = analysis.get("citation_quality", {})
        missing_sections = citation_analysis.get("commonly_missing_sections", {})
        if missing_sections:
            top_missing = list(missing_sections.items())[:3]
            suggestion_id += 1
            suggestions.append(ImprovementSuggestion(
                suggestion_id=f"SUG-{suggestion_id:03d}",
                category="retrieval",
                priority="medium",
                description="Add commonly needed product guide sections to recipe retrieval",
                evidence=[f"'{section}' missing {count} times" for section, count in top_missing],
                affected_components=["product_guide_agent", "analysis_recipes"],
                estimated_impact="Could improve citation coverage and relevance",
            ))
        
        # Check for missing evidence
        missing_data = analysis.get("missing_data", {})
        commonly_missing = missing_data.get("commonly_missing", {})
        if commonly_missing:
            top_missing = list(commonly_missing.items())[:3]
            suggestion_id += 1
            suggestions.append(ImprovementSuggestion(
                suggestion_id=f"SUG-{suggestion_id:03d}",
                category="retrieval",
                priority="medium",
                description="Add commonly missing data sources or retrieval queries",
                evidence=[f"'{term}' mentioned {count} times as missing" for term, count in top_missing],
                affected_components=["research_agent", "test_plan_agent"],
                estimated_impact="Could reduce evidence gaps",
            ))
        
        return suggestions
    
    def save_analysis(self, analysis: dict) -> str:
        """Save analysis results to disk.
        
        Returns:
            Path to saved file
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        filename = f"analysis_{timestamp}.json"
        output_path = self.feedback_dir / filename
        
        with open(output_path, "w") as f:
            json.dump(analysis, f, indent=2)
        
        return str(output_path)
