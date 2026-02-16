"""Intake Triage Agent - normalizes failure cases and identifies missing data."""

from datetime import datetime
from typing import Any, Optional

from .base_agent import BaseRCAAgent, AgentOutput
from src.schemas.case import FailureCase, CaseStatus
from config.analysis_recipes.loader import get_recipe_for_failure


class IntakeTriageAgent(BaseRCAAgent):
    """Agent that normalizes incoming failure cases and identifies gaps."""
    
    name = "IntakeTriageAgent"
    description = "Normalizes failure records, identifies missing fields, and selects analysis recipe"
    
    system_prompt = """You are an intake triage specialist for manufacturing root cause analysis.
Your job is to:
1. Review incoming failure reports and normalize the data
2. Identify any missing critical information
3. Select the appropriate analysis recipe for this failure type
4. Flag any urgent issues that need immediate attention

Be precise and systematic. Missing data should be clearly noted."""

    def execute(self, context: dict[str, Any]) -> AgentOutput:
        """Process an incoming failure case.
        
        Args:
            context: Should contain 'raw_case' with the incoming failure data
            
        Returns:
            AgentOutput with normalized case and identified gaps
        """
        self.log("Starting intake triage")
        
        raw_case = context.get("raw_case", {})
        
        # Attempt to construct a FailureCase from raw data
        try:
            case = self._normalize_case(raw_case)
        except Exception as e:
            return AgentOutput(
                agent_name=self.name,
                success=False,
                error_message=f"Failed to normalize case: {str(e)}",
                data={"raw_case": raw_case}
            )
        
        # Load analysis recipe for this failure type
        recipe = get_recipe_for_failure(case.failure_type)
        
        # Identify missing fields
        missing_fields = self._identify_missing_fields(case, recipe)
        
        # Identify data quality issues
        quality_issues = self._check_data_quality(case)
        
        # Determine urgency
        urgency = self._assess_urgency(case)
        deterministic_reasoning = f"Case normalized. {len(missing_fields)} missing fields identified."
        llm_reasoning, used_llm = self._llm_analysis_or_fallback(
            objective="Summarize intake triage findings and key data gaps.",
            context_payload={
                "case_id": case.case_id,
                "failure_type": case.failure_type,
                "missing_fields": missing_fields,
                "quality_issues": quality_issues,
                "urgency": urgency,
                "recipe_name": recipe.name if recipe else None,
            },
            fallback_reasoning=deterministic_reasoning,
        )
        
        self.log(f"Triage complete. Missing fields: {len(missing_fields)}, Quality issues: {len(quality_issues)}")
        
        return AgentOutput(
            agent_name=self.name,
            success=True,
            data={
                "case": case.model_dump(),
                "recipe_id": recipe.recipe_id if recipe else None,
                "recipe_name": recipe.name if recipe else None,
                "missing_fields": missing_fields,
                "quality_issues": quality_issues,
                "urgency": urgency,
                "required_data_sources": recipe.data_requirements if recipe else [],
                "common_hypotheses": recipe.common_hypotheses if recipe else [],
                "llm_analysis_used": used_llm,
                "llm_analysis": llm_reasoning,
            },
            confidence=1.0 if not missing_fields else 0.8,
            reasoning=llm_reasoning
        )
    
    def _normalize_case(self, raw_case: dict) -> FailureCase:
        """Normalize raw case data into a FailureCase."""
        # Handle common variations in field names
        field_mappings = {
            "part": "part_number",
            "pn": "part_number",
            "sn": "serial_number",
            "serial": "serial_number",
            "lot": "lot_number",
            "batch": "lot_number",
            "failure_date": "failure_datetime",
            "date": "failure_datetime",
            "description": "failure_description",
            "type": "failure_type",
            "station": "station_id",
            "line": "line_id",
            "value": "test_value",
            "measured": "test_value",
        }
        
        normalized = {}
        for key, value in raw_case.items():
            # Check if this is an alternate name
            normalized_key = field_mappings.get(key.lower(), key)
            normalized[normalized_key] = value
        
        # Ensure required fields
        if "case_id" not in normalized:
            normalized["case_id"] = f"CASE-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        if "failure_datetime" not in normalized:
            normalized["failure_datetime"] = datetime.utcnow()
        
        return FailureCase(**normalized)
    
    def _identify_missing_fields(
        self, 
        case: FailureCase, 
        recipe: Optional[Any]
    ) -> list[dict]:
        """Identify missing fields based on recipe requirements."""
        missing = []
        
        if recipe is None:
            # Basic required fields
            basic_required = ["failure_type", "failure_description", "part_number"]
            for field in basic_required:
                if not getattr(case, field, None):
                    missing.append({
                        "field": field,
                        "importance": "critical",
                        "reason": "Required for any analysis"
                    })
            return missing
        
        # Check recipe-specific requirements
        for field in recipe.required_case_fields:
            value = getattr(case, field, None)
            if value is None:
                missing.append({
                    "field": field,
                    "importance": "high",
                    "reason": f"Required by analysis recipe '{recipe.name}'"
                })
        
        # Check component lot traceability
        if case.component_lots is None or len(case.component_lots) == 0:
            missing.append({
                "field": "component_lots",
                "importance": "medium",
                "reason": "Component lot traceability enables root cause isolation"
            })
        
        return missing
    
    def _check_data_quality(self, case: FailureCase) -> list[dict]:
        """Check for data quality issues."""
        issues = []
        
        # Check test value against spec limits
        if case.test_value is not None:
            if case.spec_lower is not None and case.test_value < case.spec_lower:
                pass  # Expected for a failure
            elif case.spec_upper is not None and case.test_value > case.spec_upper:
                pass  # Expected for a failure
            elif case.spec_lower is not None and case.spec_upper is not None:
                if case.spec_lower <= case.test_value <= case.spec_upper:
                    issues.append({
                        "field": "test_value",
                        "issue": "Test value is within spec but case is marked as failure",
                        "severity": "warning"
                    })
        
        # Check for future dates
        if case.failure_datetime > datetime.utcnow():
            issues.append({
                "field": "failure_datetime",
                "issue": "Failure datetime is in the future",
                "severity": "error"
            })
        
        # Check description length
        if len(case.failure_description) < 20:
            issues.append({
                "field": "failure_description",
                "issue": "Description is very short, may lack detail",
                "severity": "warning"
            })
        
        return issues
    
    def _assess_urgency(self, case: FailureCase) -> dict:
        """Assess the urgency of the case."""
        urgency_level = "normal"
        reasons = []
        
        # Check for recent failure
        hours_since_failure = (datetime.utcnow() - case.failure_datetime).total_seconds() / 3600
        if hours_since_failure < 4:
            urgency_level = "high"
            reasons.append("Recent failure (< 4 hours ago)")
        
        # Check for certain failure types that might indicate safety issues
        safety_keywords = ["leak", "crack", "fracture", "safety", "critical"]
        if any(kw in case.failure_type.lower() for kw in safety_keywords):
            urgency_level = "high"
            reasons.append(f"Failure type '{case.failure_type}' may have safety implications")
        
        # Check description for urgency indicators
        urgent_keywords = ["repeat", "multiple", "increasing", "trend", "customer"]
        if any(kw in case.failure_description.lower() for kw in urgent_keywords):
            if urgency_level != "high":
                urgency_level = "elevated"
            reasons.append("Description indicates potential systemic issue")
        
        return {
            "level": urgency_level,
            "reasons": reasons,
        }
