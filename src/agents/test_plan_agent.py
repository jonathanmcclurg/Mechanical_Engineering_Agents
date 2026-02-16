"""Test Plan Agent - selects data pulls needed to test hypotheses."""

from typing import Any

from .base_agent import BaseRCAAgent, AgentOutput
from src.tools.sql_tool import SQLTool


class TestPlanAgent(BaseRCAAgent):
    """Agent that plans data pulls needed for statistical testing."""
    
    name = "TestPlanAgent"
    description = "Selects internal API/SQL pulls required to feed statistical checks"
    
    system_prompt = """You are a test planning specialist for manufacturing root cause analysis.
Your job is to:
1. Review the hypotheses and their recommended statistical tests
2. Determine what data is needed to run each test
3. Create a data pull plan that minimizes redundant queries
4. Ensure all required columns and time ranges are specified

Be efficient - combine queries where possible to reduce database load."""

    def execute(self, context: dict[str, Any]) -> AgentOutput:
        """Create data pull plan for testing hypotheses.
        
        Args:
            context: Should contain:
                - 'hypotheses': List of hypothesis dicts
                - 'recipe': Analysis recipe
                - 'case': Failure case
                - 'available_sources': List of available data sources
                
        Returns:
            AgentOutput with data pull plan
        """
        self.log("Creating test plan")
        
        hypotheses = context.get("hypotheses", [])
        recipe = context.get("recipe")
        case = context.get("case", {})
        
        # Collect all required data sources and columns
        data_requirements = {}
        
        for hyp in hypotheses:
            if isinstance(hyp, dict):
                tests = hyp.get("recommended_tests", [])
                sources = hyp.get("required_data_sources", [])
            else:
                tests = hyp.recommended_tests
                sources = hyp.required_data_sources
            
            for source in sources:
                if source not in data_requirements:
                    data_requirements[source] = {
                        "columns": set(),
                        "filters": {},
                        "time_window": "90d",
                        "hypotheses": [],
                    }
                
                data_requirements[source]["hypotheses"].append(
                    hyp.get("hypothesis_id") if isinstance(hyp, dict) else hyp.hypothesis_id
                )
            
            for test in tests:
                if isinstance(test, dict):
                    target = test.get("target_variable")
                    grouping = test.get("grouping_key")
                else:
                    target = test.target_variable
                    grouping = test.grouping_key
                
                # Add columns to all relevant sources
                for source in sources:
                    if source in data_requirements:
                        if target:
                            data_requirements[source]["columns"].add(target)
                        if grouping:
                            data_requirements[source]["columns"].add(grouping)
        
        # Add common columns needed for joining — but only if valid for the source
        common_columns = ["serial_number", "lot_number", "test_datetime", "measurement_datetime"]
        for source, reqs in data_requirements.items():
            allowed = set(
                SQLTool.ALLOWED_SOURCES.get(source, {}).get("allowed_columns", [])
            )
            for col in common_columns:
                if col in allowed:
                    reqs["columns"].add(col)
        
        # Add filters from case — only for columns that exist in the source
        for source, reqs in data_requirements.items():
            allowed = set(
                SQLTool.ALLOWED_SOURCES.get(source, {}).get("allowed_columns", [])
            )
            if case.get("lot_number") and "lot_number" in allowed:
                reqs["filters"]["lot_number"] = case["lot_number"]
            if case.get("part_number") and "part_number" in allowed:
                reqs["filters"]["part_number"] = case["part_number"]
        
        # Convert sets to lists for serialization
        pull_plan = []
        for source, reqs in data_requirements.items():
            pull_plan.append({
                "source_name": source,
                "columns": list(reqs["columns"]),
                "filters": reqs["filters"],
                "time_window": reqs["time_window"],
                "hypotheses_served": reqs["hypotheses"],
                "priority": len(reqs["hypotheses"]),  # Higher priority for more hypotheses
            })
        
        # Sort by priority
        pull_plan.sort(key=lambda x: x["priority"], reverse=True)
        deterministic_reasoning = (
            f"Created data pull plan covering {len(pull_plan)} sources for "
            f"{len(hypotheses)} hypotheses"
        )
        llm_reasoning, used_llm = self._llm_analysis_or_fallback(
            objective="Review and summarize adequacy of the statistical test data pull plan.",
            context_payload={
                "failure_type": case.get("failure_type"),
                "hypothesis_count": len(hypotheses),
                "pull_plan": pull_plan[:8],
                "recipe_name": getattr(recipe, "name", None),
            },
            fallback_reasoning=deterministic_reasoning,
        )
        
        self.log(f"Created pull plan with {len(pull_plan)} data sources")
        
        return AgentOutput(
            agent_name=self.name,
            success=True,
            data={
                "pull_plan": pull_plan,
                "total_sources": len(pull_plan),
                "estimated_queries": len(pull_plan),
                "llm_analysis_used": used_llm,
                "llm_analysis": llm_reasoning,
            },
            confidence=0.9,
            reasoning=llm_reasoning
        )
