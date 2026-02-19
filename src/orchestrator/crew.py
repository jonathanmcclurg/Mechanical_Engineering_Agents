"""CrewAI-style orchestration for the RCA agent workflow.

This module provides a CrewAI-compatible interface for orchestrating
the multi-agent RCA workflow. It can be adapted to use actual CrewAI
or run as a standalone workflow.
"""

from datetime import datetime
from typing import Any, Optional
import pandas as pd

from src.agents import (
    IntakeTriageAgent,
    ProductGuideAgent,
    PrivateDataResearchAgent,
    HypothesisAgent,
    StatsAnalysisAgent,
    TestPlanAgent,
    CriticEvidenceAgent,
    ReportAgent,
)
from src.tools.stats_tool import StatsTool
from src.tools.rag_tool import RAGTool
from src.tools.sql_tool import SQLTool
from src.tools.data_fetch_tool import DataFetchTool
from src.tools.data_catalog import DataCatalog
from config.analysis_recipes.loader import get_recipe_for_failure, normalize_recipe_mode
from config.settings import get_settings


class RCACrew:
    """Orchestrates the multi-agent RCA workflow.
    
    This class coordinates the agents in a sequential workflow:
    Intake → ProductGuide → Research → Hypothesis → TestPlan → Stats → Critic → Report
    
    With an optional loop back to Research if the Critic finds insufficient evidence.
    """
    
    def __init__(
        self,
        llm: Any = None,
        stats_tool: StatsTool = None,
        rag_tool: RAGTool = None,
        sql_tool: SQLTool = None,
        data_fetch_tool: DataFetchTool = None,
        data_catalog: DataCatalog = None,
        verbose: bool = True,
        max_research_loops: int = 2,
    ):
        self.llm = llm
        self.verbose = verbose
        self.max_research_loops = max_research_loops
        
        # Initialize tools
        self.stats_tool = stats_tool or StatsTool()
        self.rag_tool = rag_tool or RAGTool()
        self.sql_tool = sql_tool or SQLTool(mock_mode=True)
        self.data_catalog = data_catalog or DataCatalog()
        self.data_fetch_tool = data_fetch_tool or DataFetchTool(
            mock_mode=True,
            data_catalog=self.data_catalog,
        )
        
        # Initialize agents
        self._init_agents()
        
        # Workflow state
        self._workflow_log: list[dict] = []
        self._stage_order = [
            "intake",
            "product_guide",
            "research",
            "hypothesis",
            "test_plan",
            "stats",
            "critic",
            "report",
        ]
    
    def _init_agents(self):
        """Initialize all agents with shared tools."""
        common_kwargs = {"llm": self.llm, "verbose": self.verbose}
        
        self.intake_agent = IntakeTriageAgent(**common_kwargs)
        self.product_guide_agent = ProductGuideAgent(
            rag_tool=self.rag_tool, **common_kwargs
        )
        self.research_agent = PrivateDataResearchAgent(
            sql_tool=self.sql_tool,
            data_fetch_tool=self.data_fetch_tool,
            data_catalog=self.data_catalog,
            **common_kwargs
        )
        self.hypothesis_agent = HypothesisAgent(**common_kwargs)
        self.test_plan_agent = TestPlanAgent(**common_kwargs)
        self.stats_agent = StatsAnalysisAgent(
            stats_tool=self.stats_tool,
            data_fetch_tool=self.data_fetch_tool,
            **common_kwargs
        )
        self.critic_agent = CriticEvidenceAgent(**common_kwargs)
        self.report_agent = ReportAgent(**common_kwargs)
    
    def _log(self, message: str, level: str = "info"):
        """Log workflow progress."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "message": message,
        }
        self._workflow_log.append(entry)
        if self.verbose:
            print(f"[RCACrew] {message}")
    
    def run(self, raw_case: dict) -> dict:
        """Run the complete RCA workflow.
        
        Args:
            raw_case: Raw failure case data
            
        Returns:
            Dict containing the final report and all intermediate outputs
        """
        start_time = datetime.utcnow()
        # Ensure each run has an isolated workflow log.
        self._workflow_log = []
        self._reset_agent_logs()
        self._log("Starting RCA workflow")
        
        context = {"raw_case": raw_case}
        outputs = {}
        
        try:
            # Step 1: Intake Triage
            self._log("Step 1: Intake Triage")
            intake_output = self.intake_agent.execute(context)
            outputs["intake"] = intake_output
            
            if not intake_output.success:
                raise ValueError(f"Intake failed: {intake_output.error_message}")
            
            # Update context
            context["case"] = intake_output.data.get("case", {})
            
            # Load recipe
            failure_type = context["case"].get("failure_type", "")
            recipe_mode = normalize_recipe_mode(get_settings().recipe_mode)
            recipe = get_recipe_for_failure(failure_type, recipe_mode=recipe_mode)
            context["recipe"] = recipe
            self._log(
                f"Recipe mode={recipe_mode}. Loaded recipe: {recipe.name if recipe else 'None'}"
            )
            
            # Step 2: Product Guide Retrieval
            self._log("Step 2: Product Guide Retrieval")
            product_guide_output = self.product_guide_agent.execute(context)
            outputs["product_guide"] = product_guide_output
            # Keep full output shape for downstream agents that expect {"data": ...}
            context["product_guide_output"] = product_guide_output.model_dump()
            
            # Step 3: Private Data Research
            self._log("Step 3: Private Data Research")
            research_output = self.research_agent.execute(context)
            outputs["research"] = research_output
            context["research_output"] = research_output.model_dump()
            
            # Step 4: Hypothesis Generation
            self._log("Step 4: Hypothesis Generation")
            hypothesis_output = self.hypothesis_agent.execute(context)
            outputs["hypothesis"] = hypothesis_output
            context["hypothesis_output"] = hypothesis_output.model_dump()
            context["hypotheses"] = hypothesis_output.data.get("hypotheses", [])
            
            # Step 5: Test Planning
            self._log("Step 5: Test Planning")
            test_plan_output = self.test_plan_agent.execute(context)
            outputs["test_plan"] = test_plan_output
            
            # Execute data pulls based on test plan
            data_frames = self._execute_data_pulls(test_plan_output.data.get("pull_plan", []))
            context["data_frames"] = data_frames
            
            # Research loop
            research_loops = 0
            while research_loops < self.max_research_loops:
                # Step 6: Statistical Analysis
                self._log(f"Step 6: Statistical Analysis (loop {research_loops + 1})")
                stats_output = self.stats_agent.execute(context)
                outputs["stats"] = stats_output
                context["stats_output"] = stats_output.model_dump()
                context["stats_results"] = stats_output.data
                
                # Step 7: Evidence Critique
                self._log("Step 7: Evidence Critique")
                context["evidence"] = stats_output.data.get("evidence", [])
                critic_output = self.critic_agent.execute(context)
                outputs["critic"] = critic_output
                context["critic_output"] = critic_output.model_dump()
                
                # Check if more evidence is needed
                if not critic_output.data.get("needs_more_evidence", False):
                    break
                
                self._log("Critic requested more evidence, running additional research")
                research_loops += 1
                
                # Run additional research
                research_output = self.research_agent.execute(context)
                outputs["research_additional"] = research_output
                context["research_output"] = research_output.model_dump()
            
            # Step 8: Report Generation
            self._log("Step 8: Report Generation")
            report_output = self.report_agent.execute(context)
            outputs["report"] = report_output
            
            # Calculate total time
            total_time = (datetime.utcnow() - start_time).total_seconds()
            self._log(f"Workflow complete in {total_time:.1f}s")
            
            return {
                "success": True,
                "report": report_output.data.get("report"),
                "report_id": report_output.data.get("report_id"),
                "outputs": outputs,
                "workflow_log": self._workflow_log,
                "agent_logs": self._collect_agent_logs(),
                "total_time_seconds": total_time,
            }
            
        except Exception as e:
            self._log(f"Workflow failed: {str(e)}", "error")
            return {
                "success": False,
                "error": str(e),
                "outputs": outputs,
                "workflow_log": self._workflow_log,
                "agent_logs": self._collect_agent_logs(),
            }

    def get_stage_order(self) -> list[str]:
        """Get the executable stage order for manual flow debugging."""
        return self._stage_order.copy()

    def start_stage_session(self, raw_case: dict) -> dict[str, Any]:
        """Initialize a state container for running the workflow stage-by-stage."""
        self._workflow_log = []
        self._reset_agent_logs()
        self._log("Starting stage debug session")
        return {
            "context": {"raw_case": raw_case},
            "outputs": {},
            "completed_stages": [],
        }

    def run_stage(
        self,
        stage: str,
        *,
        context: dict[str, Any],
        outputs: dict[str, Any],
    ) -> dict[str, Any]:
        """Run exactly one stage and mutate context/outputs in-place."""
        if stage not in self._stage_order:
            raise ValueError(f"Unknown stage '{stage}'")

        if stage == "intake":
            self._log("Step 1: Intake Triage")
            intake_output = self.intake_agent.execute(context)
            outputs["intake"] = intake_output
            if not intake_output.success:
                raise ValueError(f"Intake failed: {intake_output.error_message}")
            context["case"] = intake_output.data.get("case", {})
            self._ensure_recipe(context)
            stage_output = intake_output
        elif stage == "product_guide":
            self._require_context_keys(context, ["case"], stage)
            self._ensure_recipe(context)
            self._log("Step 2: Product Guide Retrieval")
            product_guide_output = self.product_guide_agent.execute(context)
            outputs["product_guide"] = product_guide_output
            context["product_guide_output"] = product_guide_output.model_dump()
            stage_output = product_guide_output
        elif stage == "research":
            self._require_context_keys(context, ["case"], stage)
            self._log("Step 3: Private Data Research")
            research_output = self.research_agent.execute(context)
            outputs["research"] = research_output
            context["research_output"] = research_output.model_dump()
            stage_output = research_output
        elif stage == "hypothesis":
            self._require_context_keys(context, ["case", "research_output"], stage)
            self._log("Step 4: Hypothesis Generation")
            hypothesis_output = self.hypothesis_agent.execute(context)
            outputs["hypothesis"] = hypothesis_output
            context["hypothesis_output"] = hypothesis_output.model_dump()
            context["hypotheses"] = hypothesis_output.data.get("hypotheses", [])
            stage_output = hypothesis_output
        elif stage == "test_plan":
            self._require_context_keys(context, ["case", "hypothesis_output"], stage)
            self._log("Step 5: Test Planning")
            test_plan_output = self.test_plan_agent.execute(context)
            outputs["test_plan"] = test_plan_output
            data_frames = self._execute_data_pulls(test_plan_output.data.get("pull_plan", []))
            context["data_frames"] = data_frames
            stage_output = test_plan_output
        elif stage == "stats":
            self._require_context_keys(context, ["data_frames"], stage)
            self._log("Step 6: Statistical Analysis")
            stats_output = self.stats_agent.execute(context)
            outputs["stats"] = stats_output
            context["stats_output"] = stats_output.model_dump()
            context["stats_results"] = stats_output.data
            stage_output = stats_output
        elif stage == "critic":
            self._require_context_keys(context, ["stats_output"], stage)
            self._log("Step 7: Evidence Critique")
            context["evidence"] = context["stats_output"].get("data", {}).get("evidence", [])
            critic_output = self.critic_agent.execute(context)
            outputs["critic"] = critic_output
            context["critic_output"] = critic_output.model_dump()
            stage_output = critic_output
        else:
            self._require_context_keys(context, ["critic_output"], stage)
            self._log("Step 8: Report Generation")
            report_output = self.report_agent.execute(context)
            outputs["report"] = report_output
            stage_output = report_output

        if not stage_output.success:
            raise ValueError(
                f"Stage '{stage}' failed: {stage_output.error_message or 'Unknown error'}"
            )

        return {
            "stage": stage,
            "output": stage_output,
            "workflow_log": self.get_workflow_log(),
            "agent_logs": self._collect_agent_logs(),
        }

    def _ensure_recipe(self, context: dict[str, Any]) -> None:
        """Load recipe into context when case is available and recipe is missing."""
        if "recipe" in context:
            return
        case = context.get("case") or {}
        failure_type = case.get("failure_type", "")
        recipe_mode = normalize_recipe_mode(get_settings().recipe_mode)
        recipe = get_recipe_for_failure(failure_type, recipe_mode=recipe_mode)
        context["recipe"] = recipe
        self._log(
            f"Recipe mode={recipe_mode}. Loaded recipe: {recipe.name if recipe else 'None'}"
        )

    def _require_context_keys(
        self,
        context: dict[str, Any],
        required: list[str],
        stage: str,
    ) -> None:
        """Validate prerequisites for a stage run."""
        missing = [key for key in required if key not in context]
        if missing:
            raise ValueError(
                f"Stage '{stage}' requires context keys: {', '.join(required)}. "
                f"Missing: {', '.join(missing)}."
            )
    
    def _execute_data_pulls(self, pull_plan: list[dict]) -> dict[str, pd.DataFrame]:
        """Execute data pulls based on the test plan.
        
        Args:
            pull_plan: List of data pull specifications
            
        Returns:
            Dict mapping source names to DataFrames
        """
        data_frames = {}
        
        for pull in pull_plan:
            source_name = pull.get("source_name")
            columns = pull.get("columns", [])
            filters = pull.get("filters", {})
            time_window = pull.get("time_window", "90d")
            
            # Filter columns and filters to only those allowed for this source
            source_def = self.sql_tool.ALLOWED_SOURCES.get(source_name)
            if source_def:
                allowed = set(source_def["allowed_columns"])
                valid_columns = [c for c in columns if c in allowed or c == "*"]
                dropped_cols = set(columns) - set(valid_columns)
                if dropped_cols:
                    self._log(
                        f"Dropped invalid columns for {source_name}: {dropped_cols}",
                        "debug",
                    )
                columns = valid_columns or ["*"]
                
                valid_filters = {k: v for k, v in filters.items() if k in allowed}
                dropped_filters = set(filters) - set(valid_filters)
                if dropped_filters:
                    self._log(
                        f"Dropped invalid filter keys for {source_name}: {dropped_filters}",
                        "debug",
                    )
                filters = valid_filters
            
            try:
                result = self.sql_tool.query_for_analysis(
                    source_name=source_name,
                    columns=columns,
                    filters=filters,
                    time_window=time_window,
                )
                data_frames[source_name] = result.data
                self._log(f"Pulled {result.row_count} rows from {source_name}")
            except Exception as e:
                self._log(f"Failed to pull from {source_name}: {e}", "warning")
        
        return data_frames

    def _collect_agent_logs(self) -> dict[str, list[dict]]:
        """Collect execution logs from all agents."""
        return {
            "intake": self.intake_agent.get_execution_log(),
            "product_guide": self.product_guide_agent.get_execution_log(),
            "research": self.research_agent.get_execution_log(),
            "hypothesis": self.hypothesis_agent.get_execution_log(),
            "test_plan": self.test_plan_agent.get_execution_log(),
            "stats": self.stats_agent.get_execution_log(),
            "critic": self.critic_agent.get_execution_log(),
            "report": self.report_agent.get_execution_log(),
        }

    def _reset_agent_logs(self) -> None:
        """Reset execution logs for all agents."""
        self.intake_agent.reset_execution_log()
        self.product_guide_agent.reset_execution_log()
        self.research_agent.reset_execution_log()
        self.hypothesis_agent.reset_execution_log()
        self.test_plan_agent.reset_execution_log()
        self.stats_agent.reset_execution_log()
        self.critic_agent.reset_execution_log()
        self.report_agent.reset_execution_log()
    
    def get_workflow_log(self) -> list[dict]:
        """Get the complete workflow log."""
        return self._workflow_log.copy()

    def get_agent_logs(self) -> dict[str, list[dict]]:
        """Get current per-agent execution logs."""
        return self._collect_agent_logs()
