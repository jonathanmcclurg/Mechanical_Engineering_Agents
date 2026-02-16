"""Stats Analysis Agent - executes statistical tests to validate hypotheses."""

from typing import Any
import pandas as pd

from .base_agent import BaseRCAAgent, AgentOutput
from src.tools.stats_tool import StatsTool, StatsResult
from src.tools.data_fetch_tool import DataFetchTool
from src.schemas.hypothesis import Hypothesis, StatisticalTest
from src.schemas.evidence import Evidence, Citation, EvidenceSource


class StatsAnalysisAgent(BaseRCAAgent):
    """Agent that executes statistical analysis to test hypotheses."""
    
    name = "StatsAnalysisAgent"
    description = "Executes statistical tests (control charts, t-tests, etc.) to validate/refute hypotheses"
    
    system_prompt = """You are a statistical analysis specialist for manufacturing root cause analysis.
Your job is to:
1. Execute the recommended statistical tests for each hypothesis
2. Fetch additional test data, ROA parameters, or operator buyoffs as needed via the data API
3. Check assumptions before running tests
4. Report results with appropriate effect sizes and confidence intervals
5. Flag any data quality issues or test limitations
6. Generate clear visualizations (control charts, etc.)

Be rigorous about assumptions. If a test's assumptions are violated, use the appropriate
nonparametric alternative or clearly note the limitation.

You can request data in tabular format with columns like:
- Test IDs (e.g., LEAK_RATE, FLOW_RATE, ELECTRICAL_CONTINUITY)
- ROA parameters (e.g., TORQUE_MAIN_HOUSING, GASKET_LOT, CURE_TIME)
- Operator buyoffs (e.g., BUYOFF_SEAL_PLACEMENT, BUYOFF_VISUAL_INSPECTION)
- Process parameters (e.g., AMBIENT_TEMP, OPERATOR_ID, SHIFT)"""

    def __init__(
        self, 
        stats_tool: StatsTool = None, 
        data_fetch_tool: DataFetchTool = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.stats_tool = stats_tool or StatsTool()
        self.data_fetch_tool = data_fetch_tool or DataFetchTool(mock_mode=True)
    
    def execute(self, context: dict[str, Any]) -> AgentOutput:
        """Execute statistical tests for hypotheses.
        
        Args:
            context: Should contain:
                - 'hypotheses': List of hypothesis dicts
                - 'data_frames': Dict mapping source names to DataFrames
                - 'case': The failure case
                
        Returns:
            AgentOutput with test results and evidence
        """
        self.log("Starting statistical analysis")
        
        hypotheses_data = context.get("hypotheses", [])
        data_frames = context.get("data_frames", {})
        case = context.get("case", {})
        
        # Convert hypothesis dicts back to objects if needed
        hypotheses = []
        for h in hypotheses_data:
            if isinstance(h, dict):
                hypotheses.append(Hypothesis(**h))
            else:
                hypotheses.append(h)
        
        all_results = []
        all_evidence = []
        all_citations = []
        
        for hypothesis in hypotheses:
            self.log(f"Testing hypothesis: {hypothesis.title}")
            
            hypothesis_results = {
                "hypothesis_id": hypothesis.hypothesis_id,
                "hypothesis_title": hypothesis.title,
                "tests_run": [],
                "overall_support": "inconclusive",
            }
            
            supports = 0
            refutes = 0
            
            for test_spec in hypothesis.recommended_tests:
                self.log(f"  Running {test_spec.test_type}")
                
                try:
                    result, evidence = self._run_test(
                        test_spec=test_spec,
                        data_frames=data_frames,
                        hypothesis=hypothesis,
                        case=case,
                    )
                    
                    hypothesis_results["tests_run"].append(result.to_dict())
                    
                    if evidence:
                        all_evidence.append(evidence)
                        all_citations.extend([c.model_dump() for c in evidence.citations])
                    
                    # Tally support/refute
                    if result.is_statistically_significant:
                        if result.is_practically_significant:
                            supports += 1
                        else:
                            # Statistically but not practically significant
                            pass
                    else:
                        refutes += 1
                        
                except Exception as e:
                    self.log(f"  Failed to run {test_spec.test_type}: {e}", "warning")
                    hypothesis_results["tests_run"].append({
                        "test_type": test_spec.test_type,
                        "error": str(e),
                    })
            
            # Determine overall support
            if supports > refutes and supports > 0:
                hypothesis_results["overall_support"] = "supported"
            elif refutes > supports:
                hypothesis_results["overall_support"] = "refuted"
            else:
                hypothesis_results["overall_support"] = "inconclusive"
            
            all_results.append(hypothesis_results)
        
        # Generate summary
        summary = self._generate_summary(all_results)
        total_tests_run = sum(len(r["tests_run"]) for r in all_results)
        deterministic_reasoning = (
            f"Executed statistical tests for {len(hypotheses)} hypotheses"
        )
        llm_reasoning, used_llm = self._llm_analysis_or_fallback(
            objective="Interpret statistical outcomes and summarize evidentiary strength.",
            context_payload={
                "hypothesis_count": len(hypotheses),
                "total_tests_run": total_tests_run,
                "summary": summary,
                "sample_results": all_results[:5],
            },
            fallback_reasoning=deterministic_reasoning,
        )
        
        self.log(f"Statistical analysis complete. Ran tests for {len(hypotheses)} hypotheses")
        
        return AgentOutput(
            agent_name=self.name,
            success=True,
            data={
                "hypothesis_results": all_results,
                "evidence": [e.model_dump() for e in all_evidence],
                "summary": summary,
                "total_tests_run": total_tests_run,
                "llm_analysis_used": used_llm,
                "llm_analysis": llm_reasoning,
            },
            citations_used=all_citations,
            confidence=0.85,
            reasoning=llm_reasoning
        )
    
    def _run_test(
        self,
        test_spec: StatisticalTest,
        data_frames: dict[str, pd.DataFrame],
        hypothesis: Hypothesis,
        case: dict,
    ) -> tuple[StatsResult, Evidence | None]:
        """Run a single statistical test.
        
        Returns:
            Tuple of (StatsResult, Evidence or None)
        """
        # Find appropriate data source
        df = self._get_data_for_test(test_spec, data_frames, hypothesis)
        
        if df is None or len(df) < 5:
            raise ValueError(f"Insufficient data for test. Need data with column '{test_spec.target_variable}'")
        
        # Run the appropriate test
        if test_spec.test_type == "xbar_s_chart":
            result = self.stats_tool.xbar_s_chart(
                data=df,
                value_column=test_spec.target_variable,
                subgroup_column=test_spec.grouping_key or "lot_number",
                control_rules=["beyond_3sigma", "run_of_8", "trend_of_6"],
                title=f"X̄-S Chart: {test_spec.target_variable} by {test_spec.grouping_key}"
            )
        
        elif test_spec.test_type in ["two_sample_ttest", "welch_ttest"]:
            result = self.stats_tool.two_sample_ttest(
                data=df,
                value_column=test_spec.target_variable,
                group_column=test_spec.grouping_key or "group",
                alpha=test_spec.alpha,
                practical_threshold=test_spec.practical_threshold or 0.5,
            )
        
        elif test_spec.test_type == "capability_study":
            # Get spec limits from case
            lsl = case.get("spec_lower", 0)
            usl = case.get("spec_upper", 100)
            result = self.stats_tool.capability_study(
                data=df,
                value_column=test_spec.target_variable,
                lsl=lsl,
                usl=usl,
            )
        
        elif test_spec.test_type == "one_way_anova":
            result = self.stats_tool.one_way_anova(
                data=df,
                value_column=test_spec.target_variable,
                group_column=test_spec.grouping_key or "group",
                alpha=test_spec.alpha,
            )
        
        elif test_spec.test_type == "correlation":
            result = self.stats_tool.correlation(
                data=df,
                x_column=test_spec.grouping_key or "x",
                y_column=test_spec.target_variable,
                alpha=test_spec.alpha,
            )
        
        elif test_spec.test_type == "individuals_mr":
            result = self.stats_tool.individuals_mr_chart(
                data=df,
                value_column=test_spec.target_variable,
                order_column=test_spec.grouping_key,
                title=f"I-MR Chart: {test_spec.target_variable}",
            )
        
        else:
            # Default to control chart
            self.log(f"Unknown test type {test_spec.test_type}, defaulting to control chart", "warning")
            result = self.stats_tool.xbar_s_chart(
                data=df,
                value_column=test_spec.target_variable,
                subgroup_column=test_spec.grouping_key or "lot_number",
            )
        
        # Create evidence from result
        evidence = self._create_evidence(result, hypothesis, test_spec)
        
        return result, evidence
    
    def _get_data_for_test(
        self,
        test_spec: StatisticalTest,
        data_frames: dict[str, pd.DataFrame],
        hypothesis: Hypothesis,
    ) -> pd.DataFrame | None:
        """Get the appropriate DataFrame for a test.
        
        If the target_variable and grouping_key live in different sources,
        attempts to merge them on ``serial_number``.
        """
        required_cols = [test_spec.target_variable]
        if test_spec.grouping_key:
            required_cols.append(test_spec.grouping_key)
        
        # 1. Look for a single source that already has all required columns
        for source_name, df in data_frames.items():
            if df is not None and all(col in df.columns for col in required_cols):
                return df
        
        # 2. Try merging two sources on serial_number when columns are split
        if len(required_cols) == 2 and test_spec.grouping_key:
            target_df = None
            grouping_df = None
            for _name, df in data_frames.items():
                if df is None:
                    continue
                if test_spec.target_variable in df.columns and "serial_number" in df.columns:
                    target_df = df
                if test_spec.grouping_key in df.columns and "serial_number" in df.columns:
                    grouping_df = df
            
            if (
                target_df is not None
                and grouping_df is not None
                and target_df is not grouping_df
            ):
                merged = pd.merge(
                    target_df[["serial_number", test_spec.target_variable]].drop_duplicates("serial_number"),
                    grouping_df[["serial_number", test_spec.grouping_key]].drop_duplicates("serial_number"),
                    on="serial_number",
                    how="inner",
                )
                if len(merged) >= 5:
                    self.log(f"  Merged data for {test_spec.target_variable} + {test_spec.grouping_key} ({len(merged)} rows)")
                    return merged
        
        # 3. Fall back to source with just the target variable
        for source in hypothesis.required_data_sources:
            if source in data_frames:
                df = data_frames[source]
                if df is not None and test_spec.target_variable in df.columns:
                    return df
        
        for source_name, df in data_frames.items():
            if df is not None and test_spec.target_variable in df.columns:
                return df
        
        # 4. Generate mock data for development
        return self._generate_mock_data(test_spec)
    
    def _generate_mock_data(self, test_spec: StatisticalTest) -> pd.DataFrame:
        """Generate mock data for development/testing."""
        import numpy as np
        
        n = 100
        data = {
            test_spec.target_variable: np.random.normal(100, 10, n),
        }
        
        if test_spec.grouping_key:
            data[test_spec.grouping_key] = [f"Group_{i % 5}" for i in range(n)]
        
        return pd.DataFrame(data)
    
    def fetch_data_for_analysis(
        self,
        test_ids: list[str] = None,
        roa_parameters: list[str] = None,
        operator_buyoffs: list[str] = None,
        process_parameters: list[str] = None,
        lot_numbers: list[str] = None,
        time_window: str = "90d",
    ) -> pd.DataFrame:
        """Fetch data from internal API for statistical analysis.
        
        This allows the stats agent to request specific columns of data
        for correlation analysis, hypothesis testing, etc.
        
        Args:
            test_ids: List of test IDs to include (e.g., LEAK_RATE, FLOW_RATE)
            roa_parameters: List of ROA parameters (e.g., TORQUE_MAIN_HOUSING)
            operator_buyoffs: List of operator buyoffs (e.g., BUYOFF_SEAL_PLACEMENT)
            process_parameters: List of process parameters (e.g., AMBIENT_TEMP)
            lot_numbers: Optional list of lot numbers to filter by
            time_window: Time window for data (default 90d)
            
        Returns:
            DataFrame with serial_number and requested columns
        """
        self.log(f"Fetching analysis data: tests={test_ids}, roa={roa_parameters}")
        
        result = self.data_fetch_tool.fetch_for_analysis(
            lot_numbers=lot_numbers,
            test_ids=test_ids or [],
            roa_parameters=roa_parameters or [],
            operator_buyoffs=operator_buyoffs or [],
            process_parameters=process_parameters or [],
            time_window=time_window,
        )
        
        if result.warnings:
            for warning in result.warnings:
                self.log(warning, "warning")
        
        self.log(f"Fetched {result.row_count} rows with columns: {result.columns}")
        
        return result.data
    
    def list_available_data_fields(self) -> dict[str, list[str]]:
        """List all available data fields that can be requested.
        
        Returns:
            Dict mapping category names to lists of field IDs
        """
        return {
            "test_ids": self.data_fetch_tool.list_test_ids(),
            "roa_parameters": self.data_fetch_tool.list_roa_parameters(),
            "operator_buyoffs": self.data_fetch_tool.list_operator_buyoffs(),
            "process_parameters": self.data_fetch_tool.list_process_parameters(),
        }
    
    def _create_evidence(
        self,
        result: StatsResult,
        hypothesis: Hypothesis,
        test_spec: StatisticalTest,
    ) -> Evidence:
        """Create an Evidence object from a statistical result."""
        # Determine if evidence supports or refutes (ensure bool, not None)
        supports = bool(
            result.is_statistically_significant and result.is_practically_significant
        )
        
        # Build claim based on result
        if result.method in ("xbar_s_chart", "individuals_mr"):
            if result.signals:
                claim = f"Control chart shows {len(result.signals)} signal(s): {', '.join(s.rule for s in result.signals)}"
            else:
                claim = "Control chart shows process is in statistical control"
        elif "ttest" in result.method:
            if result.is_statistically_significant:
                claim = f"Significant difference detected (p={result.p_value:.4f}, d={result.effect_size:.2f})"
            else:
                claim = f"No significant difference (p={result.p_value:.4f})"
        elif result.method == "correlation":
            r_val = result.effect_size or 0
            claim = f"Correlation r={r_val:.3f} (p={result.p_value:.4f})" if result.p_value else result.test_name
        elif result.method == "capability_study":
            claim = f"Process capability Cpk={result.cpk:.2f}"
        else:
            claim = f"{result.test_name}: p={result.p_value:.4f}" if result.p_value else result.test_name
        
        # Create citation for the statistical analysis
        citation = Citation(
            source_type=EvidenceSource.STATISTICAL_ANALYSIS,
            source_id=f"STAT-{result.timestamp.strftime('%Y%m%d%H%M%S')}",
            source_name=result.test_name,
            excerpt=claim,
            timestamp=result.timestamp,
        )
        
        # Calculate confidence based on result quality
        confidence = 0.7
        if result.assumptions_satisfied:
            confidence += 0.1
        if result.n_total >= 30:
            confidence += 0.1
        if not result.warnings:
            confidence += 0.05
        
        return Evidence(
            evidence_id=f"EV-{hypothesis.hypothesis_id}-{test_spec.test_type}",
            claim=claim,
            supports_hypothesis=supports,
            confidence=min(0.95, confidence),
            citations=[citation],
            statistical_result=result.to_dict(),
            chart_path=result.chart_path,
            agent_name=self.name,
        )
    
    def _generate_summary(self, all_results: list[dict]) -> dict:
        """Generate summary of statistical analysis."""
        summary = {
            "hypotheses_tested": len(all_results),
            "supported": 0,
            "refuted": 0,
            "inconclusive": 0,
            "key_findings": [],
        }
        
        for result in all_results:
            support = result.get("overall_support", "inconclusive")
            summary[support] = summary.get(support, 0) + 1
            
            # Add key findings
            for test in result.get("tests_run", []):
                if test.get("is_statistically_significant") and test.get("is_practically_significant"):
                    summary["key_findings"].append({
                        "hypothesis": result["hypothesis_title"],
                        "test": test.get("test_name"),
                        "finding": f"Significant effect (p={test.get('p_value', 'N/A')}, effect={test.get('effect_size', 'N/A')})"
                    })
        
        return summary
