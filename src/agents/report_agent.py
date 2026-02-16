"""Report Agent - generates the final engineer-ready RCA report."""

from datetime import datetime
from typing import Any
from uuid import uuid4

from .base_agent import BaseRCAAgent, AgentOutput
from src.schemas.report import RCAReport, SectionContent, ReportSection
from src.schemas.hypothesis import Hypothesis
from src.schemas.evidence import Citation, EvidenceSource


class ReportAgent(BaseRCAAgent):
    """Agent that generates the final RCA report."""
    
    name = "ReportAgent"
    description = "Compiles findings into an engineer-ready report with citations"
    HYPOTHESES_DISPLAY_LIMIT = 5
    EVIDENCE_DISPLAY_LIMIT = 5
    ACTIONS_DISPLAY_LIMIT = 3
    REPORT_CITATIONS_LIMIT = 50
    SECTION_CITATIONS_LIMIT = 10
    
    system_prompt = """You are a technical report writer for manufacturing root cause analysis.
Your job is to:
1. Compile all findings into a clear, actionable report
2. Rank hypotheses by evidence strength
3. Provide specific recommendations for next actions
4. Ensure all claims are properly cited
5. Write in a clear, professional style appropriate for engineers

The report should enable an engineer to quickly understand the likely root cause
and take appropriate action."""

    def execute(self, context: dict[str, Any]) -> AgentOutput:
        """Generate the final RCA report.
        
        Args:
            context: Should contain all previous agent outputs:
                - 'case': Failure case
                - 'product_guide_output': Product guide agent output
                - 'research_output': Research agent output
                - 'hypothesis_output': Hypothesis agent output
                - 'stats_output': Stats analysis agent output
                - 'critic_output': Critic agent output
                
        Returns:
            AgentOutput with the complete RCA report
        """
        self.log("Generating RCA report")
        
        start_time = datetime.utcnow()
        
        case = context.get("case", {})
        product_guide = context.get("product_guide_output", {})
        research = context.get("research_output", {})
        hypothesis_output = context.get("hypothesis_output", {})
        stats_output = context.get("stats_output", {})
        critic_output = context.get("critic_output", {})
        
        case_id = case.get("case_id", "UNKNOWN")
        report_id = (
            f"RPT-{case_id}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{uuid4().hex[:8]}"
        )
        
        # Collect all citations
        all_citations = self._collect_all_citations(context)
        
        # Rank hypotheses
        ranked_hypotheses = self._rank_hypotheses(
            hypothesis_output.get("data", {}).get("hypotheses", []),
            stats_output.get("data", {}).get("hypothesis_results", []),
        )
        
        # Generate report sections
        sections = []
        
        # Executive Summary
        sections.append(self._generate_executive_summary(
            case, ranked_hypotheses, critic_output
        ))
        
        # Failure Description
        sections.append(self._generate_failure_description(case))
        
        # Investigation Scope
        sections.append(self._generate_investigation_scope(
            research, product_guide
        ))
        
        # Hypotheses Section
        sections.append(self._generate_hypotheses_section(
            ranked_hypotheses, stats_output
        ))
        
        # Statistical Analysis
        sections.append(self._generate_stats_section(stats_output))
        
        # Evidence Summary
        sections.append(self._generate_evidence_summary(
            stats_output, all_citations
        ))
        
        # Conclusions
        sections.append(self._generate_conclusions(
            ranked_hypotheses, critic_output
        ))
        
        # Recommended Actions
        sections.append(self._generate_recommendations(
            ranked_hypotheses, case
        ))
        
        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(
            ranked_hypotheses, critic_output
        )
        
        # Build the report
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        deterministic_reasoning = (
            f"Generated comprehensive RCA report with {len(ranked_hypotheses)} hypotheses "
            f"and {len(all_citations)} citations"
        )
        llm_reasoning, used_llm = self._llm_analysis_or_fallback(
            objective="Summarize final RCA conclusions and confidence for engineer handoff.",
            context_payload={
                "case_id": case_id,
                "top_hypothesis": ranked_hypotheses[0] if ranked_hypotheses else None,
                "overall_confidence": overall_confidence,
                "critical_issues": critic_output.get("data", {}).get("critical_issues", 0),
                "processing_time_seconds": processing_time,
            },
            fallback_reasoning=deterministic_reasoning,
        )
        
        report = RCAReport(
            report_id=report_id,
            case_id=case_id,
            title=f"Root Cause Analysis: {case.get('failure_type', 'Unknown')} - {case.get('part_number', 'Unknown')}",
            sections=sections,
            ranked_hypotheses=[Hypothesis(**h) if isinstance(h, dict) else h for h in ranked_hypotheses],
            top_hypothesis=Hypothesis(**ranked_hypotheses[0]) if ranked_hypotheses and isinstance(ranked_hypotheses[0], dict) else (ranked_hypotheses[0] if ranked_hypotheses else None),
            all_citations=[
                Citation(**self._normalize_citation_dict(c))
                if isinstance(c, dict)
                else c
                for c in all_citations[:self.REPORT_CITATIONS_LIMIT]
            ],
            immediate_actions=self._get_immediate_actions(ranked_hypotheses, case),
            investigation_actions=self._get_investigation_actions(ranked_hypotheses),
            preventive_actions=self._get_preventive_actions(ranked_hypotheses),
            overall_confidence=overall_confidence,
            caveats=self._get_caveats(critic_output),
            data_gaps=research.get("data", {}).get("summary", {}).get("data_gaps", []),
            agents_involved=[
                "IntakeTriageAgent", "ProductGuideAgent", "PrivateDataResearchAgent",
                "HypothesisAgent", "StatsAnalysisAgent", "TestPlanAgent",
                "CriticEvidenceAgent", "ReportAgent"
            ],
            processing_time_seconds=processing_time,
            is_draft=False,
        )
        
        self.log(f"Report generated: {report_id}")
        
        return AgentOutput(
            agent_name=self.name,
            success=True,
            data={
                "report": report.model_dump(),
                "report_id": report_id,
                "top_hypothesis": ranked_hypotheses[0].get("title") if ranked_hypotheses else None,
                "overall_confidence": overall_confidence,
                "llm_analysis_used": used_llm,
                "llm_analysis": llm_reasoning,
            },
            citations_used=all_citations,
            confidence=overall_confidence,
            reasoning=llm_reasoning
        )

    @staticmethod
    def _normalize_citation_dict(citation: dict) -> dict:
        """Normalize citation fields into schema-compatible values."""
        normalized = dict(citation)
        source_type = str(normalized.get("source_type", "")).strip().lower()
        allowed = {e.value for e in EvidenceSource}
        source_type_map = {
            "sql_query": EvidenceSource.HISTORICAL_CASE.value,
            "database": EvidenceSource.HISTORICAL_CASE.value,
            "sql": EvidenceSource.HISTORICAL_CASE.value,
            "rag": EvidenceSource.PRODUCT_GUIDE.value,
            "spc": EvidenceSource.SPC_DATA.value,
        }
        resolved_source_type = source_type_map.get(source_type, source_type)
        if resolved_source_type not in allowed:
            resolved_source_type = EvidenceSource.TEST_DATA.value
        normalized["source_type"] = resolved_source_type
        return normalized
    
    def _collect_all_citations(self, context: dict) -> list[dict]:
        """Collect all citations from agent outputs."""
        citations = []
        
        for key in ["product_guide_output", "research_output", "stats_output"]:
            output = context.get(key, {})
            citations.extend(output.get("citations_used", []))
        
        # Deduplicate by source_id
        seen = set()
        unique = []
        for c in citations:
            source_id = c.get("source_id") if isinstance(c, dict) else c.source_id
            if source_id not in seen:
                seen.add(source_id)
                unique.append(c)
        
        return unique
    
    def _rank_hypotheses(
        self,
        hypotheses: list[dict],
        stats_results: list[dict],
    ) -> list[dict]:
        """Rank hypotheses by evidence strength."""
        # Create a lookup for stats results
        stats_lookup = {r.get("hypothesis_id"): r for r in stats_results}
        
        for hyp in hypotheses:
            hyp_id = hyp.get("hypothesis_id")
            stats = stats_lookup.get(hyp_id, {})
            
            # Calculate posterior confidence
            prior = hyp.get("prior_confidence", 0.5)
            support = stats.get("overall_support", "inconclusive")
            
            if support == "supported":
                posterior = min(0.95, prior + 0.3)
            elif support == "refuted":
                posterior = max(0.05, prior - 0.3)
            else:
                posterior = prior
            
            hyp["posterior_confidence"] = posterior
            hyp["stats_support"] = support
        
        # Sort by posterior confidence
        hypotheses.sort(key=lambda h: h.get("posterior_confidence", 0), reverse=True)
        
        # Assign ranks
        for i, hyp in enumerate(hypotheses):
            hyp["rank"] = i + 1
        
        return hypotheses
    
    def _generate_executive_summary(
        self,
        case: dict,
        ranked_hypotheses: list[dict],
        critic_output: dict,
    ) -> SectionContent:
        """Generate executive summary section."""
        top_hyp = ranked_hypotheses[0] if ranked_hypotheses else None
        
        content = f"""## Executive Summary

**Failure:** {case.get('failure_type', 'Unknown')} on {case.get('part_number', 'Unknown')}
**Date:** {case.get('failure_datetime', 'Unknown')}
**Lot:** {case.get('lot_number', 'Unknown')}

### Most Likely Root Cause

{f"**{top_hyp.get('title', 'Unknown')}** (Confidence: {top_hyp.get('posterior_confidence', 0):.0%})" if top_hyp else "Inconclusive - additional investigation needed"}

{top_hyp.get('description', '') if top_hyp else ''}

### Key Findings

- {len(ranked_hypotheses)} hypotheses investigated
- {sum(1 for h in ranked_hypotheses if h.get('stats_support') == 'supported')} supported by statistical evidence
- {critic_output.get('data', {}).get('critical_issues', 0)} critical issues identified during review
"""
        
        return SectionContent(
            section=ReportSection.EXECUTIVE_SUMMARY,
            title="Executive Summary",
            content=content,
        )
    
    def _generate_failure_description(self, case: dict) -> SectionContent:
        """Generate failure description section."""
        content = f"""## Failure Description

**Description:** {case.get('failure_description', 'No description provided')}

### Test Details

| Parameter | Value |
|-----------|-------|
| Test Name | {case.get('test_name', 'N/A')} |
| Measured Value | {case.get('test_value', 'N/A')} |
| Lower Spec | {case.get('spec_lower', 'N/A')} |
| Upper Spec | {case.get('spec_upper', 'N/A')} |
| Station | {case.get('station_id', 'N/A')} |
| Operator | {case.get('operator_id', 'N/A')} |
"""
        
        return SectionContent(
            section=ReportSection.FAILURE_DESCRIPTION,
            title="Failure Description",
            content=content,
        )
    
    def _generate_investigation_scope(
        self,
        research: dict,
        product_guide: dict,
    ) -> SectionContent:
        """Generate investigation scope section."""
        data_sources = research.get("data", {}).get("data_sources_queried", [])
        total_records = research.get("data", {}).get("total_records", 0)
        guide_sections = product_guide.get("data", {}).get("sections_found", {})
        
        content = f"""## Investigation Scope

### Data Sources Queried

{chr(10).join(f'- {source}' for source in data_sources) if data_sources else '- No data sources queried'}

**Total Records Analyzed:** {total_records}

### Product Guide Sections Referenced

{chr(10).join(f'- {section}' for section in guide_sections.keys()) if guide_sections else '- No product guide sections referenced'}
"""
        
        return SectionContent(
            section=ReportSection.INVESTIGATION_SCOPE,
            title="Investigation Scope",
            content=content,
        )
    
    def _generate_hypotheses_section(
        self,
        ranked_hypotheses: list[dict],
        stats_output: dict,
    ) -> SectionContent:
        """Generate hypotheses section."""
        content = "## Hypotheses\n\n"
        hypothesis_count = len(ranked_hypotheses)
        if hypothesis_count > self.HYPOTHESES_DISPLAY_LIMIT:
            content += (
                f"_Showing top {self.HYPOTHESES_DISPLAY_LIMIT} of {hypothesis_count} hypotheses "
                "ranked by posterior confidence._\n\n"
            )
        
        for hyp in ranked_hypotheses[:self.HYPOTHESES_DISPLAY_LIMIT]:
            status_emoji = {
                "supported": "✅",
                "refuted": "❌",
                "inconclusive": "❓",
            }.get(hyp.get("stats_support", "inconclusive"), "❓")
            
            content += f"""### {hyp.get('rank', '?')}. {hyp.get('title', 'Unknown')} {status_emoji}

**Confidence:** {hyp.get('posterior_confidence', 0):.0%}
**Status:** {hyp.get('stats_support', 'Not tested').title()}

{hyp.get('description', '')}

**Mechanism:** {hyp.get('mechanism', 'Not specified')}

---

"""
        
        return SectionContent(
            section=ReportSection.HYPOTHESES,
            title="Hypotheses",
            content=content,
        )
    
    def _generate_stats_section(self, stats_output: dict) -> SectionContent:
        """Generate statistical analysis section."""
        results = stats_output.get("data", {}).get("hypothesis_results", [])
        summary = stats_output.get("data", {}).get("summary", {})
        
        content = "## Statistical Analysis\n\n"
        content += f"**Total Tests Run:** {stats_output.get('data', {}).get('total_tests_run', 0)}\n\n"
        
        # Key findings
        findings = summary.get("key_findings", [])
        content += "### Key Statistical Findings\n\n"
        if findings:
            for finding in findings[:5]:
                content += f"- **{finding.get('hypothesis', 'Unknown')}**: {finding.get('finding', '')}\n"
        else:
            supported = summary.get("supported", 0)
            refuted = summary.get("refuted", 0)
            inconclusive = summary.get("inconclusive", 0)
            content += (
                f"- No statistically and practically significant findings were detected.\n"
                f"- Outcome mix: {supported} supported, {refuted} refuted, {inconclusive} inconclusive hypotheses.\n"
            )
        content += "\n"
        
        # Summary by hypothesis
        content += "### Results by Hypothesis\n\n"
        for result in results[:5]:
            tests = result.get("tests_run", [])
            content += f"**{result.get('hypothesis_title', 'Unknown')}**\n"
            content += f"- Overall: {result.get('overall_support', 'inconclusive').title()}\n"
            content += f"- Tests run: {len(tests)}\n\n"
        
        return SectionContent(
            section=ReportSection.STATISTICAL_ANALYSIS,
            title="Statistical Analysis",
            content=content,
        )
    
    def _generate_evidence_summary(
        self,
        stats_output: dict,
        all_citations: list[dict],
    ) -> SectionContent:
        """Generate evidence summary section."""
        evidence = stats_output.get("data", {}).get("evidence", [])
        
        content = "## Evidence Summary\n\n"
        content += f"**Total Evidence Items:** {len(evidence)}\n"
        content += f"**Total Citations:** {len(all_citations)}\n\n"
        
        # Group by support/refute
        supporting = [e for e in evidence if e.get("supports_hypothesis")]
        refuting = [e for e in evidence if not e.get("supports_hypothesis")]
        
        content += f"### Supporting Evidence ({len(supporting)} items)\n\n"
        for ev in supporting[:self.EVIDENCE_DISPLAY_LIMIT]:
            content += f"- {ev.get('claim', 'No claim')}\n"
        if len(supporting) > self.EVIDENCE_DISPLAY_LIMIT:
            content += f"- ... {len(supporting) - self.EVIDENCE_DISPLAY_LIMIT} additional supporting items\n"
        
        content += f"\n### Refuting Evidence ({len(refuting)} items)\n\n"
        for ev in refuting[:self.EVIDENCE_DISPLAY_LIMIT]:
            content += f"- {ev.get('claim', 'No claim')}\n"
        if len(refuting) > self.EVIDENCE_DISPLAY_LIMIT:
            content += f"- ... {len(refuting) - self.EVIDENCE_DISPLAY_LIMIT} additional refuting items\n"
        
        return SectionContent(
            section=ReportSection.EVIDENCE_SUMMARY,
            title="Evidence Summary",
            content=content,
            citations=[
                Citation(**self._normalize_citation_dict(c))
                if isinstance(c, dict)
                else c
                for c in all_citations[:self.SECTION_CITATIONS_LIMIT]
            ],
        )
    
    def _generate_conclusions(
        self,
        ranked_hypotheses: list[dict],
        critic_output: dict,
    ) -> SectionContent:
        """Generate conclusions section."""
        top = ranked_hypotheses[0] if ranked_hypotheses else None
        issues = critic_output.get("data", {}).get("issues_found", [])
        
        content = "## Conclusions\n\n"
        
        if top and top.get("stats_support") == "supported":
            content += f"""Based on the statistical analysis and available evidence, the most likely root cause is:

**{top.get('title', 'Unknown')}**

This hypothesis is supported by {len([h for h in ranked_hypotheses if h.get('stats_support') == 'supported'])} statistical tests with a confidence of {top.get('posterior_confidence', 0):.0%}.
"""
        else:
            content += """The investigation did not conclusively identify a single root cause. Additional investigation may be needed.

"""
            if top:
                content += f"The most probable hypothesis is **{top.get('title', 'Unknown')}** ({top.get('posterior_confidence', 0):.0%} confidence).\n"
        
        if issues:
            content += f"\n### Limitations\n\n{len(issues)} issues were identified during review:\n\n"
            for issue in issues[:3]:
                content += f"- {issue.get('issue', 'Unknown issue')}\n"
        
        return SectionContent(
            section=ReportSection.CONCLUSIONS,
            title="Conclusions",
            content=content,
        )
    
    def _generate_recommendations(
        self,
        ranked_hypotheses: list[dict],
        case: dict,
    ) -> SectionContent:
        """Generate recommended actions section."""
        content = "## Recommended Actions\n\n"
        immediate_actions = self._get_immediate_actions(ranked_hypotheses, case)
        investigation_actions = self._get_investigation_actions(ranked_hypotheses)
        preventive_actions = self._get_preventive_actions(ranked_hypotheses)
        
        content += "### Immediate Actions\n\n"
        for action in immediate_actions[:self.ACTIONS_DISPLAY_LIMIT]:
            content += f"1. {action}\n"
        if len(immediate_actions) > self.ACTIONS_DISPLAY_LIMIT:
            content += (
                f"\n- ... {len(immediate_actions) - self.ACTIONS_DISPLAY_LIMIT} additional "
                "immediate actions omitted\n"
            )
        
        content += "\n### Further Investigation\n\n"
        for action in investigation_actions[:self.ACTIONS_DISPLAY_LIMIT]:
            content += f"- {action}\n"
        if len(investigation_actions) > self.ACTIONS_DISPLAY_LIMIT:
            content += (
                f"- ... {len(investigation_actions) - self.ACTIONS_DISPLAY_LIMIT} additional "
                "investigation actions omitted\n"
            )
        
        content += "\n### Preventive Actions\n\n"
        for action in preventive_actions[:self.ACTIONS_DISPLAY_LIMIT]:
            content += f"- {action}\n"
        if len(preventive_actions) > self.ACTIONS_DISPLAY_LIMIT:
            content += (
                f"- ... {len(preventive_actions) - self.ACTIONS_DISPLAY_LIMIT} additional "
                "preventive actions omitted\n"
            )
        
        return SectionContent(
            section=ReportSection.RECOMMENDED_ACTIONS,
            title="Recommended Actions",
            content=content,
        )
    
    def _calculate_overall_confidence(
        self,
        ranked_hypotheses: list[dict],
        critic_output: dict,
    ) -> float:
        """Calculate overall confidence in the analysis."""
        if not ranked_hypotheses:
            return 0.3
        
        top_confidence = ranked_hypotheses[0].get("posterior_confidence", 0.5)
        
        # Adjust for critic issues
        issues = critic_output.get("data", {}).get("issues_found", [])
        critical = critic_output.get("data", {}).get("critical_issues", 0)
        
        penalty = critical * 0.1 + len(issues) * 0.02
        
        return max(0.3, min(0.95, top_confidence - penalty))
    
    def _get_immediate_actions(
        self,
        ranked_hypotheses: list[dict],
        case: dict,
    ) -> list[str]:
        """Get immediate actions based on top hypothesis."""
        actions = []
        
        if ranked_hypotheses:
            top = ranked_hypotheses[0]
            title_lower = top.get("title", "").lower()
            
            if "lot" in title_lower or "component" in title_lower:
                actions.append(f"Quarantine remaining units from lot {case.get('lot_number', 'suspect lot')}")
                actions.append("Contact supplier for lot-specific information")
            
            if "torque" in title_lower or "assembly" in title_lower:
                actions.append("Verify torque wrench calibration")
                actions.append("Review recent assembly process changes")
            
            if "machine" in title_lower or "equipment" in title_lower:
                actions.append("Inspect suspected equipment")
                actions.append("Check maintenance records")
        
        if not actions:
            actions.append("Hold affected units pending further investigation")
            actions.append("Review recent process changes")
        
        return actions
    
    def _get_investigation_actions(self, ranked_hypotheses: list[dict]) -> list[str]:
        """Get further investigation actions."""
        actions = []
        
        # Add investigation for inconclusive hypotheses
        for hyp in ranked_hypotheses:
            if hyp.get("stats_support") == "inconclusive":
                actions.append(f"Gather more data to test: {hyp.get('title', 'Unknown')}")
        
        actions.append("Cross-reference with supplier quality data")
        actions.append("Review similar historical cases for patterns")
        
        return actions[:5]
    
    def _get_preventive_actions(self, ranked_hypotheses: list[dict]) -> list[str]:
        """Get preventive actions."""
        actions = []
        
        if ranked_hypotheses:
            top = ranked_hypotheses[0]
            title_lower = top.get("title", "").lower()
            
            if "lot" in title_lower:
                actions.append("Implement incoming inspection for critical components")
            if "torque" in title_lower:
                actions.append("Increase torque verification frequency")
            if "machine" in title_lower:
                actions.append("Review preventive maintenance schedule")
        
        actions.append("Update control plan based on findings")
        actions.append("Consider additional statistical process controls")
        
        return actions[:5]
    
    def _get_caveats(self, critic_output: dict) -> list[str]:
        """Get caveats and limitations."""
        caveats = []
        
        issues = critic_output.get("data", {}).get("issues_found", [])
        for issue in issues:
            if issue.get("severity") in ["high", "critical"]:
                caveats.append(issue.get("issue", "Unknown limitation"))
        
        if not caveats:
            caveats.append("Analysis limited to available data sources")
        
        return caveats[:5]
