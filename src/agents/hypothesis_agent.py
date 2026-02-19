"""Hypothesis Agent - generates testable root cause hypotheses.

Uses a structured LLM call to produce hypotheses across four origin
categories (historical precedent, unit-specific anomaly, technical
documentation, engineering reasoning).  Falls back to deterministic
template-based generation when no LLM is available.
"""

from typing import Any
import json
import uuid

from .base_agent import BaseRCAAgent, AgentOutput
from src.schemas.hypothesis import (
    Hypothesis,
    HypothesisOrigin,
    HypothesisStatus,
    StatisticalTest,
)


class HypothesisAgent(BaseRCAAgent):
    """Agent that generates testable root cause hypotheses."""

    name = "HypothesisAgent"
    description = "Synthesizes research findings into testable root cause hypotheses"

    system_prompt = """You are a senior manufacturing root cause analyst.
You generate specific, testable hypotheses for the root cause of manufacturing
failures.  Each hypothesis must identify:
- The proposed physical or process mechanism
- Expected data signatures if the hypothesis is correct
- A recommended statistical test to validate or refute it
- Required data sources

Prioritize hypotheses based on prior probability given the evidence.
Always include at least one hypothesis derived from first-principles
engineering reasoning that goes beyond the supplied documents."""

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def execute(self, context: dict[str, Any]) -> AgentOutput:
        """Generate root cause hypotheses.

        Args:
            context: Should contain:
                - 'case': The normalized failure case
                - 'recipe': The analysis recipe
                - 'product_guide_output': Output from ProductGuideAgent
                - 'research_output': Output from PrivateDataResearchAgent

        Returns:
            AgentOutput with generated hypotheses
        """
        self.log("Starting hypothesis generation")

        case = context.get("case", {})
        recipe = context.get("recipe")
        product_guide = context.get("product_guide_output", {})
        research = context.get("research_output", {})

        failure_type = case.get("failure_type", "unknown")
        case_id = case.get("case_id", "UNKNOWN")

        # Gather all available evidence
        critical_features = product_guide.get("data", {}).get("critical_features", [])
        expected_signatures = product_guide.get("data", {}).get("expected_signatures", [])
        product_guide_citations = product_guide.get("citations_used", [])
        similar_cases_block = (
            research.get("data", {})
            .get("data_retrieved", {})
            .get("similar_cases", {})
        )
        similar_cases_list = similar_cases_block.get("data", [])
        common_hypotheses = recipe.common_hypotheses if recipe else []
        catalog_context = self._extract_catalog_context(research)
        top_test_outliers = (
            research.get("data", {})
            .get("data_retrieved", {})
            .get("top_test_outliers", {})
            .get("data", [])
        )
        outlier_relevance = (
            research.get("data", {})
            .get("data_retrieved", {})
            .get("outlier_relevance", {})
        )
        unit_data_summary = {
            "warnings": (
                research.get("data", {})
                .get("data_retrieved", {})
                .get("analysis_data", {})
                .get("warnings", [])
            ),
            "top_test_outliers": top_test_outliers,
        }

        # --- Try LLM-powered hypothesis generation ---
        hypotheses: list[Hypothesis] = []
        llm_reasoning: str | None = None
        used_llm = False

        if self.llm is not None:
            try:
                prompt = self._build_hypothesis_prompt(
                    case=case,
                    recipe=recipe,
                    similar_cases=similar_cases_list,
                    critical_features=critical_features,
                    expected_signatures=expected_signatures,
                    product_guide_citations=product_guide_citations,
                    catalog_context=catalog_context,
                    unit_data_summary=unit_data_summary,
                    top_test_outliers=top_test_outliers,
                )
                raw = self._call_llm(prompt)
                parsed = self._parse_llm_hypotheses(raw, case_id, failure_type)
                hypotheses = parsed["hypotheses"]
                llm_reasoning = parsed.get("reasoning_summary", "")
                used_llm = True
                self.log(f"LLM generated {len(hypotheses)} hypotheses")
            except Exception as e:
                self.log(f"LLM hypothesis generation failed, using fallback: {e}", "warning")

        # --- Deterministic fallback ---
        if not hypotheses:
            hypotheses = self._deterministic_hypotheses(
                case_id=case_id,
                failure_type=failure_type,
                recipe=recipe,
                similar_cases=similar_cases_list,
                catalog_context=catalog_context,
            )

        # Assign / adjust prior confidences and rank
        hypotheses = self._assign_prior_confidences(hypotheses, research, product_guide)
        hypotheses = self._ensure_competing_pathways(
            hypotheses=hypotheses,
            case_id=case_id,
            failure_type=failure_type,
            outlier_relevance=outlier_relevance,
            top_test_outliers=top_test_outliers,
        )
        hypotheses.sort(key=lambda h: h.prior_confidence, reverse=True)
        for i, h in enumerate(hypotheses):
            h.rank = i + 1

        self.log(f"Generated {len(hypotheses)} hypotheses total")

        # Collect citations from upstream agents
        all_citations: list[dict] = []
        all_citations.extend(product_guide.get("citations_used", []))
        all_citations.extend(research.get("citations_used", []))

        if llm_reasoning is None:
            llm_reasoning = (
                f"Generated {len(hypotheses)} testable hypotheses based on "
                f"{len(common_hypotheses)} recipe suggestions and "
                f"{len(similar_cases_list)} similar cases"
            )

        # Origin breakdown for transparency
        origin_counts = {}
        for h in hypotheses:
            origin_counts[h.origin.value] = origin_counts.get(h.origin.value, 0) + 1
        outlier_context = self._extract_outlier_signal_context(
            outlier_relevance=outlier_relevance,
            top_test_outliers=top_test_outliers,
        )

        return AgentOutput(
            agent_name=self.name,
            success=True,
            data={
                "hypotheses": [h.model_dump() for h in hypotheses],
                "hypothesis_count": len(hypotheses),
                "top_hypothesis": hypotheses[0].title if hypotheses else None,
                "origin_breakdown": origin_counts,
                "outlier_hypotheses_count": sum(
                    1 for h in hypotheses if self._hypothesis_mentions_test_ids(h, outlier_context["all_ids"])
                ),
                "non_outlier_hypotheses_count": sum(
                    1 for h in hypotheses if not self._hypothesis_mentions_test_ids(h, outlier_context["all_ids"])
                ),
                "llm_analysis_used": used_llm,
                "llm_analysis": llm_reasoning,
            },
            citations_used=all_citations,
            confidence=0.8,
            reasoning=llm_reasoning,
        )

    # ------------------------------------------------------------------
    # LLM prompt builder
    # ------------------------------------------------------------------

    def _build_hypothesis_prompt(
        self,
        *,
        case: dict,
        recipe: Any,
        similar_cases: list[dict],
        critical_features: list,
        expected_signatures: list,
        product_guide_citations: list[dict],
        catalog_context: dict[str, list[str]],
        unit_data_summary: dict,
        top_test_outliers: list[dict],
    ) -> str:
        """Build a single structured prompt for categorized hypothesis generation."""

        failure_type = case.get("failure_type", "unknown")
        failure_desc = case.get("failure_description", "")
        part_number = case.get("part_number", "")
        lot_number = case.get("lot_number", "")
        test_value = case.get("test_value", "")
        spec_upper = case.get("spec_upper", "")

        # --- Section A: Historical precedent ---
        historical_lines: list[str] = []
        for sc in similar_cases[:8]:
            historical_lines.append(
                f"- Case {sc.get('case_id', '?')}: "
                f"root_cause=\"{sc.get('root_cause', 'unknown')}\", "
                f"resolution=\"{sc.get('resolution', 'unknown')}\", "
                f"failure_type=\"{sc.get('failure_type', '')}\""
            )
        historical_text = (
            "\n".join(historical_lines) if historical_lines else "No similar cases found."
        )

        # --- Section B: This unit's specific data ---
        unit_lines = [
            f"- Failure type: {failure_type}",
            f"- Part: {part_number}, Lot: {lot_number}",
            f"- Test value: {test_value}, Spec limit: {spec_upper}",
        ]
        for w in (unit_data_summary.get("warnings") or [])[:5]:
            unit_lines.append(f"- Data warning: {w}")
        if top_test_outliers:
            unit_lines.append("- Top outlier test IDs for this unit context:")
            for item in top_test_outliers[:10]:
                if not isinstance(item, dict):
                    continue
                unit_lines.append(
                    "- "
                    + f"{item.get('test_id', '?')} (z={item.get('stddev_from_mean', '?')}, "
                    + f"direction={item.get('direction', '?')}): "
                    + f"{item.get('description', '')}"
                )
        if catalog_context.get("test_ids"):
            unit_lines.append(
                f"- Available test fields: {', '.join(catalog_context['test_ids'][:8])}"
            )
        if catalog_context.get("outlier_test_ids"):
            unit_lines.append(
                f"- Outlier-prioritized test IDs: {', '.join(catalog_context['outlier_test_ids'][:10])}"
            )
        if catalog_context.get("roa_parameters"):
            unit_lines.append(
                f"- Available ROA fields: {', '.join(catalog_context['roa_parameters'][:8])}"
            )
        unit_text = "\n".join(unit_lines)

        # --- Section C: Technical documentation ---
        doc_lines: list[str] = []
        for feat in critical_features[:6]:
            text = feat.get("feature", str(feat)) if isinstance(feat, dict) else str(feat)
            section = feat.get("source_section", "") if isinstance(feat, dict) else ""
            doc_lines.append(f"- [§{section}] {text[:200]}")
        for sig in expected_signatures[:4]:
            text = sig.get("signature", str(sig)) if isinstance(sig, dict) else str(sig)
            doc_lines.append(f"- Expected signature: {text}")
        for cit in product_guide_citations[:4]:
            if isinstance(cit, dict):
                doc_lines.append(
                    f"- [{cit.get('section_path', '')}] "
                    f"{cit.get('excerpt', '')[:200]}"
                )
        doc_text = "\n".join(doc_lines) if doc_lines else "No product documentation retrieved."

        # --- Recipe common hypotheses ---
        common_hyps = recipe.common_hypotheses if recipe else []
        recipe_text = (
            "\n".join(f"- {h}" for h in common_hyps[:7]) if common_hyps else "None."
        )

        prompt = f"""You are a senior manufacturing root cause analyst investigating a failure.

## Failure Under Investigation
- Type: {failure_type}
- Description: {failure_desc}
- Part: {part_number} | Lot: {lot_number}
- Test value: {test_value} | Spec limit: {spec_upper}

## Evidence Section A — Previous Failures on Other Units
{historical_text}

## Evidence Section B — This Unit's Specific Data
{unit_text}

## Evidence Section C — Technical Documentation (Product Guides, Specs)
{doc_text}

## Domain Knowledge — Common Hypotheses for This Failure Type
{recipe_text}

## Your Task

Generate 5-9 testable root cause hypotheses. For EACH hypothesis, classify it into
exactly one primary origin category based on what evidence most strongly motivates it:

- **historical_precedent** — motivated by patterns in previous failures (Section A)
- **unit_specific_anomaly** — motivated by something specific in this unit's data (Section B)
- **technical_documentation** — motivated by critical features, specs, or known failure
  modes in the technical documents (Section C)
- **engineering_reasoning** — motivated by your engineering knowledge about failure
  physics, material science, or manufacturing processes, even if NOT mentioned in any
  of the evidence above

IMPORTANT RULES:
1. You MUST include at least one hypothesis from engineering_reasoning that goes BEYOND
   what the documents explicitly state. Think like a senior engineer — what would you
   check even if the documents don't mention it?
2. Each hypothesis must be specific and testable with manufacturing data.
3. If the evidence is sparse, lean more on engineering_reasoning.
4. Do NOT just rephrase the recipe hypotheses — add specificity from the evidence.
5. Prioritize hypotheses that explain the strongest outlier test-ID signals when plausible.

Return valid JSON only. No markdown fencing. Format:
{{
  "hypotheses": [
    {{
      "title": "Brief descriptive title",
      "description": "2-3 sentence description of the proposed root cause",
      "mechanism": "The physical/process mechanism by which this causes the failure",
      "origin": "historical_precedent | unit_specific_anomaly | technical_documentation | engineering_reasoning",
      "evidence_basis": "Which specific evidence item(s) motivated this hypothesis",
      "expected_signatures": ["What data pattern you'd expect if this is correct"],
      "recommended_test_type": "xbar_s_chart | two_sample_ttest | correlation | one_way_anova | individuals_mr",
      "target_variable": "the measurement column to test",
      "grouping_key": "the column to group or compare by",
      "required_data_sources": ["data source names needed"],
      "prior_confidence": 0.5
    }}
  ],
  "reasoning_summary": "2-3 sentences on your overall assessment and which direction is most promising"
}}"""
        return prompt

    # ------------------------------------------------------------------
    # LLM response parser
    # ------------------------------------------------------------------

    def _parse_llm_hypotheses(
        self,
        raw_response: str,
        case_id: str,
        failure_type: str,
    ) -> dict:
        """Parse LLM JSON response into Hypothesis objects."""
        text = raw_response.strip()

        # Strip markdown fencing if present
        if "```" in text:
            parsed = None
            for segment in text.split("```"):
                candidate = segment.strip()
                if candidate.lower().startswith("json"):
                    candidate = candidate[4:].strip()
                try:
                    parsed = json.loads(candidate)
                    break
                except json.JSONDecodeError:
                    continue
            if parsed is None:
                parsed = json.loads(text)
        else:
            parsed = json.loads(text)

        origin_map = {e.value: e for e in HypothesisOrigin}

        hypotheses: list[Hypothesis] = []
        for i, h in enumerate(parsed.get("hypotheses", [])):
            origin_str = h.get("origin", "engineering_reasoning")
            origin = origin_map.get(origin_str, HypothesisOrigin.ENGINEERING_REASONING)

            # Build a StatisticalTest from the flat fields
            target_var = h.get(
                "target_variable",
                self._resolve_measurement_column(failure_type),
            )
            grouping = h.get("grouping_key", "lot_number")
            test_type = h.get("recommended_test_type", "xbar_s_chart")
            sigs = h.get("expected_signatures", [])

            hypothesis = Hypothesis(
                hypothesis_id=f"HYP-{case_id}-{i + 1:02d}",
                case_id=case_id,
                title=h.get("title", f"Hypothesis {i + 1}"),
                description=h.get("description", ""),
                mechanism=h.get("mechanism", ""),
                origin=origin,
                evidence_basis=h.get("evidence_basis"),
                expected_signatures=sigs if sigs else ["Statistically significant deviation"],
                recommended_tests=[
                    StatisticalTest(
                        test_type=test_type,
                        target_variable=target_var,
                        grouping_key=grouping,
                        expected_signal=sigs[0] if sigs else "Significant deviation",
                    )
                ],
                required_data_sources=h.get("required_data_sources", []),
                prior_confidence=min(0.9, max(0.1, h.get("prior_confidence", 0.5))),
                rank=i + 1,
            )
            hypotheses.append(hypothesis)

        return {
            "hypotheses": hypotheses,
            "reasoning_summary": parsed.get("reasoning_summary", ""),
        }

    # ------------------------------------------------------------------
    # Deterministic fallback (existing logic)
    # ------------------------------------------------------------------

    def _deterministic_hypotheses(
        self,
        case_id: str,
        failure_type: str,
        recipe: Any,
        similar_cases: list[dict],
        catalog_context: dict[str, list[str]],
    ) -> list[Hypothesis]:
        """Generate hypotheses without an LLM using templates and similar cases."""
        common_hypotheses = recipe.common_hypotheses if recipe else []
        hypotheses: list[Hypothesis] = []

        # 1. Hypotheses from recipe's common list
        for i, hyp_template in enumerate(common_hypotheses[:5]):
            hypothesis = self._create_hypothesis_from_template(
                template=hyp_template,
                case_id=case_id,
                failure_type=failure_type,
                rank=i + 1,
                catalog_context=catalog_context,
            )
            hypotheses.append(hypothesis)

        # 2. Hypotheses from similar case root causes
        if similar_cases:
            seen_causes: set[str] = set()
            for sim_case in similar_cases[:10]:
                root_cause = sim_case.get("root_cause")
                if root_cause and root_cause not in seen_causes:
                    seen_causes.add(root_cause)
                    hypothesis = self._create_hypothesis_from_similar_case(
                        root_cause=root_cause,
                        case_id=case_id,
                        similar_case=sim_case,
                        rank=len(hypotheses) + 1,
                        failure_type=failure_type,
                    )
                    hypotheses.append(hypothesis)
                    if len(hypotheses) >= 7:
                        break

        # 3. Ensure we have at least minimum hypotheses
        if len(hypotheses) < 3:
            generic = self._create_generic_hypotheses(
                case_id,
                failure_type,
                len(hypotheses),
                catalog_context=catalog_context,
            )
            hypotheses.extend(generic)

        return hypotheses

    # ------------------------------------------------------------------
    # Template / similar-case helpers (used by deterministic fallback)
    # ------------------------------------------------------------------

    def _create_hypothesis_from_template(
        self,
        template: str,
        case_id: str,
        failure_type: str,
        rank: int,
        catalog_context: dict[str, list[str]] | None = None,
    ) -> Hypothesis:
        """Create a hypothesis from a recipe template."""
        if ":" in template:
            title, description = template.split(":", 1)
        else:
            title = template
            description = f"Investigating whether {template.lower()} is the root cause"

        tests = self._get_recommended_tests(
            title,
            failure_type,
            catalog_context=catalog_context,
        )

        return Hypothesis(
            hypothesis_id=f"HYP-{case_id}-{rank:02d}",
            case_id=case_id,
            title=title.strip(),
            description=description.strip(),
            mechanism=f"If {title.lower()}, then the failure would occur due to...",
            origin=HypothesisOrigin.RECIPE_TEMPLATE,
            expected_signatures=self._get_expected_signatures(
                title,
                catalog_context=catalog_context,
            ),
            recommended_tests=tests,
            required_data_sources=self._get_required_data_sources(
                title,
                catalog_context=catalog_context,
            ),
            prior_confidence=0.5,
            rank=rank,
        )

    def _create_hypothesis_from_similar_case(
        self,
        root_cause: str,
        case_id: str,
        similar_case: dict,
        rank: int,
        failure_type: str = "",
    ) -> Hypothesis:
        """Create a hypothesis based on a similar historical case."""
        target_var = self._resolve_measurement_column(failure_type)
        return Hypothesis(
            hypothesis_id=f"HYP-{case_id}-{rank:02d}",
            case_id=case_id,
            title=f"Similar to case {similar_case.get('case_id', 'unknown')}: {root_cause}",
            description=(
                f"Historical case had the same root cause. Resolution was: "
                f"{similar_case.get('resolution', 'unknown')}"
            ),
            mechanism="Same failure mechanism as historical case",
            origin=HypothesisOrigin.HISTORICAL_PRECEDENT,
            evidence_basis=f"Similar case {similar_case.get('case_id', '?')}",
            expected_signatures=["Pattern matches historical case data"],
            recommended_tests=[
                StatisticalTest(
                    test_type="two_sample_ttest",
                    target_variable=target_var,
                    grouping_key="lot_number",
                    expected_signal="Similar distribution to historical failing cases",
                )
            ],
            required_data_sources=["historical_cases", "leak_test_history"],
            prior_confidence=0.6,
            rank=rank,
        )

    def _create_generic_hypotheses(
        self,
        case_id: str,
        failure_type: str,
        start_rank: int,
        catalog_context: dict[str, list[str]] | None = None,
    ) -> list[Hypothesis]:
        """Create generic hypotheses when specific ones aren't available."""
        target_var = self._resolve_measurement_column(failure_type)
        generic = [
            ("Material variation", "Incoming material properties differ from specification"),
            ("Process drift", "Manufacturing process parameters have drifted from nominal"),
            ("Equipment issue", "Production equipment is not functioning within specification"),
        ]

        hypotheses: list[Hypothesis] = []
        for i, (title, description) in enumerate(generic):
            hypotheses.append(Hypothesis(
                hypothesis_id=f"HYP-{case_id}-{start_rank + i + 1:02d}",
                case_id=case_id,
                title=title,
                description=description,
                mechanism=f"Generic mechanism: {title.lower()}",
                origin=HypothesisOrigin.ENGINEERING_REASONING,
                expected_signatures=[f"Data anomaly related to {title.lower()}"],
                recommended_tests=[
                    StatisticalTest(
                        test_type="xbar_s_chart",
                        target_variable=target_var,
                        grouping_key="lot_number",
                        expected_signal="Special cause variation",
                    )
                ],
                required_data_sources=self._get_required_data_sources(
                    title,
                    catalog_context=catalog_context,
                ),
                prior_confidence=0.3,
                rank=start_rank + i + 1,
            ))

        return hypotheses

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_measurement_column(failure_type: str) -> str:
        """Resolve the primary measurement column based on failure type."""
        ft = failure_type.lower()
        if "leak" in ft:
            return "leak_rate"
        elif "dimension" in ft or "tolerance" in ft:
            return "measured_value"
        elif "torque" in ft or "assembly" in ft:
            return "torque_value"
        else:
            return "measured_value"

    def _get_recommended_tests(
        self,
        hypothesis_title: str,
        failure_type: str,
        catalog_context: dict[str, list[str]] | None = None,
    ) -> list[StatisticalTest]:
        """Get recommended statistical tests for a hypothesis."""
        title_lower = hypothesis_title.lower()
        tests: list[StatisticalTest] = []
        catalog_context = catalog_context or {}

        if any(kw in title_lower for kw in ["o-ring", "seal", "lot variation"]):
            tests.append(StatisticalTest(
                test_type="xbar_s_chart",
                target_variable="leak_rate" if "leak" in failure_type else "measured_value",
                grouping_key="component_lot",
                subgroup_size=5,
                expected_signal="Component lot shows special cause variation",
            ))
            tests.append(StatisticalTest(
                test_type="two_sample_ttest",
                target_variable="leak_rate" if "leak" in failure_type else "measured_value",
                grouping_key="lot_number",
                practical_threshold=0.5,
                expected_signal="Suspect lot differs significantly from baseline",
            ))
        elif any(kw in title_lower for kw in ["torque", "assembly", "fixture"]):
            tests.append(StatisticalTest(
                test_type="xbar_s_chart",
                target_variable="torque_value",
                grouping_key="assembly_datetime",
                expected_signal="Torque drift over time",
            ))
            tests.append(StatisticalTest(
                test_type="correlation",
                target_variable="torque_value",
                grouping_key="leak_rate",
                expected_signal="Correlation between torque and leak rate",
            ))
        elif any(kw in title_lower for kw in ["tool", "machine", "equipment"]):
            tests.append(StatisticalTest(
                test_type="xbar_s_chart",
                target_variable="measured_value",
                grouping_key="machine_id",
                expected_signal="Machine-to-machine variation",
            ))
            tests.append(StatisticalTest(
                test_type="correlation",
                target_variable="measured_value",
                grouping_key="parts_since_change",
                expected_signal="Dimension drift with tool wear",
            ))
        elif any(kw in title_lower for kw in ["surface", "contamination", "finish"]):
            tests.append(StatisticalTest(
                test_type="individuals_mr",
                target_variable="measured_value",
                grouping_key="measurement_datetime",
                expected_signal="Trend in surface quality",
            ))

        if not tests:
            target = "leak_rate" if "leak" in failure_type else "measured_value"
            if "FLOW_RATE" in catalog_context.get("test_ids", []) and "leak" not in failure_type:
                target = "flow_rate"
            tests.append(StatisticalTest(
                test_type="xbar_s_chart",
                target_variable=target,
                grouping_key="lot_number",
                expected_signal="Special cause variation by lot",
            ))

        return tests

    def _get_expected_signatures(
        self,
        hypothesis_title: str,
        catalog_context: dict[str, list[str]] | None = None,
    ) -> list[str]:
        """Get expected data signatures for a hypothesis."""
        title_lower = hypothesis_title.lower()
        catalog_context = catalog_context or {}

        signatures: list[str] = []

        if "lot" in title_lower or "o-ring" in title_lower:
            signatures.append("Specific lot shows elevated failure rate")
            signatures.append("Control chart shows out-of-control points for lot")

        if "torque" in title_lower or "assembly" in title_lower:
            signatures.append("Torque values outside specification")
            signatures.append("Correlation between torque and leak rate")

        if "drift" in title_lower or "trend" in title_lower:
            signatures.append("Time-series trend in measurement data")
            signatures.append("Control chart shows run or trend pattern")

        if "machine" in title_lower or "equipment" in title_lower:
            signatures.append("Significant difference between machines")
            signatures.append("One machine shows higher variation")

        if not signatures:
            signatures.append("Statistically significant difference from baseline")
            signatures.append("Special cause variation in control chart")

        if catalog_context.get("test_ids"):
            signatures.append(
                "Correlated behavior in selected test fields: "
                + ", ".join(catalog_context["test_ids"][:3])
            )

        return signatures

    def _get_required_data_sources(
        self,
        hypothesis_title: str,
        catalog_context: dict[str, list[str]] | None = None,
    ) -> list[str]:
        """Get required data sources for a hypothesis."""
        title_lower = hypothesis_title.lower()
        catalog_context = catalog_context or {}

        sources: list[str] = []

        if any(kw in title_lower for kw in ["o-ring", "seal", "lot", "component"]):
            sources.append("component_lots")
            sources.append("leak_test_history")

        if any(kw in title_lower for kw in ["torque", "assembly"]):
            sources.append("assembly_parameters")
            sources.append("leak_test_history")

        if any(kw in title_lower for kw in ["machine", "tool", "equipment"]):
            sources.append("machine_parameters")
            sources.append("tool_changes")

        if any(kw in title_lower for kw in ["dimension", "surface"]):
            sources.append("dimensional_history")

        if any(kw in title_lower for kw in ["material", "supplier"]):
            sources.append("material_lots")

        selected_test_ids = set(catalog_context.get("test_ids", []))
        selected_roa = set(catalog_context.get("roa_parameters", []))
        selected_process = set(catalog_context.get("process_parameters", []))

        if selected_test_ids & {"LEAK_RATE", "LEAK_RESULT", "PRESSURE_DROP", "FLOW_RATE"}:
            sources.append("leak_test_history")
        if selected_roa & {"TORQUE_MAIN_HOUSING", "TORQUE_SEAL_RING"}:
            sources.append("assembly_parameters")
        if selected_roa & {"GASKET_LOT", "SEAL_LOT", "HOUSING_LOT"}:
            sources.append("component_lots")
        if selected_process & {"AMBIENT_TEMP", "AMBIENT_HUMIDITY"}:
            sources.append("leak_test_history")
            sources.append("dimensional_history")

        if not sources:
            sources.append("leak_test_history")
            sources.append("component_lots")

        return list(set(sources))

    def _extract_catalog_context(self, research: dict) -> dict[str, list[str]]:
        """Extract selected field context from research output for hypothesis planning."""
        data_retrieved = research.get("data", {}).get("data_retrieved", {})
        analysis_data = (
            data_retrieved.get("analysis_data", {})
        )
        top_outliers = data_retrieved.get("top_test_outliers", {}).get("data", []) or []
        outlier_ids = []
        for item in top_outliers:
            if isinstance(item, dict) and isinstance(item.get("test_id"), str):
                outlier_ids.append(item["test_id"])
        return {
            "test_ids": analysis_data.get("test_ids", []) or [],
            "roa_parameters": analysis_data.get("roa_parameters", []) or [],
            "operator_buyoffs": analysis_data.get("operator_buyoffs", []) or [],
            "process_parameters": analysis_data.get("process_parameters", []) or [],
            "catalog_candidates_considered": analysis_data.get("catalog_candidates_considered", []) or [],
            "outlier_test_ids": outlier_ids,
        }

    def _assign_prior_confidences(
        self,
        hypotheses: list[Hypothesis],
        research: dict,
        product_guide: dict,
    ) -> list[Hypothesis]:
        """Assign prior confidences based on available evidence and origin."""
        similar_cases = (
            research.get("data", {})
            .get("data_retrieved", {})
            .get("similar_cases", {})
            .get("data", [])
        )
        historical_root_causes = [
            c.get("root_cause", "").lower()
            for c in similar_cases
            if c.get("root_cause")
        ]

        for hypothesis in hypotheses:
            confidence = 0.4  # Base confidence

            # Boost if matches historical root causes
            for rc in historical_root_causes:
                if any(word in hypothesis.title.lower() for word in rc.split()):
                    confidence += 0.15
                    break

            # Boost if has strong data support (multiple tests)
            if len(hypothesis.recommended_tests) >= 2:
                confidence += 0.1

            # Boost based on origin quality
            if hypothesis.origin == HypothesisOrigin.HISTORICAL_PRECEDENT:
                confidence += 0.1  # Historical precedent is strong evidence
            elif hypothesis.origin == HypothesisOrigin.TECHNICAL_DOCUMENTATION:
                confidence += 0.1  # Document-backed
            elif hypothesis.origin == HypothesisOrigin.RECIPE_TEMPLATE:
                confidence += 0.1  # Domain knowledge from recipe

            hypothesis.prior_confidence = min(0.9, confidence)

        return hypotheses

    def _extract_outlier_signal_context(
        self,
        *,
        outlier_relevance: dict,
        top_test_outliers: list[dict],
    ) -> dict[str, set[str]]:
        """Extract sets of outlier test IDs by relevance class."""
        all_ids = {
            str(item.get("test_id"))
            for item in (top_test_outliers or [])
            if isinstance(item, dict) and item.get("test_id")
        }
        likely_relevant: set[str] = set()
        likely_non_causal: set[str] = set()
        evaluations = outlier_relevance.get("evaluations", []) if isinstance(outlier_relevance, dict) else []
        for ev in evaluations:
            if not isinstance(ev, dict):
                continue
            test_id = ev.get("test_id")
            if not isinstance(test_id, str):
                continue
            cls = ev.get("classification")
            if cls == "likely_relevant":
                likely_relevant.add(test_id)
            elif cls == "likely_non_causal":
                likely_non_causal.add(test_id)
        return {
            "all_ids": all_ids,
            "likely_relevant": likely_relevant,
            "likely_non_causal": likely_non_causal,
        }

    def _hypothesis_mentions_test_ids(self, hypothesis: Hypothesis, test_ids: set[str]) -> bool:
        """Check whether hypothesis text references any test IDs."""
        if not test_ids:
            return False
        haystack = " ".join(
            [
                hypothesis.title,
                hypothesis.description,
                hypothesis.mechanism,
                hypothesis.evidence_basis or "",
                " ".join(hypothesis.expected_signatures or []),
            ]
        ).upper()
        return any(test_id.upper() in haystack for test_id in test_ids)

    def _ensure_competing_pathways(
        self,
        *,
        hypotheses: list[Hypothesis],
        case_id: str,
        failure_type: str,
        outlier_relevance: dict,
        top_test_outliers: list[dict],
    ) -> list[Hypothesis]:
        """Ensure non-outlier pathways remain viable when outlier links are weak."""
        context = self._extract_outlier_signal_context(
            outlier_relevance=outlier_relevance,
            top_test_outliers=top_test_outliers,
        )
        outlier_ids = context["all_ids"]
        if not outlier_ids:
            return hypotheses

        outlier_hyps = [h for h in hypotheses if self._hypothesis_mentions_test_ids(h, outlier_ids)]
        non_outlier_hyps = [h for h in hypotheses if not self._hypothesis_mentions_test_ids(h, outlier_ids)]

        # Penalize hypotheses tied to outliers classified as likely non-causal.
        for hypothesis in outlier_hyps:
            if self._hypothesis_mentions_test_ids(hypothesis, context["likely_non_causal"]):
                hypothesis.prior_confidence = max(0.1, hypothesis.prior_confidence - 0.12)

        # Boost hypotheses tied to outliers with strong relevance.
        for hypothesis in outlier_hyps:
            if self._hypothesis_mentions_test_ids(hypothesis, context["likely_relevant"]):
                hypothesis.prior_confidence = min(0.9, hypothesis.prior_confidence + 0.08)

        # Ensure at least one non-outlier pathway is present.
        if not non_outlier_hyps:
            target = self._resolve_measurement_column(failure_type)
            non_outlier = Hypothesis(
                hypothesis_id=f"HYP-{case_id}-{uuid.uuid4().hex[:6].upper()}",
                case_id=case_id,
                title="Non-outlier causal pathway",
                description=(
                    "Primary outlier signals may be associated but non-causal. "
                    "Investigate independent process/material pathways not anchored to top outlier tests."
                ),
                mechanism=(
                    "Failure is driven by upstream process/material factors that may not create "
                    "strong anomalies in highlighted outlier test IDs."
                ),
                origin=HypothesisOrigin.ENGINEERING_REASONING,
                expected_signatures=[
                    "Failure rate differences persist after controlling for outlier test IDs",
                    "Alternative process variables explain variance better than outlier tests",
                ],
                recommended_tests=[
                    StatisticalTest(
                        test_type="one_way_anova",
                        target_variable=target,
                        grouping_key="lot_number",
                        expected_signal="Significant group effect independent of flagged outlier tests",
                    )
                ],
                required_data_sources=["leak_test_history", "component_lots", "assembly_parameters"],
                prior_confidence=0.45,
            )
            hypotheses.append(non_outlier)
            non_outlier_hyps = [non_outlier]

        # Prevent tunnel vision: keep one non-outlier pathway competitive.
        max_outlier = max((h.prior_confidence for h in outlier_hyps), default=0.0)
        best_non_outlier = max(non_outlier_hyps, key=lambda h: h.prior_confidence)
        if max_outlier > 0 and best_non_outlier.prior_confidence < (max_outlier - 0.08):
            best_non_outlier.prior_confidence = min(0.9, max_outlier - 0.08)

        return hypotheses
