"""Private Data Research Agent - retrieves internal evidence from databases."""

import json
import re
from typing import Any

from .base_agent import BaseRCAAgent, AgentOutput
from config.settings import get_settings
from config.analysis_recipes.loader import normalize_recipe_mode
from src.tools.sql_tool import SQLTool
from src.tools.data_fetch_tool import DataFetchTool
from src.tools.data_catalog import DataCatalog, DataCategory, CatalogField


class PrivateDataResearchAgent(BaseRCAAgent):
    """Agent that retrieves internal evidence from manufacturing databases."""
    
    name = "PrivateDataResearchAgent"
    description = "Queries internal databases for historical cases, process data, and component traceability"
    
    system_prompt = """You are a data research specialist for manufacturing root cause analysis.
Your job is to:
1. Query internal databases for relevant historical data
2. Find similar past failures and their resolutions
3. Gather process parameters, component traceability, and test data
4. Fetch test data, ROA parameters, and operator buyoffs for statistical analysis
5. Identify any patterns or anomalies in the data

Be thorough but focused. Prioritize data that directly relates to the failure mode.
Always note the data source and time range for any data retrieved."""

    def __init__(
        self, 
        sql_tool: SQLTool = None, 
        data_fetch_tool: DataFetchTool = None,
        data_catalog: DataCatalog = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.sql_tool = sql_tool or SQLTool(mock_mode=True)
        self.data_fetch_tool = data_fetch_tool or DataFetchTool(mock_mode=True)
        if data_catalog is not None:
            self.data_catalog = data_catalog
        else:
            settings = get_settings()
            self.data_catalog = DataCatalog(
                catalog_dir=settings.catalog_dir,
                db_url=settings.catalog_db_url,
            )
    
    def execute(self, context: dict[str, Any]) -> AgentOutput:
        """Retrieve internal data relevant to the failure.
        
        Args:
            context: Should contain:
                - 'case': The normalized failure case
                - 'recipe': The analysis recipe
                - 'product_guide_output': Output from ProductGuideAgent (optional)
                
        Returns:
            AgentOutput with retrieved data and citations
        """
        self.log("Starting private data research")
        
        case = context.get("case", {})
        recipe = context.get("recipe")
        product_guide_output = context.get("product_guide_output", {})
        recipe_mode = normalize_recipe_mode()
        
        failure_type = case.get("failure_type", "")
        part_number = case.get("part_number", "")
        lot_number = case.get("lot_number", "")
        
        data_retrieved = {}
        citations = []
        
        # 1. Get similar historical cases
        self.log("Searching for similar historical cases")
        try:
            similar_cases = self._get_similar_cases(failure_type, part_number)
            data_retrieved["similar_cases"] = similar_cases
            citations.append(similar_cases["citation"])
        except Exception as e:
            self.log(f"Failed to get similar cases: {e}", "warning")

        # 2. Pull top outlier test IDs for this unit context
        self.log("Retrieving top outlier test IDs")
        try:
            outlier_result = self.data_fetch_tool.fetch_top_test_outliers(
                serial_number=case.get("serial_number"),
                lot_number=lot_number,
                part_number=part_number,
                failure_type=failure_type,
                top_n=get_settings().outlier_top_n,
                time_window=get_settings().outlier_time_window,
            )
            top_outliers = outlier_result.outliers
            data_retrieved["top_test_outliers"] = {
                "data": top_outliers,
                "row_count": len(top_outliers),
                "citation": outlier_result.to_citation_dict(),
                "request_id": outlier_result.request_id,
            }
            citations.append(outlier_result.to_citation_dict())
        except Exception as e:
            self.log(f"Failed to retrieve outlier test IDs: {e}", "warning")
        
        # 3. Recipe-specific pulls only in strict mode
        if recipe_mode == "strict" and recipe and hasattr(recipe, 'data_requirements'):
            for data_req in recipe.data_requirements:
                self.log(f"Querying {data_req.source_name}")
                try:
                    result = self._query_data_source(data_req, case)
                    data_retrieved[data_req.source_name] = result
                    citations.append(result["citation"])
                except Exception as e:
                    self.log(f"Failed to query {data_req.source_name}: {e}", "warning")
        
        # 4. Get component traceability if lot number available
        if lot_number:
            self.log("Getting component traceability")
            try:
                traceability = self._get_component_traceability(lot_number)
                data_retrieved["component_traceability"] = traceability
                citations.append(traceability["citation"])
            except Exception as e:
                self.log(f"Failed to get traceability: {e}", "warning")
        
        # 5. Fetch analysis data (test data, ROA, operator buyoffs) for stats
        self.log("Fetching analysis data from internal API")
        try:
            analysis_data = self._fetch_analysis_data(
                case=case,
                recipe=recipe,
                product_guide_output=product_guide_output,
                recipe_mode=recipe_mode,
                top_outlier_tests=data_retrieved.get("top_test_outliers", {}).get("data", []),
            )
            data_retrieved["analysis_data"] = analysis_data
            citations.append(analysis_data["citation"])
        except Exception as e:
            self.log(f"Failed to fetch analysis data: {e}", "warning")

        # 6. Assess whether outlier tests appear causally relevant or likely non-causal
        try:
            outlier_relevance = self._assess_outlier_relevance(
                case=case,
                top_outlier_tests=data_retrieved.get("top_test_outliers", {}).get("data", []),
                analysis_data=data_retrieved.get("analysis_data", {}),
                product_guide_output=product_guide_output,
            )
            data_retrieved["outlier_relevance"] = outlier_relevance
        except Exception as e:
            self.log(f"Failed outlier relevance scoring: {e}", "warning")
        
        # Summarize what was found
        summary = self._summarize_findings(data_retrieved, case)
        total_records = sum(
            d.get("row_count", 0) for d in data_retrieved.values()
            if isinstance(d, dict)
        )
        deterministic_reasoning = (
            f"Retrieved data from {len(data_retrieved)} sources with {total_records} total records"
        )
        llm_reasoning, used_llm = self._llm_analysis_or_fallback(
            objective="Summarize research findings, evidence quality, and data gaps.",
            context_payload={
                "failure_type": failure_type,
                "part_number": part_number,
                "sources": list(data_retrieved.keys()),
                "summary": summary,
                "total_records": total_records,
                "recipe_mode": recipe_mode,
                "outlier_relevance_summary": data_retrieved.get("outlier_relevance", {}).get("summary", {}),
            },
            fallback_reasoning=deterministic_reasoning,
        )
        
        self.log(f"Research complete. Retrieved data from {len(data_retrieved)} sources")
        
        return AgentOutput(
            agent_name=self.name,
            success=True,
            data={
                "data_sources_queried": list(data_retrieved.keys()),
                "data_retrieved": data_retrieved,
                "summary": summary,
                "total_records": total_records,
                "recipe_mode": recipe_mode,
                "recipe_applied": bool(recipe),
                "llm_analysis_used": used_llm,
                "llm_analysis": llm_reasoning,
            },
            citations_used=citations,
            confidence=min(1.0, len(data_retrieved) / 4),
            reasoning=llm_reasoning
        )
    
    def _get_similar_cases(self, failure_type: str, part_number: str) -> dict:
        """Get similar historical RCA cases."""
        result = self.sql_tool.get_similar_cases(
            failure_type=failure_type,
            part_number=part_number,
            time_window="365d",
            limit=20,
        )
        
        return {
            "data": result.data.to_dict(orient="records"),
            "row_count": result.row_count,
            "citation": result.to_citation_dict(),
            "query_id": result.query_id,
        }
    
    def _query_data_source(self, data_req: Any, case: dict) -> dict:
        """Query a data source based on recipe requirements."""
        # Build filters based on case data
        filters = {}
        
        # Common filter fields
        if case.get("lot_number") and "lot_number" in data_req.required_columns:
            filters["lot_number"] = case["lot_number"]
        if case.get("part_number") and "part_number" in data_req.required_columns:
            filters["part_number"] = case["part_number"]
        if case.get("station_id") and "station_id" in data_req.required_columns:
            filters["station_id"] = case["station_id"]
        
        result = self.sql_tool.query_for_analysis(
            source_name=data_req.source_name,
            columns=data_req.required_columns + data_req.optional_columns,
            filters=filters,
            time_window=data_req.time_window,
            limit=data_req.min_rows * 10,  # Get more than minimum
        )
        
        return {
            "data": result.data.to_dict(orient="records"),
            "row_count": result.row_count,
            "columns": result.columns_returned,
            "citation": result.to_citation_dict(),
            "query_id": result.query_id,
        }
    
    def _get_component_traceability(self, lot_number: str) -> dict:
        """Get component lot traceability for a production lot.
        
        Note: component_lots is keyed by serial_number, not lot_number.
        We retrieve all records and can cross-reference with other sources
        that link serial_number to lot_number.
        """
        result = self.sql_tool.query_for_analysis(
            source_name="component_lots",
            columns=["serial_number", "component_name", "component_lot", "supplier_id"],
            filters={},
            limit=500,
        )
        
        return {
            "data": result.data.to_dict(orient="records"),
            "row_count": result.row_count,
            "citation": result.to_citation_dict(),
            "query_id": result.query_id,
        }
    
    def _parse_json_response(self, raw_response: str) -> dict:
        """Parse JSON response from LLM, handling fenced blocks."""
        text = raw_response.strip()
        if "```" in text:
            segments = [segment.strip() for segment in text.split("```") if segment.strip()]
            for segment in segments:
                candidate = segment
                if candidate.startswith("json"):
                    candidate = candidate[len("json"):].strip()
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    continue
        return json.loads(text)

    def _extract_part_family(self, part_number: str) -> str:
        """Extract a coarse part family key for catalog pre-filtering."""
        if not part_number:
            return ""
        tokens = [token for token in part_number.replace("_", "-").split("-") if token]
        if not tokens:
            return part_number
        if len(tokens) == 1:
            return tokens[0]
        return f"{tokens[0]}-{tokens[1]}"

    def _build_enriched_catalog_query(
        self,
        case: dict,
        product_guide_output: dict,
        top_outlier_tests: list[dict] | None = None,
    ) -> str:
        """Build an enriched retrieval query from case + product guide context."""
        case_text = " ".join(
            str(value)
            for value in [
                case.get("failure_type", ""),
                case.get("failure_description", ""),
                case.get("part_number", ""),
                case.get("test_name", ""),
            ]
            if value
        )

        pg_data = product_guide_output.get("data", {}) if isinstance(product_guide_output, dict) else {}
        critical_features = pg_data.get("critical_features", [])
        expected_signatures = pg_data.get("expected_signatures", [])
        citations = product_guide_output.get("citations_used", []) if isinstance(product_guide_output, dict) else []

        feature_texts = []
        for feature in critical_features[:10]:
            if isinstance(feature, dict):
                feature_texts.append(str(feature.get("feature", "")))
            else:
                feature_texts.append(str(feature))

        signature_texts = []
        for sig in expected_signatures[:10]:
            if isinstance(sig, dict):
                signature_texts.append(str(sig.get("signature", "")))
            else:
                signature_texts.append(str(sig))

        citation_texts = [str(citation.get("excerpt", "")) for citation in citations[:5] if isinstance(citation, dict)]

        outlier_text = ""
        if top_outlier_tests:
            fragments = []
            for item in top_outlier_tests[:12]:
                if not isinstance(item, dict):
                    continue
                fragments.append(
                    " ".join(
                        str(v)
                        for v in [
                            item.get("test_id", ""),
                            item.get("description", ""),
                            f"z={item.get('stddev_from_mean', '')}",
                        ]
                        if v != ""
                    )
                )
            outlier_text = " ".join(fragments)

        return " ".join(
            fragment
            for fragment in [
                case_text,
                " ".join(feature_texts),
                " ".join(signature_texts),
                " ".join(citation_texts),
                outlier_text,
            ]
            if fragment
        ).strip()

    def _normalize_selection(
        self,
        selected: dict,
        candidate_ids: set[str],
    ) -> dict[str, list[str]]:
        """Normalize and filter selected field ids to known catalog candidates."""
        normalized = {
            "test_ids": [],
            "roa_parameters": [],
            "operator_buyoffs": [],
            "process_parameters": [],
        }
        for key in normalized:
            values = selected.get(key, []) if isinstance(selected, dict) else []
            if not isinstance(values, list):
                continue
            seen = set()
            for value in values:
                if not isinstance(value, str):
                    continue
                if value not in candidate_ids and not self.data_catalog.has_field(value):
                    continue
                if value in seen:
                    continue
                normalized[key].append(value)
                seen.add(value)
        return normalized

    def _fallback_field_selection(
        self,
        candidate_fields: list[CatalogField],
    ) -> dict[str, list[str]]:
        """Deterministic fallback when LLM field selection is unavailable."""
        selected = {
            "test_ids": [],
            "roa_parameters": [],
            "operator_buyoffs": [],
            "process_parameters": [],
        }
        max_per_category = {
            "test_ids": 6,
            "roa_parameters": 6,
            "operator_buyoffs": 6,
            "process_parameters": 5,
        }
        category_key_map = {
            DataCategory.TEST_DATA: "test_ids",
            DataCategory.ROA: "roa_parameters",
            DataCategory.OPERATOR_BUYOFF: "operator_buyoffs",
            DataCategory.PROCESS_PARAMETER: "process_parameters",
        }
        for field in candidate_fields:
            key = category_key_map.get(field.category)
            if not key:
                continue
            if len(selected[key]) >= max_per_category[key]:
                continue
            selected[key].append(field.field_id)
        return selected

    def _merge_recipe_must_include(
        self,
        selected_fields: dict[str, list[str]],
        recipe: Any = None,
        recipe_mode: str = "advisory",
    ) -> dict[str, list[str]]:
        """Merge recipe must-include fields into selected field groups."""
        must_include = {}
        if (
            recipe_mode in {"strict", "advisory"}
            and recipe
            and hasattr(recipe, "must_include_fields")
            and recipe.must_include_fields
        ):
            must_include = recipe.must_include_fields
        for key in ["test_ids", "roa_parameters", "operator_buyoffs", "process_parameters"]:
            items = must_include.get(key, []) if isinstance(must_include, dict) else []
            if not isinstance(items, list):
                continue
            for field_id in items:
                if isinstance(field_id, str) and self.data_catalog.has_field(field_id):
                    if field_id not in selected_fields[key]:
                        selected_fields[key].append(field_id)
        return selected_fields

    def _select_fields_for_analysis(
        self,
        case: dict,
        recipe: Any = None,
        product_guide_output: dict | None = None,
        recipe_mode: str = "advisory",
        top_outlier_tests: list[dict] | None = None,
    ) -> tuple[dict[str, list[str]], list[CatalogField], str]:
        """Select relevant analysis fields from catalog using product-guide context."""
        product_guide_output = product_guide_output or {}
        enriched_query = self._build_enriched_catalog_query(
            case,
            product_guide_output,
            top_outlier_tests=top_outlier_tests,
        )
        if not enriched_query:
            enriched_query = f"{case.get('failure_type', '')} {case.get('failure_description', '')}".strip()

        part_family = self._extract_part_family(case.get("part_number", ""))
        top_k = get_settings().catalog_embedding_top_k
        candidate_fields = self.data_catalog.search_fields(
            query=enriched_query,
            part_family=part_family,
            top_k=top_k,
            categories=[
                DataCategory.TEST_DATA,
                DataCategory.ROA,
                DataCategory.OPERATOR_BUYOFF,
                DataCategory.PROCESS_PARAMETER,
            ],
        )
        candidate_ids = {field.field_id for field in candidate_fields}

        selected = self._fallback_field_selection(candidate_fields)
        selection_reasoning = "deterministic candidate-based fallback"

        if self.llm is not None and candidate_fields:
            try:
                pg_data = product_guide_output.get("data", {}) if isinstance(product_guide_output, dict) else {}
                critical_features = pg_data.get("critical_features", [])
                expected_signatures = pg_data.get("expected_signatures", [])
                prompt = (
                    "Select relevant manufacturing data fields for RCA.\n"
                    "Return valid JSON only with keys: test_ids, roa_parameters, "
                    "operator_buyoffs, process_parameters, rationale.\n\n"
                    f"Failure type: {case.get('failure_type', '')}\n"
                    f"Failure description: {case.get('failure_description', '')}\n"
                    f"Part number: {case.get('part_number', '')}\n"
                    f"Lot number: {case.get('lot_number', '')}\n"
                    f"Critical features: {json.dumps(critical_features, default=str)[:1500]}\n"
                    f"Expected signatures: {json.dumps(expected_signatures, default=str)[:1500]}\n\n"
                    "Candidate fields:\n"
                    f"{self.data_catalog.format_for_prompt(candidate_fields)}\n\n"
                    "Select only from candidate field IDs."
                )
                llm_response = self._call_llm(prompt)
                parsed = self._parse_json_response(str(llm_response))
                selected = self._normalize_selection(parsed, candidate_ids)
                selection_reasoning = str(parsed.get("rationale", "llm selected fields"))
            except Exception as e:
                self.log(f"LLM field selection failed, using fallback: {e}", "warning")

        selected = self._merge_recipe_must_include(selected, recipe, recipe_mode=recipe_mode)
        outlier_ids_added: list[str] = []
        if top_outlier_tests:
            for item in top_outlier_tests:
                if not isinstance(item, dict):
                    continue
                test_id = item.get("test_id")
                if not isinstance(test_id, str):
                    continue
                if self.data_catalog.has_field(test_id) and test_id not in selected["test_ids"]:
                    selected["test_ids"].append(test_id)
                    outlier_ids_added.append(test_id)

        if outlier_ids_added:
            selection_reasoning = f"{selection_reasoning}; added outlier tests: {', '.join(outlier_ids_added[:10])}"

        baseline_process_parameters = [
            "AMBIENT_TEMP",
            "AMBIENT_HUMIDITY",
            "OPERATOR_ID",
            "STATION_ID",
            "SHIFT",
        ]
        for field_id in baseline_process_parameters:
            if self.data_catalog.has_field(field_id) and field_id not in selected["process_parameters"]:
                selected["process_parameters"].append(field_id)

        return selected, candidate_fields, selection_reasoning

    def _fetch_analysis_data(
        self,
        case: dict,
        recipe: Any = None,
        product_guide_output: dict | None = None,
        recipe_mode: str = "advisory",
        top_outlier_tests: list[dict] | None = None,
    ) -> dict:
        """Fetch test data, ROA parameters, and operator buyoffs for analysis.
        
        This uses the internal data API to retrieve tabular data suitable
        for statistical analysis by the StatsAgent.
        
        Args:
            case: The failure case with context
            recipe: Optional analysis recipe with data requirements
            
        Returns:
            Dict with DataFrame, metadata, and citation
        """
        lot_number = case.get("lot_number")
        selected_fields, candidate_fields, selection_reasoning = self._select_fields_for_analysis(
            case=case,
            recipe=recipe,
            product_guide_output=product_guide_output,
            recipe_mode=recipe_mode,
            top_outlier_tests=top_outlier_tests,
        )
        
        # Fetch the data
        result = self.data_fetch_tool.fetch_for_analysis(
            lot_numbers=[lot_number] if lot_number else None,
            test_ids=selected_fields["test_ids"],
            roa_parameters=selected_fields["roa_parameters"],
            operator_buyoffs=selected_fields["operator_buyoffs"],
            process_parameters=selected_fields["process_parameters"],
            time_window="90d",
            limit=5000,
        )
        
        return {
            "data": result.data.to_dict(orient="records"),
            "data_frame": result.data,  # Keep DataFrame for stats agent
            "row_count": result.row_count,
            "columns": result.columns,
            "test_ids": result.test_ids_found,
            "roa_parameters": result.roa_parameters_found,
            "operator_buyoffs": result.operator_buyoffs_found,
            "process_parameters": selected_fields["process_parameters"],
            "catalog_candidates_considered": [field.field_id for field in candidate_fields],
            "field_selection_reasoning": selection_reasoning,
            "top_outlier_tests_considered": top_outlier_tests or [],
            "recipe_mode": recipe_mode,
            "recipe_guardrails_applied": bool(recipe and recipe_mode in {"strict", "advisory"}),
            "missing_data_summary": result.missing_data_summary,
            "warnings": result.warnings,
            "citation": result.to_citation_dict(),
            "request_id": result.request_id,
        }

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Tokenize free text into normalized keywords."""
        return {token for token in re.split(r"[^a-z0-9]+", text.lower()) if len(token) >= 3}

    def _assess_outlier_relevance(
        self,
        case: dict,
        top_outlier_tests: list[dict],
        analysis_data: dict,
        product_guide_output: dict,
    ) -> dict:
        """Score whether each outlier test looks causally relevant vs likely non-causal."""
        if not top_outlier_tests:
            return {"evaluations": [], "summary": {"total": 0, "likely_relevant": 0, "inconclusive": 0, "likely_non_causal": 0}}

        case_text = " ".join(
            str(case.get(key, "")) for key in ["failure_type", "failure_description", "test_name", "part_number"]
        )
        case_tokens = self._tokenize(case_text)

        pg_citations = product_guide_output.get("citations_used", []) if isinstance(product_guide_output, dict) else []
        guide_text = " ".join(
            str(c.get("excerpt", "")) + " " + str(c.get("section_path", ""))
            for c in pg_citations[:40]
            if isinstance(c, dict)
        )
        guide_tokens = self._tokenize(guide_text)

        fetched_test_ids = set(analysis_data.get("test_ids", []) if isinstance(analysis_data, dict) else [])
        evaluations: list[dict] = []
        counts = {"likely_relevant": 0, "inconclusive": 0, "likely_non_causal": 0}

        for item in top_outlier_tests:
            if not isinstance(item, dict):
                continue
            test_id = str(item.get("test_id", "")).strip()
            desc = str(item.get("description", ""))
            try:
                if item.get("abs_stddev_from_mean") is not None:
                    z = float(item.get("abs_stddev_from_mean"))
                elif item.get("stddev_from_mean") is not None:
                    z = abs(float(item.get("stddev_from_mean")))
                else:
                    z = 0.0
            except Exception:
                z = 0.0

            outlier_tokens = self._tokenize(f"{test_id} {desc}")
            if not outlier_tokens:
                outlier_tokens = {test_id.lower()} if test_id else set()

            overlap_case = len(outlier_tokens & case_tokens) / max(1, len(outlier_tokens))
            overlap_guide = len(outlier_tokens & guide_tokens) / max(1, len(outlier_tokens))
            selected_for_analysis = 1.0 if test_id and test_id in fetched_test_ids else 0.0
            z_score_norm = min(1.0, z / 6.0)

            relevance_score = (
                0.25 * z_score_norm
                + 0.35 * overlap_case
                + 0.25 * overlap_guide
                + 0.15 * selected_for_analysis
            )

            if relevance_score >= 0.6:
                classification = "likely_relevant"
            elif relevance_score >= 0.4:
                classification = "inconclusive"
            else:
                classification = "likely_non_causal"
            counts[classification] += 1

            evaluations.append(
                {
                    "test_id": test_id,
                    "description": desc,
                    "stddev_from_mean": item.get("stddev_from_mean"),
                    "direction": item.get("direction"),
                    "scores": {
                        "z_outlier_strength": round(z_score_norm, 3),
                        "case_semantic_overlap": round(overlap_case, 3),
                        "guide_semantic_overlap": round(overlap_guide, 3),
                        "selected_for_analysis": round(selected_for_analysis, 3),
                        "relevance_score": round(relevance_score, 3),
                    },
                    "classification": classification,
                }
            )

        evaluations.sort(key=lambda e: e["scores"]["relevance_score"], reverse=True)
        return {
            "evaluations": evaluations,
            "summary": {
                "total": len(evaluations),
                "likely_relevant": counts["likely_relevant"],
                "inconclusive": counts["inconclusive"],
                "likely_non_causal": counts["likely_non_causal"],
            },
        }
    
    def _summarize_findings(self, data_retrieved: dict, case: dict) -> dict:
        """Summarize key findings from retrieved data."""
        summary = {
            "similar_cases_found": 0,
            "potential_patterns": [],
            "data_gaps": [],
        }
        
        # Count similar cases
        if "similar_cases" in data_retrieved:
            similar = data_retrieved["similar_cases"]
            summary["similar_cases_found"] = similar.get("row_count", 0)
            
            # Extract common root causes from similar cases
            if similar.get("data"):
                root_causes = [c.get("root_cause") for c in similar["data"] if c.get("root_cause")]
                if root_causes:
                    summary["common_root_causes"] = list(set(root_causes))[:5]
        
        # Identify data gaps
        expected_sources = ["leak_test_history", "component_lots", "assembly_parameters"]
        for source in expected_sources:
            if source not in data_retrieved:
                summary["data_gaps"].append(f"Missing data from {source}")
            elif data_retrieved[source].get("row_count", 0) < 30:
                summary["data_gaps"].append(f"Limited data from {source} ({data_retrieved[source].get('row_count', 0)} records)")
        
        return summary
