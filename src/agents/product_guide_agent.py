"""Product Guide Agent - retrieves and summarizes relevant product documentation."""

from typing import Any

from .base_agent import BaseRCAAgent, AgentOutput
from src.tools.rag_tool import RAGTool


class ProductGuideAgent(BaseRCAAgent):
    """Agent that retrieves relevant product guide sections for the failure mode."""
    
    name = "ProductGuideAgent"
    description = "Retrieves and summarizes relevant product guide sections with citations"
    
    system_prompt = """You are a product documentation specialist for manufacturing root cause analysis.
Your job is to:
1. Identify which product guide sections are relevant to the reported failure
2. Retrieve the critical features and parameters that may be involved
3. Find any documented failure modes or known issues
4. Provide clear citations for all information retrieved

Always cite the specific section, revision, and page number when providing information.
Focus on information that will help identify the root cause of the failure."""

    def __init__(self, rag_tool: RAGTool = None, **kwargs):
        super().__init__(**kwargs)
        self.rag_tool = rag_tool or RAGTool()
    
    def execute(self, context: dict[str, Any]) -> AgentOutput:
        """Retrieve relevant product guide information.
        
        Args:
            context: Should contain:
                - 'case': The normalized failure case
                - 'recipe': The analysis recipe (optional)
                
        Returns:
            AgentOutput with relevant product guide information and citations
        """
        self.log("Starting product guide retrieval")
        
        case = context.get("case", {})
        recipe = context.get("recipe")
        
        failure_type = case.get("failure_type", "unknown")
        failure_description = case.get("failure_description", "")
        part_number = case.get("part_number", "")
        
        # Build search queries
        queries = self._build_queries(failure_type, failure_description, part_number)
        
        # Get recipe-specified sections if available
        recipe_sections = []
        if recipe and hasattr(recipe, 'relevant_guide_sections'):
            recipe_sections = recipe.relevant_guide_sections
        
        # Retrieve relevant chunks
        all_citations = []
        sections_found = {}
        
        # First, search for recipe-specified sections
        if recipe_sections:
            self.log(f"Searching for {len(recipe_sections)} recipe-specified sections")
            for section in recipe_sections:
                results = self.rag_tool.retrieve_with_citations(
                    query=section,
                    top_k=2,
                    section_filter=[section]
                )
                for citation in results:
                    citation["search_type"] = "recipe_section"
                    citation["matched_query"] = section
                    all_citations.append(citation)
                    
                    section_path = citation.get("section_path", "Unknown")
                    if section_path not in sections_found:
                        sections_found[section_path] = []
                    sections_found[section_path].append(citation)
        
        # Then, do semantic search based on failure details
        self.log(f"Running {len(queries)} semantic searches")
        for query in queries:
            results = self.rag_tool.retrieve_with_citations(
                query=query,
                top_k=3,
            )
            for citation in results:
                # Avoid duplicates
                if not any(c.get("source_id") == citation.get("source_id") for c in all_citations):
                    citation["search_type"] = "semantic"
                    citation["matched_query"] = query
                    all_citations.append(citation)
        
        # Extract critical features and parameters from retrieved content
        critical_features = self._extract_critical_features(all_citations)
        expected_signatures = self._extract_expected_signatures(all_citations, failure_type)
        deterministic_reasoning = (
            f"Retrieved {len(all_citations)} relevant product guide sections covering "
            f"{len(sections_found)} topics"
        )
        llm_reasoning, used_llm = self._llm_analysis_or_fallback(
            objective="Summarize product-guide evidence relevance for this failure.",
            context_payload={
                "failure_type": failure_type,
                "part_number": part_number,
                "queries": queries[:8],
                "total_citations": len(all_citations),
                "sections_found": list(sections_found.keys())[:10],
                "critical_features_count": len(critical_features),
                "expected_signatures_count": len(expected_signatures),
            },
            fallback_reasoning=deterministic_reasoning,
        )
        
        self.log(f"Retrieved {len(all_citations)} relevant chunks, {len(critical_features)} critical features")
        
        return AgentOutput(
            agent_name=self.name,
            success=True,
            data={
                "sections_found": sections_found,
                "critical_features": critical_features,
                "expected_signatures": expected_signatures,
                "total_chunks_retrieved": len(all_citations),
                "llm_analysis_used": used_llm,
                "llm_analysis": llm_reasoning,
            },
            citations_used=all_citations,
            confidence=min(1.0, len(all_citations) / 5),  # Higher confidence with more sources
            reasoning=llm_reasoning
        )
    
    def _build_queries(
        self, 
        failure_type: str, 
        failure_description: str,
        part_number: str
    ) -> list[str]:
        """Build search queries based on failure details."""
        queries = []
        
        # Query based on failure type
        if failure_type:
            queries.append(f"failure mode {failure_type}")
            queries.append(f"troubleshooting {failure_type}")
        
        # Query based on description keywords
        if failure_description:
            # Extract key technical terms
            technical_terms = self._extract_technical_terms(failure_description)
            for term in technical_terms[:3]:
                queries.append(f"specifications {term}")
                queries.append(f"requirements {term}")
        
        # Query for part-specific info
        if part_number:
            queries.append(f"{part_number} specifications")
            queries.append(f"{part_number} critical parameters")
        
        # General queries
        queries.append("critical dimensions tolerances")
        queries.append("quality requirements acceptance criteria")
        
        return queries
    
    def _extract_technical_terms(self, text: str) -> list[str]:
        """Extract technical terms from text for targeted search."""
        # Simple extraction - in production, could use NER or domain-specific extraction
        technical_keywords = [
            "seal", "o-ring", "torque", "pressure", "leak", "dimension",
            "tolerance", "surface", "finish", "hardness", "material",
            "temperature", "force", "load", "stress", "strain",
            "clearance", "interference", "fit", "alignment", "assembly"
        ]
        
        text_lower = text.lower()
        found_terms = [term for term in technical_keywords if term in text_lower]
        
        return found_terms
    
    def _extract_critical_features(self, citations: list[dict]) -> list[dict]:
        """Extract critical features from retrieved content."""
        features = []
        
        for citation in citations:
            excerpt = citation.get("excerpt", "")
            section = citation.get("section_path", "Unknown")
            
            # Look for critical/key feature indicators
            if any(kw in excerpt.lower() for kw in ["critical", "key", "important", "required"]):
                features.append({
                    "feature": excerpt[:200],
                    "source_section": section,
                    "citation_id": citation.get("source_id"),
                })
        
        return features[:10]  # Limit to top 10
    
    def _extract_expected_signatures(
        self, 
        citations: list[dict], 
        failure_type: str
    ) -> list[dict]:
        """Extract expected data signatures based on product guide."""
        signatures = []
        
        # Based on failure type, define what data signatures to look for
        signature_map = {
            "leak_test_fail": [
                {"signature": "Leak rate correlation with O-ring lot", "data_source": "component_lots"},
                {"signature": "Assembly torque out of spec", "data_source": "assembly_parameters"},
                {"signature": "Surface finish degradation trend", "data_source": "dimensional_history"},
            ],
            "dimensional_oot": [
                {"signature": "Tool wear correlation with dimension drift", "data_source": "tool_changes"},
                {"signature": "Machine-to-machine variation", "data_source": "machine_parameters"},
                {"signature": "Material lot hardness variation", "data_source": "material_lots"},
            ],
        }
        
        base_signatures = signature_map.get(failure_type, [])
        
        # Add signatures based on retrieved content
        for citation in citations:
            excerpt = citation.get("excerpt", "").lower()
            if "specification" in excerpt or "tolerance" in excerpt:
                signatures.append({
                    "signature": f"Parameter from {citation.get('section_path', 'product guide')}",
                    "data_source": "product_guide",
                    "citation_id": citation.get("source_id"),
                })
        
        return base_signatures + signatures[:5]
