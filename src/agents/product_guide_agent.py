"""Product Guide Agent - retrieves and summarizes relevant product documentation."""

import json
import re
from collections import Counter
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
        self.max_retrieval_rounds = 3
        self.target_chunk_count = 10
        self.max_queries_per_round = 4
        self.max_coverage_queries = 3
    
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
        seen_source_ids: set[str] = set()
        
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
                    self._add_citation(
                        citation=citation,
                        all_citations=all_citations,
                        seen_source_ids=seen_source_ids,
                        sections_found=sections_found,
                    )

        # Then run iterative semantic retrieval until coverage is sufficient.
        self.log(f"Running iterative semantic retrieval from {len(queries)} seed queries")
        retrieval_iterations = self._run_iterative_retrieval(
            seed_queries=queries,
            failure_type=failure_type,
            all_citations=all_citations,
            sections_found=sections_found,
            seen_source_ids=seen_source_ids,
        )
        coverage_pass = self._run_coverage_pass(
            failure_type=failure_type,
            failure_description=failure_description,
            part_number=part_number,
            all_citations=all_citations,
            sections_found=sections_found,
            seen_source_ids=seen_source_ids,
        )
        
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
                "retrieval_iterations": retrieval_iterations,
                "coverage_pass": coverage_pass,
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
        """Extract technical terms from text for targeted search.

        Prefers LLM extraction (if available) and falls back to deterministic
        phrase scoring so behavior remains robust without external models.
        """
        if not text:
            return []

        llm_terms = self._extract_technical_terms_with_llm(text)
        if llm_terms:
            return llm_terms
        return self._extract_technical_terms_fallback(text)

    def _extract_technical_terms_with_llm(self, text: str) -> list[str]:
        """Use LLM to identify domain-specific terms from arbitrary text."""
        if self.llm is None:
            return []
        try:
            prompt = (
                "Extract up to 8 manufacturing/mechanical technical terms or short phrases "
                "that are most relevant for document retrieval.\n"
                "Return strict JSON only in the format: "
                '{"terms": ["term1", "term2"]}\n\n'
                f"Text:\n{text[:2000]}"
            )
            raw = str(self._call_llm(prompt)).strip()
            data = self._parse_json_response(raw)
            terms = data.get("terms", []) if isinstance(data, dict) else []
            if not isinstance(terms, list):
                return []
            cleaned: list[str] = []
            seen = set()
            for term in terms:
                if not isinstance(term, str):
                    continue
                normalized = term.strip().lower()
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                cleaned.append(normalized)
                if len(cleaned) >= 8:
                    break
            return cleaned
        except Exception:
            return []

    def _extract_technical_terms_fallback(self, text: str) -> list[str]:
        """Deterministically extract likely technical terms without hard-coded domain lists."""
        stopwords = {
            "the", "and", "for", "with", "from", "that", "this", "are", "was",
            "were", "have", "has", "had", "into", "onto", "over", "under",
            "then", "than", "when", "where", "what", "which", "about", "into",
            "must", "should", "may", "can", "could", "would", "after", "before",
            "between", "during", "within", "without", "failure", "failed",
            "part", "number", "test", "mode",
        }
        normalized = re.sub(r"[^a-zA-Z0-9\-\s]", " ", text.lower())
        tokens = [token for token in normalized.split() if len(token) >= 3 and token not in stopwords]
        if not tokens:
            return []

        token_counts = Counter(tokens)
        bigram_counts: Counter[str] = Counter()
        for i in range(len(tokens) - 1):
            a, b = tokens[i], tokens[i + 1]
            if a in stopwords or b in stopwords:
                continue
            if len(a) < 3 or len(b) < 3:
                continue
            bigram_counts[f"{a} {b}"] += 1

        ranked: list[str] = []
        for phrase, _count in bigram_counts.most_common(6):
            ranked.append(phrase)
        for token, _count in token_counts.most_common(12):
            ranked.append(token)

        deduped: list[str] = []
        seen = set()
        for term in ranked:
            if term in seen:
                continue
            seen.add(term)
            deduped.append(term)
            if len(deduped) >= 8:
                break
        return deduped

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

    def _add_citation(
        self,
        citation: dict,
        all_citations: list[dict],
        seen_source_ids: set[str],
        sections_found: dict[str, list[dict]],
    ) -> bool:
        """Add citation if new, returning True when a new citation is stored."""
        source_id = citation.get("source_id")
        if source_id and source_id in seen_source_ids:
            return False

        all_citations.append(citation)
        if source_id:
            seen_source_ids.add(source_id)

        section_path = citation.get("section_path", "Unknown")
        if section_path not in sections_found:
            sections_found[section_path] = []
        sections_found[section_path].append(citation)
        return True

    def _run_iterative_retrieval(
        self,
        seed_queries: list[str],
        failure_type: str,
        all_citations: list[dict],
        sections_found: dict[str, list[dict]],
        seen_source_ids: set[str],
    ) -> int:
        """Iteratively retrieve and refine queries until context is sufficient or exhausted."""
        pending_queries: list[str] = []
        seen_queries: set[str] = set()
        for query in seed_queries:
            normalized = query.strip()
            if normalized and normalized not in seen_queries:
                pending_queries.append(normalized)
                seen_queries.add(normalized)

        iterations = 0
        while (
            pending_queries
            and iterations < self.max_retrieval_rounds
            and len(all_citations) < self.target_chunk_count
        ):
            iterations += 1
            round_queries = pending_queries[: self.max_queries_per_round]
            pending_queries = pending_queries[self.max_queries_per_round :]
            self.log(
                f"Retrieval round {iterations}: evaluating {len(round_queries)} queries "
                f"(citations={len(all_citations)})"
            )

            new_round_citations: list[dict] = []
            for query in round_queries:
                results = self.rag_tool.retrieve_with_citations(
                    query=query,
                    top_k=3,
                    enable_hierarchy_expansion=True,
                    enable_coverage_gap_pass=True,
                )
                for citation in results:
                    citation["search_type"] = "semantic_iterative"
                    citation["matched_query"] = query
                    was_added = self._add_citation(
                        citation=citation,
                        all_citations=all_citations,
                        seen_source_ids=seen_source_ids,
                        sections_found=sections_found,
                    )
                    if was_added:
                        new_round_citations.append(citation)

            if not new_round_citations:
                self.log("Stopping iterative retrieval: no new citations found")
                break

            follow_up_queries = self._propose_follow_up_queries(
                citations=new_round_citations,
                failure_type=failure_type,
                seen_queries=seen_queries,
            )
            for query in follow_up_queries:
                if query not in seen_queries:
                    pending_queries.append(query)
                    seen_queries.add(query)

        return iterations

    def _propose_follow_up_queries(
        self,
        citations: list[dict],
        failure_type: str,
        seen_queries: set[str],
    ) -> list[str]:
        """Generate follow-up retrieval queries from newly discovered evidence."""
        follow_ups: list[str] = []
        for citation in citations[:5]:
            section_path = str(citation.get("section_path", "")).strip()
            if section_path:
                leaf_section = section_path.split(">")[-1].strip()
                for candidate in [leaf_section, section_path]:
                    if candidate and candidate not in seen_queries:
                        follow_ups.append(candidate)
                if leaf_section and failure_type:
                    combined = f"{failure_type} {leaf_section}".strip()
                    if combined not in seen_queries:
                        follow_ups.append(combined)

            excerpt = str(citation.get("excerpt", ""))
            for term in self._extract_technical_terms(excerpt):
                query = f"{term} specifications"
                if query not in seen_queries:
                    follow_ups.append(query)

        # De-duplicate while preserving order and keep breadth bounded.
        deduped: list[str] = []
        seen_local = set()
        for query in follow_ups:
            normalized = query.strip()
            if not normalized or normalized in seen_local:
                continue
            seen_local.add(normalized)
            deduped.append(normalized)
            if len(deduped) >= 8:
                break
        return deduped
    
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

    def _run_coverage_pass(
        self,
        failure_type: str,
        failure_description: str,
        part_number: str,
        all_citations: list[dict],
        sections_found: dict[str, list[dict]],
        seen_source_ids: set[str],
    ) -> dict[str, Any]:
        """Run targeted follow-up retrieval for uncovered terms and weak section coverage."""
        seed_text = " ".join([failure_type, failure_description, part_number]).strip()
        expected_terms = self._extract_technical_terms(seed_text)
        if failure_type:
            expected_terms.append(failure_type.lower())
        if part_number:
            expected_terms.append(part_number.lower())

        citation_text = " ".join(
            f"{c.get('section_path', '')} {c.get('excerpt', '')}".lower()
            for c in all_citations
        )
        uncovered_terms = [term for term in expected_terms if term and term not in citation_text]

        low_coverage_sections = [
            section
            for section, items in sections_found.items()
            if len(items) <= 1
        ][:2]

        follow_up_queries: list[str] = []
        for term in uncovered_terms[: self.max_coverage_queries]:
            follow_up_queries.append(f"{term} specification limits")
        for section in low_coverage_sections:
            leaf = section.split(">")[-1].strip()
            if leaf:
                follow_up_queries.append(f"{leaf} failure troubleshooting")

        deduped: list[str] = []
        seen_queries = set()
        for query in follow_up_queries:
            normalized = query.strip()
            if not normalized or normalized in seen_queries:
                continue
            seen_queries.add(normalized)
            deduped.append(normalized)
            if len(deduped) >= self.max_coverage_queries:
                break

        added = 0
        for query in deduped:
            results = self.rag_tool.retrieve_with_citations(
                query=query,
                top_k=2,
                enable_hierarchy_expansion=True,
                enable_coverage_gap_pass=True,
            )
            for citation in results:
                citation["search_type"] = "coverage_gap"
                citation["matched_query"] = query
                if self._add_citation(
                    citation=citation,
                    all_citations=all_citations,
                    seen_source_ids=seen_source_ids,
                    sections_found=sections_found,
                ):
                    added += 1

        return {
            "expected_terms": expected_terms[:8],
            "uncovered_terms": uncovered_terms[:8],
            "queries_run": deduped,
            "citations_added": added,
        }
    
    def _extract_expected_signatures(
        self, 
        citations: list[dict], 
        failure_type: str
    ) -> list[dict]:
        """Extract expected data signatures based on product guide."""
        signatures = []

        # Derive signatures from retrieved content rather than fixed failure mappings.
        for citation in citations:
            excerpt = citation.get("excerpt", "").lower()
            if "specification" in excerpt or "tolerance" in excerpt:
                signatures.append({
                    "signature": f"Parameter from {citation.get('section_path', 'product guide')}",
                    "data_source": "product_guide",
                    "citation_id": citation.get("source_id"),
                })

        # Fallback: create query-aligned signatures from discovered terms.
        if not signatures and citations:
            combined_excerpt = " ".join(str(c.get("excerpt", "")) for c in citations[:8])
            for term in self._extract_technical_terms(combined_excerpt)[:5]:
                signatures.append(
                    {
                        "signature": f"{failure_type or 'failure'} correlation with {term}",
                        "data_source": "product_guide",
                    }
                )

        return signatures[:8]
