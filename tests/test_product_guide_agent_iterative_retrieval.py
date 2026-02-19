"""Tests for iterative product-guide retrieval behavior."""

from __future__ import annotations

import unittest

from src.agents.product_guide_agent import ProductGuideAgent
from src.tools.rag_tool import DocumentChunk, RAGTool


class _StubRAGTool:
    def __init__(self, responses: dict[str, list[dict]], default: list[dict] | None = None):
        self.responses = responses
        self.default = default or []
        self.queries_seen: list[str] = []
        self.kwargs_seen: list[dict] = []

    def retrieve_with_citations(self, query: str, top_k: int = 5, **kwargs):
        self.queries_seen.append(query)
        self.kwargs_seen.append(dict(kwargs))
        items = self.responses.get(query, self.default)
        return [dict(item) for item in items[:top_k]]


class ProductGuideIterativeRetrievalTests(unittest.TestCase):
    def test_iterative_retrieval_adds_follow_up_context(self):
        seed_citation = {
            "source_id": "chunk-1",
            "source_name": "guide.md",
            "section_path": "3. Sealing System > O-Ring Specs",
            "excerpt": "Critical O-ring squeeze and leak requirements must be met.",
        }
        follow_up_citation = {
            "source_id": "chunk-2",
            "source_name": "guide.md",
            "section_path": "3. Sealing System > O-Ring Specs > Gland Dimensions",
            "excerpt": "Important gland depth tolerance is 0.125 +/- 0.002 in.",
        }
        rag_tool = _StubRAGTool(
            responses={
                "failure mode leak_test_fail": [seed_citation],
                "O-Ring Specs": [follow_up_citation],
            }
        )
        agent = ProductGuideAgent(rag_tool=rag_tool, llm=None, verbose=False)

        output = agent.execute(
            {
                "case": {
                    "failure_type": "leak_test_fail",
                    "failure_description": "",
                    "part_number": "",
                }
            }
        )

        self.assertTrue(output.success)
        self.assertGreaterEqual(output.data.get("retrieval_iterations", 0), 2)
        self.assertEqual(output.data.get("total_chunks_retrieved"), 2)
        self.assertIn("O-Ring Specs", rag_tool.queries_seen)
        self.assertIn("chunk-2", [c.get("source_id") for c in output.citations_used])

    def test_iterative_retrieval_stops_when_no_new_citations(self):
        repeated = {
            "source_id": "chunk-repeat",
            "source_name": "guide.md",
            "section_path": "2. Assembly > Torque",
            "excerpt": "Critical torque limit for assembly.",
        }
        rag_tool = _StubRAGTool(
            responses={
                "failure mode leak_test_fail": [repeated],
                "Torque": [repeated],
                "2. Assembly > Torque": [repeated],
            }
        )
        agent = ProductGuideAgent(rag_tool=rag_tool, llm=None, verbose=False)

        output = agent.execute(
            {"case": {"failure_type": "leak_test_fail", "failure_description": "", "part_number": ""}}
        )

        self.assertTrue(output.success)
        self.assertEqual(output.data.get("total_chunks_retrieved"), 1)
        self.assertLess(output.data.get("retrieval_iterations", 0), agent.max_retrieval_rounds)

    def test_dynamic_term_extraction_is_not_limited_to_fixed_keywords(self):
        rag_tool = _StubRAGTool(responses={})
        agent = ProductGuideAgent(rag_tool=rag_tool, llm=None, verbose=False)

        terms = agent._extract_technical_terms(
            "Observed cavitation erosion around diffuser vane with resonance artifacts."
        )

        self.assertTrue(terms)
        self.assertTrue(any("cavitation" in term for term in terms))

    def test_expected_signatures_fallback_uses_discovered_terms(self):
        rag_tool = _StubRAGTool(responses={})
        agent = ProductGuideAgent(rag_tool=rag_tool, llm=None, verbose=False)
        citations = [
            {
                "source_id": "chunk-x",
                "section_path": "4. Flow Path",
                "excerpt": "Cavitation erosion near diffuser vane and harmonic resonance.",
            }
        ]

        signatures = agent._extract_expected_signatures(citations, "flow_instability")

        self.assertTrue(signatures)
        self.assertTrue(any("cavitation" in s.get("signature", "") for s in signatures))

    def test_coverage_pass_adds_targeted_queries_for_uncovered_terms(self):
        seed = {
            "source_id": "chunk-seed",
            "source_name": "guide.md",
            "section_path": "1. Intro",
            "excerpt": "General reliability guidance.",
        }
        coverage = {
            "source_id": "chunk-coverage",
            "source_name": "guide.md",
            "section_path": "4. Cavitation Limits",
            "excerpt": "Cavitation erosion thresholds and acceptance criteria.",
        }
        rag_tool = _StubRAGTool(
            responses={
                "failure mode flow_instability": [seed],
                "flow_instability specification limits": [coverage],
                "cavitation specification limits": [coverage],
            }
        )
        agent = ProductGuideAgent(rag_tool=rag_tool, llm=None, verbose=False)
        agent._extract_technical_terms = lambda _text: ["cavitation"]

        output = agent.execute(
            {
                "case": {
                    "failure_type": "flow_instability",
                    "failure_description": "Observed cavitation erosion at diffuser.",
                    "part_number": "",
                }
            }
        )

        self.assertTrue(output.success)
        coverage_info = output.data.get("coverage_pass", {})
        self.assertGreaterEqual(coverage_info.get("citations_added", 0), 1)
        self.assertTrue(
            any("specification limits" in q for q in coverage_info.get("queries_run", []))
        )
        self.assertIn("chunk-coverage", [c.get("source_id") for c in output.citations_used])

    def test_hybrid_retrieve_marks_strategy_and_expands_hierarchy(self):
        rag_tool = RAGTool(embedding_model=None)
        rag_tool.clear_store()
        rag_tool._chunks = {
            "c1": DocumentChunk(
                chunk_id="c1",
                document_id="doc1",
                document_name="guide.md",
                content="O-ring gland dimensions and squeeze requirements.",
                section_path="3. Sealing System > O-Ring Specs",
            ),
            "c2": DocumentChunk(
                chunk_id="c2",
                document_id="doc1",
                document_name="guide.md",
                content="Recommended gland depth tolerance is 0.125 +/- 0.002 in.",
                section_path="3. Sealing System > O-Ring Specs > Gland Dimensions",
            ),
        }

        citations = rag_tool.retrieve_with_citations("o-ring specs", top_k=2)

        self.assertEqual(len(citations), 2)
        self.assertTrue(all(c.get("retrieval_strategy") == "hybrid_recursive" for c in citations))
        self.assertIn("c2", [c.get("source_id") for c in citations])


if __name__ == "__main__":
    unittest.main()
