"""RAG tool for document retrieval with citations.

Handles product guide ingestion, chunking, embedding, and retrieval
with proper citation tracking.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
from collections import Counter
import hashlib
import json
import re

from pydantic import BaseModel, Field


@dataclass
class DocumentChunk:
    """A chunk of a document with metadata for citation."""
    
    chunk_id: str
    document_id: str
    document_name: str
    
    # Content
    content: str
    
    # Location
    section_path: str  # e.g., "Chapter 3 > Sealing System > O-Ring Specs"
    page_number: Optional[int] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    
    # Metadata
    revision: Optional[str] = None
    effective_date: Optional[str] = None
    document_type: str = "product_guide"  # product_guide, procedure, specification
    
    # Embedding (stored separately in vector DB)
    embedding: Optional[list[float]] = None
    
    def to_citation_dict(self) -> dict:
        """Convert to citation dictionary format."""
        return {
            "source_type": self.document_type,
            "source_id": self.chunk_id,
            "source_name": self.document_name,
            "section_path": self.section_path,
            "page_number": self.page_number,
            "excerpt": self.content[:500] + "..." if len(self.content) > 500 else self.content,
            "excerpt_hash": hashlib.md5(self.content.encode()).hexdigest(),
            "revision": self.revision,
        }

    def to_dict(self) -> dict:
        """Serialize chunk to a dictionary for persistence."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "document_name": self.document_name,
            "content": self.content,
            "section_path": self.section_path,
            "page_number": self.page_number,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "revision": self.revision,
            "effective_date": self.effective_date,
            "document_type": self.document_type,
            "embedding": self.embedding,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DocumentChunk":
        """Deserialize chunk from a dictionary."""
        return cls(
            chunk_id=data["chunk_id"],
            document_id=data["document_id"],
            document_name=data["document_name"],
            content=data["content"],
            section_path=data.get("section_path", "Document Root"),
            page_number=data.get("page_number"),
            start_line=data.get("start_line"),
            end_line=data.get("end_line"),
            revision=data.get("revision"),
            effective_date=data.get("effective_date"),
            document_type=data.get("document_type", "product_guide"),
            embedding=data.get("embedding"),
        )


class ChunkingConfig(BaseModel):
    """Configuration for document chunking."""
    
    chunk_size: int = Field(default=1000, description="Target chunk size in characters")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks")
    respect_sections: bool = Field(default=True, description="Try to chunk at section boundaries")
    min_chunk_size: int = Field(default=100, description="Minimum chunk size")
    
    # Section detection patterns
    section_patterns: list[str] = Field(
        default=[
            r"^#{1,6}\s+",  # Markdown headers
            r"^\d+\.\d*\s+[A-Z]",  # Numbered sections (1. Title, 1.1 Title)
            r"^[A-Z][A-Z\s]+$",  # ALL CAPS headers
            r"^Chapter\s+\d+",  # Chapter headers
        ],
        description="Regex patterns to detect section boundaries"
    )


class ProductGuideMetadata(BaseModel):
    """Metadata for a product guide document."""
    
    document_id: str
    filename: str
    title: str
    revision: str
    effective_date: Optional[str] = None
    
    # Structure extracted from document
    sections: list[str] = Field(default_factory=list)
    
    # Special content
    critical_features: list[str] = Field(default_factory=list)
    critical_parameters: list[str] = Field(default_factory=list)
    known_failure_modes: list[str] = Field(default_factory=list)
    
    # Ingestion info
    ingested_at: datetime = Field(default_factory=datetime.utcnow)
    chunk_count: int = 0
    
    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "pg-hyd-valve-200-r3",
                "filename": "HYD-VALVE-200_Product_Guide_Rev3.pdf",
                "title": "HYD-VALVE-200 Product Guide",
                "revision": "Rev 3",
                "effective_date": "2025-01-15",
                "sections": [
                    "1. Product Overview",
                    "2. Specifications",
                    "3. Sealing System Design",
                ],
                "critical_features": [
                    "O-ring seal interface",
                    "Flow control orifice",
                ],
                "critical_parameters": [
                    "Seal gland depth: 0.125 +/- 0.002 in",
                    "Assembly torque: 35 +/- 3 ft-lb",
                ],
            }
        }


class RAGTool:
    """RAG tool for document retrieval with citation support."""
    
    def __init__(
        self,
        vector_store: Any = None,  # Will be initialized based on config
        embedding_model: Any = None,
        chunking_config: Optional[ChunkingConfig] = None,
        storage_dir: Optional[str | Path] = None,
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model or self._init_embedding_model()
        self.chunking_config = chunking_config or ChunkingConfig()
        self.storage_dir = Path(storage_dir or "./data/rag_store")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory storage for development (replace with actual vector store)
        self._documents: dict[str, ProductGuideMetadata] = {}
        self._chunks: dict[str, DocumentChunk] = {}
        self._load_store()

    def _init_embedding_model(self) -> Optional[Any]:
        """Initialize embedding model if available."""
        try:
            from config.settings import get_settings
            settings = get_settings()
            model_name = getattr(settings, "embedding_model_name", None)
            local_only = getattr(settings, "embedding_local_only", True)
        except Exception:
            model_name = None
            local_only = True

        if not model_name:
            return None

        try:
            from sentence_transformers import SentenceTransformer
        except Exception:
            return None

        try:
            if local_only:
                model_path = Path(model_name).expanduser()
                if not model_path.exists():
                    return None
                return SentenceTransformer(str(model_path))
            return SentenceTransformer(model_name)
        except Exception:
            return None

    def _embed(self, text: str) -> Optional[list[float]]:
        """Generate embedding for text if model is available."""
        if self.embedding_model is None:
            return None

        try:
            embedding = self.embedding_model.encode(text)
        except Exception:
            return None

        try:
            return embedding.tolist()
        except Exception:
            return list(embedding)

    def _tokenize(self, text: str) -> list[str]:
        """Normalize text to a stable token list for lexical retrieval."""
        normalized = re.sub(r"[^a-z0-9\s\-_]", " ", text.lower())
        return [token for token in normalized.split() if len(token) >= 2]

    def _cosine_similarity(
        self,
        lhs: Optional[list[float]],
        rhs: Optional[list[float]],
    ) -> float:
        """Compute cosine similarity without a hard dependency on NumPy."""
        if not lhs or not rhs or len(lhs) != len(rhs):
            return 0.0

        dot = 0.0
        lhs_sq = 0.0
        rhs_sq = 0.0
        for l_val, r_val in zip(lhs, rhs):
            dot += float(l_val) * float(r_val)
            lhs_sq += float(l_val) * float(l_val)
            rhs_sq += float(r_val) * float(r_val)

        denom = (lhs_sq ** 0.5) * (rhs_sq ** 0.5)
        if denom == 0.0:
            return 0.0
        return dot / denom

    def _lexical_score(self, query: str, chunk: DocumentChunk) -> float:
        """Compute a lexical relevance score with token overlap and phrase bonus."""
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return 0.0

        chunk_tokens = self._tokenize(f"{chunk.section_path} {chunk.content}")
        if not chunk_tokens:
            return 0.0

        query_counts = Counter(query_tokens)
        chunk_counts = Counter(chunk_tokens)
        overlap = 0.0
        for token, q_count in query_counts.items():
            overlap += min(float(q_count), float(chunk_counts.get(token, 0)))
        overlap_score = overlap / max(float(len(query_tokens)), 1.0)

        phrase = " ".join(query_tokens)
        phrase_bonus = 0.2 if phrase and phrase in f"{chunk.section_path} {chunk.content}".lower() else 0.0
        return min(1.0, overlap_score + phrase_bonus)

    def _hierarchy_depth(self, section_path: str) -> int:
        return len([part.strip() for part in section_path.split(">") if part.strip()])

    def _apply_hierarchical_expansion(
        self,
        scored_hits: dict[str, dict[str, Any]],
        max_expansion: int = 4,
    ) -> dict[str, dict[str, Any]]:
        """Expand retrieval around high-value chunks using section hierarchy context."""
        if not scored_hits:
            return scored_hits

        base_hits = sorted(
            scored_hits.values(),
            key=lambda item: item["score"],
            reverse=True,
        )[:max_expansion]
        expanded = dict(scored_hits)

        for hit in base_hits:
            chunk: DocumentChunk = hit["chunk"]
            section_path = chunk.section_path or "Document Root"
            section_parts = [part.strip() for part in section_path.split(">") if part.strip()]
            parent_prefixes = [
                " > ".join(section_parts[:idx])
                for idx in range(1, len(section_parts))
            ]

            for candidate in self._chunks.values():
                if candidate.document_id != chunk.document_id:
                    continue
                candidate_section = candidate.section_path or "Document Root"
                if candidate.chunk_id == chunk.chunk_id:
                    continue

                is_parent = candidate_section in parent_prefixes
                is_child = candidate_section.startswith(f"{section_path} >")
                same_parent = False
                if len(section_parts) > 1:
                    parent_path = " > ".join(section_parts[:-1])
                    same_parent = (
                        candidate_section.startswith(f"{parent_path} >")
                        and self._hierarchy_depth(candidate_section) == self._hierarchy_depth(section_path)
                    )

                if not (is_parent or is_child or same_parent):
                    continue

                neighbor_score = hit["score"] * 0.7
                existing = expanded.get(candidate.chunk_id)
                if existing is None or neighbor_score > existing["score"]:
                    expanded[candidate.chunk_id] = {
                        "chunk": candidate,
                        "score": neighbor_score,
                        "semantic_score": existing["semantic_score"] if existing else 0.0,
                        "lexical_score": existing["lexical_score"] if existing else 0.0,
                        "matched_via": "hierarchy_expansion",
                    }

        return expanded

    def _extract_coverage_terms(self, text: str) -> list[str]:
        """Extract candidate coverage terms for follow-up retrieval."""
        tokens = self._tokenize(text)
        if not tokens:
            return []

        stopwords = {
            "the", "and", "for", "with", "from", "that", "this", "are", "was",
            "were", "have", "has", "had", "into", "onto", "over", "under", "mode",
            "case", "part", "failure", "failed", "test", "guide", "section",
        }
        ranked = [token for token in tokens if token not in stopwords and len(token) >= 4]

        deduped: list[str] = []
        seen = set()
        for token in ranked:
            if token in seen:
                continue
            seen.add(token)
            deduped.append(token)
            if len(deduped) >= 10:
                break
        return deduped

    def _run_coverage_gap_pass(
        self,
        query: str,
        scored_hits: dict[str, dict[str, Any]],
        top_k: int,
    ) -> dict[str, dict[str, Any]]:
        """Find likely missing terms and run targeted retrieval for better recall."""
        if not query or not scored_hits:
            return scored_hits

        expected_terms = self._extract_coverage_terms(query)
        if not expected_terms:
            return scored_hits

        covered_text = " ".join(
            f"{item['chunk'].section_path} {item['chunk'].content}"
            for item in sorted(scored_hits.values(), key=lambda x: x["score"], reverse=True)[: max(3, top_k)]
        ).lower()
        missing_terms = [term for term in expected_terms if term not in covered_text][:3]
        if not missing_terms:
            return scored_hits

        expanded = dict(scored_hits)
        for term in missing_terms:
            for chunk in self._chunks.values():
                lexical = self._lexical_score(term, chunk)
                if lexical < 0.25:
                    continue
                boost = min(1.0, lexical * 0.8)
                existing = expanded.get(chunk.chunk_id)
                if existing is None or boost > existing["score"]:
                    expanded[chunk.chunk_id] = {
                        "chunk": chunk,
                        "score": boost,
                        "semantic_score": 0.0,
                        "lexical_score": lexical,
                        "matched_via": f"coverage_gap:{term}",
                    }
        return expanded

    def _store_paths(self) -> tuple[Path, Path]:
        """Return paths for documents and chunks store files."""
        return (
            self.storage_dir / "documents.json",
            self.storage_dir / "chunks.jsonl",
        )

    def clear_store(self) -> None:
        """Clear in-memory and on-disk RAG store."""
        self._documents = {}
        self._chunks = {}
        documents_path, chunks_path = self._store_paths()
        for path in (documents_path, chunks_path):
            if path.exists():
                path.unlink()

    def _load_store(self) -> None:
        """Load persisted documents/chunks into memory if available."""
        documents_path, chunks_path = self._store_paths()
        if documents_path.exists():
            try:
                data = json.loads(documents_path.read_text(encoding="utf-8"))
                for doc in data:
                    meta = ProductGuideMetadata(**doc)
                    self._documents[meta.document_id] = meta
            except Exception:
                pass

        if chunks_path.exists():
            try:
                for line in chunks_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    chunk = DocumentChunk.from_dict(json.loads(line))
                    self._chunks[chunk.chunk_id] = chunk
            except Exception:
                pass

    def _save_store(self) -> None:
        """Persist documents/chunks to disk."""
        documents_path, chunks_path = self._store_paths()
        documents_path.write_text(
            json.dumps(
                [m.model_dump(mode="json") for m in self._documents.values()],
                indent=2,
            ),
            encoding="utf-8",
        )
        with chunks_path.open("w", encoding="utf-8") as f:
            for chunk in self._chunks.values():
                f.write(json.dumps(chunk.to_dict()) + "\n")

    def ingest_directory(
        self,
        directory: str | Path,
        extensions: Optional[list[str]] = None,
        rebuild: bool = False,
    ) -> list[ProductGuideMetadata]:
        """Ingest all documents from a directory."""
        root = Path(directory)
        if not root.exists():
            return []

        if rebuild:
            self.clear_store()

        patterns = extensions or ["*.md", "*.txt", "*.pdf", "*.docx"]
        files: list[Path] = []
        for pattern in patterns:
            files.extend(root.glob(pattern))

        ingested: list[ProductGuideMetadata] = []
        for file_path in sorted(files):
            metadata = self.ingest_document(file_path)
            ingested.append(metadata)

        return ingested
    
    def _detect_sections(self, text: str) -> list[tuple[int, str, str]]:
        """Detect section boundaries in text.
        
        Returns:
            List of (line_number, section_title, section_path)
        """
        lines = text.split('\n')
        sections = []
        current_path = []
        
        for i, line in enumerate(lines):
            for pattern in self.chunking_config.section_patterns:
                if re.match(pattern, line.strip()):
                    # Determine section level (simplified)
                    title = line.strip()
                    
                    # Update path based on header level
                    if line.startswith('#'):
                        level = len(re.match(r'^#+', line).group())
                    elif re.match(r'^\d+\.', line):
                        level = line.count('.') + 1
                    else:
                        level = 1
                    
                    current_path = current_path[:level-1] + [title]
                    section_path = " > ".join(current_path)
                    
                    sections.append((i, title, section_path))
                    break
        
        return sections
    
    def _chunk_text(
        self,
        text: str,
        document_id: str,
        document_name: str,
        revision: Optional[str] = None,
    ) -> list[DocumentChunk]:
        """Chunk text into overlapping segments respecting section boundaries."""
        chunks = []
        
        if self.chunking_config.respect_sections:
            sections = self._detect_sections(text)
        else:
            sections = []
        
        lines = text.split('\n')
        current_chunk = []
        current_section_path = "Document Root"
        current_start_line = 0
        chunk_index = 0
        
        for i, line in enumerate(lines):
            # Check if we hit a section boundary
            section_match = next(
                (s for s in sections if s[0] == i), 
                None
            )
            if section_match:
                # Save current chunk if it has content
                if current_chunk and len('\n'.join(current_chunk)) >= self.chunking_config.min_chunk_size:
                    chunk_content = '\n'.join(current_chunk)
                    chunk_id = f"{document_id}_chunk_{chunk_index}"
                    chunks.append(DocumentChunk(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        document_name=document_name,
                        content=chunk_content,
                        section_path=current_section_path,
                        start_line=current_start_line,
                        end_line=i - 1,
                        revision=revision,
                    ))
                    chunk_index += 1
                
                current_section_path = section_match[2]
                current_chunk = []
                current_start_line = i
            
            current_chunk.append(line)
            
            # Check if chunk is getting too large
            if len('\n'.join(current_chunk)) >= self.chunking_config.chunk_size:
                chunk_content = '\n'.join(current_chunk)
                chunk_id = f"{document_id}_chunk_{chunk_index}"
                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    document_name=document_name,
                    content=chunk_content,
                    section_path=current_section_path,
                    start_line=current_start_line,
                    end_line=i,
                    revision=revision,
                ))
                chunk_index += 1
                
                # Keep overlap for next chunk
                overlap_lines = int(self.chunking_config.chunk_overlap / 50)  # ~50 chars per line
                current_chunk = current_chunk[-overlap_lines:] if overlap_lines > 0 else []
                current_start_line = i - len(current_chunk) + 1
        
        # Don't forget the last chunk
        if current_chunk and len('\n'.join(current_chunk)) >= self.chunking_config.min_chunk_size:
            chunk_content = '\n'.join(current_chunk)
            chunk_id = f"{document_id}_chunk_{chunk_index}"
            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                document_id=document_id,
                document_name=document_name,
                content=chunk_content,
                section_path=current_section_path,
                start_line=current_start_line,
                end_line=len(lines) - 1,
                revision=revision,
            ))
        
        return chunks
    
    def _extract_metadata(self, text: str, filename: str) -> ProductGuideMetadata:
        """Extract metadata from document text."""
        # Simple extraction - in production, use LLM or more sophisticated parsing
        
        # Try to find title
        lines = text.split('\n')[:50]  # Look in first 50 lines
        title = filename
        for line in lines:
            if line.strip() and len(line.strip()) > 10:
                title = line.strip()
                break
        
        # Try to find revision
        revision = "Unknown"
        rev_match = re.search(r'Rev(?:ision)?\.?\s*([A-Z0-9]+)', text[:5000], re.IGNORECASE)
        if rev_match:
            revision = f"Rev {rev_match.group(1)}"
        
        # Extract sections
        sections = self._detect_sections(text)
        section_titles = [s[1] for s in sections]
        
        # Try to find critical parameters (simplified)
        critical_params = []
        param_patterns = [
            r'(\w+(?:\s+\w+)*)\s*[:=]\s*([\d.]+\s*[+-]\s*[\d.]+\s*\w+)',  # param: value +/- tol unit
            r'(Critical|Key|Important).*?[:]\s*(.+?)(?:\n|$)',  # Critical: something
        ]
        for pattern in param_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches[:10]:  # Limit to 10
                if isinstance(match, tuple):
                    critical_params.append(f"{match[0]}: {match[1]}")
                else:
                    critical_params.append(match)
        
        document_id = hashlib.md5(f"{filename}_{revision}".encode()).hexdigest()[:12]
        
        return ProductGuideMetadata(
            document_id=document_id,
            filename=filename,
            title=title,
            revision=revision,
            sections=section_titles[:20],  # Limit to 20 sections
            critical_parameters=critical_params[:10],
        )
    
    def ingest_document(
        self,
        file_path: str | Path,
        document_type: str = "product_guide",
    ) -> ProductGuideMetadata:
        """Ingest a document into the RAG system.
        
        Args:
            file_path: Path to the document file
            document_type: Type of document (product_guide, procedure, specification)
            
        Returns:
            ProductGuideMetadata with ingestion details
        """
        file_path = Path(file_path)
        
        # Read document
        suffix = file_path.suffix.lower()
        if suffix == ".txt":
            text = file_path.read_text(encoding="utf-8")
        elif suffix == ".md":
            text = file_path.read_text(encoding="utf-8")
        elif suffix == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(str(file_path))
            pages = [page.extract_text() or "" for page in reader.pages]
            text = "\n".join(pages)
        elif suffix == ".docx":
            import docx
            doc = docx.Document(str(file_path))
            text = "\n".join(paragraph.text for paragraph in doc.paragraphs)
        else:
            raise NotImplementedError(
                f"File type {file_path.suffix} not supported. "
                "Use .txt, .md, .pdf, or .docx."
            )
        
        # Extract metadata
        metadata = self._extract_metadata(text, file_path.name)
        
        # Chunk the document
        chunks = self._chunk_text(
            text,
            document_id=metadata.document_id,
            document_name=metadata.filename,
            revision=metadata.revision,
        )
        
        # Store chunks (in production, this would go to vector store)
        for chunk in chunks:
            chunk.document_type = document_type
            self._chunks[chunk.chunk_id] = chunk
            
            # Generate embeddings if model available
            chunk.embedding = self._embed(chunk.content)
        
        metadata.chunk_count = len(chunks)
        self._documents[metadata.document_id] = metadata
        self._save_store()
        
        return metadata
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        document_filter: Optional[list[str]] = None,
        section_filter: Optional[list[str]] = None,
        min_score: float = 0.0,
        semantic_weight: float = 0.65,
        lexical_weight: float = 0.35,
        enable_hierarchy_expansion: bool = True,
        enable_coverage_gap_pass: bool = True,
    ) -> list[tuple[DocumentChunk, float]]:
        """Retrieve relevant chunks for a query.
        
        Args:
            query: Search query
            top_k: Number of results to return
            document_filter: Only search these document IDs
            section_filter: Only search sections containing these strings
            min_score: Minimum relevance score
            
        Returns:
            List of (chunk, score) tuples sorted by relevance
        """
        if not query.strip():
            return []

        # Prefer embedding similarity if available
        query_embedding = self._embed(query)
        use_embeddings = query_embedding is not None

        scored_hits: dict[str, dict[str, Any]] = {}
        for chunk in self._chunks.values():
            if document_filter and chunk.document_id not in document_filter:
                continue
            if section_filter:
                if not any(sf.lower() in chunk.section_path.lower() for sf in section_filter):
                    continue

            semantic_score = self._cosine_similarity(query_embedding, chunk.embedding) if use_embeddings else 0.0
            lexical_score = self._lexical_score(query, chunk)

            if use_embeddings:
                score = (semantic_weight * semantic_score) + (lexical_weight * lexical_score)
            else:
                score = lexical_score

            if score >= min_score:
                scored_hits[chunk.chunk_id] = {
                    "chunk": chunk,
                    "score": score,
                    "semantic_score": semantic_score,
                    "lexical_score": lexical_score,
                    "matched_via": "hybrid",
                }

        if enable_hierarchy_expansion:
            scored_hits = self._apply_hierarchical_expansion(scored_hits)
        if enable_coverage_gap_pass:
            scored_hits = self._run_coverage_gap_pass(query, scored_hits, top_k)

        ranked = sorted(scored_hits.values(), key=lambda x: x["score"], reverse=True)[:top_k]
        return [(item["chunk"], float(item["score"])) for item in ranked]
    
    def retrieve_with_citations(
        self,
        query: str,
        top_k: int = 5,
        **kwargs
    ) -> list[dict]:
        """Retrieve chunks and format as citations.
        
        Returns:
            List of citation dictionaries ready for use in Evidence
        """
        results = self.retrieve(query, top_k, **kwargs)
        
        citations = []
        for chunk, score in results:
            citation = chunk.to_citation_dict()
            citation["retrieval_score"] = score
            citation["retrieval_strategy"] = "hybrid_recursive"
            citations.append(citation)
        
        return citations
    
    def get_sections_for_failure_type(
        self,
        failure_type: str,
        recipe_sections: list[str],
    ) -> list[dict]:
        """Get product guide sections relevant to a failure type.
        
        Uses the recipe's relevant_guide_sections to find matching content.
        """
        all_citations = []
        
        for section_name in recipe_sections:
            results = self.retrieve(
                query=section_name,
                top_k=3,
                section_filter=[section_name],
            )
            for chunk, score in results:
                citation = chunk.to_citation_dict()
                citation["retrieval_score"] = score
                citation["matched_section"] = section_name
                all_citations.append(citation)
        
        return all_citations
    
    def get_document_metadata(self, document_id: str) -> Optional[ProductGuideMetadata]:
        """Get metadata for a specific document."""
        return self._documents.get(document_id)
    
    def list_documents(self) -> list[ProductGuideMetadata]:
        """List all ingested documents."""
        return list(self._documents.values())
