"""Data catalog for dynamic field selection in RCA workflows."""

from __future__ import annotations

from enum import Enum
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Optional
import json

from pydantic import BaseModel, Field


class DataCategory(str, Enum):
    """Categories of data available from the internal API."""

    TEST_DATA = "test_data"
    ROA = "record_of_assembly"
    OPERATOR_BUYOFF = "operator_buyoff"
    COMPONENT_LOT = "component_lot"
    PROCESS_PARAMETER = "process_parameter"


class DataFieldType(str, Enum):
    """Types of data fields."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    IDENTIFIER = "identifier"


class CatalogField(BaseModel):
    """Definition of an available field in the master data catalog."""

    field_id: str = Field(..., description="Unique identifier for the field")
    display_name: str = Field(..., description="Human-readable name")
    category: DataCategory = Field(..., description="Data category")
    field_type: DataFieldType = Field(..., description="Type of data")
    description: str = Field(default="", description="Description of the field")
    unit: Optional[str] = Field(None, description="Unit of measurement if applicable")
    tags: list[str] = Field(
        default_factory=list,
        description="Search tags for semantic retrieval",
    )
    applicable_part_families: list[str] = Field(
        default_factory=lambda: ["*"],
        description="Part family wildcard filters this field applies to",
    )
    embedding: Optional[list[float]] = Field(
        default=None,
        description="Cached embedding for similarity search",
    )

    def as_search_text(self) -> str:
        """Build a compact text representation used for retrieval."""
        tags_text = ", ".join(self.tags) if self.tags else "none"
        part_families = ", ".join(self.applicable_part_families)
        return (
            f"field_id={self.field_id}; name={self.display_name}; "
            f"category={self.category.value}; type={self.field_type.value}; "
            f"description={self.description}; unit={self.unit or 'none'}; "
            f"tags={tags_text}; part_families={part_families}"
        )


class DataCatalog:
    """Master catalog of available fields with semantic retrieval support."""

    FILE_MAP: dict[DataCategory, str] = {
        DataCategory.TEST_DATA: "test_ids.json",
        DataCategory.ROA: "roa_parameters.json",
        DataCategory.OPERATOR_BUYOFF: "operator_buyoffs.json",
        DataCategory.PROCESS_PARAMETER: "process_parameters.json",
    }

    def __init__(
        self,
        catalog_dir: str | Path = "./data/catalog",
        db_url: Optional[str] = None,
        embedding_model: Any = None,
        auto_load: bool = True,
    ):
        self.catalog_dir = Path(catalog_dir)
        self.db_url = db_url
        self.embedding_model = embedding_model or self._init_embedding_model()
        self._fields: dict[str, CatalogField] = {}
        self._query_embedding_cache: dict[str, list[float]] = {}
        if auto_load:
            self.load()

    def _init_embedding_model(self) -> Optional[Any]:
        """Initialize embedding model from settings, if available."""
        try:
            from config.settings import get_settings

            settings = get_settings()
            model_name = settings.embedding_model_name
            local_only = settings.embedding_local_only
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
            vector = self.embedding_model.encode(text)
        except Exception:
            return None
        try:
            return vector.tolist()
        except Exception:
            return list(vector)

    def load(self) -> None:
        """Load catalog entries from DB and/or catalog files."""
        self._fields = {}

        db_fields = self._load_from_db()
        for field in db_fields:
            self._fields[field.field_id] = field

        file_fields = self._load_from_files()
        for field in file_fields:
            self._fields[field.field_id] = field

        # Precompute embeddings for fields once at load time.
        for field in self._fields.values():
            if field.embedding is None:
                field.embedding = self._embed(field.as_search_text())

    def _load_from_db(self) -> list[CatalogField]:
        """Load fields from DB source (optional).

        This is intentionally a lightweight hook; production teams can plug in their
        own DB query implementation without changing retrieval behavior.
        """
        if not self.db_url:
            return []
        # DB loader is optional and intentionally non-fatal until production wiring.
        return []

    def _load_from_files(self) -> list[CatalogField]:
        """Load fields from category JSON files."""
        fields: list[CatalogField] = []
        if not self.catalog_dir.exists():
            return fields

        for category, file_name in self.FILE_MAP.items():
            file_path = self.catalog_dir / file_name
            if not file_path.exists():
                continue
            raw_items = json.loads(file_path.read_text(encoding="utf-8"))
            if not isinstance(raw_items, list):
                continue
            for item in raw_items:
                payload = dict(item)
                payload.setdefault("category", category.value)
                fields.append(CatalogField(**payload))
        return fields

    def list_fields(self, category: Optional[DataCategory] = None) -> list[CatalogField]:
        """List catalog fields, optionally filtered by category."""
        values = list(self._fields.values())
        if category is None:
            return values
        return [field for field in values if field.category == category]

    def get_field(self, field_id: str) -> Optional[CatalogField]:
        """Return a field by id, if present."""
        return self._fields.get(field_id)

    def has_field(self, field_id: str) -> bool:
        """Check if a field exists in the catalog."""
        return field_id in self._fields

    def _matches_part_family(self, field: CatalogField, part_family: Optional[str]) -> bool:
        """Filter by applicable part families if a part family is provided."""
        if not part_family:
            return True
        patterns = field.applicable_part_families or ["*"]
        return any(fnmatch(part_family, pattern) for pattern in patterns)

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity with minimal dependencies."""
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = 0.0
        norm_a = 0.0
        norm_b = 0.0
        for i in range(len(a)):
            dot += a[i] * b[i]
            norm_a += a[i] * a[i]
            norm_b += b[i] * b[i]
        if norm_a <= 0 or norm_b <= 0:
            return 0.0
        return dot / ((norm_a ** 0.5) * (norm_b ** 0.5))

    def _keyword_score(self, query: str, field: CatalogField) -> float:
        """Fallback lexical score when embeddings are unavailable."""
        query_terms = set(query.lower().split())
        field_terms = set(field.as_search_text().lower().split())
        if not query_terms:
            return 0.0
        overlap = len(query_terms & field_terms)
        return overlap / len(query_terms)

    def search_fields(
        self,
        query: str,
        part_family: Optional[str] = None,
        top_k: int = 50,
        categories: Optional[list[DataCategory]] = None,
    ) -> list[CatalogField]:
        """Search catalog fields by semantic similarity with optional filters."""
        if not query.strip():
            return []

        allowed_categories = set(categories) if categories else None
        pool = [
            field
            for field in self._fields.values()
            if self._matches_part_family(field, part_family)
            and (allowed_categories is None or field.category in allowed_categories)
        ]

        if not pool:
            return []

        query_vector = self._query_embedding_cache.get(query)
        if query_vector is None:
            query_vector = self._embed(query)
            if query_vector is not None:
                self._query_embedding_cache[query] = query_vector

        scored: list[tuple[CatalogField, float]] = []
        for field in pool:
            if query_vector is not None and field.embedding:
                score = self._cosine_similarity(query_vector, field.embedding)
            else:
                score = self._keyword_score(query, field)
            scored.append((field, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return [item[0] for item in scored[:top_k]]

    def format_for_prompt(self, fields: list[CatalogField]) -> str:
        """Format candidate fields for LLM prompts."""
        grouped: dict[DataCategory, list[CatalogField]] = {
            DataCategory.TEST_DATA: [],
            DataCategory.ROA: [],
            DataCategory.OPERATOR_BUYOFF: [],
            DataCategory.PROCESS_PARAMETER: [],
        }
        for field in fields:
            if field.category in grouped:
                grouped[field.category].append(field)

        lines: list[str] = []
        for category, items in grouped.items():
            if not items:
                continue
            lines.append(f"{category.value}:")
            for field in items:
                line = (
                    f"- {field.field_id}: {field.display_name}; "
                    f"type={field.field_type.value}; unit={field.unit or 'none'}; "
                    f"description={field.description}"
                )
                lines.append(line)
            lines.append("")
        return "\n".join(lines).strip()

    def get_field_ids_by_category(self) -> dict[str, list[str]]:
        """Return current field ids grouped by category."""
        return {
            "test_ids": [
                f.field_id
                for f in self.list_fields(DataCategory.TEST_DATA)
            ],
            "roa_parameters": [
                f.field_id
                for f in self.list_fields(DataCategory.ROA)
            ],
            "operator_buyoffs": [
                f.field_id
                for f in self.list_fields(DataCategory.OPERATOR_BUYOFF)
            ],
            "process_parameters": [
                f.field_id
                for f in self.list_fields(DataCategory.PROCESS_PARAMETER)
            ],
        }
