"""Evidence and citation schemas."""

from datetime import datetime
from enum import Enum
from typing import Optional, Any
from pydantic import BaseModel, Field, field_validator


class EvidenceSource(str, Enum):
    """Type of evidence source."""
    
    PRODUCT_GUIDE = "product_guide"
    HISTORICAL_CASE = "historical_case"
    TEST_DATA = "test_data"
    PROCESS_PARAMETER = "process_parameter"
    INSPECTION_RECORD = "inspection_record"
    SUPPLIER_DATA = "supplier_data"
    SPC_DATA = "spc_data"
    MAINTENANCE_LOG = "maintenance_log"
    CALIBRATION_RECORD = "calibration_record"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    SQL_QUERY = "sql_query"


class Citation(BaseModel):
    """A citation to a specific source."""
    
    source_type: EvidenceSource = Field(..., description="Type of source")
    source_id: str = Field(..., description="Unique identifier of the source record")
    source_name: str = Field(..., description="Human-readable name/title")
    
    # Location within source
    section_path: Optional[str] = Field(None, description="Section/chapter path for documents")
    page_number: Optional[int] = Field(None, description="Page number if applicable")
    line_range: Optional[tuple[int, int]] = Field(None, description="Line range (start, end)")
    
    # Content
    excerpt: str = Field(..., description="Relevant excerpt from the source")
    excerpt_hash: Optional[str] = Field(None, description="Hash of excerpt for verification")
    
    # Metadata
    timestamp: Optional[datetime] = Field(None, description="Timestamp of the source data")
    revision: Optional[str] = Field(None, description="Document revision if applicable")
    
    # Retrieval info
    retrieval_score: Optional[float] = Field(None, description="RAG retrieval confidence score")

    @field_validator("source_type", mode="before")
    @classmethod
    def _normalize_source_type(cls, value: Any) -> Any:
        """Normalize legacy/non-schema source types before enum validation."""
        if isinstance(value, EvidenceSource):
            return value
        source_type = str(value or "").strip().lower()
        source_type_map = {
            "sql_query": EvidenceSource.HISTORICAL_CASE.value,
            "sql": EvidenceSource.HISTORICAL_CASE.value,
            "database": EvidenceSource.HISTORICAL_CASE.value,
            "internal_data_api": EvidenceSource.TEST_DATA.value,
            "rag": EvidenceSource.PRODUCT_GUIDE.value,
            "spc": EvidenceSource.SPC_DATA.value,
        }
        return source_type_map.get(source_type, source_type)


class Evidence(BaseModel):
    """A piece of evidence supporting or refuting a hypothesis."""
    
    evidence_id: str = Field(..., description="Unique identifier")
    
    # What this evidence shows
    claim: str = Field(..., description="What this evidence demonstrates")
    supports_hypothesis: bool = Field(..., description="True if supports, False if refutes")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in this evidence (0-1)")
    
    # Source and citation
    citations: list[Citation] = Field(..., min_length=1, description="Citations backing this evidence")
    
    # Statistical results if applicable
    statistical_result: Optional[dict[str, Any]] = Field(
        None, 
        description="Structured statistical test results"
    )
    
    # Artifact references
    chart_path: Optional[str] = Field(None, description="Path to generated chart image")
    data_summary_path: Optional[str] = Field(None, description="Path to data summary CSV")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    agent_name: str = Field(..., description="Agent that produced this evidence")
