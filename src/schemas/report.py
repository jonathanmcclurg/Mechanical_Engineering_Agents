"""Report schemas."""

from datetime import datetime
from enum import Enum
from typing import Optional, Any
from pydantic import BaseModel, Field

from .hypothesis import Hypothesis
from .evidence import Citation


class ReportSection(str, Enum):
    """Sections of an RCA report."""
    
    EXECUTIVE_SUMMARY = "executive_summary"
    FAILURE_DESCRIPTION = "failure_description"
    INVESTIGATION_SCOPE = "investigation_scope"
    DATA_SOURCES = "data_sources"
    HYPOTHESES = "hypotheses"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    EVIDENCE_SUMMARY = "evidence_summary"
    CONCLUSIONS = "conclusions"
    RECOMMENDED_ACTIONS = "recommended_actions"
    APPENDIX = "appendix"


class SectionContent(BaseModel):
    """Content for a report section."""
    
    section: ReportSection
    title: str
    content: str = Field(..., description="Markdown-formatted content")
    citations: list[Citation] = Field(default_factory=list)
    tables: list[dict[str, Any]] = Field(default_factory=list, description="Structured tables")
    chart_paths: list[str] = Field(default_factory=list, description="Paths to chart images")


class RCAReport(BaseModel):
    """A complete Root Cause Analysis report."""
    
    report_id: str = Field(..., description="Unique identifier")
    case_id: str = Field(..., description="Case this report is for")
    
    # Header info
    title: str = Field(..., description="Report title")
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(default="1.0")
    
    # Content sections
    sections: list[SectionContent] = Field(..., description="Report sections in order")
    
    # Structured outputs
    ranked_hypotheses: list[Hypothesis] = Field(
        ..., 
        description="Hypotheses ranked by posterior confidence"
    )
    top_hypothesis: Optional[Hypothesis] = Field(
        None, 
        description="Most likely root cause"
    )
    
    # All citations for easy reference
    all_citations: list[Citation] = Field(
        default_factory=list, 
        description="All citations used in report"
    )
    
    # Recommended actions
    immediate_actions: list[str] = Field(
        default_factory=list, 
        description="Actions to take immediately"
    )
    investigation_actions: list[str] = Field(
        default_factory=list, 
        description="Additional investigations recommended"
    )
    preventive_actions: list[str] = Field(
        default_factory=list, 
        description="Long-term preventive measures"
    )
    
    # Confidence and caveats
    overall_confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Overall confidence in conclusions"
    )
    caveats: list[str] = Field(
        default_factory=list, 
        description="Limitations and caveats"
    )
    data_gaps: list[str] = Field(
        default_factory=list, 
        description="Data that was missing or insufficient"
    )
    
    # Audit trail
    agents_involved: list[str] = Field(..., description="Agents that contributed")
    processing_time_seconds: float = Field(..., description="Total processing time")
    
    # Status
    is_draft: bool = Field(default=True)
    reviewed_by: Optional[str] = Field(None, description="Engineer who reviewed")
    reviewed_at: Optional[datetime] = Field(None)
