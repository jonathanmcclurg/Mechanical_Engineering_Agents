"""Hypothesis schemas."""

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

from .evidence import Evidence


class HypothesisOrigin(str, Enum):
    """How the hypothesis was derived."""

    HISTORICAL_PRECEDENT = "historical_precedent"
    UNIT_SPECIFIC_ANOMALY = "unit_specific_anomaly"
    TECHNICAL_DOCUMENTATION = "technical_documentation"
    ENGINEERING_REASONING = "engineering_reasoning"
    RECIPE_TEMPLATE = "recipe_template"


class HypothesisStatus(str, Enum):
    """Status of a hypothesis."""
    
    PROPOSED = "proposed"
    TESTING = "testing"
    SUPPORTED = "supported"
    REFUTED = "refuted"
    INCONCLUSIVE = "inconclusive"


class StatisticalTest(BaseModel):
    """A recommended statistical test for a hypothesis."""
    
    test_type: str = Field(..., description="Type of test (xbar_s_chart, two_sample_ttest, etc.)")
    target_variable: str = Field(..., description="Variable/measurement to analyze")
    grouping_key: Optional[str] = Field(None, description="Column to group by (e.g., component_lot)")
    comparison_groups: Optional[list[str]] = Field(None, description="Groups to compare")
    
    # Parameters
    alpha: float = Field(default=0.05, description="Significance level")
    practical_threshold: Optional[float] = Field(
        None, 
        description="Threshold for practical significance"
    )
    baseline_window: Optional[str] = Field(
        None, 
        description="Time window for baseline data (e.g., '30d')"
    )
    
    # Expected outcome
    expected_signal: str = Field(
        ..., 
        description="What signal would support the hypothesis"
    )


class Hypothesis(BaseModel):
    """A root cause hypothesis to test."""
    
    hypothesis_id: str = Field(..., description="Unique identifier")
    case_id: str = Field(..., description="Case this hypothesis belongs to")
    
    # The hypothesis itself
    title: str = Field(..., description="Brief title of the hypothesis")
    description: str = Field(..., description="Detailed description of the proposed root cause")
    mechanism: str = Field(..., description="Proposed mechanism by which this causes the failure")
    
    # Testing plan
    expected_signatures: list[str] = Field(
        ..., 
        description="Data signatures we'd expect to see if this is correct"
    )
    recommended_tests: list[StatisticalTest] = Field(
        ..., 
        description="Statistical tests to run"
    )
    required_data_sources: list[str] = Field(
        ..., 
        description="Data sources needed to test this hypothesis"
    )
    
    # Origin
    origin: HypothesisOrigin = Field(
        default=HypothesisOrigin.RECIPE_TEMPLATE,
        description="How this hypothesis was derived",
    )
    evidence_basis: Optional[str] = Field(
        None,
        description="Which specific evidence items motivated this hypothesis",
    )

    # Results
    status: HypothesisStatus = Field(default=HypothesisStatus.PROPOSED)
    evidence_for: list[Evidence] = Field(default_factory=list)
    evidence_against: list[Evidence] = Field(default_factory=list)
    
    # Ranking
    prior_confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Initial confidence before testing (0-1)"
    )
    posterior_confidence: Optional[float] = Field(
        None, 
        ge=0.0, 
        le=1.0, 
        description="Confidence after testing (0-1)"
    )
    rank: Optional[int] = Field(None, description="Rank among hypotheses (1 = most likely)")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    tested_at: Optional[datetime] = Field(None)
    
    # Feedback
    engineer_verdict: Optional[str] = Field(
        None, 
        description="Engineer's final determination (correct/incorrect/partial)"
    )
    engineer_notes: Optional[str] = Field(None, description="Engineer's notes")
