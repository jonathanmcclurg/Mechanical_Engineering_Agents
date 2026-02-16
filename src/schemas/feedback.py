"""Feedback schemas for the learning loop."""

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class FeedbackOutcome(str, Enum):
    """Outcome of a hypothesis."""
    
    CORRECT = "correct"
    INCORRECT = "incorrect"
    PARTIALLY_CORRECT = "partially_correct"
    UNKNOWN = "unknown"


class HypothesisFeedback(BaseModel):
    """Feedback on a specific hypothesis."""
    
    hypothesis_id: str
    outcome: FeedbackOutcome
    was_actual_root_cause: bool = Field(
        default=False, 
        description="True if this was the confirmed root cause"
    )
    confidence_appropriate: Optional[bool] = Field(
        None, 
        description="Was the confidence level appropriate?"
    )
    notes: Optional[str] = Field(None, description="Engineer notes on this hypothesis")


class StatisticalTestFeedback(BaseModel):
    """Feedback on statistical test appropriateness."""
    
    test_type: str
    was_appropriate: bool = Field(..., description="Was this the right test to run?")
    better_alternative: Optional[str] = Field(
        None, 
        description="What test should have been used instead?"
    )
    notes: Optional[str] = None


class CitationFeedback(BaseModel):
    """Feedback on citation quality."""
    
    citation_id: str
    was_relevant: bool
    was_accurate: bool
    notes: Optional[str] = None


class EngineerFeedback(BaseModel):
    """Complete feedback from an engineer on an RCA report."""
    
    feedback_id: str = Field(..., description="Unique identifier")
    report_id: str = Field(..., description="Report being reviewed")
    case_id: str = Field(..., description="Case ID")
    
    # Reviewer info
    engineer_id: str = Field(..., description="Engineer providing feedback")
    submitted_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Overall assessment
    report_useful: bool = Field(..., description="Was the report useful overall?")
    time_saved_minutes: Optional[int] = Field(
        None, 
        description="Estimated time saved vs manual investigation"
    )
    
    # Hypothesis feedback
    hypothesis_feedback: list[HypothesisFeedback] = Field(
        ..., 
        description="Feedback on each hypothesis"
    )
    actual_root_cause: Optional[str] = Field(
        None, 
        description="What was the actual root cause if different from hypotheses?"
    )
    root_cause_was_in_top_3: bool = Field(
        default=False, 
        description="Was the actual root cause in the top 3 hypotheses?"
    )
    
    # Statistical analysis feedback
    stats_feedback: list[StatisticalTestFeedback] = Field(
        default_factory=list,
        description="Feedback on statistical tests"
    )
    stats_were_helpful: Optional[bool] = Field(
        None, 
        description="Were the statistical analyses helpful?"
    )
    
    # Citation/evidence feedback
    citation_feedback: list[CitationFeedback] = Field(
        default_factory=list
    )
    important_evidence_missing: Optional[str] = Field(
        None, 
        description="What evidence should have been included?"
    )
    
    # Product guide feedback
    product_guide_sections_helpful: list[str] = Field(
        default_factory=list,
        description="Which product guide sections were most helpful?"
    )
    product_guide_sections_missing: list[str] = Field(
        default_factory=list,
        description="Which product guide sections should have been included?"
    )
    
    # Recommended actions feedback
    actions_appropriate: bool = Field(
        default=True, 
        description="Were recommended actions appropriate?"
    )
    actions_taken: list[str] = Field(
        default_factory=list, 
        description="Which recommended actions were actually taken?"
    )
    
    # Free-form feedback
    what_worked_well: Optional[str] = Field(None)
    what_to_improve: Optional[str] = Field(None)
    additional_notes: Optional[str] = Field(None)
    
    # For learning loop
    should_update_recipe: bool = Field(
        default=False, 
        description="Should this feedback trigger recipe update review?"
    )
    suggested_recipe_changes: Optional[str] = Field(None)
