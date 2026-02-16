"""Workflow management for the RCA system.

This module provides workflow state management, persistence,
and recovery capabilities.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional
import json
import uuid

from pydantic import BaseModel, Field


class WorkflowState(str, Enum):
    """Possible states of an RCA workflow."""
    
    PENDING = "pending"
    INTAKE = "intake"
    RESEARCH = "research"
    HYPOTHESIS = "hypothesis"
    TESTING = "testing"
    REVIEW = "review"
    REPORT = "report"
    COMPLETED = "completed"
    FAILED = "failed"
    AWAITING_FEEDBACK = "awaiting_feedback"


class WorkflowCheckpoint(BaseModel):
    """A checkpoint in the workflow for recovery."""
    
    checkpoint_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    workflow_id: str
    state: WorkflowState
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Context at this checkpoint
    context: dict[str, Any] = Field(default_factory=dict)
    
    # Outputs from completed steps
    completed_steps: list[str] = Field(default_factory=list)
    outputs: dict[str, Any] = Field(default_factory=dict)
    
    # Error info if failed
    error: Optional[str] = None


class RCAWorkflow:
    """Manages the state and lifecycle of an RCA workflow instance."""
    
    def __init__(self, workflow_id: Optional[str] = None):
        self.workflow_id = workflow_id or f"WF-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.state = WorkflowState.PENDING
        self.checkpoints: list[WorkflowCheckpoint] = []
        self.context: dict[str, Any] = {}
        self.outputs: dict[str, Any] = {}
        self.completed_steps: list[str] = []
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def transition(self, new_state: WorkflowState) -> None:
        """Transition to a new state and create a checkpoint."""
        self.state = new_state
        self.updated_at = datetime.utcnow()
        self._create_checkpoint()
    
    def _create_checkpoint(self) -> WorkflowCheckpoint:
        """Create a checkpoint at the current state."""
        checkpoint = WorkflowCheckpoint(
            workflow_id=self.workflow_id,
            state=self.state,
            context=self.context.copy(),
            completed_steps=self.completed_steps.copy(),
            outputs=self.outputs.copy(),
        )
        self.checkpoints.append(checkpoint)
        return checkpoint
    
    def complete_step(self, step_name: str, output: Any) -> None:
        """Mark a step as complete and store its output."""
        self.completed_steps.append(step_name)
        self.outputs[step_name] = output
        self.updated_at = datetime.utcnow()
    
    def update_context(self, updates: dict[str, Any]) -> None:
        """Update the workflow context."""
        self.context.update(updates)
        self.updated_at = datetime.utcnow()
    
    def fail(self, error: str) -> None:
        """Mark the workflow as failed."""
        self.state = WorkflowState.FAILED
        self.updated_at = datetime.utcnow()
        
        # Create checkpoint with error
        checkpoint = WorkflowCheckpoint(
            workflow_id=self.workflow_id,
            state=self.state,
            context=self.context.copy(),
            completed_steps=self.completed_steps.copy(),
            outputs=self.outputs.copy(),
            error=error,
        )
        self.checkpoints.append(checkpoint)
    
    def recover_from_checkpoint(self, checkpoint_id: str) -> bool:
        """Recover workflow state from a checkpoint.
        
        Args:
            checkpoint_id: ID of checkpoint to recover from
            
        Returns:
            True if recovery successful, False otherwise
        """
        checkpoint = next(
            (c for c in self.checkpoints if c.checkpoint_id == checkpoint_id),
            None
        )
        
        if checkpoint is None:
            return False
        
        self.state = checkpoint.state
        self.context = checkpoint.context.copy()
        self.completed_steps = checkpoint.completed_steps.copy()
        self.outputs = checkpoint.outputs.copy()
        self.updated_at = datetime.utcnow()
        
        return True
    
    def get_next_step(self) -> Optional[str]:
        """Get the next step to execute based on current state."""
        step_map = {
            WorkflowState.PENDING: "intake",
            WorkflowState.INTAKE: "product_guide",
            WorkflowState.RESEARCH: "hypothesis",
            WorkflowState.HYPOTHESIS: "test_plan",
            WorkflowState.TESTING: "stats",
            WorkflowState.REVIEW: "report",
        }
        return step_map.get(self.state)
    
    def to_dict(self) -> dict:
        """Serialize workflow to dictionary."""
        return {
            "workflow_id": self.workflow_id,
            "state": self.state.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "completed_steps": self.completed_steps,
            "checkpoints": [
                {
                    "checkpoint_id": c.checkpoint_id,
                    "state": c.state.value,
                    "timestamp": c.timestamp.isoformat(),
                    "completed_steps": c.completed_steps,
                    "error": c.error,
                }
                for c in self.checkpoints
            ],
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "RCAWorkflow":
        """Deserialize workflow from dictionary."""
        workflow = cls(workflow_id=data["workflow_id"])
        workflow.state = WorkflowState(data["state"])
        workflow.created_at = datetime.fromisoformat(data["created_at"])
        workflow.updated_at = datetime.fromisoformat(data["updated_at"])
        workflow.completed_steps = data["completed_steps"]
        return workflow
