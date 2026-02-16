"""Base agent class with common functionality."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional
import json
from pydantic import BaseModel, Field


class AgentOutput(BaseModel):
    """Standard output format for all agents."""
    
    agent_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    success: bool = True
    error_message: Optional[str] = None
    
    # Outputs will vary by agent
    data: dict[str, Any] = Field(default_factory=dict)
    
    # Citations for audit trail
    citations_used: list[dict] = Field(default_factory=list)
    
    # For the learning loop
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    reasoning: Optional[str] = None


class BaseRCAAgent(ABC):
    """Base class for all RCA agents."""
    
    name: str = "BaseAgent"
    description: str = "Base agent class"
    
    # System prompt template
    system_prompt: str = ""
    
    def __init__(
        self,
        llm: Any = None,
        tools: dict[str, Any] = None,
        verbose: bool = False,
    ):
        self.llm = llm
        self.tools = tools or {}
        self.verbose = verbose
        self._execution_log: list[dict] = []
    
    @abstractmethod
    def execute(self, context: dict[str, Any]) -> AgentOutput:
        """Execute the agent's task.
        
        Args:
            context: Dictionary containing all relevant context
                    (case info, previous agent outputs, etc.)
        
        Returns:
            AgentOutput with results and citations
        """
        pass
    
    def log(self, message: str, level: str = "info") -> None:
        """Log a message."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "agent": self.name,
            "level": level,
            "message": message,
        }
        self._execution_log.append(entry)
        if self.verbose:
            print(f"[{self.name}] {message}")
    
    def get_execution_log(self) -> list[dict]:
        """Get the execution log for this agent."""
        return self._execution_log.copy()

    def reset_execution_log(self) -> None:
        """Clear the execution log for a new run."""
        self._execution_log = []
    
    def _format_prompt(self, template: str, **kwargs) -> str:
        """Format a prompt template with context."""
        return template.format(**kwargs)
    
    def _call_llm(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Call the LLM with a prompt."""
        if self.llm is None:
            self.log("LLM not configured, returning mock response", "warning")
            return "[Mock LLM Response - Configure LLM for real responses]"
        
        try:
            return self.llm.invoke(prompt, system_prompt=system_prompt or self.system_prompt)
        except Exception as e:
            self.log(f"LLM call failed: {e}", "error")
            raise

    def _llm_analysis_or_fallback(
        self,
        *,
        objective: str,
        context_payload: dict[str, Any],
        fallback_reasoning: str,
    ) -> tuple[str, bool]:
        """Generate agent reasoning with LLM when available, fallback otherwise."""
        if self.llm is None:
            return fallback_reasoning, False

        try:
            payload_text = json.dumps(
                context_payload,
                default=str,
                ensure_ascii=True,
            )[:5000]
            prompt = (
                f"Objective: {objective}\n"
                "You are writing a concise RCA agent analysis note.\n"
                "Return 2-4 sentences focused on findings and confidence drivers.\n"
                "Do not include markdown headings.\n\n"
                f"Context:\n{payload_text}"
            )
            llm_text = str(self._call_llm(prompt)).strip()
            if not llm_text or llm_text.startswith("[Mock LLM Response"):
                return fallback_reasoning, False
            return llm_text, True
        except Exception:
            # Keep deterministic behavior when LLM is unavailable or errors.
            return fallback_reasoning, False
