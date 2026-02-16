"""LLM adapter for multiple providers (OpenRouter, local, etc.)."""

from typing import Optional
from abc import ABC, abstractmethod
from time import perf_counter
from langchain_openai import ChatOpenAI
from config.settings import get_settings


class LLMAdapter(ABC):
    """Abstract base class for LLM adapters."""

    def __init__(self):
        self._call_count = 0
        self._total_latency_ms = 0.0
        self._last_latency_ms = 0.0
        self._last_error: Optional[str] = None
    
    @abstractmethod
    def invoke(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Invoke the LLM with a prompt."""
        pass

    def _record_call(self, latency_ms: float, error: Optional[Exception] = None) -> None:
        """Track usage metrics for observability/debugging."""
        self._call_count += 1
        self._last_latency_ms = latency_ms
        self._total_latency_ms += latency_ms
        self._last_error = str(error) if error is not None else None

    def get_usage_metrics(self) -> dict:
        """Get adapter usage metrics since process start."""
        avg_latency_ms = (
            self._total_latency_ms / self._call_count if self._call_count > 0 else 0.0
        )
        return {
            "call_count": self._call_count,
            "total_latency_ms": round(self._total_latency_ms, 2),
            "avg_latency_ms": round(avg_latency_ms, 2),
            "last_latency_ms": round(self._last_latency_ms, 2),
            "last_error": self._last_error,
        }


class OpenRouterLLM(LLMAdapter):
    """OpenRouter API adapter."""
    
    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ):
        super().__init__()
        # OpenRouter requires HTTP-Referer header (can be any URL)
        default_headers = {
            "HTTP-Referer": "https://github.com/your-org/mechanical-engineering-agents",
            "X-Title": "Manufacturing RCA System",
        }
        self.llm = ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            default_headers=default_headers,
        )
    
    def invoke(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Invoke OpenRouter API."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        started = perf_counter()
        try:
            response = self.llm.invoke(messages)
            self._record_call((perf_counter() - started) * 1000)
            return response.content
        except Exception as e:
            self._record_call((perf_counter() - started) * 1000, e)
            raise


class LocalLLM(LLMAdapter):
    """Local LLM adapter (Ollama, vLLM, etc.)."""
    
    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ):
        super().__init__()
        # For local Ollama or OpenAI-compatible APIs
        self.llm = ChatOpenAI(
            model=model_name,
            base_url=base_url,
            openai_api_key=api_key or "not-needed",
            temperature=temperature,
            max_tokens=max_tokens,
        )
    
    def invoke(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Invoke local LLM."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        started = perf_counter()
        try:
            response = self.llm.invoke(messages)
            self._record_call((perf_counter() - started) * 1000)
            return response.content
        except Exception as e:
            self._record_call((perf_counter() - started) * 1000, e)
            raise


def create_llm() -> Optional[LLMAdapter]:
    """Create LLM instance based on settings."""
    settings = get_settings()
    
    # Check if OpenRouter is configured
    if settings.llm_provider == "openrouter":
        if not settings.llm_api_key or settings.llm_api_key == "":
            print("Warning: OpenRouter provider selected but no API key provided.")
            return None
        
        return OpenRouterLLM(
            model_name=settings.llm_model_name,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url or "https://openrouter.ai/api/v1",
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
    
    # Default to local LLM
    return LocalLLM(
        base_url=settings.llm_base_url,
        model_name=settings.llm_model_name,
        api_key=settings.llm_api_key if settings.llm_api_key and settings.llm_api_key != "not-needed-for-local" else None,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
