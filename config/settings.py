"""Application settings."""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""
    
    # LLM Configuration
    llm_provider: str = Field(
        default="openrouter",
        description="LLM provider: 'local' or 'openrouter'"
    )
    llm_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="Base URL for LLM server (OpenRouter or local)"
    )
    llm_model_name: str = Field(
        default="arcee-ai/trinity-large-preview:free",
        description="Model name to use"
    )
    llm_api_key: str = Field(
        default="",
        description="API key (required for OpenRouter, optional for local)"
    )
    llm_temperature: float = Field(default=0.1, description="LLM temperature")
    llm_max_tokens: int = Field(default=4096, description="Max tokens per response")
    
    # Database
    database_url: str = Field(
        default="postgresql://user:password@localhost:5432/rca_db",
        description="PostgreSQL connection string"
    )
    
    # Vector Store
    vector_store_type: str = Field(
        default="pgvector",
        description="Vector store type: pgvector or chromadb"
    )
    chroma_persist_dir: str = Field(
        default="./data/chroma",
        description="ChromaDB persistence directory"
    )
    
    # Product Guide
    product_guide_dir: str = Field(
        default="./data/product_guides",
        description="Directory containing product guide documents"
    )
    rag_store_dir: str = Field(
        default="./data/rag_store",
        description="Directory to persist product guide chunks and metadata"
    )
    embedding_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model name for product guide retrieval"
    )
    product_guide_auto_ingest: bool = Field(
        default=True,
        description="Automatically ingest product guides on startup"
    )
    product_guide_rebuild_on_startup: bool = Field(
        default=False,
        description="Clear and rebuild RAG store on startup"
    )
    
    # Analysis
    default_alpha: float = Field(default=0.05, description="Default significance level")
    default_min_effect_size: float = Field(
        default=0.5, 
        description="Default Cohen's d for practical significance"
    )
    min_sample_size: int = Field(default=30, description="Minimum sample size for analysis")
    
    # Paths
    artifacts_dir: str = Field(
        default="./data/artifacts",
        description="Directory for generated charts and summaries"
    )
    recipes_dir: str = Field(
        default="./config/analysis_recipes",
        description="Directory containing analysis recipe YAML files"
    )
    catalog_dir: str = Field(
        default="./data/catalog",
        description="Directory containing catalog JSON files"
    )
    catalog_db_url: Optional[str] = Field(
        default=None,
        description="Optional DB URL for production catalog loading"
    )
    catalog_embedding_top_k: int = Field(
        default=50,
        description="Number of candidate fields from semantic search"
    )
    
    # Logging
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json")
    
    # API
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Singleton instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get application settings (singleton)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
