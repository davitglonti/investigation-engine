import os
from pathlib import Path
from typing import Any
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

BASE_DIR = Path(__file__).resolve().parent.parent.parent


def get_secret(key: str, default: str = "") -> str:
    """Helper to fetch secrets from environment variables or Streamlit Cloud Secrets."""
    val = os.getenv(key)
    if val:
        return val
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return default


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Project metadata
    PROJECT_NAME: str = "AI Investigation Engine"
    VERSION: str = "0.2.0"
    ENVIRONMENT: str = "production"

    # LLM Settings
    LLM_PROVIDER: str = Field(default="google", description="google, openai, or groq")
    OPENAI_API_KEY: str = Field(default_factory=lambda: get_secret("OPENAI_API_KEY"))
    GEMINI_API_KEY: str = Field(default_factory=lambda: get_secret("GEMINI_API_KEY"))
    GROQ_API_KEY: str = Field(default_factory=lambda: get_secret("GROQ_API_KEY"))
    
    PRIMARY_MODEL: str = Field(default="gemini-3.5-flash-lite")
    EMBEDDING_MODEL: str = Field(default="models/gemini-embedding-001")

    # Vector DB (Cloud & Local)
    VECTOR_DB_TYPE: str = Field(default="chroma")
    CHROMA_PERSIST_DIR: str = str(BASE_DIR / "chroma_db")
    QDRANT_URL: str = Field(default_factory=lambda: get_secret("QDRANT_URL"))
    QDRANT_API_KEY: str = Field(default_factory=lambda: get_secret("QDRANT_API_KEY"))
    QDRANT_COLLECTION: str = "investigation_corpus"

    # Storage paths
    UPLOAD_DIR: str = str(BASE_DIR / "data" / "uploaded_docs")

    # API & UI Server
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    UI_PORT: int = 8501

    # Ingestion Defaults
    CHUNK_SIZE: int = 600
    CHUNK_OVERLAP: int = 100

    # Investigation & Retrieval
    TOP_K_RETRIEVAL: int = 8
    RRF_K: int = 60  # Parameter for Reciprocal Rank Fusion


settings = Settings()
