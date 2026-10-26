import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Project metadata
    PROJECT_NAME: str = "AI Investigation Engine"
    VERSION: str = "0.1.0"
    ENVIRONMENT: str = "development"

    # LLM Settings
    LLM_PROVIDER: str = Field(default="google", description="google, openai, or groq")
    OPENAI_API_KEY: str = Field(default="")
    GEMINI_API_KEY: str = Field(default="")
    GROQ_API_KEY: str = Field(default="")
    
    PRIMARY_MODEL: str = Field(default="gemini-1.5-flash")
    EMBEDDING_MODEL: str = Field(default="models/text-embedding-004")

    # Vector DB
    VECTOR_DB_TYPE: str = Field(default="chroma")
    CHROMA_PERSIST_DIR: str = str(BASE_DIR / "chroma_db")
    QDRANT_URL: str = Field(default="")
    QDRANT_API_KEY: str = Field(default="")

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
