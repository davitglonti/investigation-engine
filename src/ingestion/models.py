from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """Represents a discrete chunk of text extracted from a Document."""
    id: str = Field(description="Unique chunk identifier, e.g., doc_id_chunk_0")
    document_id: str = Field(description="ID of the parent document")
    content: str = Field(description="Text content of the chunk")
    chunk_index: int = Field(description="Sequential position of chunk within the document")
    strategy: str = Field(description="Chunking strategy used: 'recursive' or 'semantic'")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata such as page, section header, etc.")
    token_count: Optional[int] = Field(default=None, description="Estimated token count")


class Document(BaseModel):
    """Represents an ingested source document."""
    id: str = Field(description="Unique document ID (e.g., hash or uuid)")
    title: str = Field(description="Document title or filename")
    content: str = Field(description="Full extracted textual content")
    file_type: str = Field(description="File extension: pdf, md, txt, json")
    content_hash: str = Field(description="MD5 or SHA-256 hash for exact deduplication")
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata: author, source, timestamp, etc.")
    chunks: List[Chunk] = Field(default_factory=list, description="Chunks generated from this document")
