from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "0.1.0"
    total_documents: int
    total_chunks: int


class DocumentSummary(BaseModel):
    id: str
    title: str
    file_type: str
    chunk_count: int
    char_count: int
    created_at: str


class SearchRequest(BaseModel):
    query: str = Field(..., description="Query string for retrieval")
    mode: Literal["hybrid", "dense", "sparse"] = Field(default="hybrid", description="Retrieval mode")
    top_k: int = Field(default=8, ge=1, le=50, description="Number of results to return")


class SearchHit(BaseModel):
    chunk_id: str
    document_id: str
    doc_title: str
    section_header: str
    page_number: Optional[int] = None
    content: str
    score: float
    breakdown: Optional[Dict[str, Any]] = None


class SearchResponse(BaseModel):
    query: str
    mode: str
    total_hits: int
    results: List[SearchHit]


class InvestigationRequest(BaseModel):
    query: str = Field(..., description="Investigation question in Georgian or English")
    search_mode: Literal["hybrid", "dense", "sparse"] = Field(default="hybrid")
    top_k: int = Field(default=8, ge=1, le=30)
    strict_groundedness: bool = Field(default=True)
    detect_contradictions: bool = Field(default=True)


class CitationSchema(BaseModel):
    chunk_id: str
    doc_title: str
    section_header: str
    page_number: Optional[int] = None
    snippet: str
    score: float


class InvestigationResponse(BaseModel):
    query: str
    status: str
    answer: str
    trajectory: List[str]
    citations: List[CitationSchema]
    contradictions: List[str]
