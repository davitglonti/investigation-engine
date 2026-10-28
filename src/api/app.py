import os
from pathlib import Path
from typing import List, Literal, Optional
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.core.config import settings
from src.core.logger import get_logger
from src.ingestion.document_store import DocumentStore
from src.retrieval.hybrid_search import HybridSearchService
from src.engine.investigation_agent import InvestigationAgent
from src.api.schemas import (
    CitationSchema,
    DocumentSummary,
    HealthResponse,
    InvestigationRequest,
    InvestigationResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
)

logger = get_logger(__name__)

# Initialize FastAPI App
app = FastAPI(
    title="AI Investigation Engine API",
    description="Production REST API for multi-hop document investigation, hybrid search, and evidence grounding.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for flexible frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Application Singletons
doc_store = DocumentStore(storage_dir="data/uploaded_docs")
hybrid_search = HybridSearchService()
agent = InvestigationAgent(hybrid_search)


def _seed_initial_knowledge():
    """Seed sample investigation documents for instant API usage."""
    sample_docs = [
        (
            "# Incident Post-Mortem: Payment Gateway Outage\n"
            "Date: March 14, 2024. Severity: P1-Critical.\n"
            "Summary: At 14:15 UTC, the PaymentGateway-Core service experienced complete transaction failure.\n"
            "Root Cause: Database connection pool exhaustion caused by unreleased handles in deployment release v2.4.1.\n"
            "Affected Services: PaymentGateway-Core, BillingWorker-3, and NotificationDispatcher.\n"
            "Discrepancy Note: The incident report states rollback completed at 14:45 UTC, while system audit logs record completion at 15:10 UTC.",
            "Incident-Report-Mar14.md"
        ),
        (
            "# Payment Gateway Architecture & Capacity Spec\n"
            "Default database connection pool limit is configured to 50 concurrent connections per node.\n"
            "Circuit breakers are configured to trip when latency exceeds 2500ms over a 30-second sliding window.\n"
            "All release updates must pass automated pool leak regression suites before promotion to production.",
            "Architecture-Spec.md"
        ),
        (
            "[\n"
            '  {"timestamp": "2024-03-14T13:58:00Z", "event": "Deploy release v2.4.1 to PaymentGateway-Core", "user": "deploy-bot"},\n'
            '  {"timestamp": "2024-03-14T14:15:22Z", "event": "Alarm: Pool limit 50/50 reached on cluster 3", "severity": "ERROR"},\n'
            '  {"timestamp": "2024-03-14T15:10:04Z", "event": "Audit Log: Rollback to v2.4.0 verified clean", "user": "sre-lead"}\n'
            "]",
            "Deploy-Audit-Log.json"
        )
    ]
    for content, filename in sample_docs:
        doc_store.add_document(content, filename, strategy="semantic")
    hybrid_search.index_chunks(doc_store.get_all_chunks())


_seed_initial_knowledge()


@app.get("/", tags=["Health"])
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """System health check and document index counts."""
    return HealthResponse(
        status="healthy",
        version=settings.VERSION,
        total_documents=len(doc_store.documents),
        total_chunks=len(doc_store.get_all_chunks())
    )


# --- DOCUMENT MANAGEMENT (CRUD) ---

@app.post("/api/documents/upload", status_code=status.HTTP_201_CREATED, tags=["Documents"])
async def upload_document(
    file: UploadFile = File(...),
    strategy: Literal["semantic", "recursive"] = Query("semantic")
):
    """
    Upload and index a document (PDF, Markdown, TXT, JSON).
    Extracts metadata, performs deduplication, chunks, and indexes in hybrid search.
    """
    try:
        content = await file.read()
        doc, was_added = doc_store.add_document(content, file.filename, strategy=strategy)
        
        # Refresh hybrid search index
        hybrid_search.index_chunks(doc_store.get_all_chunks())

        return {
            "document_id": doc.id,
            "filename": doc.title,
            "was_added_new": was_added,
            "total_chunks": len(doc.chunks),
            "file_type": doc.file_type,
            "message": "Document successfully ingested and indexed." if was_added else "Duplicate detected; existing record preserved."
        }
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to process document: {str(e)}")


@app.get("/api/documents", response_model=List[DocumentSummary], tags=["Documents"])
async def list_documents():
    """List all ingested documents and their chunk counts."""
    docs = doc_store.list_documents()
    return [DocumentSummary(**d) for d in docs]


@app.get("/api/documents/{doc_id}", tags=["Documents"])
async def get_document(doc_id: str):
    """Retrieve full text and metadata of a specific document."""
    doc = doc_store.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return {
        "id": doc.id,
        "title": doc.title,
        "file_type": doc.file_type,
        "content": doc.content,
        "metadata": doc.metadata,
        "chunk_count": len(doc.chunks)
    }


@app.delete("/api/documents/{doc_id}", tags=["Documents"])
async def delete_document(doc_id: str):
    """Delete a document and cascade remove all its indexed chunks."""
    success = doc_store.delete_document(doc_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Refresh hybrid index
    hybrid_search.index_chunks(doc_store.get_all_chunks())
    return {"message": f"Document '{doc_id}' successfully deleted."}


# --- RETRIEVAL & SEARCH ---

@app.post("/api/search", response_model=SearchResponse, tags=["Retrieval"])
async def search_documents(request: SearchRequest):
    """
    Execute dense, sparse (BM25), or hybrid rank-fused retrieval over indexed chunks.
    """
    hits = hybrid_search.search(
        query=request.query,
        mode=request.mode,
        top_k=request.top_k
    )

    results = []
    for chunk, score, breakdown in hits:
        results.append(SearchHit(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            doc_title=chunk.metadata.get("doc_title", "Unknown"),
            section_header=chunk.metadata.get("section_header", "General"),
            page_number=chunk.metadata.get("page_number"),
            content=chunk.content,
            score=round(score, 4),
            breakdown=breakdown
        ))

    return SearchResponse(
        query=request.query,
        mode=request.mode,
        total_hits=len(results),
        results=results
    )


# --- INVESTIGATION ENGINE ---

@app.post("/api/investigate", response_model=InvestigationResponse, tags=["Investigation"])
async def run_investigation(request: InvestigationRequest):
    """
    Run multi-hop investigation loop:
    1. Query deconstruction into investigative hypotheses.
    2. Multi-step hybrid retrieval.
    3. Cross-document contradiction analysis.
    4. Strict groundedness validation and citation synthesis.
    """
    result = agent.investigate(
        query=request.query,
        search_mode=request.search_mode,
        top_k=request.top_k,
        strict_groundedness=request.strict_groundedness,
        detect_contradictions=request.detect_contradictions
    )

    citations = [
        CitationSchema(
            chunk_id=c["chunk_id"],
            doc_title=c["doc_title"],
            section_header=c["section_header"],
            page_number=c.get("page_number"),
            snippet=c["snippet"],
            score=c["score"]
        )
        for c in result.citations
    ]

    return InvestigationResponse(
        query=result.query,
        status=result.status,
        answer=result.answer,
        trajectory=result.trajectory,
        citations=citations,
        contradictions=result.contradictions
    )
