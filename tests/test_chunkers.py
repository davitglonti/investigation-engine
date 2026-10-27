import pytest
from src.ingestion.models import Document
from src.ingestion.chunkers.recursive_chunker import RecursiveCharacterChunker
from src.ingestion.chunkers.semantic_chunker import SemanticStructureChunker


@pytest.fixture
def sample_document():
    long_markdown = """# Architecture Overview
The distributed payment engine processes millions of transactions per second.
It is built with resilient microservices communicating asynchronously over Kafka.

## Storage Layer
PostgreSQL with pgvector handles transactional state and semantic similarity indexing.
Connection pooling is strictly regulated to avoid thread exhaustion under sudden spikes.

## Incident History
In March 2024, a rogue migration caused lock contention on table user_ledgers.
Service latency spiked to 4500ms before automatic circuit breakers severed external requests.
"""
    return Document(
        id="doc_arch_001",
        title="architecture.md",
        content=long_markdown,
        file_type="md",
        content_hash="abc12345hash"
    )


def test_recursive_chunker(sample_document):
    chunker = RecursiveCharacterChunker(chunk_size=150, chunk_overlap=30)
    chunks = chunker.chunk(sample_document)

    assert len(chunks) > 1
    assert all(c.strategy == "recursive" for c in chunks)
    assert all(c.document_id == sample_document.id for c in chunks)
    assert all(len(c.content) <= 200 for c in chunks)  # respects approximate size boundary


def test_semantic_chunker(sample_document):
    chunker = SemanticStructureChunker(chunk_size=200, chunk_overlap=40)
    chunks = chunker.chunk(sample_document)

    assert len(chunks) > 1
    assert all(c.strategy == "semantic" for c in chunks)
    # Check that section headers are captured in metadata
    headers_found = [c.metadata.get("section_header") for c in chunks]
    assert any("Storage Layer" in h or "Incident History" in h for h in headers_found if h)


def test_chunking_comparison(sample_document):
    """Compare Recursive vs Semantic chunking on the same document."""
    rec_chunker = RecursiveCharacterChunker(chunk_size=180, chunk_overlap=30)
    sem_chunker = SemanticStructureChunker(chunk_size=180, chunk_overlap=30)

    rec_chunks = rec_chunker.chunk(sample_document)
    sem_chunks = sem_chunker.chunk(sample_document)

    # Both produce chunks, but semantic chunks preserve headers and structure
    assert len(rec_chunks) >= 1
    assert len(sem_chunks) >= 1
    assert "section_header" in sem_chunks[0].metadata
