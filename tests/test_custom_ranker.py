import pytest
from src.ingestion.models import Chunk
from src.retrieval.custom_ranker import CustomRankFusion
from src.retrieval.bm25_search import BM25SearchEngine


@pytest.fixture
def sample_chunks():
    c1 = Chunk(
        id="c1",
        document_id="d1",
        content="Payment gateway memory leak occurred on March 14.",
        chunk_index=0,
        strategy="semantic",
        metadata={"doc_title": "Incident-March.md", "section_header": "Root Cause"}
    )
    c2 = Chunk(
        id="c2",
        document_id="d2",
        content="User authentication service restarted successfully.",
        chunk_index=0,
        strategy="semantic",
        metadata={"doc_title": "Auth.md", "section_header": "Overview"}
    )
    c3 = Chunk(
        id="c3",
        document_id="d3",
        content="Payment gateway connection pool limit reached 50/50.",
        chunk_index=1,
        strategy="semantic",
        metadata={"doc_title": "Capacity.md", "section_header": "Thresholds"}
    )
    return [c1, c2, c3]


def test_bm25_search(sample_chunks):
    engine = BM25SearchEngine()
    engine.index_chunks(sample_chunks)

    results = engine.search("payment gateway memory leak", top_k=2)
    assert len(results) > 0
    top_chunk, score = results[0]
    assert top_chunk.id == "c1"
    assert score > 0


def test_custom_rank_fusion(sample_chunks):
    c1, c2, c3 = sample_chunks
    dense_results = [(c3, 0.95), (c1, 0.88)]
    sparse_results = [(c1, 4.2), (c3, 2.1)]

    ranker = CustomRankFusion(rrf_k=60)
    fused = ranker.fuse(
        dense_results=dense_results,
        sparse_results=sparse_results,
        query="payment gateway memory leak",
        top_k=3
    )

    assert len(fused) == 2
    # c1 should be boosted to top because it has exact term coverage + strong rankings in both
    top_chunk, final_score, breakdown = fused[0]
    assert top_chunk.id == "c1"
    assert final_score > 0
    assert "base_rrf" in breakdown
    assert "boost_factor" in breakdown
