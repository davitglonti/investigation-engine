from typing import Dict, List, Literal, Tuple
from src.ingestion.models import Chunk
from src.retrieval.bm25_search import BM25SearchEngine
from src.retrieval.vector_store import LocalVectorStore
from src.retrieval.custom_ranker import CustomRankFusion
from src.core.logger import get_logger

logger = get_logger(__name__)


class HybridSearchService:
    """
    Coordinates Dense Vector Search, Sparse BM25 Search, and Custom Algorithmic Fusion.
    """

    def __init__(self, rrf_k: int = 60):
        self.vector_store = LocalVectorStore()
        self.bm25_engine = BM25SearchEngine()
        self.rank_fusion = CustomRankFusion(rrf_k=rrf_k)
        self.all_chunks: List[Chunk] = []

    def index_chunks(self, chunks: List[Chunk]) -> None:
        """Index chunks across both sparse BM25 and dense vector backends."""
        self.all_chunks = list(chunks)
        self.bm25_engine.index_chunks(chunks)
        self.vector_store.add_chunks(chunks)
        logger.info(f"Hybrid index refreshed with {len(chunks)} chunks.")

    def search(
        self,
        query: str,
        mode: Literal["hybrid", "dense", "sparse"] = "hybrid",
        top_k: int = 10
    ) -> List[Tuple[Chunk, float, Dict[str, float]]]:
        """
        Execute search across configured search mode.
        Returns: List of (Chunk, score, breakdown).
        """
        if not self.all_chunks:
            return []

        if mode == "dense":
            dense_hits = self.vector_store.search(query, top_k=top_k)
            return [(c, s, {"dense_score": s}) for c, s in dense_hits]

        elif mode == "sparse":
            sparse_hits = self.bm25_engine.search(query, top_k=top_k)
            return [(c, s, {"sparse_score": s}) for c, s in sparse_hits]

        else:  # hybrid
            # Retrieve expanded candidates from both branches
            dense_candidates = self.vector_store.search(query, top_k=top_k * 2)
            sparse_candidates = self.bm25_engine.search(query, top_k=top_k * 2)

            fused_results = self.rank_fusion.fuse(
                dense_results=dense_candidates,
                sparse_results=sparse_candidates,
                query=query,
                top_k=top_k
            )
            return fused_results
