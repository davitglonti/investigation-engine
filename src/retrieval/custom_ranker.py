from typing import Dict, List, Optional, Tuple
from src.ingestion.models import Chunk


class CustomRankFusion:
    """
    Custom Algorithmic Component for Multi-Engine Search Fusion & Re-Ranking.
    Implements:
    1. Reciprocal Rank Fusion (RRF) across Dense and Sparse retrieval rankings.
    2. Contextual Re-Ranking using query term coverage, title matching, and header relevance.
    """

    def __init__(self, rrf_k: int = 60, dense_weight: float = 0.5, sparse_weight: float = 0.5):
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    def fuse(
        self,
        dense_results: List[Tuple[Chunk, float]],
        sparse_results: List[Tuple[Chunk, float]],
        query: str,
        top_k: int = 10
    ) -> List[Tuple[Chunk, float, Dict[str, float]]]:
        """
        Merge and algorithmically re-rank dense and sparse candidates.
        Returns: List of (Chunk, final_score, score_breakdown).
        """
        chunk_map: Dict[str, Chunk] = {}
        dense_ranks: Dict[str, int] = {}
        dense_raw_scores: Dict[str, float] = {}

        for rank, (chunk, score) in enumerate(dense_results):
            chunk_map[chunk.id] = chunk
            dense_ranks[chunk.id] = rank + 1
            dense_raw_scores[chunk.id] = score

        sparse_ranks: Dict[str, int] = {}
        sparse_raw_scores: Dict[str, float] = {}

        for rank, (chunk, score) in enumerate(sparse_results):
            chunk_map[chunk.id] = chunk
            sparse_ranks[chunk.id] = rank + 1
            sparse_raw_scores[chunk.id] = score

        # Compute RRF score for all candidate chunks
        all_ids = set(chunk_map.keys())
        scored_candidates = []

        query_terms = set(query.lower().split())

        for chunk_id in all_ids:
            chunk = chunk_map[chunk_id]
            
            # Dense RRF component
            dense_rank = dense_ranks.get(chunk_id, 9999)
            dense_rrf = self.dense_weight / (self.rrf_k + dense_rank) if dense_rank != 9999 else 0.0

            # Sparse RRF component
            sparse_rank = sparse_ranks.get(chunk_id, 9999)
            sparse_rrf = self.sparse_weight / (self.rrf_k + sparse_rank) if sparse_rank != 9999 else 0.0

            raw_rrf = dense_rrf + sparse_rrf
            
            # Normalize RRF into [0, 1] range based on theoretical maximum
            max_rrf = (self.dense_weight + self.sparse_weight) / (self.rrf_k + 1)
            base_rrf = raw_rrf / max_rrf if max_rrf > 0 else raw_rrf

            # Custom Algorithmic Re-scoring / Boosting
            boost = self._calculate_contextual_boost(chunk, query, query_terms)
            final_score = base_rrf * boost

            breakdown = {
                "base_rrf": round(base_rrf, 4),
                "dense_rank": dense_rank if dense_rank != 9999 else None,
                "sparse_rank": sparse_rank if sparse_rank != 9999 else None,
                "dense_score": dense_raw_scores.get(chunk_id, 0.0),
                "sparse_score": sparse_raw_scores.get(chunk_id, 0.0),
                "boost_factor": round(boost, 3)
            }

            scored_candidates.append((chunk, final_score, breakdown))

        # Sort by final fused score descending
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return scored_candidates[:top_k]

    def _calculate_contextual_boost(self, chunk: Chunk, query: str, query_terms: set) -> float:
        """
        Custom heuristic scoring:
        - Term coverage ratio (fraction of query words present in chunk content)
        - Section header relevance
        - Document title relevance
        - Exact phrase match
        """
        boost = 1.0
        content_lower = chunk.content.lower()
        title_lower = chunk.metadata.get("doc_title", "").lower()
        header_lower = chunk.metadata.get("section_header", "").lower()

        # 1. Exact phrase match bonus
        if query.lower().strip() in content_lower:
            boost += 0.35

        # 2. Query Term Coverage
        if query_terms:
            matched_terms = sum(1 for term in query_terms if term in content_lower)
            coverage_ratio = matched_terms / len(query_terms)
            boost += 0.25 * coverage_ratio

        # 3. Title match bonus
        if any(term in title_lower for term in query_terms if len(term) > 3):
            boost += 0.15

        # 4. Header match bonus
        if any(term in header_lower for term in query_terms if len(term) > 3):
            boost += 0.10

        # 5. Introductory Page 1 / Overview Bonus for exploratory questions
        is_exploratory = any(k in query.lower() for k in ["რაზეა", "შინაარსი", "რა არის", "about", "summary", "overview", "პდფ", "pdf"])
        if is_exploratory:
            if chunk.metadata.get("page_number") == 1 or chunk.chunk_index == 0:
                boost += 0.40
            if "აღწერა" in content_lower or "description" in content_lower or "overview" in content_lower:
                boost += 0.30

        return boost
