import re
from typing import Dict, List, Tuple
from rank_bm25 import BM25Okapi
from src.ingestion.models import Chunk


class BM25SearchEngine:
    """
    Sparse keyword search engine using BM25Okapi.
    Supports multilingual text (Georgian, English, etc.) via Unicode word tokenization.
    """

    def __init__(self):
        self.chunks: List[Chunk] = []
        self.corpus_tokens: List[List[str]] = []
        self.bm25: BM25Okapi = None

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Unicode-aware tokenization supporting Georgian, English, and symbols."""
        cleaned = re.sub(r"[^\w\s]", " ", text.lower())
        tokens = cleaned.split()
        return [t for t in tokens if len(t) > 1]

    def index_chunks(self, chunks: List[Chunk]) -> None:
        """Build BM25 index over provided chunks."""
        self.chunks = list(chunks)
        self.corpus_tokens = [self.tokenize(c.content) for c in self.chunks]
        if self.corpus_tokens:
            self.bm25 = BM25Okapi(self.corpus_tokens)
        else:
            self.bm25 = None

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Chunk, float]]:
        """
        Search for query keywords and return ranked (Chunk, score) tuples.
        """
        if not self.bm25 or not self.chunks:
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results = []
        for idx in ranked_indices[:top_k]:
            score = float(scores[idx])
            if score > 0:
                results.append((self.chunks[idx], score))

        return results
