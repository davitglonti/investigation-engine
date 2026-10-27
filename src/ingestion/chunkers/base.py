from abc import ABC, abstractmethod
from typing import List
from src.ingestion.models import Chunk, Document


class BaseChunker(ABC):
    """Abstract base class for all chunking strategies."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def chunk(self, document: Document) -> List[Chunk]:
        """Split a document into discrete chunks."""
        pass

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough estimation of token count (~4 characters per token)."""
        return max(1, len(text) // 4)
