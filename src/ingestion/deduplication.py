import hashlib
import re
from typing import Dict, List, Optional, Set, Tuple
from src.ingestion.models import Document
from src.core.logger import get_logger

logger = get_logger(__name__)


class Deduplicator:
    """
    Handles exact and near-duplicate detection across documents.
    Prevents storage bloat and redundant retrieval noise.
    """

    def __init__(self, near_duplicate_threshold: float = 0.88):
        self.exact_hashes: Dict[str, str] = {}  # hash -> doc_id
        self.near_duplicate_threshold = near_duplicate_threshold
        self.minhash_signatures: Dict[str, Set[str]] = {}  # doc_id -> shingle set

    @staticmethod
    def normalize_text(text: str) -> str:
        """Strip punctuation, excessive whitespace, and lower-case text."""
        cleaned = re.sub(r"[^\w\s]", "", text.lower())
        return re.sub(r"\s+", " ", cleaned).strip()

    def check_duplicate(self, doc: Document) -> Tuple[bool, Optional[str], str]:
        """
        Check if document is a duplicate.
        Returns: (is_duplicate: bool, existing_doc_id: Optional[str], match_type: str)
        """
        # 1. Exact Content Hash
        if doc.content_hash in self.exact_hashes:
            existing_id = self.exact_hashes[doc.content_hash]
            logger.info(f"Exact duplicate detected: '{doc.title}' matches existing '{existing_id}'")
            return True, existing_id, "exact"

        # 2. Normalized Text Hash
        norm_hash = hashlib.md5(self.normalize_text(doc.content).encode("utf-8")).hexdigest()
        if norm_hash in self.exact_hashes:
            existing_id = self.exact_hashes[norm_hash]
            logger.info(f"Normalized exact duplicate detected: '{doc.title}' matches '{existing_id}'")
            return True, existing_id, "normalized_exact"

        # 3. Near-Duplicate (Shingle Jaccard Similarity)
        doc_shingles = self._compute_shingles(doc.content)
        for existing_id, existing_shingles in self.minhash_signatures.items():
            similarity = self._jaccard_similarity(doc_shingles, existing_shingles)
            if similarity >= self.near_duplicate_threshold:
                logger.info(
                    f"Near-duplicate detected ({similarity:.2f} similarity): "
                    f"'{doc.title}' matches '{existing_id}'"
                )
                return True, existing_id, f"near_duplicate_{similarity:.2f}"

        return False, None, "unique"

    def register_document(self, doc: Document) -> None:
        """Register document hashes to track future duplicates."""
        self.exact_hashes[doc.content_hash] = doc.id
        norm_hash = hashlib.md5(self.normalize_text(doc.content).encode("utf-8")).hexdigest()
        self.exact_hashes[norm_hash] = doc.id
        self.minhash_signatures[doc.id] = self._compute_shingles(doc.content)

    def unregister_document(self, doc_id: str, content_hash: str) -> None:
        """Remove document hashes upon deletion."""
        self.minhash_signatures.pop(doc_id, None)
        keys_to_remove = [k for k, v in self.exact_hashes.items() if v == doc_id]
        for k in keys_to_remove:
            self.exact_hashes.pop(k, None)

    def _compute_shingles(self, text: str, k: int = 4) -> Set[str]:
        words = self.normalize_text(text).split()
        if len(words) < k:
            return set(words)
        return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}

    @staticmethod
    def _jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0
