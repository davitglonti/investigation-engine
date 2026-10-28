import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import chromadb
from chromadb.config import Settings as ChromaSettings
from sklearn.feature_extraction.text import HashingVectorizer
import numpy as np

from src.ingestion.models import Chunk
from src.core.config import settings
from src.core.logger import get_logger

logger = get_logger(__name__)


class LocalVectorStore:
    """
    Vector Database wrapper using ChromaDB with resilient in-memory fallback.
    Supports persistent collection storage, dense cosine retrieval,
    and automatic embedding fallback (API or Local Hashing Vectorizer).
    """

    def __init__(self, collection_name: str = "investigation_corpus", persist_dir: Optional[str] = None):
        self.persist_dir = persist_dir or settings.CHROMA_PERSIST_DIR
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)

        self.collection_name = collection_name
        self.chunk_cache: Dict[str, Chunk] = {}
        self.chunk_vectors: Dict[str, np.ndarray] = {}
        self.vectorizer = HashingVectorizer(n_features=384, alternate_sign=False)
        self.client = None
        self.collection = None

        # 1. Try Persistent ChromaDB
        try:
            from chromadb.config import Settings
            self.client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=Settings(anonymized_telemetry=False, allow_reset=True, is_persistent=True)
            )
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("ChromaDB PersistentClient successfully initialized.")
        except Exception as e1:
            logger.warning(f"Persistent ChromaDB failed ({e1}), falling back to EphemeralClient...")
            # 2. Try Ephemeral ChromaDB
            try:
                from chromadb.config import Settings
                self.client = chromadb.EphemeralClient(
                    settings=Settings(anonymized_telemetry=False, allow_reset=True)
                )
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info("ChromaDB EphemeralClient initialized.")
            except Exception as e2:
                # 3. Resilient Pure NumPy Fallback for cutting-edge runtimes (e.g. Python 3.14)
                logger.warning(f"ChromaDB unavailable on runtime ({e2}). Using Pure NumPy In-Memory Vector Index.")
                self.client = None
                self.collection = None

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Embed and index chunks in ChromaDB or In-Memory Vector Store."""
        if not chunks:
            return

        ids = [c.id for c in chunks]
        documents = [c.content for c in chunks]
        metadatas = [
            {
                "document_id": c.document_id,
                "strategy": c.strategy,
                "doc_title": str(c.metadata.get("doc_title", "")),
                "section_header": str(c.metadata.get("section_header", "")),
                "page_number": int(c.metadata.get("page_number", 0))
            }
            for c in chunks
        ]

        # Generate embeddings
        embeddings = self._generate_embeddings(documents)

        # Cache in memory for instant lookup and fallback
        for c, emb in zip(chunks, embeddings):
            self.chunk_cache[c.id] = c
            self.chunk_vectors[c.id] = np.array(emb, dtype=np.float32)

        if self.collection is not None:
            try:
                self.collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas
                )
            except Exception as e:
                logger.warning(f"Chroma upsert error ({e}), recreating collection...")
                try:
                    self.client.delete_collection(self.collection_name)
                    self.collection = self.client.get_or_create_collection(
                        name=self.collection_name,
                        metadata={"hnsw:space": "cosine"}
                    )
                    self.collection.upsert(
                        ids=ids,
                        embeddings=embeddings,
                        documents=documents,
                        metadatas=metadatas
                    )
                except Exception as inner_e:
                    logger.warning(f"Chroma upsert failed completely ({inner_e}), using pure vector fallback.")

        logger.info(f"Indexed {len(chunks)} chunks in Vector Store (Total: {len(self.chunk_cache)}).")

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Chunk, float]]:
        """Dense semantic similarity search with ChromaDB or NumPy cosine similarity."""
        if not self.chunk_cache:
            return []

        query_embedding = self._generate_embeddings([query])[0]
        q_vec = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)

        # If ChromaDB collection is active, attempt Chroma query
        if self.collection is not None:
            try:
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(top_k, len(self.chunk_cache))
                )
                if results and results.get("ids") and results["ids"][0]:
                    matched_ids = results["ids"][0]
                    distances = results["distances"][0] if results.get("distances") else [0.0] * len(matched_ids)
                    retrieved: List[Tuple[Chunk, float]] = []
                    for chunk_id, dist in zip(matched_ids, distances):
                        if chunk_id in self.chunk_cache:
                            similarity = max(0.0, 1.0 - (float(dist) / 2.0))
                            retrieved.append((self.chunk_cache[chunk_id], similarity))
                    return retrieved
            except Exception as e:
                logger.warning(f"Chroma query failed ({e}), falling back to direct cosine similarity.")

        # Robust NumPy Cosine Search Fallback
        scores = []
        for chunk_id, c_vec in self.chunk_vectors.items():
            c_norm = np.linalg.norm(c_vec)
            if q_norm > 0 and c_norm > 0:
                sim = float(np.dot(q_vec, c_vec) / (q_norm * c_norm))
            else:
                sim = 0.0
            scores.append((self.chunk_cache[chunk_id], max(0.0, min(1.0, sim))))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate vector embeddings via configured provider or local Hashing Vectorizer."""
        gemini_key = os.getenv("GEMINI_API_KEY") or settings.GEMINI_API_KEY
        if gemini_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=gemini_key)
                vectors = []
                for text in texts:
                    res = genai.embed_content(
                        model=settings.EMBEDDING_MODEL,
                        content=text,
                        task_type="retrieval_query"
                    )
                    vectors.append(res["embedding"])
                return vectors
            except Exception as e:
                logger.warning(f"Remote embedding failed ({e}), using local vector fallback.")

        # Local Hashing Vectorizer (always exactly 384 dimensions)
        return self._local_dense_embeddings(texts)

    def _local_dense_embeddings(self, texts: List[str]) -> List[List[float]]:
        matrix = self.vectorizer.transform(texts).toarray()
        normalized = []
        for row in matrix:
            norm = np.linalg.norm(row)
            if norm > 0:
                normalized.append((row / norm).tolist())
            else:
                seed = np.zeros(384)
                seed[0] = 1.0
                normalized.append(seed.tolist())
        return normalized

    def clear(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception:
            pass
        self.chunk_cache.clear()
