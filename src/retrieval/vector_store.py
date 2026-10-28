import os
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from src.core.config import settings
from src.core.logger import get_logger
from src.ingestion.models import Chunk

logger = get_logger(__name__)


class LocalVectorStore:
    """
    Production-grade Vector Database Engine supporting:
    1. Cloud Vector DB: Qdrant Cloud Free Tier (persistent cluster)
    2. Local Vector DB: Persistent ChromaDB with Ephemeral and In-Memory fallbacks
    3. Neural Embeddings: Gemini Dense Neural Embeddings (gemini-embedding-001)
       with Latent Semantic Analysis (LSA) offline fallback.
    """

    def __init__(self, collection_name: str = "investigation_corpus", persist_dir: Optional[str] = None):
        self.persist_dir = persist_dir or settings.CHROMA_PERSIST_DIR
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)

        self.collection_name = collection_name
        self.chunk_cache: Dict[str, Chunk] = {}
        self.chunk_vectors: Dict[str, np.ndarray] = {}
        
        # State indicators
        self.is_qdrant_cloud = False
        self.qdrant_client = None
        self.chroma_client = None
        self.chroma_collection = None

        # 1. Check for Qdrant Cloud Configuration
        qdrant_url = os.getenv("QDRANT_URL") or settings.QDRANT_URL
        qdrant_key = os.getenv("QDRANT_API_KEY") or settings.QDRANT_API_KEY
        if qdrant_url and qdrant_key:
            try:
                from qdrant_client import QdrantClient
                from qdrant_client.http import models as q_models
                self.qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_key, timeout=10.0)
                # Check / create collection
                collections = [c.name for c in self.qdrant_client.get_collections().collections]
                embedding_dim = 3072 if (os.getenv("GEMINI_API_KEY") or settings.GEMINI_API_KEY) else 384
                if self.collection_name not in collections:
                    self.qdrant_client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=q_models.VectorParams(size=embedding_dim, distance=q_models.Distance.COSINE)
                    )
                self.is_qdrant_cloud = True
                logger.info(f"Connected to Qdrant Cloud cluster at {qdrant_url}")
            except Exception as e_q:
                logger.warning(f"Qdrant Cloud connection failed ({e_q}), falling back to ChromaDB.")
                self.is_qdrant_cloud = False

        # 2. Local ChromaDB initialization (if not using Qdrant Cloud)
        if not self.is_qdrant_cloud:
            self._init_chroma()

    def _init_chroma(self) -> None:
        """Initialize ChromaDB with persistent and ephemeral fallbacks."""
        import chromadb
        from chromadb.config import Settings
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=Settings(anonymized_telemetry=False, allow_reset=True, is_persistent=True)
            )
            self.chroma_collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("ChromaDB PersistentClient successfully initialized.")
        except Exception as e1:
            logger.warning(f"Persistent ChromaDB unavailable ({e1}), attempting EphemeralClient...")
            try:
                self.chroma_client = chromadb.EphemeralClient(
                    settings=Settings(anonymized_telemetry=False, allow_reset=True)
                )
                self.chroma_collection = self.chroma_client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info("ChromaDB EphemeralClient initialized.")
            except Exception as e2:
                logger.warning(f"ChromaDB completely unavailable ({e2}). Using In-Memory Dense Vector Index.")
                self.chroma_client = None
                self.chroma_collection = None

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Embed and index chunks using Neural Embeddings in Qdrant or ChromaDB."""
        if not chunks:
            return

        documents = [c.content for c in chunks]
        embeddings = self._generate_embeddings(documents)

        # Cache in memory
        for c, emb in zip(chunks, embeddings):
            self.chunk_cache[c.id] = c
            self.chunk_vectors[c.id] = np.array(emb, dtype=np.float32)

        # 1. Upsert into Qdrant Cloud if active
        if self.is_qdrant_cloud and self.qdrant_client:
            try:
                from qdrant_client.http import models as q_models
                points = []
                for c, emb in zip(chunks, embeddings):
                    pt_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, c.id))
                    payload = {
                        "chunk_id": c.id,
                        "document_id": c.document_id,
                        "content": c.content,
                        "strategy": c.strategy,
                        "doc_title": str(c.metadata.get("doc_title", "")),
                        "section_header": str(c.metadata.get("section_header", "")),
                        "page_number": int(c.metadata.get("page_number", 0))
                    }
                    points.append(q_models.PointStruct(id=pt_id, vector=emb, payload=payload))
                
                self.qdrant_client.upsert(collection_name=self.collection_name, points=points)
                logger.info(f"Indexed {len(chunks)} chunks into Qdrant Cloud collection '{self.collection_name}'.")
                return
            except Exception as e_q:
                logger.error(f"Failed to upsert to Qdrant Cloud ({e_q}), falling back to local index.")

        # 2. Upsert into ChromaDB
        if self.chroma_collection is not None:
            ids = [c.id for c in chunks]
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
            try:
                self.chroma_collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas
                )
            except Exception as e_c:
                logger.warning(f"Chroma upsert error ({e_c}), recreating collection to match neural dimensions...")
                try:
                    self.chroma_client.delete_collection(self.collection_name)
                    self.chroma_collection = self.chroma_client.get_or_create_collection(
                        name=self.collection_name,
                        metadata={"hnsw:space": "cosine"}
                    )
                    self.chroma_collection.upsert(
                        ids=ids,
                        embeddings=embeddings,
                        documents=documents,
                        metadatas=metadatas
                    )
                    logger.info(f"Re-created Chroma collection with neural embedding dimension ({len(embeddings[0])}).")
                except Exception as inner_e:
                    logger.warning(f"Chroma recreation error ({inner_e}), using in-memory cosine fallback.")

        logger.info(f"Indexed {len(chunks)} chunks in Vector Store (Total: {len(self.chunk_cache)}).")

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Chunk, float]]:
        """Dense semantic similarity search with true Neural Embeddings."""
        if not self.chunk_cache:
            return []

        query_embedding = self._generate_embeddings([query])[0]

        # 1. Search in Qdrant Cloud
        if self.is_qdrant_cloud and self.qdrant_client:
            try:
                results = self.qdrant_client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding,
                    limit=top_k
                )
                retrieved: List[Tuple[Chunk, float]] = []
                for pt in results:
                    chunk_id = pt.payload.get("chunk_id")
                    if chunk_id in self.chunk_cache:
                        retrieved.append((self.chunk_cache[chunk_id], float(pt.score)))
                if retrieved:
                    return retrieved
            except Exception as e_q:
                logger.warning(f"Qdrant search error ({e_q}), falling back to local query.")

        # 2. Search in ChromaDB
        if self.chroma_collection is not None:
            try:
                results = self.chroma_collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(top_k, len(self.chunk_cache))
                )
                if results and results.get("ids") and results["ids"][0]:
                    matched_ids = results["ids"][0]
                    distances = results["distances"][0] if results.get("distances") else [0.0] * len(matched_ids)
                    retrieved = []
                    for chunk_id, dist in zip(matched_ids, distances):
                        if chunk_id in self.chunk_cache:
                            similarity = max(0.0, 1.0 - (float(dist) / 2.0))
                            retrieved.append((self.chunk_cache[chunk_id], similarity))
                    return retrieved
            except Exception as e_c:
                logger.warning(f"Chroma query failed ({e_c}), using in-memory cosine fallback.")

        # 3. Dense In-Memory Cosine Similarity
        q_vec = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
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
        """
        Generate dense neural embeddings using Google Gemini API.
        Falls back to Latent Semantic Analysis (LSA) dense projection offline.
        """
        gemini_key = os.getenv("GEMINI_API_KEY") or settings.GEMINI_API_KEY
        if gemini_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=gemini_key)
                
                batch_size = 50
                all_embeddings = []
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    res = genai.embed_content(
                        model=settings.EMBEDDING_MODEL,
                        content=batch
                    )
                    if "embedding" in res:
                        emb = res["embedding"]
                        if isinstance(emb, list) and len(emb) > 0 and isinstance(emb[0], list):
                            all_embeddings.extend(emb)
                        elif isinstance(emb, list) and len(emb) > 0 and isinstance(emb[0], (int, float)):
                            all_embeddings.append(emb)
                if len(all_embeddings) == len(texts):
                    return all_embeddings
            except Exception as e:
                logger.warning(f"Remote Gemini neural embedding error ({e}), using dense LSA fallback.")

        # Offline Dense Latent Semantic Analysis (LSA) Fallback
        return self._dense_lsa_embeddings(texts)

    def _dense_lsa_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Dense Latent Semantic Analysis (LSA) embedding projection.
        Derives dense semantic representations from term co-occurrences.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        
        vec = TfidfVectorizer(max_features=500, stop_words="english", token_pattern=r"(?u)\b\w+\b")
        try:
            tfidf = vec.fit_transform(texts)
            n_comp = min(64, tfidf.shape[0], tfidf.shape[1] - 1)
            if n_comp >= 2:
                svd = TruncatedSVD(n_components=n_comp, random_state=42)
                dense = svd.fit_transform(tfidf)
            else:
                dense = tfidf.toarray()
            
            result = []
            for row in dense:
                target = np.zeros(384, dtype=np.float32)
                l = min(len(row), 384)
                target[:l] = row[:l]
                norm = np.linalg.norm(target)
                if norm > 0:
                    target = target / norm
                else:
                    target[0] = 1.0
                result.append(target.tolist())
            return result
        except Exception:
            return [[1.0 / np.sqrt(384)] * 384 for _ in texts]

    def clear(self) -> None:
        """Clear vector collection."""
        if self.is_qdrant_cloud and self.qdrant_client:
            try:
                self.qdrant_client.delete_collection(self.collection_name)
            except Exception:
                pass
        if self.chroma_client and self.chroma_collection:
            try:
                self.chroma_client.delete_collection(self.collection_name)
                self.chroma_collection = self.chroma_client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
            except Exception:
                pass
        self.chunk_cache.clear()
        self.chunk_vectors.clear()
