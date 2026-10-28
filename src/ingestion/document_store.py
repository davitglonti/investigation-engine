import json
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Tuple, Union
from src.ingestion.models import Chunk, Document
from src.ingestion.parsers.factory import ParserFactory
from src.ingestion.chunkers.recursive_chunker import RecursiveCharacterChunker
from src.ingestion.chunkers.semantic_chunker import SemanticStructureChunker
from src.ingestion.deduplication import Deduplicator
from src.core.logger import get_logger

logger = get_logger(__name__)


class DocumentStore:
    """
    In-memory and persistent document repository.
    Handles complete lifecycle: Ingestion -> Parsing -> Deduplication -> Chunking -> CRUD.
    """

    def __init__(self, storage_dir: Optional[Union[str, Path]] = None):
        self.storage_dir = Path(storage_dir) if storage_dir else None
        self.documents: Dict[str, Document] = {}
        self.deduplicator = Deduplicator()
        
        # Initialize Chunkers
        self.chunker_recursive = RecursiveCharacterChunker()
        self.chunker_semantic = SemanticStructureChunker()

        if self.storage_dir:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            self._load_persisted_documents()

    def _save_persisted_document(self, doc: Document) -> None:
        """Persist document metadata, content, and chunks to JSON file."""
        if not self.storage_dir:
            return
        doc_path = self.storage_dir / f"{doc.id}.json"
        try:
            data = {
                "id": doc.id,
                "title": doc.title,
                "content": doc.content,
                "file_type": doc.file_type,
                "content_hash": doc.content_hash,
                "metadata": doc.metadata,
                "created_at": doc.created_at,
                "chunks": [c.model_dump() for c in doc.chunks]
            }
            with open(doc_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to persist document '{doc.id}' to disk ({e}).")

    def _load_persisted_documents(self) -> None:
        """Load all previously stored documents from disk on boot."""
        if not self.storage_dir or not self.storage_dir.exists():
            return
        for json_file in self.storage_dir.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                chunks = [Chunk(**c) for c in data.get("chunks", [])]
                doc = Document(
                    id=data["id"],
                    title=data["title"],
                    content=data["content"],
                    file_type=data.get("file_type", "txt"),
                    content_hash=data.get("content_hash", ""),
                    metadata=data.get("metadata", {}),
                    chunks=chunks,
                    created_at=data.get("created_at", "")
                )
                self.documents[doc.id] = doc
                self.deduplicator.register_document(doc)
            except Exception as e:
                logger.warning(f"Error loading persisted document from {json_file}: {e}")
        if self.documents:
            logger.info(f"Loaded {len(self.documents)} persisted documents from {self.storage_dir}.")

    def add_document(
        self,
        source: Union[str, Path, BinaryIO, bytes],
        filename: str,
        strategy: str = "semantic",
        allow_duplicates: bool = False
    ) -> Tuple[Document, bool]:
        """
        Ingest, parse, deduplicate, and chunk a document.
        Returns: (Document, was_added_new: bool)
        """
        # 1. Parse Document
        doc = ParserFactory.parse_file(source, filename)

        # 2. Check Deduplication
        if not allow_duplicates:
            is_dup, existing_id, match_type = self.deduplicator.check_duplicate(doc)
            if is_dup and existing_id in self.documents:
                logger.info(f"Skipping duplicate document '{filename}'. Matches '{existing_id}' ({match_type}).")
                return self.documents[existing_id], False

        # 3. Apply Chunking Strategy
        chunker = self.chunker_semantic if strategy == "semantic" else self.chunker_recursive
        chunks = chunker.chunk(doc)
        doc.chunks = chunks

        # 4. Store Document & Register Hashes
        self.documents[doc.id] = doc
        self.deduplicator.register_document(doc)
        self._save_persisted_document(doc)

        logger.info(f"Successfully added document '{doc.title}' [ID: {doc.id}] with {len(chunks)} chunks.")
        return doc, True

    def get_document(self, doc_id: str) -> Optional[Document]:
        """Retrieve document by ID."""
        return self.documents.get(doc_id)

    def update_document(
        self,
        doc_id: str,
        source: Union[str, Path, BinaryIO, bytes],
        filename: str,
        strategy: str = "semantic"
    ) -> Optional[Document]:
        """Update existing document content and regenerate its chunks."""
        if doc_id not in self.documents:
            logger.warning(f"Cannot update non-existent document ID '{doc_id}'.")
            return None

        old_doc = self.documents[doc_id]
        self.deduplicator.unregister_document(old_doc.id, old_doc.content_hash)

        # Re-parse
        updated_doc = ParserFactory.parse_file(source, filename)
        updated_doc.id = doc_id  # maintain ID consistency

        # Re-chunk
        chunker = self.chunker_semantic if strategy == "semantic" else self.chunker_recursive
        updated_doc.chunks = chunker.chunk(updated_doc)

        self.documents[doc_id] = updated_doc
        self.deduplicator.register_document(updated_doc)

        logger.info(f"Updated document '{filename}' [ID: {doc_id}] with {len(updated_doc.chunks)} new chunks.")
        return updated_doc

    def delete_document(self, doc_id: str) -> bool:
        """Delete document and all associated chunks."""
        if doc_id not in self.documents:
            return False

        doc = self.documents.pop(doc_id)
        self.deduplicator.unregister_document(doc.id, doc.content_hash)
        
        # Remove from persistent storage
        if self.storage_dir:
            fpath = self.storage_dir / f"{doc_id}.json"
            if fpath.exists():
                try:
                    fpath.unlink()
                except Exception:
                    pass

        logger.info(f"Deleted document '{doc.title}' [ID: {doc_id}].")
        return True

    def list_documents(self) -> List[Dict]:
        """List summary of all ingested documents."""
        summary = []
        for doc in self.documents.values():
            summary.append({
                "id": doc.id,
                "title": doc.title,
                "file_type": doc.file_type,
                "chunk_count": len(doc.chunks),
                "char_count": len(doc.content),
                "created_at": doc.created_at
            })
        return summary

    def get_all_chunks(self) -> List[Chunk]:
        """Retrieve flat list of all active chunks across documents."""
        all_chunks = []
        for doc in self.documents.values():
            all_chunks.extend(doc.chunks)
        return all_chunks

    def clear(self) -> None:
        """Clear all stored documents and deduplication hashes."""
        self.documents.clear()
        self.deduplicator = Deduplicator()
