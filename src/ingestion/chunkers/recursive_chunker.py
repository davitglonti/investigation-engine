from typing import List
from src.ingestion.models import Chunk, Document
from src.ingestion.chunkers.base import BaseChunker


class RecursiveCharacterChunker(BaseChunker):
    """
    Strategy 1: Recursive Character Chunking.
    Splits text by hierarchical separators (\n\n, \n, sentence endings, space)
    to keep chunks within chunk_size while maintaining semantic cohesion and overlap.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        separators: List[str] = None
    ):
        super().__init__(chunk_size, chunk_overlap)
        self.separators = separators or ["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""]

    def chunk(self, document: Document) -> List[Chunk]:
        raw_chunks = self._split_text(document.content, self.separators)
        
        chunks: List[Chunk] = []
        for idx, text in enumerate(raw_chunks):
            chunk_obj = Chunk(
                id=f"{document.id}_rec_{idx}",
                document_id=document.id,
                content=text,
                chunk_index=idx,
                strategy="recursive",
                metadata={
                    "doc_title": document.title,
                    "file_type": document.file_type,
                    "char_length": len(text)
                },
                token_count=self.estimate_tokens(text)
            )
            chunks.append(chunk_obj)
            
        return chunks

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        final_chunks = []
        separator = separators[-1]
        new_separators = []

        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        splits = text.split(separator) if separator else list(text)

        good_splits = []
        current_chunk = []
        total_len = 0

        for s in splits:
            s_len = len(s)
            if total_len + s_len + (len(separator) if total_len > 0 else 0) <= self.chunk_size:
                current_chunk.append(s)
                total_len += s_len + (len(separator) if total_len > 0 else 0)
            else:
                if current_chunk:
                    doc = separator.join(current_chunk).strip()
                    if doc:
                        good_splits.append(doc)
                    
                    # Compute overlap
                    overlap_size = 0
                    overlap_chunk = []
                    for item in reversed(current_chunk):
                        if overlap_size + len(item) <= self.chunk_overlap:
                            overlap_chunk.insert(0, item)
                            overlap_size += len(item)
                        else:
                            break
                    current_chunk = overlap_chunk
                    total_len = sum(len(x) for x in current_chunk) + max(0, len(current_chunk) - 1) * len(separator)

                if s_len > self.chunk_size and new_separators:
                    sub_chunks = self._split_text(s, new_separators)
                    good_splits.extend(sub_chunks)
                else:
                    current_chunk.append(s)
                    total_len += s_len

        if current_chunk:
            doc = separator.join(current_chunk).strip()
            if doc:
                good_splits.append(doc)

        return [c for c in good_splits if c.strip()]
