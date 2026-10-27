import io
from pathlib import Path
from typing import BinaryIO, Union
import pypdf
from src.ingestion.models import Document
from src.ingestion.parsers.base import BaseParser
from src.core.logger import get_logger

logger = get_logger(__name__)


class PDFParser(BaseParser):
    """Extracts text and page-level metadata from PDF documents."""

    def parse(self, source: Union[str, Path, BinaryIO, bytes], filename: str) -> Document:
        pdf_stream = self._get_stream(source)
        reader = pypdf.PdfReader(pdf_stream)
        
        extracted_pages = []
        page_metadata = []

        for page_idx, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            page_text = page_text.strip()
            if page_text:
                extracted_pages.append(f"[Page {page_idx + 1}]\n{page_text}")
                page_metadata.append({
                    "page_number": page_idx + 1,
                    "char_count": len(page_text)
                })

        full_content = "\n\n".join(extracted_pages)
        content_hash = self.compute_hash(full_content)
        
        # Extract embedded metadata if available
        meta = reader.metadata or {}
        doc_metadata = {
            "total_pages": len(reader.pages),
            "author": getattr(meta, "author", None) or "Unknown",
            "title": getattr(meta, "title", None) or filename,
            "creator": getattr(meta, "creator", None) or "PDF",
            "pages": page_metadata
        }

        logger.info(f"Parsed PDF '{filename}' with {len(reader.pages)} pages ({len(full_content)} chars)")

        return Document(
            id=f"doc_{content_hash[:12]}",
            title=filename,
            content=full_content,
            file_type="pdf",
            content_hash=content_hash,
            metadata=doc_metadata
        )

    def _get_stream(self, source: Union[str, Path, BinaryIO, bytes]) -> BinaryIO:
        if isinstance(source, bytes):
            return io.BytesIO(source)
        elif isinstance(source, (str, Path)):
            return open(source, "rb")
        elif hasattr(source, "read"):
            return source
        raise ValueError(f"Unsupported source type: {type(source)}")
