import io
import re
from pathlib import Path
from typing import BinaryIO, Union
import pypdf
from src.ingestion.models import Document
from src.ingestion.parsers.base import BaseParser
from src.core.logger import get_logger

logger = get_logger(__name__)


class PDFParser(BaseParser):
    """Extracts text and page-level metadata from PDF documents with running-header filtering."""

    RUNNING_HEADERS = [
        "talent growth program",
        "applied ai – ai investigation engine task",
        "applied ai - ai investigation engine task"
    ]

    def parse(self, source: Union[str, Path, BinaryIO, bytes], filename: str) -> Document:
        pdf_stream = self._get_stream(source)
        reader = pypdf.PdfReader(pdf_stream)
        
        extracted_pages = []
        page_metadata = []

        for page_idx, page in enumerate(reader.pages):
            raw_page_text = page.extract_text() or ""
            cleaned = self._clean_page_text(raw_page_text)
            
            if cleaned:
                extracted_pages.append(f"[Page {page_idx + 1}]\n{cleaned}")
                page_metadata.append({
                    "page_number": page_idx + 1,
                    "char_count": len(cleaned)
                })

        full_content = "\n\n".join(extracted_pages)
        content_hash = self.compute_hash(full_content)
        
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

    def _clean_page_text(self, text: str) -> str:
        """Strip running headers, fix vertical word breaks, and normalize whitespace."""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        
        lines = text.split("\n")
        filtered_lines = []
        for line in lines:
            line_str = line.strip()
            # Drop empty or running repetitive header lines
            if not line_str:
                continue
            if any(rh in line_str.lower() for rh in self.RUNNING_HEADERS):
                continue
            filtered_lines.append(line_str)

        reconstructed = "\n".join(filtered_lines)
        
        # Merge artificial single-word broken lines into continuous sentences
        reconstructed = re.sub(r"(?<!\n)\n(?!\n|[•\-\*\d+\.]|[A-ZА-Яა-ჰ0-9\.\s]{1,4}\))", " ", reconstructed)
        reconstructed = re.sub(r"[ \t]+", " ", reconstructed)
        reconstructed = re.sub(r"\n{3,}", "\n\n", reconstructed)
        return reconstructed.strip()

    def _get_stream(self, source: Union[str, Path, BinaryIO, bytes]) -> BinaryIO:
        if isinstance(source, bytes):
            return io.BytesIO(source)
        elif isinstance(source, (str, Path)):
            return open(source, "rb")
        elif hasattr(source, "read"):
            return source
        raise ValueError(f"Unsupported source type: {type(source)}")
