import io
import re
from pathlib import Path
from typing import BinaryIO, List, Union
from src.ingestion.models import Document
from src.ingestion.parsers.base import BaseParser
from src.core.logger import get_logger

logger = get_logger(__name__)


class MarkdownParser(BaseParser):
    """Parses Markdown files, extracting header hierarchy, links, and sections."""

    def parse(self, source: Union[str, Path, BinaryIO, bytes], filename: str) -> Document:
        raw_text = self._read_text(source)
        content_hash = self.compute_hash(raw_text)

        # Extract markdown headers (# Heading 1, ## Heading 2, etc.)
        headers = self._extract_headers(raw_text)
        links = self._extract_links(raw_text)

        doc_metadata = {
            "headings": headers,
            "heading_count": len(headers),
            "links_count": len(links),
            "char_count": len(raw_text)
        }

        logger.info(f"Parsed Markdown '{filename}' with {len(headers)} headings")

        return Document(
            id=f"doc_{content_hash[:12]}",
            title=filename,
            content=raw_text,
            file_type="md",
            content_hash=content_hash,
            metadata=doc_metadata
        )

    def _extract_headers(self, text: str) -> List[dict]:
        header_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
        headers = []
        for match in header_pattern.finditer(text):
            level = len(match.group(1))
            title = match.group(2).strip()
            headers.append({
                "level": level,
                "title": title,
                "position": match.start()
            })
        return headers

    def _extract_links(self, text: str) -> List[str]:
        link_pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
        return [match.group(2) for match in link_pattern.finditer(text)]

    def _read_text(self, source: Union[str, Path, BinaryIO, bytes]) -> str:
        if isinstance(source, bytes):
            return source.decode("utf-8", errors="replace")
        elif isinstance(source, (str, Path)) and Path(source).exists():
            with open(source, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        elif hasattr(source, "read"):
            data = source.read()
            return data.decode("utf-8", errors="replace") if isinstance(data, bytes) else str(data)
        elif isinstance(source, str):
            return source
        raise ValueError(f"Unable to read markdown text from: {type(source)}")
