from abc import ABC, abstractmethod
import hashlib
from pathlib import Path
from typing import BinaryIO, Union
from src.ingestion.models import Document


class BaseParser(ABC):
    """Abstract base class for all document parsers."""

    @abstractmethod
    def parse(self, source: Union[str, Path, BinaryIO, bytes], filename: str) -> Document:
        """Parse the input source into a standardized Document object."""
        pass

    @staticmethod
    def compute_hash(text: str) -> str:
        """Generate MD5 hash of text content for deduplication."""
        return hashlib.md5(text.strip().encode("utf-8")).hexdigest()
