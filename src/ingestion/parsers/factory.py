from pathlib import Path
from typing import BinaryIO, Union
from src.ingestion.models import Document
from src.ingestion.parsers.base import BaseParser
from src.ingestion.parsers.pdf_parser import PDFParser
from src.ingestion.parsers.markdown_parser import MarkdownParser
from src.ingestion.parsers.text_json_parser import TextAndJSONParser


class ParserFactory:
    """Factory to dispatch appropriate parser based on file extension."""

    @staticmethod
    def get_parser(filename: str) -> BaseParser:
        ext = Path(filename).suffix.lower()
        if ext == ".pdf":
            return PDFParser()
        elif ext in [".md", ".markdown"]:
            return MarkdownParser()
        elif ext in [".txt", ".json", ".log", ".csv"]:
            return TextAndJSONParser()
        else:
            # Fallback to Text parser
            return TextAndJSONParser()

    @classmethod
    def parse_file(cls, source: Union[str, Path, BinaryIO, bytes], filename: str) -> Document:
        parser = cls.get_parser(filename)
        return parser.parse(source, filename)
