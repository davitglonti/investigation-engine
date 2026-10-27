import io
import json
from pathlib import Path
from typing import BinaryIO, Union
from src.ingestion.models import Document
from src.ingestion.parsers.base import BaseParser
from src.core.logger import get_logger

logger = get_logger(__name__)


class TextAndJSONParser(BaseParser):
    """Parses plain text and structured JSON files."""

    def parse(self, source: Union[str, Path, BinaryIO, bytes], filename: str) -> Document:
        raw_text = self._read_text(source)
        file_ext = Path(filename).suffix.lower().lstrip(".") or "txt"
        
        extracted_content = raw_text
        doc_metadata = {"format": file_ext, "char_count": len(raw_text)}

        if file_ext == "json":
            try:
                parsed_json = json.loads(raw_text)
                doc_metadata["is_valid_json"] = True
                
                # If JSON is a list of records or dictionary
                if isinstance(parsed_json, list):
                    doc_metadata["record_count"] = len(parsed_json)
                    formatted_records = []
                    for idx, record in enumerate(parsed_json):
                        formatted_records.append(f"[Record {idx + 1}]\n{json.dumps(record, indent=2)}")
                    extracted_content = "\n\n".join(formatted_records)
                elif isinstance(parsed_json, dict):
                    doc_metadata["keys"] = list(parsed_json.keys())
                    extracted_content = json.dumps(parsed_json, indent=2)
            except Exception as e:
                logger.warning(f"File '{filename}' has .json extension but failed JSON parsing: {e}")
                doc_metadata["is_valid_json"] = False

        content_hash = self.compute_hash(extracted_content)

        return Document(
            id=f"doc_{content_hash[:12]}",
            title=filename,
            content=extracted_content,
            file_type=file_ext,
            content_hash=content_hash,
            metadata=doc_metadata
        )

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
        raise ValueError(f"Unable to read text from source: {type(source)}")
