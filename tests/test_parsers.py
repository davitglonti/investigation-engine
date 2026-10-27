import io
import pytest
import pypdf
from src.ingestion.parsers.factory import ParserFactory
from src.ingestion.parsers.markdown_parser import MarkdownParser
from src.ingestion.parsers.text_json_parser import TextAndJSONParser
from src.ingestion.parsers.pdf_parser import PDFParser


def create_sample_pdf_bytes() -> bytes:
    """Helper to generate a minimal in-memory PDF."""
    writer = pypdf.PdfWriter()
    page1 = writer.add_blank_page(width=300, height=300)
    # We can write text onto the page or save blank with metadata
    writer.add_metadata({
        "/Title": "Incident Report March 2024",
        "/Author": "DevOps Lead"
    })
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def test_markdown_parser():
    md_content = """# System Incident Report
## Root Cause Analysis
A memory leak occurred in service PaymentGateway at 14:00 UTC.
For more details see [Architecture Doc](https://wiki.internal/payment).

### Mitigation Steps
1. Rollback to v2.4.0
2. Restart workers
"""
    parser = MarkdownParser()
    doc = parser.parse(md_content, "incident.md")

    assert doc.file_type == "md"
    assert doc.title == "incident.md"
    assert "System Incident Report" in doc.content
    assert doc.metadata["heading_count"] == 3
    assert doc.metadata["links_count"] == 1
    assert doc.content_hash is not None


def test_text_and_json_parser():
    json_data = """[
        {"incident_id": "INC-101", "service": "PaymentGateway", "severity": "P1"},
        {"incident_id": "INC-102", "service": "AuthService", "severity": "P2"}
    ]"""
    parser = TextAndJSONParser()
    doc = parser.parse(json_data, "events.json")

    assert doc.file_type == "json"
    assert doc.metadata["is_valid_json"] is True
    assert doc.metadata["record_count"] == 2
    assert "INC-101" in doc.content


def test_parser_factory():
    p_pdf = ParserFactory.get_parser("report.pdf")
    p_md = ParserFactory.get_parser("readme.md")
    p_txt = ParserFactory.get_parser("log.txt")

    assert isinstance(p_pdf, PDFParser)
    assert isinstance(p_md, MarkdownParser)
    assert isinstance(p_txt, TextAndJSONParser)


def test_pdf_parser_bytes():
    pdf_bytes = create_sample_pdf_bytes()
    parser = PDFParser()
    doc = parser.parse(pdf_bytes, "sample.pdf")

    assert doc.file_type == "pdf"
    assert doc.metadata["total_pages"] == 1
    assert doc.metadata["author"] == "DevOps Lead"
