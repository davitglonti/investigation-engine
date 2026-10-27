import re
from typing import List
from src.ingestion.models import Chunk, Document
from src.ingestion.chunkers.base import BaseChunker


class SemanticStructureChunker(BaseChunker):
    """
    Strategy 2: Structure & Semantic Section Chunking.
    Preserves document structure (Markdown headings, PDF pages, JSON records)
    and attaches structural context (section headers, page numbers) to each chunk.
    Avoids arbitrary word-cuts and preserves semantic units.
    """

    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 80):
        super().__init__(chunk_size, chunk_overlap)

    def chunk(self, document: Document) -> List[Chunk]:
        content = document.content
        
        # Step 1: Detect structural boundaries
        # For PDF: split by [Page X]
        # For Markdown: split by headers (#, ##, ###)
        # For others: split by double newlines (paragraphs)
        if document.file_type == "pdf" and "[Page " in content:
            sections = self._split_by_pdf_pages(content)
        elif document.file_type in ["md", "markdown"]:
            sections = self._split_by_markdown_headers(content)
        else:
            sections = self._split_by_paragraphs(content)

        chunks: List[Chunk] = []
        chunk_idx = 0

        for section in sections:
            header_context = section.get("header", "General")
            page_context = section.get("page")
            text = section.get("text", "").strip()

            if not text:
                continue

            if len(text) <= self.chunk_size:
                chunks.append(self._create_chunk(
                    document=document,
                    text=text,
                    index=chunk_idx,
                    header=header_context,
                    page=page_context
                ))
                chunk_idx += 1
            else:
                # Subdivide oversized section by sentences
                sub_texts = self._split_into_sentences(text)
                current_buf = []
                current_len = 0

                for sentence in sub_texts:
                    if current_len + len(sentence) <= self.chunk_size:
                        current_buf.append(sentence)
                        current_len += len(sentence)
                    else:
                        if current_buf:
                            chunk_text = " ".join(current_buf).strip()
                            chunks.append(self._create_chunk(
                                document=document,
                                text=chunk_text,
                                index=chunk_idx,
                                header=header_context,
                                page=page_context
                            ))
                            chunk_idx += 1
                        current_buf = [sentence]
                        current_len = len(sentence)

                if current_buf:
                    chunk_text = " ".join(current_buf).strip()
                    chunks.append(self._create_chunk(
                        document=document,
                        text=chunk_text,
                        index=chunk_idx,
                        header=header_context,
                        page=page_context
                    ))
                    chunk_idx += 1

        return chunks

    def _create_chunk(
        self,
        document: Document,
        text: str,
        index: int,
        header: str,
        page: int = None
    ) -> Chunk:
        meta = {
            "doc_title": document.title,
            "file_type": document.file_type,
            "section_header": header,
            "char_length": len(text)
        }
        if page:
            meta["page_number"] = page

        # Prepend header context if meaningful
        formatted_content = f"[{header}]\n{text}" if header and header != "General" else text

        return Chunk(
            id=f"{document.id}_sem_{index}",
            document_id=document.id,
            content=formatted_content,
            chunk_index=index,
            strategy="semantic",
            metadata=meta,
            token_count=self.estimate_tokens(formatted_content)
        )

    def _split_by_pdf_pages(self, content: str) -> List[dict]:
        parts = re.split(r"\[Page (\d+)\]", content)
        sections = []
        if parts[0].strip():
            sections.append({"header": "Intro", "page": 1, "text": parts[0].strip()})
        
        for i in range(1, len(parts), 2):
            page_num = int(parts[i])
            text = parts[i + 1].strip() if (i + 1) < len(parts) else ""
            if text:
                sections.append({
                    "header": f"Page {page_num}",
                    "page": page_num,
                    "text": text
                })
        return sections

    def _split_by_markdown_headers(self, content: str) -> List[dict]:
        lines = content.splitlines()
        sections = []
        current_header = "Preamble"
        current_lines = []

        for line in lines:
            header_match = re.match(r"^(#{1,6})\s+(.+)$", line)
            if header_match:
                if current_lines:
                    sections.append({
                        "header": current_header,
                        "text": "\n".join(current_lines).strip()
                    })
                    current_lines = []
                current_header = header_match.group(2).strip()
            else:
                current_lines.append(line)

        if current_lines:
            sections.append({
                "header": current_header,
                "text": "\n".join(current_lines).strip()
            })
        return sections

    def _split_by_paragraphs(self, content: str) -> List[dict]:
        paragraphs = content.split("\n\n")
        return [{"header": "General", "text": p.strip()} for p in paragraphs if p.strip()]

    def _split_into_sentences(self, text: str) -> List[str]:
        sentence_endings = re.compile(r'(?<=[.!?]) +')
        return [s.strip() for s in sentence_endings.split(text) if s.strip()]
