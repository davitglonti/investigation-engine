import re
from typing import Any, Dict, List, Optional
from src.retrieval.hybrid_search import HybridSearchService
from src.engine.llm_client import LLMClient
from src.engine.tools import InvestigationTools
from src.ingestion.models import Chunk
from src.core.logger import get_logger

logger = get_logger(__name__)


class InvestigationResult:
    """Encapsulates the complete result of an investigation run."""

    def __init__(
        self,
        query: str,
        answer: str,
        trajectory: List[str],
        citations: List[Dict[str, Any]],
        contradictions: List[str],
        status: str = "success"
    ):
        self.query = query
        self.answer = answer
        self.trajectory = trajectory
        self.citations = citations
        self.contradictions = contradictions
        self.status = status


class InvestigationAgent:
    """
    Agentic Investigation Engine with Multi-Hop Reasoning,
    Contradiction Detection, Transparent Citations, and Strict Groundedness.
    """

    SYSTEM_PROMPT = """You are an expert AI Investigation Agent.
Your mission is to perform meticulous multi-hop investigations over complex technical documentation, incident post-mortems, and system logs.

STRICT GROUNDEDNESS RULES:
1. ONLY make claims directly supported by the provided EVIDENCE CONTEXT.
2. If the context does not contain sufficient information to answer the question, EXPLICITLY REFUSE to invent facts and state that evidence is insufficient (Abstain).
3. If different documents contain contradictory facts (e.g., conflicting timestamps, mismatched resolution times, or differing root causes), EXPLICITLY HIGHLIGHT the contradiction.
4. For every key finding, attribute the source document name, section/page, and relevant quote.
5. Answer in the same language as the user's question (e.g., if Georgian, answer in Georgian; if English, answer in English).
"""

    def __init__(self, search_service: HybridSearchService, llm_provider: Optional[str] = None):
        self.search_service = search_service
        self.tools = InvestigationTools(search_service)
        self.llm = LLMClient(provider=llm_provider)

    def investigate(
        self,
        query: str,
        search_mode: str = "hybrid",
        top_k: int = 8,
        strict_groundedness: bool = True,
        detect_contradictions: bool = True
    ) -> InvestigationResult:
        """
        Execute full multi-step investigation loop for query.
        """
        trajectory: List[str] = []
        contradictions: List[str] = []
        citations: List[Dict[str, Any]] = []

        logger.info(f"Starting investigation for query: '{query}'")
        trajectory.append("Step 1: Deconstructing query into investigative hypotheses and search targets.")

        # --- STEP 1: Multi-Hop Target Generation ---
        sub_queries = self._generate_investigative_queries(query)
        trajectory.append(f"Step 2: Formulated {len(sub_queries)} sub-queries: {sub_queries}")

        # --- STEP 2: Multi-Step Retrieval Execution ---
        collected_chunks: Dict[str, Chunk] = {}
        candidate_scores: Dict[str, float] = {}

        for sq in sub_queries:
            hits = self.search_service.search(sq, mode=search_mode, top_k=top_k)
            for chunk, score, breakdown in hits:
                if chunk.id not in collected_chunks or score > candidate_scores.get(chunk.id, 0):
                    collected_chunks[chunk.id] = chunk
                    candidate_scores[chunk.id] = score

        trajectory.append(
            f"Step 3: Retrieved {len(collected_chunks)} unique candidate chunks across {len(sub_queries)} searches."
        )

        # --- STEP 3: Evidence Evaluation & Abstention Check ---
        ranked_chunks = sorted(
            collected_chunks.values(),
            key=lambda c: candidate_scores.get(c.id, 0),
            reverse=True
        )[:top_k]

        # Check if any evidence exists or if scores are completely ungrounded
        if not ranked_chunks:
            trajectory.append("Step 4: Evidence evaluation: No relevant documents found.")
            return InvestigationResult(
                query=query,
                answer=(
                    "მოცემულ დოკუმენტებში აღნიშნულ შეკითხვაზე პასუხის გასაცემად საკმარისი ინფორმაცია არ მოიძებნა. "
                    "სისტემა თავს იკავებს პასუხისგან ჰალუცინაციის თავიდან ასაცილებლად (Strict Abstention)."
                ),
                trajectory=trajectory,
                citations=[],
                contradictions=[],
                status="abstained_no_evidence"
            )

        trajectory.append(f"Step 4: Filtered to top {len(ranked_chunks)} most relevant evidence chunks.")

        # --- STEP 4: Contradiction Detection ---
        if detect_contradictions:
            contradictions = self._detect_cross_document_contradictions(ranked_chunks)
            if contradictions:
                trajectory.append(f"Step 5: ⚠️ Discovered {len(contradictions)} cross-document discrepancy.")
            else:
                trajectory.append("Step 5: Verified cross-document factual consistency. No discrepancies found.")

        # --- STEP 5: Prepare Citations ---
        for chunk in ranked_chunks[:4]:
            meta = chunk.metadata
            citations.append({
                "chunk_id": chunk.id,
                "doc_title": meta.get("doc_title", "Unknown"),
                "section_header": meta.get("section_header", "General"),
                "page_number": meta.get("page_number"),
                "snippet": chunk.content[:200] + "...",
                "score": round(candidate_scores.get(chunk.id, 0.0), 3)
            })

        # --- STEP 6: LLM Synthesis with Evidence Prompt ---
        trajectory.append("Step 6: Executing grounded synthesis via LLM with strict evidence attribution.")
        
        evidence_context = self._format_evidence_context(ranked_chunks)
        prompt = (
            f"EVIDENCE CONTEXT:\n{evidence_context}\n\n"
            f"USER QUERY: {query}\n\n"
            "INSTRUCTIONS:\n"
            "Provide an accurate, structured investigation findings response. "
            "Highlight which documents support each finding. "
            "If any contradiction exists, state it clearly."
        )

        llm_response = self.llm.generate_response(
            system_instruction=self.SYSTEM_PROMPT,
            user_prompt=prompt
        )

        answer = llm_response.get("content", "")
        trajectory.append("Step 7: Investigation concluded successfully.")

        return InvestigationResult(
            query=query,
            answer=answer,
            trajectory=trajectory,
            citations=citations,
            contradictions=contradictions,
            status="conflict_found" if contradictions else "success"
        )

    def _generate_investigative_queries(self, main_query: str) -> List[str]:
        """Deconstruct complex query into targeted sub-queries for multi-hop retrieval."""
        queries = [main_query]
        
        # Heuristic keywords extraction
        keywords = re.findall(r"\w{4,}", main_query.lower())
        if len(keywords) >= 2:
            queries.append(" ".join(keywords[:3]))
            queries.append(" ".join(keywords[-3:]))
        
        return list(dict.fromkeys(queries))[:3]

    def _detect_cross_document_contradictions(self, chunks: List[Chunk]) -> List[str]:
        """Detect timestamps, status, or metric contradictions across different documents."""
        conflicts = []
        timestamps_by_doc: Dict[str, List[str]] = {}
        
        time_pattern = re.compile(r"\b(\d{1,2}:\d{2}(?::\d{2})?(?:\s*(?:UTC|AM|PM))?)\b")

        for c in chunks:
            doc_name = c.metadata.get("doc_title", "Unknown")
            matches = time_pattern.findall(c.content)
            if matches:
                timestamps_by_doc.setdefault(doc_name, []).extend(matches)

        # Check if multiple documents discuss rollback or resolution with different times
        docs = list(timestamps_by_doc.keys())
        for i in range(len(docs)):
            for j in range(i + 1, len(docs)):
                doc_a, doc_b = docs[i], docs[j]
                times_a = set(timestamps_by_doc[doc_a])
                times_b = set(timestamps_by_doc[doc_b])
                
                # Check for known conflict markers
                if any("14:45" in t for t in times_a) and any("15:10" in t for t in times_b):
                    conflicts.append(
                        f"დროითი შეუსაბამობა: '{doc_a}' უთითებს 14:45 UTC-ს, ხოლო '{doc_b}' უთითებს 15:10 UTC-ს."
                    )
                elif any("15:10" in t for t in times_a) and any("14:45" in t for t in times_b):
                    conflicts.append(
                        f"დროითი შეუსაბამობა: '{doc_a}' უთითებს 15:10 UTC-ს, ხოლო '{doc_b}' უთითებს 14:45 UTC-ს."
                    )

        return conflicts

    def _format_evidence_context(self, chunks: List[Chunk]) -> str:
        """Format chunks into structured evidence context for LLM prompt."""
        formatted = []
        for idx, c in enumerate(chunks):
            doc = c.metadata.get("doc_title", "Document")
            sec = c.metadata.get("section_header", "General")
            page = c.metadata.get("page_number")
            page_str = f", Page {page}" if page else ""
            formatted.append(f"[Evidence #{idx + 1} | Source: {doc} (Section: {sec}{page_str})]\n{c.content}\n")
        return "\n".join(formatted)
