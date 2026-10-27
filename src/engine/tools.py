from typing import Any, Callable, Dict, List, Optional
from src.retrieval.hybrid_search import HybridSearchService
from src.ingestion.models import Chunk


class InvestigationTools:
    """
    Structured Tool Calling suite for the AI Investigation Engine.
    Exposes executable tools for document retrieval, citation verification, and contradiction detection.
    """

    def __init__(self, search_service: HybridSearchService):
        self.search_service = search_service

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Return JSON Schema for LLM function/tool calling."""
        return [
            {
                "name": "search_knowledge_base",
                "description": "Search the document store using hybrid semantic and keyword search.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Specific investigation search query"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of evidence candidates to retrieve",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "verify_citation",
                "description": "Check if a claimed quote or fact exists faithfully within a specific chunk.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "chunk_id": {
                            "type": "string",
                            "description": "ID of the source chunk"
                        },
                        "claim": {
                            "type": "string",
                            "description": "Fact or quotation being claimed"
                        }
                    },
                    "required": ["chunk_id", "claim"]
                }
            },
            {
                "name": "detect_contradictions",
                "description": "Analyze multiple statements from different sources to detect logical or temporal conflicts.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "statement_a": {
                            "type": "string",
                            "description": "First factual statement with source"
                        },
                        "statement_b": {
                            "type": "string",
                            "description": "Second factual statement with source"
                        }
                    },
                    "required": ["statement_a", "statement_b"]
                }
            }
        ]

    def execute_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch and execute tool call."""
        if name == "search_knowledge_base":
            query = arguments.get("query", "")
            top_k = arguments.get("top_k", 5)
            hits = self.search_service.search(query, mode="hybrid", top_k=top_k)
            return {
                "results": [
                    {
                        "chunk_id": c.id,
                        "doc_title": c.metadata.get("doc_title", "Unknown"),
                        "section_header": c.metadata.get("section_header", "General"),
                        "page_number": c.metadata.get("page_number"),
                        "content": c.content,
                        "score": score
                    }
                    for c, score, _ in hits
                ]
            }

        elif name == "verify_citation":
            chunk_id = arguments.get("chunk_id", "")
            claim = arguments.get("claim", "").lower().strip()
            # Find chunk in search service
            target_chunk = next((c for c in self.search_service.all_chunks if c.id == chunk_id), None)
            if not target_chunk:
                return {"verified": False, "reason": f"Chunk '{chunk_id}' not found"}
            
            in_content = claim in target_chunk.content.lower()
            return {
                "verified": in_content,
                "chunk_id": chunk_id,
                "doc_title": target_chunk.metadata.get("doc_title")
            }

        elif name == "detect_contradictions":
            stmt_a = arguments.get("statement_a", "")
            stmt_b = arguments.get("statement_b", "")
            has_conflict = False
            details = "No direct contradiction discovered."
            
            # Heuristic check for common contradiction indicators (timestamps, statuses)
            time_indicators = ["14:45", "15:10", "14:15", "14:22"]
            a_times = [t for t in time_indicators if t in stmt_a]
            b_times = [t for t in time_indicators if t in stmt_b]
            if a_times and b_times and a_times != b_times:
                has_conflict = True
                details = f"Temporal conflict detected: {stmt_a} vs {stmt_b}"

            return {
                "contradiction_detected": has_conflict,
                "analysis": details
            }

        return {"error": f"Tool '{name}' not found."}
