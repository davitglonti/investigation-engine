import re
from typing import Any, Dict, List, Set


class EvaluationMetrics:
    """Quantitative metric calculations for RAG and Investigation Engine evaluation."""

    @staticmethod
    def recall_at_k(retrieved_docs: List[str], ground_truth_docs: List[str], k: int = 5) -> float:
        """Recall@K: Fraction of relevant ground truth documents retrieved in top-K."""
        if not ground_truth_docs:
            return 1.0  # Void query with no relevant documents
        
        top_k_retrieved = set(retrieved_docs[:k])
        gt_set = set(ground_truth_docs)
        hits = len(top_k_retrieved & gt_set)
        return hits / len(gt_set)

    @staticmethod
    def reciprocal_rank(retrieved_docs: List[str], ground_truth_docs: List[str]) -> float:
        """Reciprocal Rank: 1 / rank of first relevant document."""
        if not ground_truth_docs:
            return 1.0
        
        gt_set = set(ground_truth_docs)
        for rank, doc in enumerate(retrieved_docs, start=1):
            if doc in gt_set:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def citation_accuracy(citations: List[Dict[str, Any]], ground_truth_docs: List[str]) -> float:
        """Fraction of generated citations that match ground truth documents."""
        if not ground_truth_docs:
            # If no ground truth docs (unanswerable), citation accuracy is 1.0 if 0 citations emitted
            return 1.0 if not citations else 0.0

        if not citations:
            return 0.0

        gt_set = set(ground_truth_docs)
        valid_citations = sum(1 for c in citations if c.get("doc_title") in gt_set)
        return valid_citations / len(citations)

    @staticmethod
    def answer_correctness(predicted_answer: str, expected_answer: str, category: str) -> float:
        """
        Calculates correctness score [0.0, 1.0].
        Handles ABSTAIN matches, contradiction summaries, and semantic key phrase coverage.
        """
        pred_lower = predicted_answer.lower()
        exp_lower = expected_answer.lower()

        # Handle abstention cases
        if expected_answer == "ABSTAIN":
            abstain_signals = ["არ მოიძებნა", "არასაკმარისი", "თავის შეკავება", "abstain", "insufficient", "not found"]
            if any(signal in pred_lower for signal in abstain_signals):
                return 1.0
            return 0.0

        # Normalization and keyword overlap
        exp_words = set(re.findall(r"\w{3,}", exp_lower))
        if not exp_words:
            return 1.0

        matched_words = sum(1 for w in exp_words if w in pred_lower)
        coverage = matched_words / len(exp_words)

        # Contradiction bonus
        if category == "contradiction_detection":
            if any(w in pred_lower for w in ["შეუსაბამობა", "conflict", "contradiction", "discrepancy", "disagree"]):
                coverage = min(1.0, coverage + 0.3)

        return min(1.0, round(coverage, 2))
