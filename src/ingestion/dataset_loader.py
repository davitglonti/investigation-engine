import json
import os
from pathlib import Path
from typing import Dict, Generator, List, Optional
from src.ingestion.models import Document
from src.core.logger import get_logger

logger = get_logger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"


class DatasetLoader:
    """
    Manages large-scale dataset ingestion (50,000+ documents).
    Uses HotpotQA / 2WikiMultihopQA style multi-hop corpus with full source attribution.
    """

    CORPUS_FILE = DATA_DIR / "corpus_50k.jsonl"

    @classmethod
    def load_or_generate_corpus(
        cls,
        target_count: int = 50000,
        use_huggingface: bool = True
    ) -> Path:
        """
        Ensures a corpus of at least target_count (default 50,000) documents exists.
        Saves locally in JSONL format for reproducible benchmarking.
        """
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        if cls.CORPUS_FILE.exists():
            existing_count = cls.count_documents(cls.CORPUS_FILE)
            if existing_count >= target_count:
                logger.info(f"Using existing cached corpus with {existing_count} documents.")
                return cls.CORPUS_FILE

        logger.info(f"Preparing 50,000-document investigation corpus at {cls.CORPUS_FILE}...")
        
        # Try HuggingFace HotpotQA first
        success = False
        if use_huggingface:
            try:
                success = cls._download_hotpotqa(target_count)
            except Exception as e:
                logger.warning(f"Failed to load from HuggingFace directly ({e}). Falling back to generator.")

        if not success:
            cls._generate_synthetic_investigation_corpus(target_count)

        total = cls.count_documents(cls.CORPUS_FILE)
        logger.info(f"Corpus ready! Total documents: {total}")
        return cls.CORPUS_FILE

    @classmethod
    def _download_hotpotqa(cls, target_count: int) -> bool:
        """Download paragraphs from HotpotQA multi-hop benchmark."""
        logger.info("Attempting to load HotpotQA paragraphs via `datasets`...")
        from datasets import load_dataset
        
        # Load validation/distractor split which is fast and rich
        ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train", streaming=True)
        
        doc_count = 0
        seen_titles = set()

        with open(cls.CORPUS_FILE, "w", encoding="utf-8") as out_f:
            for item in ds:
                context = item.get("context", {})
                titles = context.get("title", [])
                sentences_list = context.get("sentences", [])

                for title, sentences in zip(titles, sentences_list):
                    if title in seen_titles:
                        continue
                    seen_titles.add(title)
                    
                    full_text = " ".join(sentences).strip()
                    if not full_text:
                        continue

                    doc_entry = {
                        "id": f"hotpot_{doc_count}",
                        "title": title,
                        "content": full_text,
                        "file_type": "txt",
                        "metadata": {
                            "source": "hotpot_qa_multihop",
                            "sentence_count": len(sentences)
                        }
                    }
                    out_f.write(json.dumps(doc_entry, ensure_ascii=False) + "\n")
                    doc_count += 1

                    if doc_count >= target_count:
                        break

                if doc_count >= target_count:
                    break

        return doc_count >= target_count

    @classmethod
    def _generate_synthetic_investigation_corpus(cls, target_count: int) -> None:
        """
        Generates realistic enterprise IT post-mortems, change records, and architectural docs.
        Guarantees reproducible 50,000 records if network access to HF is restricted.
        """
        logger.info(f"Generating reproducible corpus of {target_count} investigation documents...")

        services = [
            "PaymentGateway", "AuthService", "UserRegistry", "BillingEngine", "NotificationHub",
            "KafkaBroker", "OrderProcessor", "InventoryManager", "AuditLogService", "SessionStore",
            "DataWarehouse", "ReportingPipeline", "CacheCluster", "ApiGateway", "WebhookDispatcher"
        ]
        incident_types = [
            "Connection Pool Exhaustion", "Memory Leak after Deployment", "DNS Resolution Failure",
            "TLS Certificate Expiration", "Cascading Retry Storm", "Deadlock on Order Table",
            "Redis Eviction Spike", "Kafka Rebalance Latency", "Rate Limiter Misconfiguration"
        ]
        severities = ["P1-Critical", "P2-High", "P3-Medium", "P4-Low"]
        months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October"]

        with open(cls.CORPUS_FILE, "w", encoding="utf-8") as out_f:
            for idx in range(target_count):
                svc = services[idx % len(services)]
                incident = incident_types[idx % len(incident_types)]
                sev = severities[idx % len(severities)]
                month = months[idx % len(months)]
                year = 2024 + (idx // 10000)
                version = f"v{1 + (idx % 8)}.{idx % 20}.{idx % 10}"

                title = f"{svc} Incident Report #{idx + 1000} - {month} {year}"
                text = (
                    f"Document ID: INC-{idx + 1000}. Service Affected: {svc}. "
                    f"Timestamp: {month} {1 + (idx % 28)}, {year} at {10 + (idx % 12)}:{idx % 60:02d} UTC. "
                    f"Severity Level: {sev}. Incident Cause: {incident}. "
                    f"Triggered by deployment release {version}. "
                    f"Telemetry metrics recorded error rate surge to {5 + (idx % 45)}% on worker node cluster {idx % 16}. "
                    f"Resolution: Engineering team initiated rollback to {version}-stable. "
                    f"Mitigation completed in {15 + (idx % 120)} minutes."
                )

                doc_entry = {
                    "id": f"doc_corp_{idx:05d}",
                    "title": title,
                    "content": text,
                    "file_type": "txt",
                    "metadata": {
                        "service": svc,
                        "incident_type": incident,
                        "severity": sev,
                        "month": month,
                        "year": year,
                        "source": "enterprise_post_mortems"
                    }
                }
                out_f.write(json.dumps(doc_entry, ensure_ascii=False) + "\n")

    @classmethod
    def count_documents(cls, file_path: Path) -> int:
        if not file_path.exists():
            return 0
        count = 0
        with open(file_path, "r", encoding="utf-8") as f:
            for _ in f:
                count += 1
        return count

    @classmethod
    def stream_documents(
        cls,
        file_path: Optional[Path] = None,
        offset: int = 0,
        limit: Optional[int] = 50,
        search_query: Optional[str] = None
    ) -> Generator[Dict, None, None]:
        """Stream documents lazily with offset, limit, and search filter."""
        path = file_path or cls.CORPUS_FILE
        if not path.exists():
            return

        skipped = 0
        emitted = 0
        query_lower = search_query.lower() if search_query else None

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                
                # Filter if search query provided
                if query_lower:
                    searchable = f"{record.get('title', '')} {record.get('content', '')}".lower()
                    if query_lower not in searchable:
                        continue

                # Handle offset
                if skipped < offset:
                    skipped += 1
                    continue

                yield record
                emitted += 1
                if limit and emitted >= limit:
                    break


def main():
    """Standalone CLI entry point for 50,000+ dataset batch processing."""
    import argparse
    import time
    from src.ingestion.models import Document
    from src.ingestion.chunkers.recursive_chunker import RecursiveCharacterChunker
    from src.retrieval.hybrid_search import HybridSearchService

    parser = argparse.ArgumentParser(description="AI Investigation Engine 50k Dataset Ingestion CLI")
    parser.add_argument("--dataset", choices=["hotpot_qa", "synthetic_incidents"], default="hotpot_qa", help="Dataset source")
    parser.add_argument("--target-records", type=int, default=50000, help="Number of records to process (default: 50000)")
    parser.add_argument("--batch-size", type=int, default=250, help="Batch size for vector ingestion (default: 250)")
    parser.add_argument("--index", action="store_true", help="Index into Vector Store and BM25")

    args = parser.parse_args()

    print(f"[Dataset Loader CLI] Starting ingestion for {args.target_records:,} records ({args.dataset})...")
    start_time = time.time()

    # Step 1: Ensure 50,000 corpus file exists
    corpus_file = DatasetLoader.load_or_generate_corpus(
        target_count=args.target_records,
        use_huggingface=(args.dataset == "hotpot_qa")
    )
    total_docs = DatasetLoader.count_documents(corpus_file)
    print(f"[OK] Verified corpus file at: {corpus_file} ({total_docs:,} total documents)")

    # Step 2: Batch indexing if requested
    if args.index:
        search_service = HybridSearchService()
        chunker = RecursiveCharacterChunker(chunk_size=600, chunk_overlap=80)
        batch = []
        indexed_count = 0

        print(f"[INDEX] Streaming and indexing in batches of {args.batch_size}...")
        for doc_data in DatasetLoader.stream_documents(corpus_file, limit=args.target_records):
            doc = Document(
                id=doc_data["id"],
                title=doc_data["title"],
                content=doc_data["content"],
                file_type=doc_data.get("file_type", "txt"),
                metadata=doc_data.get("metadata", {})
            )
            chunks = chunker.chunk(doc)
            batch.extend(chunks)

            if len(batch) >= args.batch_size:
                search_service.index_chunks(batch)
                indexed_count += len(batch)
                batch = []
                elapsed = time.time() - start_time
                throughput = indexed_count / elapsed if elapsed > 0 else 0
                print(f"   Indexed {indexed_count:,} chunks | Throughput: {throughput:.1f} chunks/sec")

        if batch:
            search_service.index_chunks(batch)
            indexed_count += len(batch)

        print(f"[SUCCESS] Successfully indexed {indexed_count:,} chunks into hybrid retrieval engine!")

    elapsed_total = time.time() - start_time
    print(f"[DONE] Ingestion pipeline finished in {elapsed_total:.2f}s.")


if __name__ == "__main__":
    main()
