import pytest
from src.ingestion.document_store import DocumentStore
from src.ingestion.deduplication import Deduplicator
from src.ingestion.models import Document


def test_deduplicator_exact():
    dedup = Deduplicator()
    doc1 = Document(
        id="doc_1",
        title="incident_1.txt",
        content="Service outage reported on node 4 at 10:00 AM.",
        file_type="txt",
        content_hash="hash1"
    )
    doc2 = Document(
        id="doc_2",
        title="incident_1_copy.txt",
        content="Service outage reported on node 4 at 10:00 AM.",
        file_type="txt",
        content_hash="hash1"
    )

    dedup.register_document(doc1)
    is_dup, match_id, match_type = dedup.check_duplicate(doc2)

    assert is_dup is True
    assert match_id == "doc_1"
    assert match_type == "exact"


def test_document_store_crud():
    store = DocumentStore()

    # 1. CREATE (Add)
    content = "# Incident 500\nPayment service failed due to connection timeout."
    doc, was_added = store.add_document(content, "incident_500.md", strategy="semantic")
    assert was_added is True
    assert doc.id in store.documents
    assert len(doc.chunks) > 0

    # Duplicate check
    _, dup_added = store.add_document(content, "incident_500_copy.md", strategy="semantic")
    assert dup_added is False  # filtered as duplicate

    # 2. READ (Get & List)
    retrieved = store.get_document(doc.id)
    assert retrieved is not None
    assert retrieved.title == "incident_500.md"
    
    doc_list = store.list_documents()
    assert len(doc_list) == 1
    assert doc_list[0]["title"] == "incident_500.md"

    # 3. UPDATE
    updated_content = "# Incident 500 (Resolved)\nPayment service restored after pool expansion."
    updated_doc = store.update_document(doc.id, updated_content, "incident_500.md")
    assert updated_doc is not None
    assert "Resolved" in updated_doc.content

    # 4. DELETE
    deleted = store.delete_document(doc.id)
    assert deleted is True
    assert store.get_document(doc.id) is None
    assert len(store.list_documents()) == 0
