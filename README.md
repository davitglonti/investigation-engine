# 🔍 Applied AI – AI Investigation Engine

> Intelligent Multi-Hop Investigation & Reasoning System for Complex Document Analysis

[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Architecture](https://img.shields.io/badge/architecture-Agentic_RAG-orange.svg)](#architecture)

---

## 📌 Overview

**AI Investigation Engine** is an advanced artificial intelligence system built to investigate, correlate, and analyze large-scale, multi-source document collections.

Unlike standard conversational "Chat with PDF" wrappers that retrieve single top-ranked chunks, this engine operates as an **investigative multi-step reasoning agent**:
- Correlates facts across disjoint documents (Multi-Hop QA).
- Detects contradictions and conflicting reports.
- Enforces strict groundedness and transparent source attribution (citations with snippet & line references).
- Refuses to hallucinate when evidence is absent or insufficient.

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    subgraph Ingestion ["1. Document Ingestion & Storage"]
        RawDocs["Multi-Format Input\n(PDF, Markdown, TXT, JSON)"] --> Parser["Document Parsers\n& Metadata Extractor"]
        Parser --> Deduplicator["MD5 / Content Deduplication"]
        Deduplicator --> Chunking["Dual Chunking Strategies\n(Recursive vs Semantic)"]
        Chunking --> DocStore["Document Store (CRUD)"]
    end

    subgraph Indexing ["2. Dual-Engine Indexing"]
        DocStore --> DenseVector["Dense Embeddings\n(Vector DB: Chroma / Qdrant)"]
        DocStore --> SparseBM25["Sparse Keyword Index\n(BM25)"]
    end

    subgraph HybridEngine ["3. Algorithmic Hybrid Retrieval"]
        Query["User Investigation Query"] --> SparseBM25
        Query --> DenseVector
        SparseBM25 --> CustomRRF["Custom Algorithmic Fusion\n(Reciprocal Rank Fusion + Re-scoring)"]
        DenseVector --> CustomRRF
        CustomRRF --> RankedContext["Top-K Evidence Candidates"]
    end

    subgraph AgenticLoop ["4. Agentic Investigation & Reasoning"]
        RankedContext --> Agent["Investigation Agent\n(Multi-Hop Reasoning Loop)"]
        Agent <--> Tools["Tool Calling\n- search_documents\n- verify_citation\n- check_contradiction"]
        Agent --> Decision{"Evidence Sufficient?"}
        Decision -- No / Conflicting --> ConflictAbstain["Abstention / Conflict Flagging"]
        Decision -- Yes --> GroundedSynthesis["Grounded Synthesis & Citations"]
    end

    subgraph Presentation ["5. Delivery & Interfaces"]
        ConflictAbstain --> FastAPI["FastAPI Backend"]
        GroundedSynthesis --> FastAPI
        FastAPI --> StreamlitUI["Interactive Web UI\n(Evidence Explorer & Citations)"]
    end
```

---

## 🚀 Key Features

| Capability | Implementation |
|---|---|
| **Multi-Format Ingestion** | Native extraction for PDF (`pypdf`), Markdown, Plain Text & JSON with metadata extraction. |
| **Dual Chunking** | Comparison of *Recursive Character Chunking* vs *Semantic/Section-based Chunking*. |
| **Hybrid Search** | Combines Dense Semantic Retrieval with Sparse BM25 via custom **Reciprocal Rank Fusion (RRF)**. |
| **Investigation Agent** | Structured Tool Calling (`search_documents`, `verify_citation`, `detect_contradiction`). |
| **Safety & Truthfulness** | Built-in contradiction detection and reliable abstention against ungrounded hallucinations. |
| **Automated Evaluation** | Quantitative benchmark suite testing 40 diverse scenarios across Recall@K, MRR, Correctness, and Citation Accuracy. |

---

## ⚙️ Installation & Quickstart

### 1. Prerequisites
- Python 3.11+
- Git

### 2. Clone Repository
```bash
git clone https://github.com/your-username/ai-investigation-engine.git
cd ai-investigation-engine
```

### 3. Virtual Environment
```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Environment Variables
Copy `.env.example` to `.env` and fill in your API keys:
```bash
cp .env.example .env
```

---

## 💻 Running the Services

### Start FastAPI Backend
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```
Interactive API docs available at: `http://localhost:8000/docs`

### Start Web UI (Streamlit)
```bash
streamlit run src/ui/app.py
```
Access the application at: `http://localhost:8501`

---

## 🧪 Evaluation Suite

To run the automated 40-case evaluation benchmark:
```bash
pytest tests/ -v
python -m src.evaluation.evaluator --config configs/eval_config.yaml
```
Results and comparative metrics are saved to `docs/EVALUATION_REPORT.md`.
