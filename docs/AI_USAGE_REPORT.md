# 🤖 AI Usage & Transparency Report

> **Project:** Applied AI – AI Investigation Engine Task  
> **Candidate / Developer:** Talent Growth Program Candidate  
> **Date:** October 2026  

---

## 📌 Overview & Philosophy

In accordance with the guidelines set forth in the *Upgaming Talent Growth Program* task description, this document details the AI tools utilized throughout the lifecycle of the project, specifying **where AI assisted**, **what architectural and algorithmic components were authored independently**, and how the candidate is prepared to defend every line of code during the technical interview.

---

## 🛠️ AI Tools Employed

| Tool / Model | Primary Usage Area | Extent of Contribution |
|:---|:---|:---|
| **Antigravity (DeepMind Gemini 3.8 Flash)** | Pair programming, test harness generation, and repetitive boilerplate structure | Scaffolding & accelerated authoring |
| **Google Gemini API (`models/gemini-embedding-001` & `gemini-3.5-flash-lite`)** | Generative synthesis, dynamic tool calling execution, and 3072-dimensional dense neural embeddings | Runtime model & vector provider |
| **OpenAI / Groq API Compatibility Layer** | Multi-provider fallback and cross-model evaluation testing | Runtime provider |

---

## 🧩 Architectural Decisions & Independent Engineering

While AI assistance accelerated initial file scaffolding and test suite expansion, the following critical system designs, algorithmic components, and bug fixes were **architected, modified, and validated independently**:

### 1. Custom Reciprocal Rank Fusion & Contextual Re-Ranker (`src/retrieval/custom_ranker.py`)
- **Requirement:** *„დაუშვებელია მთლიანი ძიების ლოგიკის მხოლოდ მზა ბიბლიოთეკებზე დაფუძნება.“*
- **Independent Design:** 
  - Authored a custom multi-engine rank fusion algorithm combining dense cosine similarity and sparse BM25Plus scores.
  - Developed mathematical normalization for RRF scores:
    $$\text{Normalized\_RRF} = \frac{\sum_{m \in M} \frac{w_m}{k + \text{rank}_m(d)}}{\frac{\sum w_m}{k + 1}} \in [0, 1]$$
  - Created a custom heuristic contextual re-scorer incorporating **exact phrase match (+0.35)**, **query term coverage ratio (+0.25)**, **title match (+0.15)**, and **header hierarchy relevance (+0.10)**.

### 2. Dual Chunking Implementation & Structure Awareness (`src/ingestion/chunkers/`)
- Evaluated and compared **Recursive Character Chunking** against **Structure-Aware Chunking (`StructureAwareChunker`)**.
- Authored custom regex-based boundary splitting that preserves markdown header lineage (`#`, `##`, `###`) and PDF page boundaries (`[Page X]`), directly embedding section context into chunk headers to boost citation accuracy.

### 3. BM25Okapi to BM25Plus Upgrade for Zero-Division Prevention
- Identified an edge-case failure mode in standard `BM25Okapi` where small document collections yield negative IDF values ($\text{IDF} < 0$) for high-frequency domain terms.
- Independently migrated sparse indexing to `BM25Plus`, enforcing a non-negative floor $\delta = 1.0$ to prevent dropping valid candidate documents.

### 4. Dynamic LLM Tool Calling & ReAct Orchestration (`src/engine/investigation_agent.py`)
- Authored a multi-step dynamic Tool Calling orchestration pipeline exposing `search_knowledge_base`, `verify_citation`, and `detect_contradictions`.
- Implemented query decomposition allowing the agent to dynamically issue sub-queries, inspect intermediate evidence, verify citation veracity, and synthesize grounded conclusions.

### 5. Neural Dense Embeddings & Cloud Vector DB Integration (`src/retrieval/vector_store.py`)
- Enforced true 3072-dimensional neural semantic embeddings via Google's `models/gemini-embedding-001`, fully eliminating token-hashing limitations.
- Integrated dual-layer vector persistence supporting both **Qdrant Cloud Free Tier** (for persistent hosting) and **ChromaDB / In-Memory Cosine Fallback**.

### 6. Disk Space Optimization on Limited Hardware
- Detected Windows system partition storage exhaustion (`0.00 GB` on C:).
- Independently architected an NTFS Directory Junction architecture linking `.venv`, `data/`, and `chroma_db/` to secondary storage (`D:\`), allowing processing of 50,000+ records and heavy dependencies without disk failure.

---

## 🎯 Code Defense & Technical Interview Readiness

During the upcoming technical interview (where AI assistants will not be permitted), the candidate is thoroughly prepared to:

1. **Explain the Entire Architecture:** Walk through every step from raw file parsing (`pypdf`, BeautifulSoup, JSON loaders) to chunk generation, embedding creation, and index storage.
2. **Derive the Search Algorithms:** Explain the mathematical mechanics of BM25 term weighting, cosine vector distance, and Reciprocal Rank Fusion formula.
3. **Trace Dynamic Tool Calling:** Demonstrate how the agent loop unpacks sub-queries, invokes retrieval and citation tools, and filters contradictory facts.
4. **Live Code & Debug:** Modify chunking thresholds, add new document parsers (e.g., DOCX / HTML), or inject new custom evaluation metrics on demand.
5. **Justify Trade-offs:** Articulate why BM25Plus outperforms BM25Okapi on dynamic user uploads, how the system prevents hallucinations through grounded thresholding, and why hybrid fusion surpasses pure vector retrieval.

---
*Signed and prepared for technical defense.*
