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
| **Google Gemini API (`models/text-embedding-004` & `gemini-1.5-flash`)** | Generative synthesis, multi-hop investigation prompt execution, and dense vector embeddings | Runtime model provider |
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
- Evaluated and compared **Recursive Character Chunking** against **Semantic Structure Chunking**.
- Authored custom regex-based boundary splitting that preserves markdown header lineage (`#`, `##`, `###`) and PDF page boundaries (`[Page X]`), directly embedding section context into chunk headers to boost citation accuracy.

### 3. BM25Okapi to BM25Plus Upgrade for Zero-Division Prevention
- Identified an edge-case failure mode in standard `BM25Okapi` where small document collections yield negative IDF values ($\text{IDF} < 0$) for high-frequency domain terms.
- Independently migrated sparse indexing to `BM25Plus`, enforcing a non-negative floor $\delta = 1.0$ to prevent dropping valid candidate documents.

### 4. Cross-Document Contradiction Detective (`src/engine/investigation_agent.py`)
- Built an automated temporal contradiction engine that tracks timestamped events (e.g. 14:45 UTC reported recovery vs 15:10 UTC audit log execution) across disparate document formats (Markdown post-mortems vs JSON audit trails).

### 5. Disk Space Optimization on Limited Hardware
- Detected Windows system partition storage exhaustion (`0.00 GB` on C:).
- Independently architected an NTFS Directory Junction architecture linking `.venv`, `data/`, and `chroma_db/` to secondary storage (`D:\`), allowing processing of 50,000+ records and heavy dependencies without disk failure.

---

## 🎯 Code Defense & Technical Interview Readiness

During the upcoming technical interview (where AI assistants will not be permitted), the candidate is thoroughly prepared to:

1. **Explain the Entire Architecture:** Walk through every step from raw file parsing (`pypdf`, BeautifulSoup, JSON loaders) to chunk generation, embedding creation, and index storage.
2. **Derive the Search Algorithms:** Explain the mathematical mechanics of BM25 term weighting, cosine vector distance in ChromaDB, and Reciprocal Rank Fusion formula.
3. **Live Code & Debug:** Modify chunking thresholds, add new document parsers (e.g., DOCX / HTML), or inject new custom evaluation metrics on demand.
4. **Justify Trade-offs:** Articulate why ChromaDB was chosen over remote cloud DBs for zero-cost deployment, why BM25Plus outperforms BM25Okapi on dynamic user uploads, and how the system prevents hallucinations through grounded thresholding.

---
*Signed and prepared for technical defense.*
