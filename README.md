# ⚖️ Nyaya RAG Engine

**Nyaya RAG Engine** is a specialized Retrieval-Augmented Generation system designed for the legal domain. It automates the ingestion, indexing, and retrieval of legal documents (case files, statutes, contracts) to provide accurate, context-aware answers with citations.

This project implements a complete end-to-end LLM pipeline, moving from raw data ingestion to a production-ready UI.

![Project Workflow](Screenshot%202025-11-18%20at%2011.51.21%E2%80%AFPM.png)
*(Workflow architecture: Ingestion → Embedding → Retrieval → Generation → Orchestration → Evaluation → Deployment → UI)*

---

## 🚀 Key Features

* **Legal Data Ingestion:** Parsers for PDF, DOCX, and TXT legal texts.
* **Semantic Search:** Vector-based retrieval to find relevant case laws and clauses.
* **Citations:** The generation layer references specific source documents to minimize hallucinations.
* **Evaluation Loop:** Automated testing for answer faithfulness and context relevancy.
* **Containerized Deployment:** Docker support for easy productionisation.

---

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **Orchestration:** LangChain / LlamaIndex
* **LLM:** OpenAI GPT-4 / Llama 3 (via Groq or Ollama)
* **Vector Database:** ChromaDB / Pinecone / Qdrant
* **Backend API:** FastAPI
* **Frontend:** Streamlit / React
* **Containerization:** Docker

---

## 📂 Repository Structure

```bash
Nyaya-RAG-Engine/
├── data/                   # Raw legal documents for ingestion
├── src/
│   ├── ingestion/          # Step 1 & 2: Data loading & Vector indexing
│   ├── retrieval/          # Step 3: Semantic search logic
│   ├── generation/         # Step 4: LLM Prompting & Answer formulation
│   ├── orchestration/      # Step 5: Chains and Agent logic
│   └── evaluation/         # Step 6: RAGAS/TruLens eval scripts
├── ui/                     # Step 8: Frontend application
├── docker-compose.yml      # Step 7: Deployment config
├── requirements.txt        # Python dependencies
└── README.md
