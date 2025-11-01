# ⚖️ Nyaya RAG Engine

## 🎯 **Project Goal**
This project implements a **high-performance Semantic Search Engine** for a massive corpus of **legal judgments**, enabling **fast, natural language query** capabilities for legal analysis and retrieval.

---

## 🧩 **Key Technical Architecture**

| **Component** | **Technology / Role** | **Scalability & Efficiency** |
|----------------|------------------------|-------------------------------|
| **Application Logic** | **Streamlit (Python)** | Provides an interactive, pure-Python frontend for demonstration and real-time query display. |
| **Data Retrieval** | **s3fs + @st.cache_resource** | **Critical:** Uses `s3fs` to stream the index from **S3** and the `@st.cache_resource` decorator to load the **3GB index** into memory only once per server instance — minimizing latency and cost. |
| **Core Functionality** | **RAG / Vector Index (FAISS or HNSW)** | Enables **semantic meaning-based queries** rather than keyword matching, improving accuracy in legal text retrieval. |

---

