⚖️ Nyaya RAG Engine
Project Goal
This project implements a high-performance Semantic Search Engine for a massive corpus of legal judgments, providing fast, natural language query capabilities for legal analysis and retrieval.


Key Technical Architecture->
Component,Technology / Role,Scalability & Efficiency

Application Logic,Streamlit (Python),"Provides an interactive, pure-Python frontend for demonstration and real-time query display."
Data Retrieval,s3fs + @st.cache_resource,Critical: The app.py uses s3fs to stream the index from S3 and the @st.cache_resource decorator to load the entire 3 GB index into memory only once per server instance. This minimizes latency and cost.
Core Functionality,RAG/Vector Index (Implied by index files),Enables queries based on semantic meaning rather than just keyword matching.

📁 File Structure
.
├── Data/
│   ├── index_to_chunk_map.json     # Metadata mapping index vectors to documents
│   └── judgments.index             # The 3GB vector index (e.g., FAISS/HNSW)
├── app.py                            # Core Streamlit application and loading logic
└── requirements.txt                  # Essential Python package dependencies

ocal Execution
To run the project locally (for demonstration):

Install Dependencies: pip install -r requirements.txt

Set Environment Variables: Set your S3 credentials as local environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY).

Run: streamlit run app.py
