import os
import re
import json
import time
import gc
import streamlit as st
import requests
import asyncio
import httpx
import numpy as np
import faiss
from urllib.parse import quote_plus
from sentence_transformers import SentenceTransformer

# -----------------------------
# Config / Constants
# -----------------------------
st.set_page_config(layout="wide", page_title="Judicium AI — Nyaya RAG Engine")

DATA_DIR = "./Data"
INDEX_FILENAME = "judgments.index"
MAP_FILENAME = "index_to_chunk_map.json"
DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"

# Read secrets (set these in Streamlit Cloud: Settings → Secrets)
# GDRIVE_INDEX_ID: file ID or full share URL for judgments.index
# GDRIVE_MAP_ID: file ID or full share URL for index_to_chunk_map.json
# GEMINI_API_KEY: your Gemini / Google generative API key
# GEMINI_MODEL (optional): model name, e.g. "gemini-2.5-flash"
GDRIVE_INDEX = st.secrets.get("GDRIVE_INDEX_ID") if "GDRIVE_INDEX_ID" in st.secrets else os.environ.get("GDRIVE_INDEX_ID")
GDRIVE_MAP = st.secrets.get("GDRIVE_MAP_ID") if "GDRIVE_MAP_ID" in st.secrets else os.environ.get("GDRIVE_MAP_ID")
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") if "GEMINI_API_KEY" in st.secrets else os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = st.secrets.get("GEMINI_MODEL") if "GEMINI_MODEL" in st.secrets else os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# Ensure Data dir exists
os.makedirs(DATA_DIR, exist_ok=True)

# -----------------------------
# Helper: Google Drive downloader
# -----------------------------
def extract_gdrive_id(url_or_id: str) -> str:
    if not url_or_id:
        return ""
    if "drive.google.com" in url_or_id:
        m = re.search(r"/d/([a-zA-Z0-9_-]+)", url_or_id)
        if m:
            return m.group(1)
        m = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url_or_id)
        if m:
            return m.group(1)
    return url_or_id

def download_from_gdrive(file_id_or_url: str, dest_path: str, chunk_size: int = 32768):
    """
    Downloads a file from Google Drive to dest_path. Works for large files (handles confirm tokens).
    If file exists locally, it returns immediately.
    """
    if not file_id_or_url:
        raise ValueError("No Google Drive file id/url provided.")
    if os.path.exists(dest_path):
        st.info(f"Using existing file: {dest_path}")
        return

    file_id = extract_gdrive_id(file_id_or_url)
    base_url = "https://drive.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(base_url, params={"id": file_id}, stream=True)
    token = None
    for k, v in session.cookies.items():
        if k.startswith("download_warning"):
            token = v
            break

    if token:
        response = session.get(base_url, params={"id": file_id, "confirm": token}, stream=True)

    # Fallback: search for confirm token in HTML
    if "Content-Disposition" not in response.headers and response.status_code == 200:
        txt = response.text
        m = re.search(r"confirm=([0-9A-Za-z-_]+)&", txt)
        if m:
            token = m.group(1)
            response = session.get(base_url, params={"id": file_id, "confirm": token}, stream=True)

    response.raise_for_status()

    total = 0
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                total += len(chunk)

    st.info(f"Downloaded {os.path.basename(dest_path)} ({total} bytes)")

# -----------------------------
# Load models & data (cached)
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_all_models_and_data():
    """
    Downloads (if needed) FAISS index + mapping from Google Drive (using secrets),
    then loads the index, mapping and embedding model.
    """
    try:
        st.info("Initializing: checking & downloading data if needed...")

        index_path = os.path.join(DATA_DIR, INDEX_FILENAME)
        map_path = os.path.join(DATA_DIR, MAP_FILENAME)

        # Download from Drive if secrets provided; otherwise assume files are in repo
        if GDRIVE_INDEX:
            st.info("Attempting to download FAISS index from Google Drive...")
            download_from_gdrive(GDRIVE_INDEX, index_path)
        else:
            st.info("GDRIVE_INDEX_ID not set — assuming FAISS index exists in repo at " + index_path)

        if GDRIVE_MAP:
            st.info("Attempting to download mapping JSON from Google Drive...")
            download_from_gdrive(GDRIVE_MAP, map_path)
        else:
            st.info("GDRIVE_MAP_ID not set — assuming mapping JSON exists in repo at " + map_path)

        st.info("Loading FAISS index and mapping from disk...")
        index = faiss.read_index(index_path)

        with open(map_path, "r", encoding="utf-8") as f:
            index_to_chunk_map = json.load(f)

        st.info("Loading embedding model (SentenceTransformer). This may take a while on first run...")
        embedding_model = SentenceTransformer(DEFAULT_EMBED_MODEL)

        st.success("✅ Models and data loaded successfully!")
        return index, index_to_chunk_map, embedding_model

    except Exception as e:
        st.error("--- Application Initialization Failed ---")
        st.error(f"Error loading FAISS or data: {e}")
        # Return None triple so UI can check and show friendly message
        return None, None, None

# -----------------------------
# Retrieval
# -----------------------------
def retrieve_chunks(query: str, index: faiss.Index, embedding_model, top_k: int = 6):
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, top_k)
    return indices[0]

# -----------------------------
# Gemini API (async) with backoff
# -----------------------------
async def generate_response_with_gemini(query: str, context: str):
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set. Please add it to Streamlit secrets.")
    system_prompt = (
        "You are a professional legal expert for 'Judicium AI'. Your sole task is to provide a comprehensive, "
        "clear, and well-structured legal analysis of the user's question, strictly based *only* on the provided "
        "legal precedents and context. Do not use outside knowledge. The answer must be objective and formal."
    )

    user_query = f"""
QUESTION: {query}

CONTEXT (Retrieved Legal Precedents for Synthesis):
---
{context}
---
"""

    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
    }

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

    max_retries = 3
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                result = resp.json()
            # Extract text safely
            text = None
            if isinstance(result, dict):
                # Newer API forms may use 'candidates' or 'outputs' — check both
                if "candidates" in result and result["candidates"]:
                    c = result["candidates"][0]
                    if "content" in c and isinstance(c["content"], list) and c["content"]:
                        text = c["content"][0].get("text")
                elif "outputs" in result and result["outputs"]:
                    # Some responses embed text differently
                    out = result["outputs"][0]
                    if "content" in out and isinstance(out["content"], list) and out["content"]:
                        text = out["content"][0].get("text")
            if not text:
                # Last resort: try to stringify the response
                text = json.dumps(result)[:4000]
            return text
        except httpx.HTTPStatusError as e:
            status = e.response.status_code if e.response is not None else None
            if status in [429, 500, 503] and attempt < max_retries - 1:
                wait = 2 ** attempt
                st.info(f"API server responded {status}. Retrying in {wait}s...")
                await asyncio.sleep(wait)
            else:
                st.error(f"Gemini API HTTP error: {e}")
                raise
        except Exception as e:
            st.error(f"Unexpected error calling Gemini API: {e}")
            raise

    raise RuntimeError("Failed to get response from Gemini after retries.")

# -----------------------------
# Post-process
# -----------------------------
def post_process_answer(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    if not text:
        return ""
    # Capitalize first character and after periods
    text = text[0].upper() + text[1:]
    text = re.sub(r'(?<=\.\s)([a-z])', lambda m: m.group(1).upper(), text)
    return text

# -----------------------------
# UI
# -----------------------------
st.title("⚖️ Nyaya — RAG Engine (Judicium AI)")
st.markdown("Enter a question about Indian Supreme Court case law to get an AI-synthesized, evidence-backed answer.")
st.markdown("---")

# Load on demand (cached)
index, index_to_chunk_map, embedding_model = load_all_models_and_data()

if index is None:
    st.error("Application initialization failed. Please check logs and Streamlit Secrets.")
    st.info("Make sure you set the following in Streamlit Secrets: GDRIVE_INDEX_ID, GDRIVE_MAP_ID, GEMINI_API_KEY (optional GEMINI_MODEL).")
else:
    query = st.text_input("Enter your legal query:", placeholder="e.g., What are the current rules on capital vs revenue receipts from timber sale?")

    if st.button("Generate Answer"):
        if not query:
            st.warning("Please enter a query.")
        else:
            with st.spinner("Analyzing precedents and synthesizing answer..."):
                try:
                    retrieved_indices = retrieve_chunks(query, index, embedding_model, top_k=6)
                    context = ""
                    retrieved_chunks_data = []
                    unique_doc_names = set()

                    for idx in retrieved_indices:
                        # Defensive: some FAISS indexes may return -1 for missing
                        if int(idx) < 0:
                            continue
                        chunk_info = index_to_chunk_map.get(str(int(idx))) or index_to_chunk_map.get(int(idx))
                        if not chunk_info:
                            # try list indexing if map is a list
                            try:
                                chunk_info = index_to_chunk_map[int(idx)]
                            except Exception:
                                chunk_info = None
                        if not chunk_info:
                            continue
                        # accumulate context
                        context += chunk_info.get('chunk_text', '') + "\n\n"
                        retrieved_chunks_data.append(chunk_info)
                        unique_doc_names.add(chunk_info.get('doc_name', f"doc_{idx}"))

                    if not retrieved_chunks_data:
                        st.warning("No relevant chunks were retrieved. Try reformulating your query.")
                    else:
                        # Call Gemini async API
                        try:
                            raw_answer = asyncio.run(generate_response_with_gemini(query, context))
                            final_answer = post_process_answer(raw_answer)
                        except Exception as e:
                            final_answer = f"Error during AI generation: {e}"

                        # Display results
                        st.markdown("### 📜 AI-Synthesized Analysis (Summary)")
                        st.info(final_answer)

                        st.markdown("### 📚 Key Supporting Cases (documents used)")
                        sorted_docs = sorted(list(unique_doc_names))
                        for doc_name in sorted_docs:
                            search_term = os.path.splitext(os.path.basename(doc_name))[0].replace("_", " ")
                            google_search_url = f"https://www.google.com/search?q={quote_plus(search_term)}"
                            st.markdown(f"- **[{doc_name}]({google_search_url})**")

                        st.markdown("### 🔎 Retrieved Chunks (for reference)")
                        for i, chunk in enumerate(retrieved_chunks_data, start=1):
                            st.markdown(f"**Chunk #{i}** — Document: {chunk.get('doc_name', 'unknown')}")
                            st.write(chunk.get('chunk_text', '')[:2000])  # truncate for UI sanity

                        # cleanup a bit
                        gc.collect()

                except Exception as e:
                    st.error(f"Unexpected error during processing: {e}")
