# app.py
import os
import re
import json
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
# Page config
# -----------------------------
st.set_page_config(layout="wide", page_title="Judicium AI — Nyaya RAG Engine")

# -----------------------------
# Constants & secrets
# -----------------------------
DATA_DIR = "./Data"
INDEX_FILENAME = "judgments.index"
MAP_FILENAME = "index_to_chunk_map.json"
DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"

# Secrets (Streamlit Secrets or environment fallback)
GDRIVE_INDEX_ID = st.secrets.get("GDRIVE_INDEX_ID") if "GDRIVE_INDEX_ID" in st.secrets else os.environ.get("GDRIVE_INDEX_ID")
GDRIVE_MAP_ID = st.secrets.get("GDRIVE_MAP_ID") if "GDRIVE_MAP_ID" in st.secrets else os.environ.get("GDRIVE_MAP_ID")
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") if "GEMINI_API_KEY" in st.secrets else os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = st.secrets.get("GEMINI_MODEL") if "GEMINI_MODEL" in st.secrets else os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

os.makedirs(DATA_DIR, exist_ok=True)

# -----------------------------
# Robust Google Drive downloader
# -----------------------------
def extract_gdrive_id(url_or_id: str) -> str:
    """Return file id from either a full google drive url or raw id."""
    if not url_or_id:
        return ""
    if "drive.google.com" in url_or_id:
        m = re.search(r"/d/([^/]+)/", url_or_id)
        if m:
            return m.group(1)
        m = re.search(r"[?&]id=([^&]+)", url_or_id)
        if m:
            return m.group(1)
    return url_or_id

def download_file_from_gdrive(file_id_or_url: str, dest_path: str):
    """
    Reliable GDrive downloader.
    - Handles large-files confirm token flow
    - Detects HTML preview and raises with debug info
    """
    if not file_id_or_url:
        raise ValueError("No Google Drive id/url provided for downloading.")

    file_id = extract_gdrive_id(file_id_or_url)
    if os.path.exists(dest_path):
        st.info(f"Using cached file: {dest_path} ({os.path.getsize(dest_path)} bytes)")
        return

    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()

    # initial request
    response = session.get(URL, params={'id': file_id}, stream=True)

    # check for confirm token in cookies (common for large files)
    token = None
    for key, val in session.cookies.items():
        if key.startswith("download_warning") or key.startswith("download"):
            token = val
            break

    if token:
        response = session.get(URL, params={'id': file_id, 'confirm': token}, stream=True)

    # If response is still HTML, try a forced confirm parameter
    ctype = response.headers.get("Content-Type", "").lower()
    if "text/html" in ctype:
        response = session.get(f"https://drive.google.com/uc?export=download&confirm=1&id={file_id}", stream=True)

    # final safety: if HTML still returned, save debug and raise
    final_ctype = response.headers.get("Content-Type", "").lower()
    resp_head = (response.text[:600] if "text/html" in final_ctype else "")
    if "text/html" in final_ctype or response.headers.get("Content-Disposition", "") == "":
        # Save debug HTML for inspection
        debug_path = os.path.join(DATA_DIR, f"debug_drive_{os.path.basename(dest_path)}.html")
        try:
            with open(debug_path, "w", encoding="utf-8") as fh:
                fh.write(response.text)
            st.error(f"Google Drive returned HTML (preview or permission page). Saved debug HTML at: {debug_path}")
            st.code(resp_head)
        except Exception:
            st.error("Google Drive returned HTML and debug save failed.")
            st.code(resp_head)
        raise RuntimeError("Drive returned HTML (preview or permission page). Ensure file is public and uploaded as raw file (not Google Doc).")

    # At this point we expect binary content
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    total = 0
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)
                total += len(chunk)
    st.success(f"Downloaded {os.path.basename(dest_path)} ({total} bytes)")

# -----------------------------
# Load models & data (cached)
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_all_models_and_data():
    """
    Download (if needed) the FAISS index + mapping and load the index + embedding model.
    """
    try:
        st.info("Initialization: checking & downloading data if needed...")

        index_path = os.path.join(DATA_DIR, INDEX_FILENAME)
        map_path = os.path.join(DATA_DIR, MAP_FILENAME)

        if GDRIVE_INDEX_ID:
            st.info("Attempting to download FAISS index from Google Drive...")
            download_file_from_gdrive(GDRIVE_INDEX_ID, index_path)
        else:
            st.info("GDRIVE_INDEX_ID not set — assuming index exists locally at " + index_path)

        if GDRIVE_MAP_ID:
            st.info("Attempting to download mapping JSON from Google Drive...")
            download_file_from_gdrive(GDRIVE_MAP_ID, map_path)
        else:
            st.info("GDRIVE_MAP_ID not set — assuming mapping exists locally at " + map_path)

        st.info("Loading FAISS index from disk...")
        index = faiss.read_index(index_path)

        st.info("Loading mapping JSON...")
        with open(map_path, "r", encoding="utf-8") as f:
            index_to_chunk_map = json.load(f)

        st.info("Loading SentenceTransformer embedding model (may take time on first run)...")
        embedding_model = SentenceTransformer(DEFAULT_EMBED_MODEL)

        st.success("✅ Models and data loaded successfully!")
        return index, index_to_chunk_map, embedding_model

    except Exception as e:
        st.error("--- Application Initialization Failed ---")
        st.error(f"Error loading FAISS or data: {e}")
        # If debug html file exists, show path
        debug_files = [p for p in os.listdir(DATA_DIR) if p.startswith("debug_drive_")]
        if debug_files:
            st.error(f"Debug HTML files: {debug_files}")
        return None, None, None

# -----------------------------
# Retrieval helper
# -----------------------------
def retrieve_chunks(query: str, index: faiss.Index, embedding_model, top_k: int = 6):
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, top_k)
    return indices[0]

# -----------------------------
# Gemini async call (with retries)
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
                if "candidates" in result and result["candidates"]:
                    c = result["candidates"][0]
                    if "content" in c and isinstance(c["content"], list) and c["content"]:
                        text = c["content"][0].get("text")
                elif "outputs" in result and result["outputs"]:
                    out = result["outputs"][0]
                    if "content" in out and isinstance(out["content"], list) and out["content"]:
                        text = out["content"][0].get("text")
            if not text:
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
# Post-process helper
# -----------------------------
def post_process_answer(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    if not text:
        return ""
    text = text[0].upper() + text[1:]
    text = re.sub(r'(?<=\.\s)([a-z])', lambda m: m.group(1).upper(), text)
    return text

# -----------------------------
# UI
# -----------------------------
st.title("⚖️ Nyaya — RAG Engine (Judicium AI)")
st.markdown("Enter a question about Indian Supreme Court case law to get an AI-synthesized, evidence-backed answer.")
st.markdown("---")

index, index_to_chunk_map, embedding_model = load_all_models_and_data()

if index is None:
    st.error("Application initialization failed. Check Streamlit Secrets and logs.")
    st.info("Required secrets: GDRIVE_INDEX_ID, GDRIVE_MAP_ID, GEMINI_API_KEY (optional GEMINI_MODEL).")
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
                        if int(idx) < 0:
                            continue
                        # index_to_chunk_map may be list-indexed or dict keyed by str(int)
                        key1 = str(int(idx))
                        chunk_info = None
                        if isinstance(index_to_chunk_map, dict):
                            chunk_info = index_to_chunk_map.get(key1) or index_to_chunk_map.get(int(idx))
                        else:
                            try:
                                chunk_info = index_to_chunk_map[int(idx)]
                            except Exception:
                                chunk_info = None
                        if not chunk_info:
                            continue
                        context += chunk_info.get("chunk_text", "") + "\n\n"
                        retrieved_chunks_data.append(chunk_info)
                        unique_doc_names.add(chunk_info.get("doc_name", f"doc_{idx}"))

                    if not retrieved_chunks_data:
                        st.warning("No relevant chunks retrieved. Try reformulating your query.")
                    else:
                        try:
                            raw_answer = asyncio.run(generate_response_with_gemini(query, context))
                            final_answer = post_process_answer(raw_answer)
                        except Exception as e:
                            final_answer = f"Error during AI generation: {e}"

                        st.markdown("### 📜 AI-Synthesized Analysis (Summary)")
                        st.info(final_answer)

                        st.markdown("### 📚 Key Supporting Cases (documents used)")
                        for doc_name in sorted(list(unique_doc_names)):
                            search_term = os.path.splitext(os.path.basename(doc_name))[0].replace("_", " ")
                            google_search_url = f"https://www.google.com/search?q={quote_plus(search_term)}"
                            st.markdown(f"- **[{doc_name}]({google_search_url})**")

                        st.markdown("### 🔎 Retrieved Chunks (for reference)")
                        for i, chunk in enumerate(retrieved_chunks_data, start=1):
                            st.markdown(f"**Chunk #{i}** — Document: {chunk.get('doc_name', 'unknown')}")
                            st.write(chunk.get("chunk_text", "")[:2000])  # truncated display

                        gc.collect()

                except Exception as e:
                    st.error(f"Unexpected processing error: {e}")
