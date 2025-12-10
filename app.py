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
# Note: these variables can hold direct URLs (HuggingFace/S3/GCS) or local file paths.
GDRIVE_INDEX_ID = st.secrets.get("GDRIVE_INDEX_ID") if "GDRIVE_INDEX_ID" in st.secrets else os.environ.get("GDRIVE_INDEX_ID")
GDRIVE_MAP_ID = st.secrets.get("GDRIVE_MAP_ID") if "GDRIVE_MAP_ID" in st.secrets else os.environ.get("GDRIVE_MAP_ID")
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") if "GEMINI_API_KEY" in st.secrets else os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = st.secrets.get("GEMINI_MODEL") if "GEMINI_MODEL" in st.secrets else os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

os.makedirs(DATA_DIR, exist_ok=True)

# -----------------------------
# Simple HTTP downloader for direct URLs
# -----------------------------
def download_from_url(url: str, dest_path: str):
    """
    Download a file from a direct HTTP(S) URL (Hugging Face, S3, GCS...). Streams to disk.
    If dest_path already exists the function returns immediately.
    """
    if not url:
        raise ValueError("No URL provided for download.")
    if os.path.exists(dest_path):
        st.info(f"Using cached file: {dest_path} ({os.path.getsize(dest_path)} bytes)")
        return

    st.info(f"Downloading {os.path.basename(dest_path)} ... This may take a while for large files.")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
        total = 0
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)
                    total += len(chunk)
    st.success(f"Downloaded {os.path.basename(dest_path)} ({os.path.getsize(dest_path)} bytes)")

# -----------------------------
# Load models & data (cached)
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_all_models_and_data():
    """
    Download (if URL provided) and load FAISS index + mapping and the embedding model.
    """
    try:
        st.info("Initialization: checking & downloading data if needed...")

        index_path = os.path.join(DATA_DIR, INDEX_FILENAME)
        map_path = os.path.join(DATA_DIR, MAP_FILENAME)

        # If secret is an http(s) URL, download it. If it's a local path, assume it's already present.
        if GDRIVE_INDEX_ID:
            if isinstance(GDRIVE_INDEX_ID, str) and GDRIVE_INDEX_ID.startswith("http"):
                download_from_url(GDRIVE_INDEX_ID, index_path)
            else:
                # treat as local path or id - we expect a path if not http
                if not os.path.exists(index_path) and os.path.exists(GDRIVE_INDEX_ID):
                    # copy local file (user provided full path in secret)
                    os.replace(GDRIVE_INDEX_ID, index_path)
                    st.info(f"Copied local index to {index_path}")
                else:
                    st.info(f"Assuming index exists in repo at {index_path}")
        else:
            st.info("GDRIVE_INDEX_ID not set — assuming index exists locally at " + index_path)

        if GDRIVE_MAP_ID:
            if isinstance(GDRIVE_MAP_ID, str) and GDRIVE_MAP_ID.startswith("http"):
                download_from_url(GDRIVE_MAP_ID, map_path)
            else:
                if not os.path.exists(map_path) and os.path.exists(GDRIVE_MAP_ID):
                    os.replace(GDRIVE_MAP_ID, map_path)
                    st.info(f"Copied local mapping to {map_path}")
                else:
                    st.info(f"Assuming mapping exists in repo at {map_path}")
        else:
            st.info("GDRIVE_MAP_ID not set — assuming mapping exists locally at " + map_path)

        st.info("Loading FAISS index from disk...")
        index = faiss.read_index(index_path)

        st.info("Loading mapping JSON...")
        with open(map_path, "r", encoding="utf-8") as f:
            index_to_chunk_map = json.load(f)

        st.info("Loading SentenceTransformer embedding model (this can take time on first run)...")
        embedding_model = SentenceTransformer(DEFAULT_EMBED_MODEL)

        st.success("Models and data loaded successfully!")
        return index, index_to_chunk_map, embedding_model

    except Exception as e:
        st.error("--- Application Initialization Failed ---")
        st.error(f"Error loading FAISS or data: {e}")
        # List debug files if any
        debug_files = [p for p in os.listdir(DATA_DIR) if p.startswith("debug_")]
        if debug_files:
            st.error(f"Debug files: {debug_files}")
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
    st.info("Required secrets: GDRIVE_INDEX_ID (direct URL or path), GDRIVE_MAP_ID (direct URL or path), GEMINI_API_KEY (optional GEMINI_MODEL).")
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
                        # index_to_chunk_map may be dict keyed by str indices or list
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

                        st.markdown("### AI-Synthesized Analysis (Summary)")
                        st.info(final_answer)

                        st.markdown("### Key Supporting Cases (documents used)")
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
