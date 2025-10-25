import streamlit as st
import json
import numpy as np
import faiss
import os
import gc
import re
import time
from urllib.parse import quote_plus
from sentence_transformers import SentenceTransformer
import asyncio
import httpx # We'll use the httpx library for async API calls

# --- CONFIGURATION ---

# The relative path to your data folder
DATA_DIR = "./Data"
# IMPORTANT: Use your actual API Key here if you have one, or leave blank to use the canvas secret.
GEMINI_API_KEY = "AIzaSyAqw6TEyG6Y7grnHle9X7ffa_v7T4ImFmE" 
GEMINI_MODEL = "gemini-2.5-flash" 
# We use gemini-2.5-flash for speed, which provides excellent quality for RAG synthesis.


# --- MODEL LOADING (Runs only once) ---

@st.cache_resource
def load_all_models_and_data():
    """
    Loads FAISS index and Embedding Model into memory.
    """
    try:
        st.info("1/2 - Loading RAG Assets (FAISS Index and Mapping)...")
        INDEX_PATH = os.path.join(DATA_DIR, "judgments.index")
        MAPPING_PATH = os.path.join(DATA_DIR, "index_to_chunk_map.json")
        MODEL_NAME = 'all-MiniLM-L6-v2'
        
        # Load FAISS index
        index = faiss.read_index(INDEX_PATH)
        
        # Load chunk mapping (contains the actual text)
        with open(MAPPING_PATH, 'r') as f:
            index_to_chunk_map = json.load(f)
            
        # Load embedding model
        embedding_model = SentenceTransformer(MODEL_NAME)
        
        st.success("✅ 2/2 - Models and data loaded successfully!")
        return index, index_to_chunk_map, embedding_model
        
    except Exception as e:
        st.error("--- Application Initialization Failed ---")
        st.error(f"Error loading FAISS or data. Check your terminal for details.")
        st.error(f"Error: {e}")
        return None, None, None

# --- RAG CORE FUNCTIONS ---

def retrieve_chunks(query, index, embedding_model, top_k=6): 
    """Retrieves the top_k most relevant chunk indices from FAISS."""
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, top_k)
    return indices[0]

# ⭐ NEW: Asynchronous function to call the Gemini API
async def generate_response_with_gemini(query, context):
    """Calls the Gemini API with RAG context."""
    
    # Define the professional prompt
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
    
    # Construct the API payload
    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
    }
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    
    # Implement exponential backoff for resilience
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Use httpx for asynchronous request
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                
            result = response.json()
            
            # Extract the text
            text = result['candidates'][0]['content']['parts'][0]['text']
            return text
        
        except httpx.HTTPStatusError as e:
            if e.response.status_code in [429, 500, 503] and attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"API Rate Limit/Server Error ({e.response.status_code}). Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                st.error(f"Gemini API Error: {e}")
                raise
        except Exception as e:
            st.error(f"An unexpected error occurred during API call: {e}")
            raise
    
    return "Error: Could not generate response from AI model after several retries."


def post_process_answer(text):
    """Ensures the text starts with a capital letter and capitalizes sentences."""
    if text:
        text = text.strip()
        text = text[0].upper() + text[1:]
        text = re.sub(r'(?<=\.\s)([a-z])', lambda m: m.group(1).upper(), text)
    return text

# --- STREAMLIT UI ---
st.set_page_config(layout="wide", page_title="Judicium AI")
st.title("⚖️ Nyaya RAG Engine")
st.markdown("---")

# Load components
index, index_to_chunk_map, embedding_model = load_all_models_and_data()

if index is not None:
    st.write("Welcome to the public query interface. Enter a question about Indian Supreme Court case law to get an AI-synthesized, evidence-backed answer.")
    
    # Input area
    query = st.text_input("Enter your legal query:", placeholder="e.g., What are the current rules on capital vs revenue receipts from timber sale?")

    if st.button("Generate Answer"):
        if not query:
            st.warning("Please enter a query to analyze the judgments.")
        else:
            with st.spinner("Analyzing precedents and synthesizing answer with Gemini 2.5 Flash..."):
                # 1. Retrieve the top 6 relevant chunks
                retrieved_indices = retrieve_chunks(query, index, embedding_model)
                
                context = ""
                retrieved_chunks_data = []
                unique_doc_names = set() 
                
                for idx in retrieved_indices:
                    chunk_info = index_to_chunk_map[idx]
                    context += chunk_info['chunk_text'] + "\n\n"
                    retrieved_chunks_data.append(chunk_info)
                    unique_doc_names.add(chunk_info['doc_name'])

                # 2. Generate the Summary using the async function
                # Streamlit's "st.run_in_thread" or "st.experimental_fragment" 
                # (or just asyncio.run) are needed to call async functions in synchronous Streamlit.
                try:
                    raw_answer = asyncio.run(generate_response_with_gemini(query, context))
                    final_answer = post_process_answer(raw_answer)
                except Exception as e:
                    final_answer = f"Error during AI Generation: {e}"
            
            # --- DISPLAY RESULTS ---
            
            # 1. Display the Summary first
            st.markdown("### 📜 AI-Synthesized Analysis (Summary)")
            st.info(final_answer)

            # 2. Display the Top Cases as clickable links
            st.markdown("### 📚 Key Supporting Cases (6 Cases Used)")
            
            sorted_unique_docs = sorted(list(unique_doc_names))
            
            for doc_name in sorted_unique_docs:
                search_term = os.path.splitext(os.path.basename(doc_name))[0].replace('_', ' ')
                google_search_url = f"https://www.google.com/search?q={quote_plus(search_term)}"
                st.markdown(f"- **[{doc_name}]({google_search_url})**")

           
