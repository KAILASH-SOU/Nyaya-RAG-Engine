# app.py (temporary minimal test)
import streamlit as st
st.set_page_config(page_title="Minimal Test", layout="centered")
st.title("Minimal Streamlit App — sanity check")
st.write("If you see this, Streamlit started successfully.")
st.write("Environment variables (sample):")
st.write({
    "GDRIVE_INDEX_ID": bool(st.secrets.get("GDRIVE_INDEX_ID")),
    "GDRIVE_MAP_ID": bool(st.secrets.get("GDRIVE_MAP_ID")),
    "GEMINI_API_KEY": bool(st.secrets.get("GEMINI_API_KEY")),
})
