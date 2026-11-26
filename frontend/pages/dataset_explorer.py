"""Dataset Explorer Page with Pickle Store Inspection."""
import streamlit as st
import requests
import pandas as pd
from pathlib import Path
from utils.pickle_utils import load_pickle
from backend.config import settings

def render(api_url: str):
    st.markdown('<div class="main-header">📦 Dataset & Pickle Store Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Inspection of train/val/test splits, tool distributions, and mandatory pickle payloads.</div>', unsafe_allow_html=True)

    meta_file = settings.PICKLE_STORE_DIR / "dataset_metadata.pkl"
    proc_file = settings.PICKLE_STORE_DIR / "processed_dataset.pkl"

    if not meta_file.exists() or not proc_file.exists():
        st.warning("Processed dataset pickle files not found. Run `python -m training.prepare_dataset`.")
        return

    meta = load_pickle(meta_file)
    dataset = load_pickle(proc_file)

    counts = meta.get("sample_counts", {})
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Train Samples", counts.get("train", 0))
    with c2:
        st.metric("Validation Samples", counts.get("val", 0))
    with c3:
        st.metric("Test Samples", counts.get("test", 0))
    with c4:
        st.metric("Total Valid", counts.get("total_valid", 0))

    st.divider()

    st.subheader("Split Inspection")
    split = st.selectbox("Select Split", ["train", "val", "test"])
    records = dataset.get(split, [])
    
    if records:
        df = pd.DataFrame([{
            "ID": r.get("id"),
            "Query": r.get("query"),
            "Tool": r.get("tool_call", {}).get("name"),
            "Arguments": str(r.get("tool_call", {}).get("arguments")),
            "Domain": r.get("domain")
        } for r in records])
        
        st.dataframe(df, use_container_width=True, height=350)
        
        st.subheader("Tool Call Distribution")
        st.bar_chart(df["Tool"].value_counts())
    else:
        st.info("No records in this split.")

