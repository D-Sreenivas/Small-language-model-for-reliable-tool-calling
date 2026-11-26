"""Streamlit Frontend Application: Small Language Model for Reliable Tool Calling."""
import streamlit as st
import requests
import json
import os
import sys
from pathlib import Path

# Adjust path to find backend and utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.config import settings

st.set_page_config(
    page_title="SLM Reliable Tool Calling",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Enterprise CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .status-badge-ok {
        background-color: #DCFCE7;
        color: #166534;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .status-badge-err {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("⚡ SLM Tool Engine")
st.sidebar.caption("1–3B Parameter Reliable Function Calling")

api_url = st.sidebar.text_input("FastAPI Base URL", value=settings.API_BASE_URL)

try:
    health_resp = requests.get(f"{api_url}/health", timeout=2)
    if health_resp.status_code == 200:
        st.sidebar.markdown('**Backend Status:** <span class="status-badge-ok">CONNECTED</span>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('**Backend Status:** <span class="status-badge-err">ERROR</span>', unsafe_allow_html=True)
except Exception:
    st.sidebar.markdown('**Backend Status:** <span class="status-badge-err">OFFLINE</span>', unsafe_allow_html=True)

st.sidebar.divider()
st.sidebar.info(
    "**Architecture Stack:**\n"
    "- Model: Qwen2.5-1.5B-Instruct\n"
    "- LoRA / QLoRA 4-bit\n"
    "- Constrained Grammar Decoding\n"
    "- Safe Sandboxed Execution"
)

# Navigation
page = st.sidebar.radio(
    "Navigation",
    [
        "Tool Calling Playground",
        "Model Information",
        "Dataset Explorer",
        "Training Dashboard",
        "Evaluation & Latency Dashboard"
    ]
)

# Routing
if page == "Tool Calling Playground":
    from frontend.pages import playground
    playground.render(api_url)
elif page == "Model Information":
    from frontend.pages import model_info
    model_info.render(api_url)
elif page == "Dataset Explorer":
    from frontend.pages import dataset_explorer
    dataset_explorer.render(api_url)
elif page == "Training Dashboard":
    from frontend.pages import training_dashboard
    training_dashboard.render(api_url)
elif page == "Evaluation & Latency Dashboard":
    from frontend.pages import evaluation_dashboard
    evaluation_dashboard.render(api_url)

