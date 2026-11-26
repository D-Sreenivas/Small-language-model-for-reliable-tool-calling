"""Model Information Page."""
import streamlit as st
import requests

def render(api_url: str):
    st.markdown('<div class="main-header">🤖 Model & Runtime Architecture</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Inspection of Gemini 3.8 Flash / SLM parameters, active backend, and telemetry.</div>', unsafe_allow_html=True)

    try:
        resp = requests.get(f"{api_url}/model/info", timeout=5)
        info = resp.json() if resp.status_code == 200 else {}
    except Exception:
        info = {"status": "Backend Unreachable"}

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Base Model", info.get("model_name", "gemini-3.8-flash"))
    with col2:
        st.metric("Active Backend", info.get("backend", "gemini"))
    with col3:
        st.metric("Execution Target", info.get("device", "cloud_api"))
    with col4:
        st.metric("API Key Configured", "Yes" if info.get("api_key_configured") else "Pending (.env)")

    st.divider()

    st.subheader("Gemini 3.8 Flash vs Edge SLM Architecture")
    st.markdown("""
    | Configuration | Gemini 3.8 Flash | Local Qwen2.5-1.5B (4-bit) |
    | :--- | :--- | :--- |
    | **Provider / Engine** | Google Gemini Cloud API | Local PyTorch / llama.cpp GGUF |
    | **VRAM Requirement** | 0 GB (Cloud Hosted) | ~1.94 GB (Fits in 4GB VRAM) |
    | **Inference Speed** | High throughput Flash | Sub-200ms TTFT on local GPU |
    | **Tool Calling Accuracy** | Frontier-level Structured Output | Fine-tuned LoRA with Constrained Grammar |
    | **Configuration** | `GEMINI_API_KEY` in `.env` | `MODEL_BACKEND=huggingface` in `.env` |
    """)

    st.subheader("Available Registered Tools & Schemas")
    try:
        t_resp = requests.get(f"{api_url}/tools", timeout=5)
        if t_resp.status_code == 200:
            tools = t_resp.json().get("tools", [])
            for t in tools:
                with st.expander(f"Tool: {t['name']}"):
                    st.write(t["description"])
                    st.json(t["parameters"])
    except Exception as e:
        st.error(f"Could not fetch tools: {e}")
