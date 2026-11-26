"""Training Dashboard Page."""
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from utils.pickle_utils import load_pickle
from backend.config import settings

def render(api_url: str):
    st.markdown('<div class="main-header">📈 Model Training Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Supervised Fine-Tuning (SFT), DPO, and LoRA adapter checkpoints telemetry.</div>', unsafe_allow_html=True)

    history_path = settings.PICKLE_STORE_DIR / "training_history.pkl"
    
    if not history_path.exists():
        st.info("Training history pickle (`training_history.pkl`) not generated yet. Execute training pipeline to populate live loss curves.")
        st.markdown("""
        ### Ready Training Pipelines:
        - **SFT with QLoRA:** `python -m training.train_sft`
        - **DPO Alignment:** `python -m training.train_dpo`
        - **GRPO Verifiable Rewards:** `python -m training.train_grpo`
        """)
        return

    history = load_pickle(history_path)
    loss_history = history.get("loss_history", [])

    if loss_history:
        df = pd.DataFrame(loss_history)
        fig = px.line(df, x="step", y="loss", title="SFT Training Loss Curve (Qwen2.5-1.5B LoRA)", markers=True)
        st.plotly_chart(fig, use_container_width=True)

    st.json(history.get("config", {}))

