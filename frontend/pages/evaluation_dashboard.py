"""Evaluation & Latency Telemetry Dashboard."""
import streamlit as st
import pandas as pd
import plotly.express as px
from utils.pickle_utils import load_pickle
from backend.config import settings

def render(api_url: str):
    st.markdown('<div class="main-header">📊 Benchmark Evaluation & Latency Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Rigorous metrics tracking: JSON validity, schema match, BFCL accuracy, and P95 latency.</div>', unsafe_allow_html=True)

    eval_file = settings.PICKLE_STORE_DIR / "evaluation_results.pkl"

    if not eval_file.exists():
        st.info("Evaluation results pickle (`evaluation_results.pkl`) not found. Run `python -m evaluation.metrics`.")
        return

    results = load_pickle(eval_file)
    summary = results.get("summary", {})

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("JSON Validity", f"{summary.get('json_validity_rate', 0.0) * 100:.1f}%")
    with c2:
        st.metric("Schema Validity", f"{summary.get('schema_validity_rate', 0.0) * 100:.1f}%")
    with c3:
        st.metric("Tool Selection Acc", f"{summary.get('tool_selection_accuracy', 0.0) * 100:.1f}%")
    with c4:
        st.metric("Execution Success", f"{summary.get('execution_success_rate', 0.0) * 100:.1f}%")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Latency Distribution (ms)")
        lat_data = results.get("latency_records", [])
        if lat_data:
            df_lat = pd.DataFrame(lat_data)
            fig = px.histogram(df_lat, x="latency_ms", nbins=20, title="Inference Latency Distribution")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Latency Key Percentiles")
        st.markdown(f"""
        - **Mean Latency:** {summary.get('avg_latency_ms', 0.0):.2f} ms
        - **P50 Latency:** {summary.get('p50_latency_ms', 0.0):.2f} ms
        - **P95 Latency:** {summary.get('p95_latency_ms', 0.0):.2f} ms
        - **Target TTFT:** < 200 ms (Observed: {summary.get('avg_latency_ms', 0.0):.2f} ms)
        """)

