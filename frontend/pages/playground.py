"""Tool Calling Playground Page."""
import streamlit as st
import requests
import json
import time

def render(api_url: str):
    st.markdown('<div class="main-header">🎯 Tool Calling Interactive Playground</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Convert natural language instructions to schema-compliant function calls using <b>Gemini 3.8 Flash</b> and execute safely.</div>', unsafe_allow_html=True)

    # Gemini Key Status Indicator in Playground
    with st.expander("🔑 Gemini Configuration & API Key Status", expanded=False):
        st.write("You can configure your `GEMINI_API_KEY` either in your `.env` file or provide a session override below:")
        session_key = st.text_input("Session Gemini API Key (optional override)", type="password", placeholder="AIzaSy...")
        if session_key:
            st.success("Session API Key provided for requests.")

    # Sample Presets
    presets = {
        "Select a sample prompt...": "",
        "Math: Multiply 458 by 32": "What is 458 multiplied by 32?",
        "Math: Square Root of 256": "Calculate the square root of 256 please.",
        "Unit Conversion: Celsius to Fahrenheit": "Convert 37 degrees Celsius to Fahrenheit.",
        "Unit Conversion: Meters to Feet": "How many feet are in 50 meters?",
        "Database: Customer Email Lookup": "Retrieve customer profile for email alex@enterprise.com from database.",
        "Database: Check Inventory Stock": "Find inventory details for SKU-4001 in our stock.",
        "Search: Python Asyncio": "Search for documentation on python asyncio task group."
    }

    selected_preset = st.selectbox("Preset Query Examples", list(presets.keys()))
    default_text = presets[selected_preset] if selected_preset != "Select a sample prompt..." else ""

    user_query = st.text_area("User Natural Language Query", value=default_text, height=100, placeholder="e.g. Can you convert 125 meters to feet?")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        execute_now = st.checkbox("Execute Tool Immediately", value=True)
    with col2:
        use_all_tools = st.checkbox("Expose All Registered Tools", value=True)

    if st.button("Generate & Validate Tool Call", type="primary"):
        if not user_query.strip():
            st.warning("Please provide a natural language query.")
            return

        with st.spinner("Executing Tool Calling via Gemini 3.8 Flash..."):
            try:
                payload = {
                    "query": user_query,
                    "execute_immediately": execute_now
                }
                resp = requests.post(f"{api_url}/predict", json=payload, timeout=30)
                
                if resp.status_code != 200:
                    st.error(f"Backend returned error code {resp.status_code}: {resp.text}")
                    return
                    
                data = resp.json()
                
                # Metrics Row
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Latency", f"{data['latency_ms']} ms")
                with m2:
                    st.metric("JSON Valid", "Yes" if data["is_valid_json"] else "No")
                with m3:
                    st.metric("Schema Valid", "Yes" if data["is_schema_valid"] else "No")
                with m4:
                    st.metric("Model", data["model_name"])

                st.divider()

                # Detailed Result Columns
                c_left, c_right = st.columns(2)
                
                with c_left:
                    st.subheader("1. Predicted Tool Call")
                    if data.get("tool_call"):
                        st.json(data["tool_call"])
                    else:
                        st.code(data["raw_output"], language="json")

                    if data.get("validation_error"):
                        st.error(f"Validation Failure: {data['validation_error']}")

                with c_right:
                    st.subheader("2. Safe Execution Result")
                    if data.get("execution_result"):
                        res = data["execution_result"]
                        if res.get("success"):
                            st.success(f"Executed Tool '{res['tool_name']}' in {res.get('execution_time_ms', 0)} ms")
                            st.json(res["output"])
                        else:
                            st.error(f"Execution Error: {res.get('error_message')}")
                    else:
                        st.info("Tool execution was not requested or validation failed.")

            except Exception as e:
                st.error(f"Failed to communicate with API server: {e}")
