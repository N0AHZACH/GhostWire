import sys
import os

# This tells the computer to look at the main project folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import streamlit as st
import pandas as pd
import plotly.express as px
import json
import time
from src.core.engine import GhostwireEngine # Role 3: Pipeline Architect's logic

# --- UI CONFIGURATION ---
st.set_page_config(page_title="GhostWire | Hallucination Detector", layout="wide")

# --- CUSTOM STYLES ---
st.markdown("""
    <style>
    .status-green { color: #2ecc71; font-weight: bold; font-size: 24px; }
    .status-red { color: #e74c3c; font-weight: bold; font-size: 24px; }
    .metric-container { background-color: #1e1e1e; padding: 15px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER ---
st.title("🛡️ GhostWire")
st.caption("AI Hallucination Detection using Judge-Model Architecture")

# --- SIDEBAR: Configuration for Role 1 & 6 ---
with st.sidebar:
    st.header("⚙️ Settings")
    model_provider = st.selectbox("Judge Model", ["Gemini Pro (Default)", "GPT-4o", "Claude 3.5"])
    st.divider()
    threshold = st.slider("Risk Sensitivity Threshold", 1, 5, 3)
    st.info("GhostWire identifies factual inconsistencies between a Subject Model and Ground Truth.")

# --- TABS: Single Test vs Bulk ---
tab1, tab2 = st.tabs(["🎯 Single Test", "📦 Bulk Evaluation"])

# --- TAB 1: SINGLE TESTING ---
with tab1:
    col_in, col_out = st.columns([1, 1])
    
    with col_in:
        st.subheader("Input Playground")
        context_data = st.text_area("Ground Truth / Context", 
                                  placeholder="Paste the factual source here...", height=150)
        model_output = st.text_area("Model Output to Audit", 
                                  placeholder="Paste the AI's response here...", height=150)
        
        if st.button("Run GhostWire Audit", use_container_width=True):
            if not model_output or not context_data:
                st.warning("Please provide both context and prompt.")
            else:
                # INTEGRATION: Initializing the real engine from Role 3
                engine = GhostwireEngine() 
                
                with st.spinner("Judge Model is auditing..."):
                    # Calling the real audit pipeline
                    result = engine.run_audit(prompt=model_output, context=context_data)
                
                st.session_state['latest_result'] = result

    with col_out:
        st.subheader("Audit Verdict")
        if 'latest_result' in st.session_state:
            res = st.session_state['latest_result']
            audit_data = res.get('audit_data', {})
            
            # THE VISUAL INDICATORS (Traffic Lights)
            if res.get('status') == 'error':
                st.markdown('<p class="status-red">🔥 PIPELINE ERROR</p>', unsafe_allow_html=True)
                st.error(f"**Message:** {res.get('message', 'Unknown error.')}")
            elif audit_data.get('is_hallucination'):
                st.markdown('<p class="status-red">🔴 HALLUCINATION DETECTED</p>', unsafe_allow_html=True)
                st.error(f"**Explanation:** {audit_data.get('auditor_notes', 'No explanation provided.')}")
            else:
                st.markdown('<p class="status-green">🟢 NO HALLUCINATION FOUND</p>', unsafe_allow_html=True)
                st.success("The output is grounded in the provided context.")

            # RELIABILITY METRICS
            m1, m2 = st.columns(2)
            with m1:
                st.metric("Confidence Score", f"{audit_data.get('confidence_score', 0)}%")
            with m2:
                # Risk level visual comparison to threshold
                risk = audit_data.get('risk_level', 0)
                st.metric("Risk Level", f"{risk}/5", 
                          delta="High Risk" if risk >= threshold else "Acceptable",
                          delta_color="inverse" if risk >= threshold else "normal")
            
            with st.expander("View Raw JSON Verdict"):
                st.json(res)
        else:
            st.info("Run an audit to see results.")

# --- TAB 2: BULK EVALUATION (Role 1 Integration) ---
with tab2:
    st.subheader("Dataset Stress Test")
    uploaded_file = st.file_uploader("Upload Prompt Dataset (CSV)", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:", df.head(3))
        
        if st.button("Start Bulk Audit"):
            engine = GhostwireEngine()
            progress_bar = st.progress(0)
            bulk_results = []
            
            for i in range(len(df)):
                # In production, this would call engine.run_audit for each row
                # For UI demo, we simulate processing time
                time.sleep(0.1)
                simulated_score = (i * 17) % 100 
                bulk_results.append(simulated_score)
                progress_bar.progress((i + 1) / len(df))
            
            df['Reliability_Score'] = bulk_results
            
            # VISUAL CHARTS
            st.divider()
            fig = px.histogram(df, x="Reliability_Score", 
                               title="Reliability Distribution across Dataset",
                               color_discrete_sequence=['#2ecc71'])
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(df)
            st.download_button("💾 Download Audit Report", df.to_csv(index=False), "ghostwire_report.csv")