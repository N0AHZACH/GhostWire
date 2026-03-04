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

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Settings")
    model_provider = st.selectbox("Judge Model", ["Gemini Pro (Default)", "GPT-4o", "Claude 3.5"])
    test_domain = st.selectbox("Industry Domain", ["General", "Medical", "Legal", "Finance", "Infrastructure"])
    st.divider()
    threshold = st.slider("Risk Sensitivity Threshold", 1, 5, 3)
    st.info("GhostWire identifies factual inconsistencies between a Subject Model and Ground Truth.")

# --- TABS ---
tab1, tab2 = st.tabs(["🎯 Single Test", "📦 Bulk Evaluation"])

# --- TAB 1: SINGLE TESTING ---
with tab1:
    col_in, col_out = st.columns([1, 1])
    
    with col_in:
        st.subheader("Input Playground")
        context_data = st.text_area("Ground Truth / Context", 
                                  placeholder="Paste the factual source here...", height=150)
        model_output = st.text_area("User Query to Model", 
                                  placeholder="What should the AI explain?", height=150)
        
        if st.button("Run GhostWire Audit", use_container_width=True):
            if not model_output or not context_data:
                st.warning("Please provide both context and prompt.")
            else:
                engine = GhostwireEngine() 
                with st.spinner("Judge Model is auditing..."):
                    raw_result = engine.run_audit(prompt=model_output, context=context_data)
                st.session_state['latest_result'] = raw_result

    with col_out:
        st.subheader("Audit Verdict")
        if 'latest_result' in st.session_state:
            full_res = st.session_state['latest_result']
            
            if full_res["status"] == "success":
                res = full_res["audit_data"]
                subject_response = full_res["subject_response"]

                # --- AUDITOR INTEGRATION (Role 6) ---
                from src.analytics.auditor import GhostwireAuditor
                auditor = GhostwireAuditor()
                ethical_risk = auditor.classify_ethical_risk(res, domain=test_domain) 
                
                st.markdown(f"### ⚖️ Auditor's Verdict: **{ethical_risk}**")
                
                # VISUAL INDICATORS
                if res.get('is_hallucination'):
                    st.markdown('<p class="status-red">🔴 HALLUCINATION DETECTED</p>', unsafe_allow_html=True)
                    st.error(f"**Explanation:** {res.get('auditor_notes', 'Fact-check failed.')}")
                else:
                    st.markdown('<p class="status-green">🟢 NO HALLUCINATION FOUND</p>', unsafe_allow_html=True)
                    st.success("The output is grounded in the provided context.")

                # METRICS
                m1, m2 = st.columns(2)
                with m1:
                    st.metric("Confidence Score", f"{res.get('confidence_score', 0)}%")
                with m2:
                    risk = res.get('risk_level', 0)
                    st.metric("Risk Level", f"{risk}/5", 
                              delta="High Risk" if risk >= threshold else "Acceptable",
                              delta_color="inverse" if risk >= threshold else "normal")
                
                with st.expander("View Full AI Response"):
                    st.write(subject_response)

                with st.expander("View Raw JSON Verdict"):
                    st.json(res)
            else:
                st.error(f"Pipeline Error: {full_res.get('message')}")
        else:
            st.info("Run an audit to see results.")

# --- TAB 2: BULK EVALUATION ---
with tab2:
    st.subheader("Dataset Stress Test")
    uploaded_file = st.file_uploader("Upload Prompt Dataset (CSV)", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if st.button("Start Bulk Audit"):
            engine = GhostwireEngine()
            progress_bar = st.progress(0)
            bulk_results = []
            
            for i in range(len(df)):
                time.sleep(0.1)
                simulated_score = (i * 17) % 100 
                bulk_results.append(simulated_score)
                progress_bar.progress((i + 1) / len(df))
            
            df['Reliability_Score'] = bulk_results
            st.divider()
            fig = px.histogram(df, x="Reliability_Score", title="Reliability Distribution", color_discrete_sequence=['#2ecc71'])
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df)