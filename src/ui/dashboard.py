"""
GhostWire Dashboard — Streamlit-based UI for hallucination auditing.
Enhanced by Full-Stack UI Developer (Role 5).
"""

from __future__ import annotations
import time
import json
import sys
import pandas as pd
from pathlib import Path

import streamlit as st
import google.api_core.exceptions as google_exceptions

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.engine import GhostWireEngine, AuditResult 
from src.analytics.scoring import HallucinationScorer 

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="GhostWire — AI Hallucination Detector",
    page_icon="🔍",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown("---")

api_key = st.sidebar.text_input(
    "Google API Key",
    type="password",
    help="Your Google AI API key. Get one at aistudio.google.com",
)

# Added "-latest" and Flash-Lite options to help avoid 404/429 errors
subject_model = st.sidebar.selectbox(
    "Subject Model",
    ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-flash-latest"],
    index=0,
)

judge_model = st.sidebar.selectbox(
    "Judge Model",
    ["gemini-2.5-pro", "gemini-3.1-pro-preview", "gemini-pro-latest"],
    index=0,
)

# ---------------------------------------------------------------------------
# Main Content Header
# ---------------------------------------------------------------------------
st.title("🔍 GhostWire")
st.markdown(
    "**AI Hallucination Detection Tool** — Audit any LLM response against "
    "ground‑truth context using a Judge‑Model architecture."
)
st.markdown("---")

# ---------------------------------------------------------------------------
# Tabs for Navigation (Role 5 Feature)
# ---------------------------------------------------------------------------
tab_single, tab_bulk = st.tabs(["🎯 Single Audit", "📦 Bulk Analysis"])

# --- TAB 1: SINGLE AUDIT ---
with tab_single:
    col_input, col_result = st.columns([1, 1])

    with col_input:
        st.subheader("📝 Audit Input")
        prompt = st.text_area(
            "Prompt",
            placeholder="Enter the question or instruction to audit…",
            height=120,
        )
        context = st.text_area(
            "Context (ground truth)",
            placeholder="Paste reference material the Judge will use to verify...",
            height=200,
        )
        run_button = st.button("🚀 Run Audit", type="primary", use_container_width=True)

    with col_result:
        st.subheader("📊 Audit Result")
        if run_button:
            if not api_key:
                st.warning("Please provide an API Key in the sidebar.")
            elif not prompt:
                st.warning("Please enter a prompt to audit.")
            else:
                try:
                    engine = GhostWireEngine(
                        subject_model=subject_model,
                        judge_model=judge_model,
                        api_key=api_key or None,
                    )

                    with st.spinner("Querying Models..."):
                        result: AuditResult = engine.run_audit(prompt, context)

                    # Display verdict
                    if result.is_hallucination:
                        st.error(f"⚠️ **Hallucination Detected** (Risk Level: {result.risk_level}/5)")
                    else:
                        st.success("✅ **No Hallucination Detected**")

                    st.metric("Confidence", f"{result.confidence}%")
                    st.markdown(f"**Explanation:** {result.explanation}")

                    with st.expander("🔎 Subject Answer"):
                        st.write(result.subject_answer)

                except Exception as exc:
                    st.error(f"Error: {exc}")

# --- TAB 2: BULK ANALYSIS (New Feature) ---
with tab_bulk:
    st.subheader("📦 Bulk Processing")
    uploaded_file = st.file_uploader("Upload Adversarial JSON", type="json", key="bulk_uploader")
    
    if uploaded_file is not None:
        data = json.load(uploaded_file)
        df = pd.DataFrame(data)
        st.success(f"Loaded {len(df)} prompts.")
        st.dataframe(df.head())

        if st.button("▶️ Start Batch Audit"):
            results_list = []
            progress_bar = st.progress(0)
            
            engine = GhostWireEngine(
                subject_model=subject_model,
                judge_model=judge_model,
                api_key=api_key or None
            )

            # Loop through the rows with Error Handling
            # Loop through the rows
            for index, row in df.iterrows():
                with st.spinner(f"Checking question {index+1}..."):
                    try:
                        res = engine.run_audit(row['prompt'], row['context'])
                        results_list.append({
                            "Prompt": row['prompt'],
                            "Hallucination": "⚠️ YES" if res.is_hallucination else "✅ NO",
                            "Confidence": res.confidence,
                            "Risk": res.risk_level,
                            "Explanation": res.explanation
                        })
                    except Exception as e:
                        st.error(f"Google is busy! Skipping row {index+1}")
                        continue
                
                # THIS IS THE MOST IMPORTANT LINE:
                time.sleep(10) # Tells the computer to wait 10 seconds
                
                progress_bar.progress((index + 1) / len(df))

            # Display final batch results
            if results_list:
                st.markdown("---")
                st.subheader("🏁 Final Audit Report")
                results_df = pd.DataFrame(results_list)
                st.dataframe(results_df, use_container_width=True)

                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Results CSV", data=csv, file_name="audit_results.csv", mime="text/csv")
            else:
                st.error("No results were generated. Check your API key and Quota.")