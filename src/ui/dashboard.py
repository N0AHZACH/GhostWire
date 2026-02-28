"""
GhostWire Dashboard — Streamlit-based UI for hallucination auditing.

This module is owned by the Frontend Developer (Role 6). It provides a
web interface to run audits and visualize results.

Run with:
    streamlit run src/ui/dashboard.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so imports resolve when running via
# `streamlit run src/ui/dashboard.py` from the project root.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.engine import GhostWireEngine, AuditResult  # noqa: E402
from src.analytics.scoring import HallucinationScorer  # noqa: E402

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="GhostWire — AI Hallucination Detector",
    page_icon="🔍",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown("---")

api_key = st.sidebar.text_input(
    "Google API Key",
    type="password",
    help="Your Google AI API key. Can also be set via GOOGLE_API_KEY env var.",
)

subject_model = st.sidebar.selectbox(
    "Subject Model",
    ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
    index=0,
)

judge_model = st.sidebar.selectbox(
    "Judge Model",
    ["gemini-1.5-pro", "gemini-2.0-flash", "gemini-1.5-flash"],
    index=0,
)

# ---------------------------------------------------------------------------
# Main Content
# ---------------------------------------------------------------------------

st.title("🔍 GhostWire")
st.markdown(
    "**AI Hallucination Detection Tool** — Audit any LLM response against "
    "ground‑truth context using a Judge‑Model architecture."
)
st.markdown("---")

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
        placeholder="Paste reference material the Judge will use to verify the answer…",
        height=200,
    )

    run_button = st.button("🚀 Run Audit", type="primary", use_container_width=True)

with col_result:
    st.subheader("📊 Audit Result")

    if run_button:
        if not prompt:
            st.warning("Please enter a prompt to audit.")
        else:
            try:
                engine = GhostWireEngine(
                    subject_model=subject_model,
                    judge_model=judge_model,
                    api_key=api_key or None,
                )

                with st.spinner("Querying Subject model…"):
                    result: AuditResult = engine.run_audit(prompt, context)

                # Display verdict.
                if result.is_hallucination:
                    st.error(f"⚠️ **Hallucination Detected** (Risk Level: {result.risk_level}/5)")
                else:
                    st.success("✅ **No Hallucination Detected**")

                st.metric("Confidence", f"{result.confidence}%")
                st.markdown(f"**Explanation:** {result.explanation}")

                with st.expander("🔎 Subject Answer"):
                    st.write(result.subject_answer)

                with st.expander("📋 Full JSON Result"):
                    st.json(result.to_dict())

                # Store result in session for batch analytics.
                if "results" not in st.session_state:
                    st.session_state["results"] = []
                st.session_state["results"].append(result)

            except EnvironmentError as exc:
                st.error(f"Configuration error: {exc}")
            except Exception as exc:
                st.error(f"Unexpected error: {exc}")

# ---------------------------------------------------------------------------
# Batch Analytics (if results exist in session)
# ---------------------------------------------------------------------------

if st.session_state.get("results"):
    st.markdown("---")
    st.subheader("📈 Session Analytics")

    results = st.session_state["results"]
    scorer = HallucinationScorer()
    report = scorer.generate_report(results)

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Total Audits", report["total_audits"])
    col_b.metric("Hallucination Rate", f"{report['hallucination_rate']:.0%}")
    col_c.metric("Avg Confidence", f"{report['average_confidence']:.0f}%")

    with st.expander("📊 Risk Distribution"):
        st.bar_chart(report["risk_distribution"])

    with st.expander("📋 Full Report JSON"):
        st.json(report)
