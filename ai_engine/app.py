import streamlit as st
import tempfile
from pipeline import run_pipeline

st.set_page_config(page_title="IntelliSecure", layout="centered")

st.title("🚨 IntelliSecure: Smart Surveillance System for Predictive Crowd Risk and Threat Detection")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_file:

    # Save uploaded video temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    st.video(tfile.name)

    with st.spinner("🔍 Analyzing video..."):
        result = run_pipeline(tfile.name)

    violence = result["violence_score"]
    stampede = result["stampede_score"]
    people = result["max_people"]

    st.subheader("📊 Results")

    # -------- SMART PRIORITY LOGIC --------
    if violence > 0.8:
        st.error("VIOLENCE DETECTED", icon="🚨")

    elif stampede > 0.7:
        st.error("HIGH STAMPEDE RISK", icon="⚠️")

    elif stampede > 0.45:
        st.warning("CROWD BUILDING / MODERATE RISK", icon="⚠️")

    else:
        st.success("SAFE ENVIRONMENT", icon="✅")

    # -------- DETAILS --------
    st.markdown("---")
    st.subheader("🔍 Detailed Metrics")

    st.write(f"**Violence Score:** {violence:.2f}")
    st.write(f"**Stampede Score:** {stampede:.2f}")
    st.write(f"**Max People Detected:** {people}")