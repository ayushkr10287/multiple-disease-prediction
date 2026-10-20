import sys
from pathlib import Path


def _project_root() -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / "src").is_dir():
            return p
    raise RuntimeError("Could not locate project root (folder containing 'src')")


sys.path.insert(0, str(_project_root()))

import streamlit as st

from src.frontend.config.settings import Settings
from src.frontend.ui import DISCLAIMER, render_backend_status, setup_page

API_URL = Settings().api_url

setup_page("Dr. ML", "🩺", "AI-assisted health risk screening")

render_backend_status(API_URL)

st.write("")
c1, c2 = st.columns(2)

with c1:
    with st.container(border=True):
        st.subheader("🩺 Diabetes")
        st.write("Risk estimate from 8 routine clinical measurements.")
        st.page_link("pages/1_Diabetes.py", label="Open predictor →")

with c2:
    with st.container(border=True):
        st.subheader("❤️ Heart Disease")
        st.write("Risk estimate from 13 cardiac and clinical indicators.")
        st.page_link("pages/2_Heart_Disease.py", label="Open predictor →")

st.write("")
st.subheader("How it works")
s1, s2, s3 = st.columns(3)
s1.markdown("**1. Enter details**  \nFill in the patient's clinical values.")
s2.markdown("**2. Assess risk**  \nThe values are sent to a trained ML model.")
s3.markdown("**3. Review result**  \nSee the estimated probability and risk level.")

st.divider()
st.caption(DISCLAIMER)
