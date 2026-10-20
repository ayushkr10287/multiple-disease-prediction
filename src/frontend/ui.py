"""Shared UI helpers for the Dr. ML Streamlit frontend."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

APP_NAME = "Dr. ML"

DISCLAIMER = (
    "This tool is an educational machine-learning demo and is **not** a medical "
    "diagnosis. Please consult a qualified healthcare professional."
)

CSS = """
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 2rem; max-width: 1100px;}
.result-card {
    border-radius: 12px; padding: 1rem 1.25rem;
    border: 1px solid; margin-bottom: .75rem;
}
.result-card h3 {margin: 0;}
.result-card p {margin: .25rem 0 0;}
.low  {background: #ECFDF5; border-color: #A7F3D0; color: #065F46;}
.mid  {background: #FFFBEB; border-color: #FDE68A; color: #92400E;}
.high {background: #FEF2F2; border-color: #FECACA; color: #991B1B;}
</style>
"""


# --------------------------------------------------------------------------- #
# Page setup
# --------------------------------------------------------------------------- #
def setup_page(title: str, icon: str, subtitle: str | None = None) -> None:
    """Configure the page, inject CSS, render the sidebar and page header."""
    page_title = title if title == APP_NAME else f"{APP_NAME} · {title}"
    st.set_page_config(page_title=page_title, page_icon=icon, layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    _render_sidebar()
    st.title(f"{icon} {title}")
    if subtitle:
        st.caption(subtitle)


def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown(f"### {APP_NAME}")
        st.caption("AI-assisted health risk screening")
        st.divider()
        st.markdown("**About the models**")
        # TODO: replace with your real model details
        st.caption(
            "- Diabetes: Pima Indians dataset\n"
            "- Heart disease: UCI Cleveland dataset\n"
            "- Predictions are served by a FastAPI backend"
        )
        st.divider()
        st.caption(DISCLAIMER)


# --------------------------------------------------------------------------- #
# Input helpers
# --------------------------------------------------------------------------- #
def choice(label: str, options: dict, default=None, help: str | None = None):
    """Selectbox showing readable labels, returning the underlying code."""
    labels = list(options)
    index = next((i for i, l in enumerate(labels) if options[l] == default), 0)
    return options[st.selectbox(label, labels, index=index, help=help)]


def name_of(options: dict, code) -> str:
    """Reverse lookup: code -> readable label."""
    return next((l for l, c in options.items() if c == code), str(code))


# --------------------------------------------------------------------------- #
# Backend
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=15, show_spinner=False)
def _backend_online(api_url: str) -> bool:
    base = api_url.rsplit("/", 1)[0]
    try:
        # Any non-5xx response means the server is up (works even without /health)
        return requests.get(f"{base}/health", timeout=3).status_code < 500
    except requests.RequestException:
        return False


def render_backend_status(api_url: str) -> bool:
    online = _backend_online(api_url)
    if online:
        st.success("Prediction service online", icon="✅")
    else:
        st.error(
            "Prediction service offline. Start the FastAPI backend and refresh.",
            icon="⚠️",
        )
    return online


def predict(api_url: str, payload: dict) -> dict | None:
    """POST to the backend. Returns the JSON result, or None after showing an error."""
    try:
        with st.spinner("Analyzing..."):
            response = requests.post(api_url, json=payload, timeout=20)
            response.raise_for_status()
        return response.json()
    except requests.RequestException:
        st.error("The prediction service is unavailable. Please try again shortly.")
        return None
    except ValueError:
        st.error("The prediction service returned an unexpected response.")
        return None


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
def risk_level(p: float) -> tuple[str, str]:
    if p < 0.33:
        return "Low risk", "low"
    if p < 0.66:
        return "Moderate risk", "mid"
    return "High risk", "high"


def save_result(key: str, result: dict, inputs: dict) -> None:
    st.session_state[key] = {
        "probability": float(result["probability"]),
        "prediction": int(result["prediction"]),
        "inputs": inputs,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


def clear_result(key: str) -> None:
    st.session_state.pop(key, None)


def show_saved_result(key: str, label: str, file_stem: str) -> None:
    """Render the last stored result for this page (persists across reruns)."""
    data = st.session_state.get(key)
    if not data:
        return

    probability = data["probability"]
    level, css = risk_level(probability)

    st.divider()
    st.subheader("Result")
    left, right = st.columns(2)

    with left:
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                number={"suffix": "%", "valueformat": ".1f"},
                title={"text": f"{label} probability"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#0F766E"},
                    "steps": [
                        {"range": [0, 33], "color": "#D1FAE5"},
                        {"range": [33, 66], "color": "#FEF3C7"},
                        {"range": [66, 100], "color": "#FEE2E2"},
                    ],
                },
            )
        )
        fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        flag = "Elevated risk flagged" if data["prediction"] == 1 else "Not flagged"
        st.markdown(
            f'<div class="result-card {css}">'
            f"<h3>{level}</h3>"
            f"<p>Estimated probability: <b>{probability:.1%}</b></p>"
            f"<p>Model classification: <b>{flag}</b></p>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.caption(DISCLAIMER)

        report = pd.DataFrame(
            [
                *[(k, v) for k, v in data["inputs"].items()],
                ("Estimated probability", f"{probability:.1%}"),
                ("Risk level", level),
                ("Generated", data["time"]),
            ],
            columns=["Field", "Value"],
        )
        st.download_button(
            "Download report (CSV)",
            report.to_csv(index=False).encode("utf-8"),
            file_name=f"{file_stem}_report.csv",
            mime="text/csv",
        )

    with st.expander("Submitted values"):
        st.table(
            pd.DataFrame(
                list(data["inputs"].items()), columns=["Field", "Value"]
            ).set_index("Field")
        )
