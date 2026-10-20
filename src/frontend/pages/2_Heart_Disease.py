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
from src.frontend.ui import (
    choice,
    clear_result,
    name_of,
    predict,
    save_result,
    setup_page,
    show_saved_result,
)

API_URL = Settings().api_url
KEY = "heart_result"

# NOTE: verify these encodings against the dataset your model was trained on.
SEX = {"Female": 0, "Male": 1}
CP = {
    "Typical angina": 0,
    "Atypical angina": 1,
    "Non-anginal pain": 2,
    "Asymptomatic": 3,
}
RESTECG = {
    "Normal": 0,
    "ST-T wave abnormality": 1,
    "Left ventricular hypertrophy": 2,
}
SLOPE = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
THAL = {"Fixed defect": 1, "Normal": 2, "Reversible defect": 3}

setup_page(
    "Heart Disease Risk Predictor",
    "❤️",
    "Enter the patient's details across the three tabs, then click Assess risk.",
)

with st.form("heart_form"):
    tab1, tab2, tab3 = st.tabs(["Demographics", "Vitals & Labs", "Symptoms & ECG"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age (years)", 1, 120, 52)
        with c2:
            sex = choice("Sex", SEX, default=1)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            trestbps = st.number_input(
                "Resting blood pressure (mm Hg)", 50, 250, 125,
                help="Systolic blood pressure at rest.",
            )
            chol = st.number_input("Serum cholesterol (mg/dL)", 100, 600, 212)
        with c2:
            thalach = st.number_input(
                "Maximum heart rate achieved (bpm)", 50, 250, 168
            )
            fbs = int(st.toggle("Fasting blood sugar > 120 mg/dL", value=False))

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            cp = choice("Chest pain type", CP, default=0)
            restecg = choice("Resting ECG result", RESTECG, default=1)
            exang = int(st.toggle("Exercise-induced angina", value=False))
            oldpeak = st.number_input(
                "ST depression (oldpeak)", 0.0, 10.0, 1.0, step=0.1,
                help="ST depression induced by exercise relative to rest.",
            )
        with c2:
            slope = choice(
                "Slope of peak exercise ST segment", SLOPE, default=2
            )
            ca = st.slider(
                "Major vessels coloured by fluoroscopy", 0, 4, 0,
                help="Number of major vessels (0-4).",
            )
            thal = choice("Thalassemia (thal)", THAL, default=2)

    submitted = st.form_submit_button(
        "Assess risk", type="primary", use_container_width=True
    )

if submitted:
    clear_result(KEY)
    payload = {
        "disease": "heart_disease",
        "features": {
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal,
        },
    }
    result = predict(API_URL, payload)
    if result:
        save_result(
            KEY,
            result,
            {
                "Age (years)": age,
                "Sex": name_of(SEX, sex),
                "Resting blood pressure (mm Hg)": trestbps,
                "Serum cholesterol (mg/dL)": chol,
                "Maximum heart rate (bpm)": thalach,
                "Fasting blood sugar > 120 mg/dL": "Yes" if fbs else "No",
                "Chest pain type": name_of(CP, cp),
                "Resting ECG": name_of(RESTECG, restecg),
                "Exercise-induced angina": "Yes" if exang else "No",
                "ST depression (oldpeak)": oldpeak,
                "Slope": name_of(SLOPE, slope),
                "Major vessels (ca)": ca,
                "Thalassemia": name_of(THAL, thal),
            },
        )

show_saved_result(KEY, "Heart disease", "heart_disease")
