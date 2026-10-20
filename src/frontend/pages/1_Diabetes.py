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
    clear_result,
    predict,
    save_result,
    setup_page,
    show_saved_result,
)

API_URL = Settings().api_url
KEY = "diabetes_result"

setup_page(
    "Diabetes Risk Predictor",
    "🩺",
    "Enter the patient's clinical measurements, then click Assess risk.",
)

with st.form("diabetes_form"):
    st.subheader("Patient Details")
    c1, c2 = st.columns(2)

    with c1:
        age = st.number_input("Age (years)", 1, 120, 35)
        pregnancies = st.number_input("Pregnancies", 0, 20, 2)
        bmi = st.number_input("BMI (kg/m²)", 10.0, 70.0, 28.5, step=0.1)
        dpf = st.number_input(
            "Diabetes pedigree function",
            0.0,
            3.0,
            0.45,
            step=0.01,
            help="Score reflecting family history of diabetes.",
        )

    with c2:
        glucose = st.number_input(
            "Glucose (mg/dL)",
            40,
            400,
            120,
            help="Plasma glucose concentration, 2-hour oral glucose tolerance test.",
        )
        blood_pressure = st.number_input("Diastolic blood pressure (mm Hg)", 30, 250, 70)
        skin_thickness = st.number_input("Triceps skin fold (mm)", 0, 100, 25)
        insulin = st.number_input("Insulin (µU/mL)", 0, 900, 80, help="2-hour serum insulin.")

    submitted = st.form_submit_button(
        "Assess risk", type="primary", use_container_width=True
    )

if submitted:
    clear_result(KEY)
    payload = {
        "disease": "diabetes",
        "features": {
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": dpf,
            "Age": age,
        },
    }
    result = predict(API_URL, payload)
    if result:
        save_result(
            KEY,
            result,
            {
                "Age (years)": age,
                "Pregnancies": pregnancies,
                "BMI (kg/m²)": bmi,
                "Diabetes pedigree function": dpf,
                "Glucose (mg/dL)": glucose,
                "Diastolic blood pressure (mm Hg)": blood_pressure,
                "Triceps skin fold (mm)": skin_thickness,
                "Insulin (µU/mL)": insulin,
            },
        )

show_saved_result(KEY, "Diabetes", "diabetes")
