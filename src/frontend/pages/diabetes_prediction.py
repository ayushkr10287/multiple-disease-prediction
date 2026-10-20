import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

import requests
import streamlit as st

from src.frontend.config.settings import Settings

settings = Settings()
API_URL = settings.api_url


st.set_page_config(
    page_title="Dr.ML - Diabetes Prediction",
    page_icon="🩺",
    layout="centered",
)

st.title("🩺 Diabetes Risk Predictor")
st.write("Enter patient details and click **Predict**")

st.subheader("Patient Details")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", 0, 20, 2)
    glucose = st.number_input("Glucose", 0, 400, 120)
    blood_pressure = st.number_input("Blood Pressure", 0, 250, 70)
    skin_thickness = st.number_input("Skin Thickness", 0, 100, 25)

with col2:
    insulin = st.number_input("Insulin", 0, 900, 80)
    bmi = st.number_input("BMI", 0.0, 70.0, 28.5)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.45)
    age = st.number_input("Age", 1, 120, 35)

if st.button("🔍 Predict", use_container_width=True):
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

    try:
        response = requests.post(API_URL, json=payload, timeout=20)
        response.raise_for_status()
    except requests.RequestException as e:
        st.error("Unable to reach prediction service.")
        st.caption(str(e))
        st.stop()

    result = response.json()
    prediction = int(result["prediction"])
    probability = float(result["probability"])

    st.divider()
    st.metric("Diabetes Probability", f"{probability:.2f}")

    if prediction == 1:
        st.error("⚠️ Model Prediction: Diabetes")
    else:
        st.success("✅ Model Prediction: No Diabetes")
