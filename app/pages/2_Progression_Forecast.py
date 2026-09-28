"""
Mode 2: Progression Forecast -- shows a patient's real glucose trajectory
(a pre-loaded MIMIC-IV example, or a real app-tracked patient) plotted
alongside the trained model's forecast for the next reading.

Backed by Track B (src/models/train_track_b.py -- glucose is the primary
signal here, not HbA1c; see README.md for why).
"""

import streamlit as st
import pandas as pd
import pickle
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import MIMIC_TRAJECTORIES, MODELS_SAVED
from src.storage.db import list_patients_with_history, get_patient_history
from src.models.train_track_b import make_features, WINDOW

st.set_page_config(page_title="Mode 2: Progression Forecast", page_icon="\U0001F4C8")
st.title("Mode 2 -- Progression Forecast")
st.caption("Backed by the MIMIC-IV-trained glucose forecaster (Track B). "
           "Glucose is used because it has far more real readings than HbA1c in this data.")


def load_forecaster():
    path = MODELS_SAVED / "mimic_glucose_forecaster.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def show_forecast(readings):
    """readings: list of real glucose values, oldest first."""
    model = load_forecaster()
    if model is None:
        st.warning("No trained forecaster found. Run `python src/models/train_track_b.py` first.")
        return
    if len(readings) < WINDOW - 1:
        st.info(f"Need at least {WINDOW - 1} readings to forecast the next one -- this case has {len(readings)}.")
        return
    last_4 = readings[-(WINDOW - 1):]
    x = [make_features(last_4)]
    forecast = model.predict(x)[0]
    st.metric("Forecasted next glucose reading", f"{forecast:.0f} mg/dL",
              delta=f"{forecast - readings[-1]:+.0f} vs last reading")
    st.caption("Model tested by leave-one-patient-out cross-validation, average error ~58 mg/dL. "
               "This is a small-scale proof of concept, not a clinical-grade forecast.")


source = st.radio("Choose a case", ["Pre-loaded MIMIC-IV example patient", "A real tracked patient"])

if source == "Pre-loaded MIMIC-IV example patient":
    if MIMIC_TRAJECTORIES.exists():
        traj = pd.read_csv(MIMIC_TRAJECTORIES)
        glucose_patients = traj[traj["lab_name"] == "glucose"]["subject_id"].unique()
        chosen = st.selectbox("Select a patient case", sorted(glucose_patients))
        patient_data = traj[(traj["subject_id"] == chosen) & (traj["lab_name"] == "glucose")].sort_values("charttime")
        st.line_chart(patient_data.set_index("charttime")["valuenum"])
        st.caption(f"{len(patient_data)} real glucose readings for patient {chosen} (MIMIC-IV Demo, de-identified)")
        show_forecast(patient_data["valuenum"].tolist())
    else:
        st.error(f"mimic_trajectories.csv not found at {MIMIC_TRAJECTORIES} -- run "
                 f"src/data_prep/build_mimic_trajectories.py first.")
else:
    real_patients = list_patients_with_history(min_visits=2)
    if not real_patients:
        st.info("No real patients with 2+ logged visits yet. Use Mode 1 to log check-ins first.")
    else:
        patient_ids = [p[0] for p in real_patients]
        chosen = st.selectbox("Select a tracked patient", patient_ids)
        history = get_patient_history(chosen)
        df = pd.DataFrame(history, columns=["visit_date", "hba1c", "glucose", "bmi", "systolic_bp", "diastolic_bp"])
        st.line_chart(df.set_index("visit_date")["glucose"])
        st.caption(f"{len(df)} real check-ins logged for patient {chosen}")
        show_forecast(df["glucose"].dropna().tolist())
