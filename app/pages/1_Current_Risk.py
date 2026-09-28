"""
Mode 1: Current Risk Check -- manual entry of a patient's current values,
returns multi-complication risk + SHAP explanation.

Backed by Track A (NHANES stacking ensemble). See Product_Story_Build_to_Use.md
Part 5 for the full walkthrough of what this page does from a clinician's
perspective.

FLOW (not yet wired to real models -- see TODOs)
    1. Clinician enters: age, sex, HbA1c, fasting glucose, lipid panel, BMI,
       blood pressure, smoking/alcohol/activity, income-to-poverty ratio.
    2. On submit: run feature_engineering.py's ratio/flag logic on the input,
       feed into the trained stacking ensemble (stacking_ensemble.py) to get
       per-complication risk probabilities.
    3. Run shap_explainer.py on this one patient's prediction to get top-5
       contributing features.
    4. Pass those SHAP values into llm_interpreter.py's generate_explanation()
       for the plain-language narrative.
    5. Display: risk bars/numbers, SHAP bar chart, LLM narrative text.
    6. If the clinician names a patient_id, log this check-in via
       src.storage.db.add_visit() -- this is what lets Mode 2 later show a
       REAL trajectory for this patient, not just pre-loaded MIMIC-IV examples.
    7. Log the prediction itself via src.storage.db.log_prediction() for the
       audit trail.
"""

import streamlit as st
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.storage.db import add_visit, log_prediction, get_patient_history
from src.models.predict import generate_clinical_recommendations, compute_risk_delta

st.set_page_config(page_title="Mode 1: Current Risk", page_icon="\U0001FA7A")
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg, #f5f9ff 0%, #edf3ff 100%); }
    div[data-testid="stMetricValue"] { font-size: 1.4rem; }
    div[data-testid="stProgressBar"] > div { background: linear-gradient(90deg, #4f46e5 0%, #14b8a6 100%); }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Mode 1 -- Current Risk Check")
st.caption("Backed by the NHANES-trained multi-label complication model (Track A).")

with st.form("current_risk_form"):
    st.subheader("Patient values")
    col1, col2 = st.columns(2)
    with col1:
        patient_id = st.text_input("Patient ID (optional -- enables trend tracking in Mode 2)")
        age = st.number_input("Age", min_value=18, max_value=100, value=50)
        hba1c = st.number_input("HbA1c (%)", min_value=3.0, max_value=15.0, value=6.5, step=0.1)
        glucose = st.number_input("Fasting glucose (mg/dL)", min_value=50, max_value=400, value=100)
        bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=27.0, step=0.1)
    with col2:
        systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=70, max_value=250, value=120)
        diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=150, value=80)
        ldl = st.number_input("LDL cholesterol (mg/dL)", min_value=30, max_value=300, value=100)
        hdl = st.number_input("HDL cholesterol (mg/dL)", min_value=15, max_value=120, value=50)

    submitted = st.form_submit_button("Calculate risk")

if submitted:
    from src.storage.db import get_active_model_version

    values = {
        "RIDAGEYR": age, "LBXGH": hba1c, "LBXGLU": glucose, "BMXBMI": bmi,
        "BPXOSY1": systolic_bp, "BPXODI1": diastolic_bp, "LBDLDL": ldl, "LBDHDD": hdl,
    }

    try:
        from src.explainability.explain import explain_patient
        out = explain_patient(values)
        risks = out["risks"]
    except Exception as e:
        st.warning(f"Explanation layer unavailable ({e}). Showing risk only. "
                    f"Run `pip install shap groq` and set GROQ_API_KEY to enable it.")
        from src.models.predict import predict_patient
        risks, _, _ = predict_patient(values)
        out = None

    previous_risks = None
    if patient_id:
        history = get_patient_history(patient_id)
        if history:
            last = history[-1]
            prev_values = {
                "RIDAGEYR": age,
                "LBXGH": last[1] if len(last) > 1 else hba1c,
                "LBXGLU": last[2] if len(last) > 2 else glucose,
                "BMXBMI": last[3] if len(last) > 3 else bmi,
                "BPXOSY1": last[4] if len(last) > 4 else systolic_bp,
                "BPXODI1": last[5] if len(last) > 5 else diastolic_bp,
            }
            from src.models.predict import predict_patient
            previous_risks, _, _ = predict_patient(prev_values)

    st.subheader("Patient summary")
    summary_cols = st.columns(4)
    summary_cols[0].metric("Age", age)
    summary_cols[1].metric("BMI", f"{bmi:.1f}")
    summary_cols[2].metric("BP", f"{systolic_bp}/{diastolic_bp}")
    summary_cols[3].metric("HbA1c", f"{hba1c:.1f}%")

    st.subheader("Estimated risk")
    if previous_risks is not None:
        avg_delta, trend = compute_risk_delta(risks, previous_risks)
        if trend == "up":
            st.info(f"Compared with the most recent visit, the average complication risk is up by {abs(avg_delta):.1%}. This pattern warrants closer follow-up and risk-reduction planning.")
        elif trend == "down":
            st.success(f"Compared with the most recent visit, the average complication risk is down by {abs(avg_delta):.1%}. Continue the current management plan and monitor the trajectory.")
        else:
            st.warning(f"Compared with the most recent visit, the average complication risk is stable at about {avg_delta:+.1%}. Continue routine monitoring.")
    else:
        st.info("No prior visit is available for this patient. This check-in provides a fresh baseline for future trend monitoring.")

    def risk_band(risk):
        if risk >= 0.7:
            return "High"
        if risk >= 0.4:
            return "Moderate"
        return "Low"

    risk_cols = st.columns(len(risks))
    for idx, (complication, risk) in enumerate(risks.items()):
        delta = None
        direction = "new-patient"
        if previous_risks is not None:
            delta, direction = compute_risk_delta({complication: risk}, previous_risks)
        label = complication.capitalize()
        if previous_risks is not None and direction == "up":
            trend_text = f"↑ {abs(delta):.1%} vs last visit"
        elif previous_risks is not None and direction == "down":
            trend_text = f"↓ {abs(delta):.1%} vs last visit"
        elif previous_risks is not None and direction == "stable":
            trend_text = "stable vs last visit"
        else:
            trend_text = "new patient baseline"

        with risk_cols[idx]:
            st.markdown(f"### {label}")
            st.metric("Risk", f"{risk:.0%}", delta=trend_text)
            st.caption(f"Clinical band: {risk_band(risk)}")
            st.progress(min(max(risk, 0.0), 1.0))
            if out:
                st.caption(out["narrative"].get(complication, ""))
    st.caption("Decision-support estimate only -- not a diagnosis.")

    rec_cols = st.columns(2)
    recommendations = generate_clinical_recommendations(values, risks)
    with rec_cols[0]:
        st.subheader("Clinician follow-up plan")
        for rec in recommendations:
            st.markdown(f"- {rec}")
    with rec_cols[1]:
        st.subheader("Visit context")
        if patient_id:
            st.write(f"**Patient ID:** {patient_id}")
            st.write(f"**History:** {len(get_patient_history(patient_id)) + 1} recorded visit(s)")
        else:
            st.write("**Patient ID:** not provided")
            st.write("**History:** no prior records available")
        st.write(f"**Fasting glucose:** {glucose} mg/dL")
        st.write(f"**LDL:** {ldl} mg/dL")
        st.write(f"**HDL:** {hdl} mg/dL")

    active = get_active_model_version("nhanes_active")
    if active:
        log_prediction("nhanes_active", active["version_id"], risks, patient_id or None)

    if patient_id:
        add_visit(patient_id, hba1c=hba1c, glucose=glucose, bmi=bmi,
                   systolic_bp=systolic_bp, diastolic_bp=diastolic_bp)
        st.info(f"Check-in saved for patient '{patient_id}'.")
