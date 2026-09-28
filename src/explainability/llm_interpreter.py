"""
LLM interpretability layer.

WHAT THIS DOES
    SHAP already tells us *which* features drove a prediction and by how much
    (that's the ground-truth explanation, and it's what gets reported in the paper).
    What SHAP doesn't do is turn "URXUMA: +0.31, BPQ020: +0.18, BMXBMI: +0.12"
    into a sentence a clinician can read in two seconds. That's this module's
    only job: translate real, already-computed SHAP numbers into a short,
    plain-language narrative -- it does not make its own predictions or
    invent any numbers.

WHY THIS MATTERS FOR THE PROJECT
    This is what turns Mode 1 and Mode 2 of the demo app from "a bar chart"
    into "a bar chart *and* a sentence a doctor would actually read." It's
    the natural-language layer on top of the Explainability layer in the
    architecture, not a replacement for SHAP.

GROUNDING RULE (important, do not relax this)
    The prompt sent to the LLM contains ONLY the real SHAP values, the real
    feature values, and the real model prediction for one specific patient.
    The LLM is explicitly instructed to use only those numbers and never
    introduce a statistic, study finding, or claim that wasn't given to it.
    This prevents hallucinated medical claims from reaching a clinician.

STATUS
    This module is fully written and runnable right now against the example
    at the bottom of the file. It cannot yet run against REAL patients
    because src/models/ and src/explainability/shap_explainer.py don't exist
    yet (Steps 2-4). Once those exist, swap out `EXAMPLE_SHAP_OUTPUT` below
    for real output from the trained model + SHAP explainer.

SETUP
    pip install groq
    export GROQ_API_KEY=gsk_...
    Free key: https://console.groq.com/keys (no credit card required)
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import LLM_MODEL

try:
    from groq import Groq
except ImportError:
    Groq = None


# Human-readable names for the raw NHANES column codes, so the LLM prompt
# (and the final explanation) never shows a clinician a raw variable code.
FEATURE_LABELS = {
    "LBXGH": "HbA1c (%)",
    "LBXGLU": "fasting glucose (mg/dL)",
    "BMXBMI": "Body Mass Index",
    "BPXOSY1": "systolic blood pressure (mmHg)",
    "BPXODI1": "diastolic blood pressure (mmHg)",
    "URXUMA": "urine albumin (mg/L) -- kidney marker",
    "URXUCR": "urine creatinine (mg/dL) -- kidney marker",
    "LBXSCR": "serum creatinine (mg/dL) -- kidney function",
    "LBDLDL": "LDL cholesterol (mg/dL)",
    "LBDHDD": "HDL cholesterol (mg/dL)",
    "LBXTLG": "triglycerides (mg/dL)",
    "RIDAGEYR": "age (years)",
    "PAD680": "sedentary minutes per day",
    "SMQ020": "smoking history",
    "ALQ121": "alcohol use frequency",
    "INDFMPIR": "income-to-poverty ratio",
}


def build_prompt(patient_id, complication_name, predicted_risk, shap_values, feature_values):
    """
    Constructs a strictly grounded prompt: every number the LLM is allowed
    to reference is listed explicitly. Nothing else is supplied, so there is
    nothing else for it to draw on.
    """
    lines = []
    for feature_code, contribution in sorted(shap_values.items(), key=lambda x: -abs(x[1]))[:5]:
        label = FEATURE_LABELS.get(feature_code, feature_code)
        value = feature_values.get(feature_code, "unknown")
        direction = "increased" if contribution > 0 else "decreased"
        lines.append(f"- {label} = {value} ({direction} risk, SHAP contribution {contribution:+.3f})")
    factor_block = "\n".join(lines)

    prompt = f"""You are helping a clinician read a machine learning model's risk explanation for one patient (ID: {patient_id}).

Predicted risk of {complication_name}: {predicted_risk:.1%}

The model's top contributing factors for THIS patient, from SHAP analysis (positive = pushes risk up, negative = pushes risk down):
{factor_block}

Write a 3-4 sentence plain-language explanation for a clinician, in a neutral clinical tone.
Rules you must follow exactly:
- Only reference the factors and numbers listed above. Do not introduce any other statistic, study, or claim.
- Do not state or imply a diagnosis. This is a risk estimate from a decision-support tool, not a diagnosis.
- Do not recommend a specific treatment or medication.
- End with one neutral sentence reminding the reader this is a decision-support estimate, not a clinical diagnosis."""
    return prompt


def generate_explanation(patient_id, complication_name, predicted_risk, shap_values, feature_values):
    """
    Calls the Groq API to turn one patient's SHAP output into a short
    clinical narrative. Returns the explanation text, or a clear error
    message if the API key isn't set / the call fails -- this function
    should never crash the demo app if the LLM call has a problem, since
    the SHAP bar chart is still shown regardless.
    """
    if Groq is None:
        return "[LLM interpretability layer unavailable: run `pip install groq`]"

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return "[LLM interpretability layer unavailable: set the GROQ_API_KEY environment variable]"

    prompt = build_prompt(patient_id, complication_name, predicted_risk, shap_values, feature_values)

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=LLM_MODEL,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[LLM interpretability layer error: {e}]"


# ---------------------------------------------------------------------------
# Runnable example using placeholder SHAP output (not a real patient yet).
# Once shap_explainer.py exists (Step 4), replace EXAMPLE_* below with real
# output pulled from the trained model.
# ---------------------------------------------------------------------------
EXAMPLE_SHAP_OUTPUT = {
    "URXUMA": 0.31,
    "BPQ020": 0.18,
    "BMXBMI": 0.12,
    "LBXGH": 0.09,
    "PAD680": -0.05,
}
EXAMPLE_FEATURE_VALUES = {
    "URXUMA": "142 mg/L (elevated)",
    "BPQ020": "Yes (hypertension diagnosed)",
    "BMXBMI": "31.4",
    "LBXGH": "7.2%",
    "PAD680": "180 min/day",
}

if __name__ == "__main__":
    explanation = generate_explanation(
        patient_id="EXAMPLE_001",
        complication_name="nephropathy",
        predicted_risk=0.64,
        shap_values=EXAMPLE_SHAP_OUTPUT,
        feature_values=EXAMPLE_FEATURE_VALUES,
    )
    print("--- Generated explanation ---")
    print(explanation)
