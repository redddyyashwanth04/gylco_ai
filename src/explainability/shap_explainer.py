"""
SHAP explainability -- the ground-truth, paper-reported explanation layer
for both tracks. This is what llm_interpreter.py (already built) consumes
to generate the plain-language narrative; SHAP values themselves are what
get reported as numbers in the paper's Results section.

WHAT TO CHECK AGAINST THE BASE PAPERS ONCE THIS RUNS
    Zamani et al. (2026) found: macrovascular complications driven mainly by
    LDL cholesterol and diastolic BP; microvascular by drug-use history,
    fasting glucose, and HDL; age and BMI showed MINIMAL importance.
    Run this on the Track A NHANES model and compare directly -- confirming
    or contradicting this finding is itself a reportable result, per
    Base_Papers_Deep_Dive_Blueprint.md.
"""

import shap
import pandas as pd


def explain_tree_model(model, X, feature_names=None):
    """
    SHAP TreeExplainer for tree-based models (Random Forest, LightGBM,
    CatBoost, XGBoost) -- fast, exact, appropriate for the stacking ensemble
    and static baseline models in both tracks.
    Returns (explainer, shap_values) -- callers use shap_values for both
    plotting and for feeding into llm_interpreter.py's per-patient prompt.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return explainer, shap_values


def top_features_for_patient(shap_values, feature_names, patient_idx, top_n=5):
    """
    Returns the top_n (feature_name, shap_value) pairs for one patient --
    this is the exact input format llm_interpreter.py's build_prompt()
    expects as its shap_values dict.
    """
    patient_values = shap_values[patient_idx]
    paired = list(zip(feature_names, patient_values))
    paired.sort(key=lambda x: -abs(x[1]))
    return dict(paired[:top_n])


def summarize_global_importance(shap_values, feature_names):
    """
    Mean absolute SHAP value per feature across all patients -- this is
    what gets compared against Zamani et al.'s reported top features
    (LDL, diastolic BP, drug-use history, fasting glucose, HDL) and their
    minimal-importance finding for age/BMI.
    """
    import numpy as np
    mean_abs = np.abs(shap_values).mean(axis=0)
    ranked = sorted(zip(feature_names, mean_abs), key=lambda x: -x[1])
    return ranked


if __name__ == "__main__":
    print("SHAP explainability module ready.")
    print("Requires a trained model (from stacking_ensemble.py or")
    print("mimic_static_baseline.py) and its feature matrix to run against.")
