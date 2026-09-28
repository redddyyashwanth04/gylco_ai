"""
Track B static (first-visit-only) baseline, replicating Huang et al. (2025)
exactly, as the benchmark before the LSTM extension is added.

ARCHITECTURE (matching the base paper, per Base_Papers_Deep_Dive_Blueprint.md)
    - Four models compared: Random Forest, XGBoost, SVM, Logistic Regression
    - LASSO feature selection narrowing available variables down to a final
      predictor set (paper used 56 -> 12; our available MIMIC-IV Demo
      variables will likely start smaller, so document the actual before/after
      counts rather than assuming the paper's exact numbers transfer)
    - Uses only each patient's FIRST recorded values -- deliberately static,
      matching the base paper's design, so the comparison against the LSTM
      (which uses the FULL trajectory) is fair and specific

REQUIRED: report the train-vs-validation AUC gap for every model here,
benchmarked against the base paper's disclosed 0.219 gap (their training
AUC 0.999 vs validation AUC 0.780) -- given our even smaller cohort, this
check matters more here than it did for them, not less.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import MIMIC_PATIENT_SUMMARY

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.feature_selection import SelectFromModel


def build_lasso_selector():
    """Matches the base paper's LASSO feature-reduction step."""
    return SelectFromModel(LassoCV(cv=5, max_iter=5000))


def build_models():
    """The four models compared in the base paper -- returns a dict of unfitted estimators."""
    import xgboost as xgb
    return {
        "random_forest": RandomForestClassifier(class_weight="balanced", n_estimators=300),
        "xgboost": xgb.XGBClassifier(eval_metric="logloss"),
        "svm": SVC(probability=True, class_weight="balanced"),
        "logistic_regression": LogisticRegression(class_weight="balanced", max_iter=1000),
    }


def get_first_visit_features(patient_summary_df):
    """
    Extracts the "static, first-visit-only" feature view from
    mimic_patient_summary.csv -- deliberately uses only first_hba1c (not
    last_hba1c, not hba1c_trend), to match the base paper's single-snapshot
    design. This is the feature set the static baseline trains on; contrast
    with the LSTM in lstm_trajectory.py which uses the full trajectory.
    """
    # TODO: select first_hba1c, demographics, and any other first-visit-only
    # columns from mimic_patient_summary.csv; explicitly EXCLUDE hba1c_trend,
    # last_hba1c, follow_up_days, n_hba1c_readings -- those encode
    # information from later visits, which the static baseline must not see.
    raise NotImplementedError("First-visit feature extraction not yet implemented")


def main():
    print(f"Expecting patient summary data at: {MIMIC_PATIENT_SUMMARY}")
    print(f"Exists: {MIMIC_PATIENT_SUMMARY.exists()}")
    print("\nModels defined: random_forest, xgboost, svm, logistic_regression")
    print("TODO: extract first-visit features, apply LASSO selection, fit each")
    print("model via lopo_cv.py, report train/val AUC gap vs. base paper's 0.219.")


if __name__ == "__main__":
    main()
