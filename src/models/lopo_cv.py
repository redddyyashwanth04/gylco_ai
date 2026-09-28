"""
Leave-one-patient-out cross-validation (LOPO-CV) for Track B.

WHY THIS EXISTS
    With only 13-35 patients, a single 80/20 train/validation split wastes
    data and gives a noisy, unstable performance estimate -- one unlucky
    split could make a good model look bad, or vice versa. LOPO-CV trains on
    all-but-one patient and tests on the held-out one, repeating for every
    patient, so every single patient is used for both training and testing
    (across different folds) and the reported average is far more credible
    at this sample size. This is standard practice for small clinical
    cohorts, not a workaround specific to this project.

    Contrast with Huang et al. (2025, our Track B base paper), who use a
    single 80/20 split on their much larger cohort (1,313 patients) -- that
    choice is reasonable at their scale and would NOT be reasonable at ours.

USAGE
    Works with any model that has a scikit-learn-style .fit(X, y) /
    .predict_proba(X) interface, including a wrapped PyTorch model.
"""

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score, f1_score


def run_lopo_cv(X, y, patient_ids, train_fn, predict_fn):
    """
    X: feature matrix (n_samples x n_features)
    y: binary target array
    patient_ids: array-like, same length as X/y, identifying which patient
        each row belongs to (a patient may contribute multiple rows if using
        window-sliced augmented data -- see augmentation.py -- in which case
        ALL of that patient's rows, real and augmented, must stay together
        in the same fold to avoid leakage)
    train_fn(X_train, y_train) -> trained model
    predict_fn(model, X_test) -> predicted probabilities

    Returns a dict with per-fold and averaged AUC/F1, plus the full list of
    (patient_id, true_label, predicted_probability) for inspection.
    """
    logo = LeaveOneGroupOut()
    fold_results = []
    predictions = []

    for train_idx, test_idx in logo.split(X, y, groups=patient_ids):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        held_out_patient = patient_ids[test_idx[0]]

        model = train_fn(X_train, y_train)
        y_pred = predict_fn(model, X_test)

        predictions.extend(
            zip([held_out_patient] * len(y_test), y_test, y_pred)
        )

        # AUC is undefined for a single test point with only one class present
        # (which is the norm here, since each fold holds out one patient) --
        # so per-fold AUC isn't meaningful; only the aggregated AUC across all
        # held-out predictions (computed after the loop) is reported.
        fold_results.append({"patient_id": held_out_patient, "y_true": y_test, "y_pred": y_pred})

    all_true = np.array([p[1] for p in predictions])
    all_pred = np.array([p[2] for p in predictions])

    overall_auc = roc_auc_score(all_true, all_pred) if len(set(all_true)) > 1 else None
    overall_f1 = f1_score(all_true, (all_pred >= 0.5).astype(int)) if len(set(all_true)) > 1 else None

    return {
        "n_folds": len(fold_results),
        "overall_auc": overall_auc,
        "overall_f1": overall_f1,
        "predictions": predictions,
    }


if __name__ == "__main__":
    print("LOPO-CV harness ready. Wire in train_fn/predict_fn from a real model")
    print("(e.g. src/models/mimic_static_baseline.py) once training data is prepared.")
