"""
Shared evaluation utilities for both tracks -- one place for metrics, so
every model (Track A's traditional/stacking ensemble, Track B's static
baseline/LSTM) is scored the same way and every results table in the paper
is directly comparable.

METRICS MATCHING EACH BASE PAPER (for direct numeric comparison)
    Track A (vs. Zamani et al. 2026): Hamming Loss, F1, AUC
    Track B (vs. Huang et al. 2025): AUC, with train-vs-validation gap
        reported for every model, benchmarked against their disclosed 0.219
        gap

REQUIRED PRACTICE, NOT OPTIONAL
    report_train_val_gap() must be called for every model trained in this
    project, not just Track B's -- given our overall small sample sizes
    (8,153 for Track A is fine; 13-35 for Track B is not), overfitting risk
    is a project-wide concern, not just a Track B one.
"""

from sklearn.metrics import hamming_loss, f1_score, roc_auc_score


def evaluate_multilabel(y_true, y_pred, y_pred_proba=None):
    """
    Track A style evaluation -- matches Zamani et al. (2026)'s reported
    metrics exactly so results are directly comparable.
    y_true, y_pred: (n_samples, n_labels) binary arrays
    y_pred_proba: (n_samples, n_labels) probabilities, needed for AUC
    """
    results = {
        "hamming_loss": hamming_loss(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }
    if y_pred_proba is not None:
        # TODO: per-label AUC then average, since roc_auc_score needs care
        # with multi-label input -- confirm sklearn version/API before
        # finalizing this call.
        results["auc_macro"] = None
    return results


def report_train_val_gap(train_auc, val_auc, comparison_gap=0.219, comparison_label="Huang et al. (2025)"):
    """
    Prints and returns the train/validation AUC gap alongside the disclosed
    gap from the relevant base paper, flagging anything approaching or
    exceeding it as a warning -- this is a REQUIRED check per model, not
    just for Track B.
    """
    gap = train_auc - val_auc
    flag = gap >= comparison_gap * 0.8  # flag once we're within 80% of their disclosed overfitting gap
    print(f"Train AUC: {train_auc:.3f} | Validation AUC: {val_auc:.3f} | Gap: {gap:.3f}")
    print(f"  (compare to {comparison_label}'s disclosed gap of {comparison_gap:.3f})")
    if flag:
        print(f"  WARNING: gap is approaching the base paper's overfitting signature -- "
              f"consider regularizing further or reducing model complexity.")
    return {"train_auc": train_auc, "val_auc": val_auc, "gap": gap, "flagged": flag}


if __name__ == "__main__":
    # smoke test with made-up numbers
    print("Smoke test of report_train_val_gap():")
    report_train_val_gap(train_auc=0.95, val_auc=0.80)
