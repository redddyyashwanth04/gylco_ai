"""
Explanation-stability testing -- one of the project's stated differentiators
(from the general literature survey; the MDPI/Joseph et al. paper does this,
Zamani et al. 2026 does NOT, which is one of the gaps our Track A work
explicitly fills per Base_Papers_Deep_Dive_Blueprint.md).

WHAT THIS CHECKS
    If we perturb a patient's input slightly (small realistic noise on their
    lab values), do the model's SHAP-ranked top features stay roughly the
    same, or do they flip around? An explanation that changes its "reasons"
    under tiny, clinically-insignificant noise isn't trustworthy enough to
    show a clinician -- this check turns that concern into a reportable
    number instead of leaving it as an assumption.

REPORTING
    Report this as a single stability score (see below), not just a
    qualitative "explanations seemed stable" -- the MDPI paper's version of
    this test used only 20 patients due to compute cost; we should use as
    many as feasible from our validation set and state exactly how many,
    matching their honest disclosure of that limitation.
"""

import numpy as np


def perturb_input(x, noise_std_frac=0.02, n_perturbations=10, seed=None):
    """
    Returns n_perturbations slightly-noised copies of one patient's feature
    vector x -- noise_std_frac matches the same small, clinically-plausible
    scale used in augmentation.py's jitter() function, intentionally, so
    the perturbation here represents realistic measurement noise, not an
    arbitrary stress test.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    return [x + rng.normal(0, noise_std_frac * np.abs(x)) for _ in range(n_perturbations)]


def top_k_overlap(shap_values_a, shap_values_b, feature_names, k=5):
    """
    Jaccard-style overlap between the top-k SHAP features of two
    explanations for what should be "the same" patient (original vs.
    perturbed) -- 1.0 means identical top-k feature sets, 0.0 means no
    overlap at all.
    """
    top_a = set(sorted(zip(feature_names, shap_values_a), key=lambda x: -abs(x[1]))[:k])
    top_b = set(sorted(zip(feature_names, shap_values_b), key=lambda x: -abs(x[1]))[:k])
    names_a = {f for f, _ in top_a}
    names_b = {f for f, _ in top_b}
    if not names_a and not names_b:
        return 1.0
    return len(names_a & names_b) / len(names_a | names_b)


def stability_score_for_patient(model_explainer_fn, x, feature_names, n_perturbations=10, k=5, seed=None):
    """
    model_explainer_fn(x) -> shap_values for one patient's feature vector.
    Returns the mean top-k overlap between the original explanation and
    n_perturbations perturbed versions -- this single number IS the
    stability score to report for this patient.
    """
    original_shap = model_explainer_fn(x)
    perturbed_inputs = perturb_input(x, n_perturbations=n_perturbations, seed=seed)
    overlaps = [
        top_k_overlap(original_shap, model_explainer_fn(px), feature_names, k=k)
        for px in perturbed_inputs
    ]
    return float(np.mean(overlaps))


if __name__ == "__main__":
    print("Explanation-stability module ready.")
    print("Requires a fitted model's SHAP explainer (from shap_explainer.py)")
    print("wrapped as model_explainer_fn(x) -> shap values for one patient.")
