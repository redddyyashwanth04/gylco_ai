"""
Track A model: stacking ensemble replicating Zamani et al. (2026), plus the
traditional-baseline comparison their paper omits.

ARCHITECTURE (matching the base paper exactly, per Base_Papers_Deep_Dive_Blueprint.md)
    - Base learners: Random Forest, LightGBM, CatBoost
    - Two multi-label wrapper strategies, both to be trained and compared:
        - Binary Relevance: one independent model per complication
        - Classifier Chain: each complication's model gets earlier
          complications' predictions as extra input features
    - Class-weighted training (complications are minority classes)
    - Meta-learner: multi-output logistic regression combining the three
      base learners' outputs

ALSO INCLUDED: the traditional baseline
    Single-label logistic regression, one per complication, with NO
    ensembling and NO cross-complication information sharing. This produces
    the number our own problem statement is actually about -- the gap
    between this and the stacking ensemble above IS the central empirical
    claim of the paper.

TARGETS: hypertension, nephropathy, cardiovascular, obesity (from
build_targets.py) -- diabetes itself is typically NOT included as a
complication target since it's the primary condition, not a complication,
but check with build_targets.py's TARGET_COLUMNS before finalizing.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import DATA_MODEL_READY

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.multioutput import MultiOutputClassifier, ClassifierChain

COMPLICATION_TARGETS = ["hypertension", "nephropathy", "cardiovascular", "obesity"]


def build_traditional_baseline():
    """One independent logistic regression per complication -- no ensembling."""
    return MultiOutputClassifier(
        LogisticRegression(class_weight="balanced", max_iter=1000)
    )


def build_stacking_base_learners():
    """
    Returns the three base learners matching the paper. LightGBM and
    CatBoost need `pip install lightgbm catboost` -- imported lazily here so
    this file can at least be imported/inspected without those installed yet.
    """
    import lightgbm as lgb
    import catboost as cb

    return [
        ("random_forest", RandomForestClassifier(class_weight="balanced", n_estimators=300)),
        ("lightgbm", lgb.LGBMClassifier(class_weight="balanced")),
        ("catboost", cb.CatBoostClassifier(verbose=0, auto_class_weights="Balanced")),
    ]


def build_stacking_binary_relevance():
    """Binary Relevance: complications predicted independently."""
    base_learners = build_stacking_base_learners()
    stack = StackingClassifier(
        estimators=base_learners,
        final_estimator=LogisticRegression(max_iter=1000),
    )
    return MultiOutputClassifier(stack)


def build_stacking_classifier_chain(random_state=42):
    """Classifier Chain: each complication's prediction can use earlier ones."""
    base_learners = build_stacking_base_learners()
    stack = StackingClassifier(
        estimators=base_learners,
        final_estimator=LogisticRegression(max_iter=1000),
    )
    return ClassifierChain(stack, random_state=random_state)


def main():
    data_path = DATA_MODEL_READY / "nhanes_model_ready.csv"
    print(f"Expecting model-ready data at: {data_path}")
    print(f"Exists: {data_path.exists()}")
    print("\nModels defined:")
    print("  - build_traditional_baseline()      : single-label logistic regression")
    print("  - build_stacking_binary_relevance()  : RF+LightGBM+CatBoost, BR")
    print("  - build_stacking_classifier_chain()  : RF+LightGBM+CatBoost, CC")
    print("\nTODO: load nhanes_model_ready.csv, split by patient, fit each model,")
    print("evaluate with evaluate.py, compare against paper's F1=0.752 / AUC=0.857.")


if __name__ == "__main__":
    main()
