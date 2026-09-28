"""
Track A training pipeline. Run this to actually train models on real NHANES
data and save them.

Run: python src/models/train.py
"""

import sys
import pickle
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import DATA_MODEL_READY, MODELS_SAVED
from src.storage.db import init_db, register_model_version, promote_model

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import roc_auc_score, f1_score

# obesity deliberately NOT a target: it is defined as BMI>=30 and BMI is a feature (definitional leak)
TARGET_COLUMNS = ["hypertension", "nephropathy", "cardiovascular"]
# columns that were used to BUILD the targets above (build_targets.py) --
# these must be excluded from features, or the model just reads the answer
# directly (this was caught by seeing a suspicious perfect 1.0 AUC on the
# first training run -- always be suspicious of a perfect score)
LEAKAGE_COLUMNS = ["BPQ020", "KIQ022", "URXUMA", "MCQ160B", "MCQ160C", "MCQ160E", "MCQ160F", "OBESITY_FLAG"]
DROP_COLUMNS = ["SEQN", "diabetes", "obesity"] + TARGET_COLUMNS + LEAKAGE_COLUMNS


def load_data():
    path = DATA_MODEL_READY / "nhanes_model_ready.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run preprocess_nhanes.py, feature_engineering.py, "
            f"and build_targets.py first."
        )
    return pd.read_csv(path)


def split_data(df):
    feature_cols = [c for c in df.columns if c not in DROP_COLUMNS]
    X = df[feature_cols]
    y = df[TARGET_COLUMNS]
    # split by row is fine here -- each NHANES row is already one unique patient (SEQN)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, feature_cols


def train_traditional_baseline(X_train, y_train):
    """One independent logistic regression per complication -- the traditional-model comparison."""
    model = MultiOutputClassifier(make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000)))
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    """The nonlinear comparison model."""
    model = MultiOutputClassifier(RandomForestClassifier(class_weight="balanced", n_estimators=300, random_state=42))
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting(X_train, y_train):
    """Gradient boosting -- usually the strongest model on tabular clinical data."""
    model = MultiOutputClassifier(HistGradientBoostingClassifier(class_weight="balanced", random_state=42))
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)  # list of arrays, one per target
    aucs, f1s = [], []
    for i, col in enumerate(TARGET_COLUMNS):
        try:
            auc = roc_auc_score(y_test[col], y_proba[i][:, 1])
        except ValueError:
            auc = None  # only one class present in this split
        f1 = f1_score(y_test[col], y_pred[:, i])
        aucs.append(auc)
        f1s.append(f1)
        print(f"  {col:15s} AUC={auc if auc is None else round(auc,3)}  F1={round(f1,3)}")
    valid_aucs = [a for a in aucs if a is not None]
    mean_auc = sum(valid_aucs) / len(valid_aucs) if valid_aucs else None
    mean_f1 = sum(f1s) / len(f1s)
    return mean_auc, mean_f1


def save_model(model, name):
    MODELS_SAVED.mkdir(parents=True, exist_ok=True)
    path = MODELS_SAVED / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return path


def main():
    init_db()
    df = load_data()
    X_train, X_test, y_train, y_test, feature_cols = split_data(df)
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows | Features: {len(feature_cols)}")

    print("\n--- Traditional baseline (logistic regression) ---")
    baseline = train_traditional_baseline(X_train, y_train)
    baseline_auc, baseline_f1 = evaluate(baseline, X_test, y_test)
    print(f"  Mean AUC={baseline_auc}  Mean F1={round(baseline_f1,3)}")
    baseline_path = save_model(baseline, "nhanes_traditional_baseline")

    print("\n--- Nonlinear model 1: Random Forest ---")
    rf_model = train_random_forest(X_train, y_train)
    rf_auc, rf_f1 = evaluate(rf_model, X_test, y_test)
    print(f"  Mean AUC={round(rf_auc,3)}  Mean F1={round(rf_f1,3)}")
    save_model(rf_model, "nhanes_random_forest")

    print("\n--- Nonlinear model 2: Gradient Boosting ---")
    gb_model = train_gradient_boosting(X_train, y_train)
    gb_auc, gb_f1 = evaluate(gb_model, X_test, y_test)
    print(f"  Mean AUC={round(gb_auc,3)}  Mean F1={round(gb_f1,3)}")
    save_model(gb_model, "nhanes_gradient_boosting")

    print("\n--- Comparison (central claim of the project) ---")
    print(f"  Logistic regression (traditional): AUC={round(baseline_auc,3)}  F1={round(baseline_f1,3)}")
    print(f"  Random Forest:                     AUC={round(rf_auc,3)}  F1={round(rf_f1,3)}")
    print(f"  Gradient Boosting:                 AUC={round(gb_auc,3)}  F1={round(gb_f1,3)}")

    # the best nonlinear model (by AUC) becomes the active model the app uses
    best_name, best_auc, best_f1 = max(
        [("nhanes_random_forest", rf_auc, rf_f1), ("nhanes_gradient_boosting", gb_auc, gb_f1)],
        key=lambda t: t[1])
    version_id = register_model_version(
        model_name="nhanes_active", n_original_rows=len(X_train), n_app_rows=0,
        val_auc=best_auc, val_f1=best_f1, notes=f"best nonlinear model: {best_name}")
    promote_model(version_id, "nhanes_active")
    with open(MODELS_SAVED / "nhanes_active_name.txt", "w") as f:
        f.write(best_name)
    print(f"\nActive model: {best_name} (registry version {version_id})")

    # save feature column order alongside the model -- predict.py needs this
    with open(MODELS_SAVED / "nhanes_feature_columns.pkl", "wb") as f:
        pickle.dump(feature_cols, f)


if __name__ == "__main__":
    main()
