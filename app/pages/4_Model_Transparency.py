"""
Model Transparency page -- shows the full model registry: every trained
version, its validation metrics, and whether it's currently active.

WHY THIS PAGE EXISTS (don't cut it even if time is short)
    The retraining pipeline (src/models/retrain_pipeline.py) can promote a
    new model version automatically once it passes the validation gate. A
    clinical tool that silently changes its own behavior with no visibility
    is worse than one that doesn't retrain at all. This page is what makes
    the feedback loop trustworthy instead of a black box -- see
    Tech_Stack_Methodology_Architecture.md Section 1.6 for the full reasoning.
"""

import streamlit as st
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.storage.db import get_connection

st.set_page_config(page_title="Model Transparency", page_icon="\U0001F50D")
st.title("Model Transparency")
st.caption("Every trained model version, its validation performance, and which one is currently active.")

conn = get_connection()
rows = conn.execute(
    """SELECT model_name, version_id, trained_at, n_original_rows, n_app_rows,
              val_auc, val_f1, is_active, notes
       FROM model_registry ORDER BY model_name, trained_at DESC"""
).fetchall()
conn.close()

if not rows:
    st.info(
        "No model versions registered yet. This page will populate once "
        "src/models/retrain_pipeline.py's register_model_version() has been "
        "called by a real training run."
    )
else:
    for model_name, version_id, trained_at, n_orig, n_app, val_auc, val_f1, is_active, notes in rows:
        badge = "\U0001F7E2 ACTIVE" if is_active else "inactive"
        with st.container(border=True):
            st.write(f"**{model_name}** -- version {version_id} -- {badge}")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Validation AUC", f"{val_auc:.3f}" if val_auc else "N/A")
            col2.metric("Validation F1", f"{val_f1:.3f}" if val_f1 else "N/A")
            col3.metric("Original rows", n_orig)
            col4.metric("Real app-collected rows", n_app)
            st.caption(f"Trained: {trained_at}")
            if notes:
                st.caption(notes)
