"""
Home / Overview page -- the entry point of the 5-page clinician-facing app.

WHAT THIS PAGE SHOWS
    - A plain-language summary of what the tool does and which two tracks
      it's built on (mirrors the framing in Product_Story_Build_to_Use.md).
    - Current model status pulled LIVE from the model registry
      (src/storage/db.py's get_active_model_version()) -- active version,
      when it was last trained, its validation performance. This is the
      transparency the retraining pipeline needs to be trustworthy rather
      than a black box quietly changing behavior.
    - Navigation guidance to the other 4 pages (Streamlit's multi-page app
      support handles the actual navigation via the pages/ folder -- this
      page doesn't need to build its own nav).

RUN WITH: streamlit run app/Home.py
"""

import streamlit as st
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.storage.db import init_db, get_active_model_version

st.set_page_config(page_title="T2D Progression Tool", page_icon="\U0001F4CA", layout="wide")

init_db()  # safe to call every startup -- creates tables if they don't exist yet

st.title("Type-2 Diabetes Progression Prediction")
st.caption("A clinician-facing decision-support tool -- not a diagnostic device.")

st.markdown("""
This tool supports two kinds of clinical questions:

- **Current Risk** -- given a patient's values today, what's their multi-complication risk picture?
- **Progression Forecast** -- given a patient's history, where is their glycemic trajectory heading?

Use the sidebar to navigate between pages.
""")

st.subheader("Current model status")

# Show the models actually trained and registered in this project.
for model_name in ["nhanes_active", "mimic_glucose_forecaster_lstm"]:
    active = get_active_model_version(model_name)
    if active is None:
        st.info(f"**{model_name}**: no trained version registered yet.")
    else:
        val_auc = active.get("val_auc")
        val_text = f", validation AUC {val_auc:.3f}" if val_auc is not None else ""
        st.success(
            f"**{model_name}**: version {active['version_id']}, "
            f"trained {active['trained_at']}{val_text}"
        )
