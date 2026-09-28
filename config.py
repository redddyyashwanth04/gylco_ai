"""
Central configuration: file paths and shared settings.
Every script in src/ should import paths from here rather than hardcoding them,
so moving the project directory only ever requires changing this one file.
"""

from pathlib import Path

# project root = the folder this file lives in
ROOT = Path(__file__).resolve().parent

# data
DATA_RAW_NHANES = ROOT / "data" / "raw" / "nhanes"
DATA_RAW_MIMIC = ROOT / "data" / "raw" / "mimic_demo"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_MODEL_READY = ROOT / "data" / "model_ready"

NHANES_MERGED = DATA_PROCESSED / "nhanes_merged.csv"
MIMIC_TRAJECTORIES = DATA_PROCESSED / "mimic_trajectories.csv"
MIMIC_PATIENT_SUMMARY = DATA_PROCESSED / "mimic_patient_summary.csv"

# models
MODELS_SAVED = ROOT / "models_saved"

# LLM interpretability layer -- using Groq's free tier (no credit card required,
# rate-limited but generous: currently ~14,400 requests/day on Llama models).
# Set your API key as an environment variable, never hardcode it:
#   export GROQ_API_KEY=gsk_...      (Mac/Linux)
#   setx GROQ_API_KEY "gsk_..."      (Windows, new terminal after)
# Get a free key at: https://console.groq.com/keys
#
# IMPORTANT: Groq's free-tier model catalog changes over time (models get
# added/retired). Before relying on the default below, verify it's still
# live by running: python -c "from groq import Groq; import os;
# print([m.id for m in Groq(api_key=os.environ['GROQ_API_KEY']).models.list().data])"
LLM_MODEL = "llama-3.3-70b-versatile"  # strong general instruction-following --
                                         # kept general-purpose deliberately, not a medical
                                         # fine-tune, since llm_interpreter.py's task is strict
                                         # grounding to given SHAP numbers, not medical recall
                                         # (see reports/ discussion on domain-specific LLMs)
