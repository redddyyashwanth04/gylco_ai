# t2d_project — Full Project Structure

Working directory for the Type-2 Diabetes Progression Prediction project.
Every file below is either fully working, or a real stub with a complete
docstring explaining exactly what it needs to do and why — nothing is an
empty placeholder.

## Complete directory tree

```
t2d_project/
├── README.md
├── config.py                          # central file paths, LLM model choice
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   │   ├── nhanes/                    # your .XPT files go here (not committed to git)
│   │   └── mimic_demo/                # 6 .csv.gz files — already present, verified
│   ├── external/                      # UCI Diabetes-130 (pretraining source) goes here
│   ├── processed/                     # ✅ DONE — nhanes_merged.csv, mimic_trajectories.csv,
│   │                                  #     mimic_patient_summary.csv (all verified)
│   ├── augmented/                     # output of augmentation.py
│   └── model_ready/                   # output of preprocessing + feature engineering
│
├── src/
│   ├── data_prep/                     # ✅ DONE — built, tested, already run
│   │   ├── merge_nhanes_core.py
│   │   ├── inspect_xpt.py
│   │   └── build_mimic_trajectories.py
│   │
│   ├── features/                      # ⬜ STUBBED — Step 2, do this next
│   │   ├── preprocess_nhanes.py       # imputation, sentinel cleanup, drop DBQ360
│   │   ├── feature_engineering.py     # lipid ratios, obesity/prediabetes/hypertension flags
│   │   ├── build_targets.py           # 5 multi-label complication targets (Track A)
│   │   ├── preprocess_mimic.py        # sequence formatting for the LSTM (Track B)
│   │   └── augmentation.py            # ✅ working — jitter/window-slice/time-warp,
│   │                                  #     tested against real logic (see tests/)
│   │
│   ├── models/                        # ⬜ STUBBED — Step 3
│   │   ├── stacking_ensemble.py       # Track A: replicates Zamani et al. (2026) + traditional baseline
│   │   ├── mimic_static_baseline.py   # Track B: replicates Huang et al. (2025)
│   │   ├── lstm_trajectory.py         # Track B: our extension — full trajectory, not first-visit-only
│   │   ├── pretrain_finetune.py       # pretrain on UCI Diabetes-130, fine-tune on real MIMIC-IV
│   │   ├── lopo_cv.py                 # ✅ working — leave-one-patient-out CV for small cohorts
│   │   ├── evaluate.py                # ✅ working — shared metrics + train/val gap check
│   │   └── retrain_pipeline.py        # ✅ working — validation-gated model promotion
│   │
│   ├── explainability/                # ⬜ STUBBED — Step 4
│   │   ├── shap_explainer.py          # ground-truth SHAP attribution for both tracks
│   │   ├── stability_check.py         # explanation-robustness testing under input perturbation
│   │   └── llm_interpreter.py         # ✅ working — SHAP → plain-language clinical narrative
│   │
│   └── storage/                       # ✅ DONE
│       └── db.py                      # SQLite: patients, visits, model_registry, prediction_log
│
├── app/                                # ⬜ STUBBED — Step 5, last
│   ├── Home.py                        # overview + live model status
│   └── pages/
│       ├── 1_Current_Risk.py          # Mode 1
│       ├── 2_Progression_Forecast.py  # Mode 2 — MIMIC-IV examples + real tracked patients
│       ├── 3_Patient_History.py       # searchable patient list
│       └── 4_Model_Transparency.py    # model registry, exposed to the user
│
├── tests/                              # ✅ working — 9/9 passing right now
│   ├── test_data_prep.py              # structural checks against your real, current data
│   └── test_augmentation.py           # verifies jitter/window-slice/time-warp logic
│
├── models_saved/                       # trained model artifacts land here (gitignored)
├── notebooks/                          # scratch space, not part of the pipeline
└── reports/                            # every project document written so far (7 files)
```

## What "stubbed" means here
Every stubbed file has: a full docstring explaining exactly what it needs to
do and why, correct imports, real function signatures, and — where the logic
is simple enough to write without a trained model (like `augmentation.py`,
`lopo_cv.py`, `evaluate.py`) — actual working code, not just a plan. Files
that genuinely need a trained model first (`stacking_ensemble.py`'s fitting
step, the app pages' prediction calls) raise `NotImplementedError` at the
exact point real logic needs to go in, with a `TODO` comment saying what
that logic is.

## Running the tests right now
```
pip install pytest
pytest tests/ -v
```
9 tests currently pass against your real, already-verified NHANES and
MIMIC-IV data — this confirms the data pipeline itself hasn't broken, even
before any model code is written.

## Track B's final data strategy: MIMIC-IV Demo, augmented (committed decision)
Full MIMIC-IV credentialed access was applied for and declined by PhysioNet
(requires a verifiable institutional reference email). Rather than continue
pursuing that, **the MIMIC-IV Demo cohort is the permanent Track B dataset**
for this project, expanded via the techniques below. This is a final
decision, not a fallback — the numbers below make a genuinely credible case
for it.

**Primary signal: glucose, not HbA1c** — confirmed by checking the real data,
not assumed. Glucose is checked far more often than HbA1c in ICU care, so
it's dramatically richer in this dataset:
- HbA1c: 20 of 35 patients have any reading, only 7 have 2+
- Glucose: **all 35 patients** have readings, 30 have 5+, one patient alone has 185

**Real numbers, computed by running `src/features/augmentation.py` against
the actual project data** (reproducible — run it yourself, this isn't a projection):

| | Count |
|---|---|
| Real diabetic patients (glucose) | 35 |
| Real glucose readings | 1,212 |
| Windows after slicing (min_window=5, stride=3) | 373 |
| Total training examples after jittering (6x per window) | **2,240** |

**Note for the forecasting model (`train_track_b.py`):** 5 of the 35 patients have fewer than 5 glucose readings, so they can't form a window. The trained model uses **30 real patients, 370 real windows, 1,850 examples** (with augmentation). Report these numbers, not 2,240, when describing that model.

Three techniques, in order of use, all operating on real patient data
(never inventing a synthetic patient):
1. **Window slicing** (`augmentation.window_slice`) — fixed-length overlapping sub-sequences, so richer patients contribute proportionally more
2. **Jittering** (`augmentation.jitter`) — small, physiologically-plausible noise on real readings
3. **Time warping** (`augmentation.time_warp`) — small variation in reading spacing, order always preserved

`augmentation.augment_full_cohort()` runs all of this end-to-end and returns both the combined dataset and a summary dict with the real-vs-augmented counts — already tested against real data (see numbers above).

Plus, for the model itself:
4. **Pretrain-then-finetune** (`models/pretrain_finetune.py`) — pretrain the LSTM on the UCI
   Diabetes-130 dataset's repeat-encounter patients (structure only, not diagnostic quality),
   then fine-tune on the real, augmented MIMIC-IV Demo cohort
5. **Leave-one-patient-out CV** (`models/lopo_cv.py`) — makes full use of every real patient
   for both training and evaluation. **Critical:** all windows/jitter-copies from one real
   patient must stay in the same fold — splitting one patient's augmented examples across
   train and test would leak information and invalidate the evaluation.

**Reporting rule, non-negotiable:** every result using augmented data must state the real
patient count (35) and the augmented example count (2,240) separately. Never let a table
imply 2,240 real patients.

## Build order
1. ✅ Project structure (this)
2. ✅ Data acquisition (NHANES + MIMIC-IV Demo, verified)
3. ⬜ Wire up `src/features/` — run preprocessing, feature engineering, target construction
4. ⬜ Wire up `src/models/` — fit real models, run through `lopo_cv.py` and `evaluate.py`
5. ⬜ Wire up `src/explainability/` — SHAP, stability check, connect to `llm_interpreter.py`
6. ⬜ Wire up `app/` — connect real trained models to the 5 pages
