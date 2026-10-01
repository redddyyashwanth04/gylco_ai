# Carepath — T2D Decision-Support Demo

A clinician-facing prototype for Type-2 Diabetes progression prediction.
Two tracks: **Track A** (NHANES Random Forest, complication risk) and
**Track B** (MIMIC-IV Ridge glucose forecaster).

---

## Quick start — demo walkthrough

### 1. Start the server

```
cd t2d_project
python -m src.api_server
```

Open **http://127.0.0.1:5001/** in your browser. The server also hosts the
frontend, so no separate Vite / static server is needed.

### 2. Sign in

On the login screen, click **Fill dr.smith details** (or type manually):

| Field    | Value      |
|----------|------------|
| Username | `dr.smith` |
| Password | `clinic123` |

### 3. Load an example patient

Navigate to **Current assessment** (sidebar → ◫).
Click **Load example** — this fills in a pre-set 58-year-old patient
(HbA1c 7.4 %, glucose 150 mg/dL, BMI 31).

### 4. Review visit → complication risk + SHAP drivers

Click **Review visit →**.

The Random Forest model (Track A, trained on NHANES) returns three risk
estimates. Beneath the estimates the **Top drivers** panel shows a SHAP bar
chart — orange bars raise risk, green bars lower it.  Fields left blank are
imputed from training medians and labelled *"typical value assumed"*.

### 5. Save the visit

With the example patient ID (`DEMO-1042`) in place, click
**Save visit to patient record**. The measurements and risk snapshot are
written to the local SQLite database.

### 6. Switch to a MIMIC-IV demo case

Navigate to **Progression forecast** (sidebar → ⌁).
In the *Patient or example case* selector, pick any **MIMIC-IV demo case**
from the dropdown (ICU glucose trajectories, de-identified).

### 7. Run the glucose forecast

Click **Forecast next reading**. The Ridge forecaster (Track B, trained on
MIMIC-IV Demo, 30 patients) predicts the next glucose value, plotted on the
chart with a ± error bar (leave-one-patient-out MAE).

---

## Running the Streamlit app (optional, separate interface)

```
streamlit run app/Home.py
```

The Streamlit pages (`app/`) are an independent view of the same models and
database — useful for development. The primary demo interface is the
browser-based frontend served by `python -m src.api_server`.

---

## Project layout

```
t2d_project/
├── src/
│   ├── api_server.py          # HTTP API + static frontend server (port 5001)
│   ├── data_prep/             # NHANES merge, MIMIC trajectory builder
│   ├── features/              # preprocessing, feature engineering, augmentation
│   ├── models/
│   │   ├── predict.py         # Track A: Random Forest predict + SHAP
│   │   └── train_track_b.py   # Track B: Ridge glucose forecaster
│   ├── explainability/
│   │   └── explain.py         # SHAP values + LLM/template narrative
│   └── storage/db.py          # SQLite: visits, model registry, prediction log
├── frontend/                  # SPA served by api_server
│   ├── index.html
│   ├── login.html
│   ├── app.js
│   └── styles.css
├── app/                       # Streamlit interface (optional)
│   ├── Home.py
│   └── pages/
│       ├── 1_Current_Risk.py
│       ├── 2_Progression_Forecast.py
│       ├── 3_Patient_History.py
│       └── 4_Model_Transparency.py
├── models_saved/
│   ├── nhanes_random_forest.pkl       # Track A active model
│   ├── nhanes_feature_columns.pkl
│   ├── nhanes_active_name.txt         # points to nhanes_random_forest
│   └── mimic_glucose_forecaster.pkl  # Track B — Ridge
├── data/
│   ├── raw/nhanes/            # .XPT survey files (not committed)
│   ├── raw/mimic_demo/        # 6 .csv.gz files (present)
│   ├── processed/             # merged CSVs
│   └── model_ready/           # feature-engineered training tables
├── tests/                     # pytest suite
│   ├── test_data_prep.py
│   ├── test_augmentation.py
│   └── test_api.py            # HTTP tests for predict/explain/forecast/save-visit
├── reports/                   # project documents
└── config.py                  # central paths
```

---

## Track A — NHANES Random Forest

- **Data:** NHANES population survey
- **Target:** multi-label risk (hypertension, nephropathy, cardiovascular)
- **Model:** Random Forest (active), with Logistic Regression and
  Gradient Boosting trained for comparison (see Model Transparency)
- **Explainability:** per-prediction SHAP + LLM/template narrative
- **API:** `POST /api/predict`, `POST /api/explain`

## Track B — Ridge glucose forecaster

- **Data:** MIMIC-IV Demo (30 patients, 370 real windows, augmented to 1,850)
- **Input:** last 4 glucose readings → predict reading 5
- **Model:** Ridge regression (best LOPO-CV error vs naive / gradient boosting)
- **Validation:** leave-one-patient-out MAE (printed during training)
- **API:** `POST /api/forecast`

---

## Running tests

```
pip install pytest
pytest tests/ -v
```

The `tests/test_api.py` suite starts the server in a background thread and
issues real HTTP requests to all four primary endpoints.

---

## Honest limits

- Research prototype, not a validated medical device.
- Track A labels come from self-reported NHANES survey data.
- Track B is trained on 30 ICU patients; outpatient use is out of
  distribution. The error bar shown with every forecast reflects this.
- Model retraining from app data is designed (validation-gated) but not yet
  enabled in the UI.
