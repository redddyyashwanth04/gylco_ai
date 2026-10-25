# Carepath: Detailed Project Report

## Explainable machine-learning decision support for Type-2 Diabetes risk and progression

**Project:** `t2d_project`  
**Application name:** Carepath  
**Report purpose:** complete technical, functional, methodological, operational,
and presentation reference for review by a project panel  
**Primary runtime:** Python local HTTP service at `127.0.0.1:5001`  
**Status:** research prototype; not a certified medical device

---

## Executive summary

Carepath is a clinician-facing prototype for supporting Type-2 Diabetes (T2D)
management. It combines two related but distinct machine-learning problems.

The first problem is **multi-label complication-risk estimation**. Given
demographic, physiological, behavioural, and socioeconomic measurements, the
system estimates the probability of three outcomes: hypertension,
nephropathy, and cardiovascular disease. This track uses NHANES data because
NHANES provides a relatively broad population-health view with laboratory,
examination, questionnaire, and demographic variables.

The second problem is **short-horizon glucose forecasting**. Given a patient’s
recent glucose trajectory, the system predicts the next glucose reading. This
track uses repeated observations from MIMIC-IV Demo because a cross-sectional
survey cannot represent a time sequence.

The project is more than a model-training script. It includes:

- reproducible data-preparation modules;
- feature engineering and target construction;
- traditional and nonlinear model comparisons;
- an LSTM/Ridge forecasting path;
- grouped validation to reduce patient leakage;
- SHAP explanations for individual risk predictions;
- an optional grounded natural-language explanation layer;
- a local browser application;
- SQLite persistence for patients, visits, model versions, and prediction logs;
- endpoint-level automated tests; and
- timestamped training reports.

The latest recorded training run reported a Track A mean AUC of **0.8179** for
the selected Random Forest model. Track B’s LSTM + Ridge blend reported a
pooled leave-one-patient-out MAE of **45.45 mg/dL**, compared with **54.51
mg/dL** for the naive “repeat the last value” baseline. These results
demonstrate technical feasibility; they do not establish clinical efficacy.

---

# 1. Background and problem definition

## 1.1 Clinical context

Type-2 Diabetes is a chronic metabolic disease in which insulin resistance and
progressive beta-cell dysfunction can lead to prolonged elevation of blood
glucose. The clinical challenge is not limited to identifying whether a person
has diabetes. Clinicians must also monitor control over time and identify
patients who may be at increased risk of complications.

Important measurements include:

- HbA1c, which reflects average glycaemic exposure over approximately three
  months;
- fasting or measured glucose;
- body mass index;
- systolic and diastolic blood pressure;
- LDL, HDL, triglycerides, and total cholesterol;
- serum and urine kidney markers;
- age, sex, and ethnicity;
- smoking, alcohol, and sedentary behaviour; and
- socioeconomic variables that can affect access to care and health behaviour.

These variables are not independent. For example, BMI, blood pressure, lipid
ratios, glycaemic control, kidney markers, and activity may interact. A simple
linear score may not capture all such interactions.

## 1.2 Computational problem

The project decomposes the broad problem into two measurable tasks.

### Task A: current complication risk

For an input feature vector \(x\), estimate:

\[
\hat{p}(y_k = 1 \mid x)
\]

for each complication \(k\) in:

\[
\{\text{hypertension}, \text{nephropathy}, \text{cardiovascular}\}
\]

This is a multi-label problem because several outcomes may be positive for one
person. The output is a vector of probabilities, not a single mutually
exclusive class.

### Task B: next-glucose forecasting

For an ordered glucose sequence:

\[
(g_{t-n+1}, \ldots, g_t)
\]

predict:

\[
\hat{g}_{t+1}
\]

The primary metric is mean absolute error:

\[
\text{MAE} =
\frac{1}{N}\sum_{i=1}^{N} |y_i-\hat{y}_i|
\]

where the unit is mg/dL.

## 1.3 Project objective

The objective is to build a technically defensible prototype that:

1. uses appropriate public or research data;
2. compares against interpretable baselines;
3. handles missing values and class imbalance explicitly;
4. prevents patient-level leakage during evaluation;
5. explains individual predictions;
6. makes uncertainty and limitations visible;
7. supports repeated patient visits; and
8. provides a usable clinician-oriented interface.

---

# 2. Intended users and practical value

## 2.1 Primary user

The primary user is an endocrinologist, physician, diabetes specialist, or
trained clinical professional. The user enters current measurements, reviews
the model output, examines the drivers, saves the visit, and compares later
visits.

## 2.2 Beneficiary

The patient is the intended beneficiary rather than the direct operator. Better
structured review may support earlier follow-up or more focused clinical
discussion. A patient-facing product would require separate health-literacy,
communication, safety, and emotional-design work.

## 2.3 Research and engineering users

Researchers can use the project to compare models and validation strategies.
ML engineers can inspect artifacts and training logs. Supervisors can inspect
model version information and prediction history.

## 2.4 Workflow benefit

In a manual workflow, a clinician must gather measurements, compare historical
values, mentally combine risk factors, and document the reasoning. Carepath
centralises these steps:

1. validate the entered data;
2. calculate three risk probabilities;
3. show the main positive and negative drivers;
4. compare the result with the most recent saved visit;
5. display the previous clinical note when available;
6. save the measurements and risk snapshot; and
7. forecast the next glucose value when a trajectory is available.

The system is intended to reduce information-synthesis time, not to replace
clinical reasoning.

---

# 3. System architecture

## 3.1 High-level architecture

```text
Browser
  login.html / login.js
  index.html / app.js / styles.css
              |
              | HTTP loopback
              v
src/api_server.py
  validation, routing, static files, API handlers
       |             |                 |
       v             v                 v
Track A         Explainability      SQLite
predict.py     SHAP + narrative     app_state.db
       |
       v
Track B inference
LSTM checkpoint or Ridge pickle
```

Training is separate from serving:

```text
NHANES raw files
  -> merge
  -> clean and impute
  -> engineer features
  -> build targets
  -> train/evaluate models
  -> save model artifacts and registry row

MIMIC-IV Demo files
  -> build trajectories
  -> filter and window
  -> augment training windows
  -> grouped validation
  -> train Ridge/LSTM/blend
  -> save checkpoint and metrics
```

## 3.2 Runtime boundary

`src/api_server.py` binds to `127.0.0.1`, so the intended demo is local to the
machine. This reduces accidental network exposure and avoids transmitting
entered values to a remote service. The design is not equivalent to a
production security architecture; hospital deployment would require identity,
authorisation, encryption, monitoring, and governance.

## 3.3 Model loading

Track A reads `models_saved/nhanes_active_name.txt` to identify the active
artifact. The model, feature list, and training medians are cached. The cache
is refreshed when the saved model modification time changes.

Track B checks for the LSTM checkpoint and Ridge pickle. The health endpoint
reports whether forecasting artifacts exist. The application can therefore
continue to provide risk assessment even if forecasting is unavailable.

---

# 4. Data sources

## 4.1 NHANES

NHANES is a large population-health survey from the U.S. Centers for Disease
Control and Prevention. The project combines multiple component files into a
single merged table.

The data contributes:

- laboratory values;
- examination measurements;
- questionnaire responses;
- demographic variables;
- behavioural variables; and
- socioeconomic variables.

NHANES is appropriate for Track A because the task is a broad complication-risk
screening estimate from a feature-rich health survey.

### NHANES limitations

NHANES is not a continuous clinical record. It has survey-specific missingness,
skip patterns, and self-reported outcomes. The model should therefore be
interpreted as an association-based research estimate, not a causal model or
universal clinical risk calculator.

## 4.2 MIMIC-IV Demo

MIMIC-IV Demo is a de-identified clinical dataset containing repeated
observations from hospital encounters. The project extracts glucose
measurements and orders them by time for each subject.

MIMIC-IV Demo is useful for demonstrating sequence modelling, but it is not a
representative outpatient T2D population. ICU measurement frequency, illness
severity, and treatment patterns may differ substantially from routine
diabetes clinics.

### MIMIC-IV limitations

- small number of real patients;
- ICU-oriented population;
- irregular measurement times;
- possible distribution shift to outpatient use;
- no claim of prospective clinical validation.

## 4.3 Data governance

Development uses public, de-identified, or research-access data. The local
application is designed to keep entered values on the local machine. Any
production deployment would need institutional approval, privacy review,
retention policy, access controls, and a formal clinical safety process.

---

# 5. Data preparation

## 5.1 NHANES preparation sequence

The intended order is:

1. merge the component files;
2. remove unusable columns;
3. convert sentinel values;
4. impute remaining missing values;
5. engineer derived features;
6. construct outcome labels; and
7. produce the model-ready table.

### Removing unusable columns

`DBQ360` is dropped because it has approximately 96% missingness caused by a
survey skip pattern. Treating this as an ordinary numeric missing-value
problem would create a large amount of artificial information.

### Sentinel conversion

NHANES `PAD680` uses values such as `7777` and `9999` for “don't know” or
“refused” responses. These are converted to missing values before imputation.
Keeping them as literal minutes would distort the sedentary-behaviour feature.

### Imputation

The current preprocessing module uses:

- median imputation for numeric columns; and
- most-frequent imputation for categorical columns such as sex and
  race/ethnicity.

At inference, the serving code uses medians derived from the model-ready
training table. This keeps training and serving aligned.

## 5.2 Feature engineering

`src/features/feature_engineering.py` creates:

| Feature | Definition | Purpose |
|---|---|---|
| `TG_HDL_RATIO` | triglycerides / HDL | lipid balance signal |
| `LDL_HDL_RATIO` | LDL / HDL | atherogenic lipid relationship |
| `TC_HDL_RATIO` | total cholesterol / HDL | total-to-protective lipid relationship |
| `OBESITY_FLAG` | BMI >= 30 | clinically interpretable obesity indicator |
| `PREDIABETES_FLAG` | HbA1c 5.7–6.4 or glucose 100–125 | glycaemic-range indicator |
| `LAB_HYPERTENSION_FLAG` | systolic >= 130 or diastolic >= 80 | measured BP indicator |

Division by zero and missing HDL values are guarded.

## 5.3 Target construction

`src/features/build_targets.py` creates:

- `hypertension`: self-reported diagnosis `BPQ020 == 1`;
- `nephropathy`: self-reported kidney problem or `URXUMA >= 30`;
- `cardiovascular`: any configured heart disease, heart attack, heart failure,
  or stroke indicator;
- `obesity`: copied from `OBESITY_FLAG`; and
- `diabetes`: survey diabetes label.

The served complication task uses the first three labels. Obesity and diabetes
remain available for analysis and comparison.

## 5.4 Leakage prevention

The training script excludes direct target-source columns such as `BPQ020`,
`KIQ022`, `URXUMA`, cardiovascular diagnosis columns, target columns, and
identifier fields from the model feature matrix. This prevents the model from
simply copying the label source.

---

# 6. Track A model methodology

## 6.1 Compared models

### Logistic regression

Logistic regression is the traditional baseline. It is scaled and class
balanced. It provides a useful answer to the question: does a nonlinear model
improve ranking or classification compared with a conventional additive
method?

### Random Forest

Random Forest is an ensemble of decision trees. It can represent nonlinear
relationships and interactions without manually specifying every interaction.
The comparison model uses class-balanced trees. Separate per-label forests are
also trained for stable, target-specific SHAP explanations.

### Gradient Boosting

Gradient boosting builds a sequence of weak learners, each correcting errors
from earlier learners. It provides another nonlinear tabular baseline.

## 6.2 Training split

The recorded Track A run used:

- 6,522 training rows;
- 1,631 held-out test rows;
- 36 model features; and
- a fixed random seed of 42.

The held-out test set is used for final comparison. Threshold tuning is
performed inside the training partition using stratified cross-validation.

## 6.3 Threshold tuning

Risk probabilities are not automatically converted to binary decisions using
0.5. The training pipeline searches thresholds from 0.15 to 0.65 and selects
the threshold that maximises F1 on cross-validation predictions from the
training partition. The selected thresholds are saved in
`nhanes_thresholds.json`.

This is important for imbalanced outcomes because a 0.5 threshold may be
inappropriate when positive cases are less frequent.

## 6.4 Serving process

For a new patient:

1. validate required fields and ranges;
2. copy the input dictionary;
3. add engineered features;
4. align columns with the saved feature list;
5. fill missing columns with training medians;
6. run `predict_proba`;
7. return one probability for each target; and
8. generate clinician-review recommendations.

The API returns probabilities even when thresholds are tuned, because
probabilities are needed for explanation and ranking.

---

# 7. Track B forecasting methodology

## 7.1 Forecasting target

Track B predicts the next glucose reading from recent readings. The project has
both a Ridge path and an LSTM v4 path. The latest recorded unified training run
selected the LSTM + Ridge blend.

## 7.2 Window construction

Each patient’s glucose values are ordered by `charttime`. A window contains
recent values and one target value. The v4 path uses a window size of eight:
seven values provide sequence context and the eighth is the target.

The earlier Ridge-only path uses four inputs plus engineered statistics:

- the four previous readings;
- mean;
- standard deviation; and
- difference between the latest and earliest input.

## 7.3 Gap filtering

The v4 pipeline filters out windows where the target gap exceeds 24 hours.
Large irregular gaps may contain changes that cannot be inferred from the
recent readings alone. This is a data-driven restriction, not a claim that
long-horizon prediction is solved.

## 7.4 Augmentation

The augmentation module performs:

- overlapping window slicing;
- small multiplicative Gaussian jitter; and
- optional time warping support.

Augmentation does not create new real patients. Every generated example retains
its source patient identifier and an `is_augmented` flag.

## 7.5 Patient-grouped validation

All windows and augmented copies from one real patient remain in the same
leave-one-patient-out fold. This prevents a near-duplicate of a patient’s
trajectory from appearing in both training and testing.

## 7.6 LSTM v4 design

The LSTM uses:

- two recurrent layers;
- hidden size 64;
- dropout 0.3;
- SmoothL1 loss;
- AdamW optimisation;
- weight decay;
- gradient clipping;
- early stopping;
- three random seeds in the recorded full run;
- time-gap inputs;
- horizon bucket embedding; and
- a patient context vector.

The context vector contains scaled patient-level information such as mean,
standard deviation, slope, and the latest-reading z-score. The model predicts
the change from the last reading, which is then converted back to glucose
scale.

## 7.7 Blend

The LSTM captures sequential structure while Ridge provides a stable
low-variance tabular estimate. Their predictions are blended, with the blend
weight learned from training-fold information. The final recorded blend
outperformed either component slightly.

---

# 8. Explainability

## 8.1 SHAP

SHAP decomposes a prediction into feature contributions relative to a baseline.
For each complication, the application returns up to five top factors.

- positive SHAP contribution: increases the model estimate;
- negative SHAP contribution: decreases the model estimate;
- absolute magnitude: strength of contribution for that patient.

The system uses separate per-label models to avoid a shared explanation budget
across different outcomes.

## 8.2 Correlated-feature handling

The explanation module deduplicates pairs such as:

- total cholesterol and total/HDL ratio;
- LDL and LDL/HDL ratio.

When both appear, the ratio is retained and the raw feature’s contribution is
merged. This avoids presenting the same physiological signal twice as if it
were independent evidence.

## 8.3 Missing-field visibility

The explanation includes an `entered` flag. When a top factor was not entered,
the application can show that the value was filled from a typical training
median. This is important because an imputed value is not the same as an
observed measurement.

## 8.4 Grounded LLM narrative

The optional LLM layer receives only:

- complication name;
- predicted risk;
- selected feature values; and
- SHAP contributions.

It is instructed not to diagnose, prescribe, introduce unsupported statistics,
or add unrelated medical claims. If Groq is unavailable, the system uses a
deterministic local template. The LLM never produces the risk value and is not
part of the predictive model.

---

# 9. API and backend implementation

## 9.1 Request handling

`src/api_server.py` uses Python’s `ThreadingHTTPServer` and
`BaseHTTPRequestHandler`. It serves static frontend files and exposes API
routes.

The server:

- requires JSON content for POST requests;
- caps request bodies at 64 KB;
- rejects non-object JSON bodies;
- validates finite numeric values;
- validates configured clinical ranges;
- rejects untrusted origins;
- blocks static-file path traversal;
- sends JSON with no-store and `nosniff` headers; and
- returns explicit 400, 403, 404, 500, or 503 responses as appropriate.

## 9.2 Endpoint contracts

### `GET /api/health`

Reports:

- whether the service can load the active model;
- active model name;
- feature count;
- registry version;
- whether a forecast artifact is available; and
- forecast artifact type.

### `POST /api/predict`

Required fields:

- `RIDAGEYR`;
- `LBXGH`; and
- `LBXGLU`.

Returns:

- `risks`;
- `modelName`;
- `recommendations`;
- `trends`;
- `previousNote`; and
- `previousNoteDate`.

When a patient ID is provided, the server compares the current estimate with
the latest saved risk snapshot.

### `POST /api/explain`

Uses the same required fields and returns:

- risk probabilities;
- factors for each target;
- SHAP values;
- human-readable labels;
- model input values;
- whether each value was entered; and
- a narrative plus its source.

### `POST /api/forecast`

Accepts a patient or case trajectory and returns the predicted next glucose,
model information, and expected validation error where available.

### `GET /api/visits`

Returns saved visits ordered from newest to oldest.

### `POST /api/visits`

Requires a patient ID. The server recomputes the risk snapshot itself rather
than trusting client-supplied risk values. It stores measurements, risks,
active model version, timestamp, and optional note.

### `GET /api/patients`

Returns patient summaries and the prediction count.

### `GET /api/patients/{id}/visits`

Returns the visit history for one patient.

### `GET /api/models`

Returns model registry rows including metrics, active status, original rows,
app rows, and notes.

## 9.3 Error handling

Invalid input is returned as a 400 response with a message. Unexpected
failures are logged server-side and returned as a generic 500 message rather
than exposing internal details. Health failures return 503.

---

# 10. Persistence and auditability

`src/storage/db.py` creates and migrates `data/app_state.db`.

## 10.1 Tables

### `patients`

Stores a unique patient ID and creation time.

### `visits`

Stores:

- patient ID;
- visit date;
- HbA1c;
- glucose;
- BMI;
- blood pressure;
- serialized risk snapshot;
- model version ID; and
- clinician note.

### `model_registry`

Stores:

- model name;
- training time;
- original training row count;
- app-collected row count;
- validation AUC;
- validation F1;
- validation MAE;
- active flag; and
- notes.

### `prediction_log`

Stores:

- timestamp;
- model name;
- model version;
- patient ID when supplied; and
- serialized prediction.

## 10.2 Version promotion

Candidate model versions are registered before promotion. Promotion first
deactivates the previous model with the same name and then marks the candidate
active. The retraining design includes a validation gate so a candidate that
regresses beyond tolerance is not promoted.

## 10.3 Current retraining limitation

The retraining pipeline correctly separates training, validation, and
promotion, but the current app-collected visit schema is not yet automatically
mapped into the original NHANES/MIMIC feature schemas. Therefore, the
retraining design should be described as prepared infrastructure rather than a
fully operational online learning system.

---

# 11. Frontend and user experience

The browser interface is implemented in:

- `frontend/login.html`;
- `frontend/login.js`;
- `frontend/index.html`;
- `frontend/app.js`; and
- `frontend/styles.css`.

The optional Streamlit interface is implemented under `app/`.

## 11.1 Login

The demonstration login is session-based and intended only for local
demonstration. It must not be described as production authentication.

## 11.2 Current Risk workflow

The clinician enters required values, optionally enters additional
measurements, reviews the result, examines SHAP drivers, and saves the visit.
The interface indicates assumed values and shows recommendations as review
considerations rather than orders.

## 11.3 Forecast workflow

The clinician selects a tracked patient or de-identified case. The chart shows
historical readings and the next forecast. An error indicator communicates that
the prediction is uncertain.

## 11.4 Patient History

The patient history view supports searching and reviewing saved visits,
measurements, risk snapshots, dates, and notes.

## 11.5 Model Transparency

The model registry view makes model changes visible. It can show training
dates, row counts, AUC, F1, MAE, active status, and notes.

---

# 12. Performance results

The figures below are taken from
`reports/training_runs/20261006_213220.json` and its Markdown companion.

## 12.1 Track A

| Model | Mean AUC | Mean F1 |
|---|---:|---:|
| Logistic regression | 0.8166 | 0.5083 |
| Random Forest | **0.8179** | 0.4592 |
| Gradient boosting | 0.8132 | **0.5213** |

Random Forest was selected by mean AUC. Gradient Boosting produced the highest
mean F1. This distinction must be stated honestly: the “best model” depends on
the selection objective.

### Random Forest per-label results

| Target | AUC | F1 |
|---|---:|---:|
| Hypertension | 0.8223 | 0.6824 |
| Nephropathy | 0.7910 | 0.3138 |
| Cardiovascular | 0.8405 | 0.3812 |

The weaker nephropathy F1 indicates that this outcome is more difficult and/or
imbalanced. It is not appropriate to claim equal performance across all
complications.

## 12.2 Track B

| Model | Pooled MAE (mg/dL) | Per-patient MAE (mg/dL) |
|---|---:|---:|
| Naive | 54.51 | 48.11 |
| Ridge | 46.26 | 40.54 |
| LSTM v4 | 45.80 | 40.59 |
| LSTM v4 + Ridge | **45.45** | **40.36** |

The blend improves over the naive baseline by 9.06 mg/dL, or 16.6%.

## 12.3 Correct interpretation of metrics

AUC measures ranking discrimination, not calibration or clinical utility. F1
depends on a threshold and reflects precision/recall balance. MAE is an average
absolute error and does not mean every forecast is within that distance.

These metrics are internal validation evidence. They do not prove that the
system improves outcomes in a real clinic.

---

# 13. Testing and reproducibility

## 13.1 Automated tests

`tests/test_api.py` starts the API on a separate local port and sends real HTTP
requests. It checks:

- health response;
- risk response structure;
- required-field rejection;
- out-of-range rejection;
- optional-field handling;
- recommendations;
- patient trend response;
- SHAP factor structure;
- narrative response; and
- visit persistence.

Other tests cover data preparation and augmentation.

## 13.2 Reproducibility controls

- central paths in `config.py`;
- fixed random seeds;
- saved feature column order;
- serialized model artifacts;
- timestamped JSON and Markdown training reports;
- SQLite model registry;
- explicit README commands; and
- endpoint tests against real saved artifacts.

## 13.3 Recommended verification commands

From the `t2d_project` directory:

```text
pytest tests/ -v
python -m src.api_server
```

Training commands are documented in the repository’s model modules and
`README.md`. A fresh run should be performed before presenting live metrics.

---

# 14. File-by-file implementation reference

## Configuration

### `config.py`

Defines the project root, raw-data directories, processed-data directories,
model directory, trajectory paths, and LLM model name. The purpose is to avoid
hard-coded paths scattered throughout the codebase.

### `requirements.txt`

Declares pandas, NumPy, scikit-learn, XGBoost, LightGBM, CatBoost,
imbalanced-learn, PyTorch, SHAP, Groq, Streamlit, and pytest.

## Data preparation

### `src/data_prep/inspect_xpt.py`

Inspects NHANES SAS transport files and helps identify columns and formats.

### `src/data_prep/merge_nhanes_core.py`

Merges NHANES component tables into the processed core dataset.

### `src/data_prep/build_mimic_trajectories.py`

Extracts and organises MIMIC observations into patient-level time-ordered
trajectory rows.

### `src/data_prep/generate_synthetic_mimic.py`

Creates synthetic trajectory material for pretraining or controlled
experiments. Synthetic rows must remain distinguishable from real rows.

## Feature and target construction

### `src/features/preprocess_nhanes.py`

Drops unusable columns, recodes sentinels, and imputes missing values.

### `src/features/preprocess_mimic.py`

Contains the sequence-preparation direction for converting trajectories into
model tensors and preserving patient/group metadata.

### `src/features/feature_engineering.py`

Adds lipid ratios and clinical flags.

### `src/features/build_targets.py`

Builds the complication and auxiliary labels.

### `src/features/augmentation.py`

Implements jittering, window slicing, time warping, source-patient tagging, and
cohort augmentation.

## Models

### `src/models/train.py`

Provides Track A training support.

### `src/models/train_all.py`

Runs the unified Track A and Track B training flow, compares models, saves
artifacts, registers versions, and writes training metrics.

### `src/models/train_track_b.py`

Implements the simpler Ridge, gradient-boosting, and naive glucose-forecast
comparison path.

### `src/models/train_lstm.py`

Implements the v4 LSTM data loader, model architecture, training loop,
early-stopping logic, inference, and blending support.

### `src/models/predict.py`

Loads the active Track A model, creates engineered features, fills medians,
returns risks, computes risk deltas, and produces clinician-review
recommendations.

### `src/models/evaluate.py`

Defines multi-label evaluation helpers and train-validation gap reporting.

### `src/models/lopo_cv.py`

Provides a reusable leave-one-patient-out cross-validation harness.

### `src/models/mimic_static_baseline.py`

Provides a static baseline for MIMIC experiments.

### `src/models/lstm_trajectory.py`

Provides trajectory-model support.

### `src/models/stacking_ensemble.py`

Supports ensemble experimentation.

### `src/models/pretrain_finetune.py`

Supports pretraining and later fine-tuning experiments.

### `src/models/retrain_pipeline.py`

Defines minimum-data, regression-tolerance, candidate-registration, and
promotion-gate logic for future retraining.

## Explainability

### `src/explainability/explain.py`

Coordinates per-label models, SHAP calculation, feature deduplication, factor
ranking, and narrative generation.

### `src/explainability/shap_explainer.py`

Contains SHAP helper functionality.

### `src/explainability/llm_interpreter.py`

Builds the grounded prompt and optionally calls Groq. It returns an explicit
unavailable/error marker so the caller can use a deterministic fallback.

### `src/explainability/stability_check.py`

Supports checking whether top explanations remain stable under small input
perturbations.

## Storage and metrics

### `src/storage/db.py`

Creates and migrates SQLite tables, records visits, manages model versions,
promotes active models, and logs predictions.

### `src/metrics/training_log.py`

Writes timestamped JSON and Markdown reports containing model parameters,
dataset counts, and performance metrics.

## Serving and user interface

### `src/api_server.py`

Provides static-file serving, request validation, API routing, model health,
prediction, explanation, forecast, visit, patient, and registry handlers.

### `frontend/app.js`

Coordinates browser state, API calls, charts, risk displays, patient history,
and model transparency.

### `frontend/index.html`

Defines the application shell and page containers.

### `frontend/styles.css`

Defines the visual design and responsive layout.

### `frontend/login.html` and `frontend/login.js`

Implement the local demonstration login flow.

### `app/Home.py` and `app/pages/`

Provide an optional Streamlit view of the same project capabilities.

## Tests

### `tests/test_api.py`

Exercises the local server over HTTP.

### `tests/test_data_prep.py`

Tests preparation logic and expected transformations.

### `tests/test_augmentation.py`

Tests jittering, slicing, and augmentation invariants.

---

# 15. Limitations and risk analysis

## Data limitations

- NHANES includes self-reported labels.
- MIMIC-IV Demo is small and ICU-specific.
- There is no external-site validation.
- The datasets have different populations and collection processes.
- The project does not prove causation.

## Modelling limitations

- Track A F1 varies substantially by complication.
- Probabilities are not yet demonstrated to be clinically calibrated.
- Track B MAE is too large for autonomous dosing or treatment decisions.
- Input distribution shift can invalidate predictions.
- Median imputation may affect explanations.
- A single held-out split for Track A can be less stable than repeated or
  external validation.

## Application limitations

- demonstration login is not production authentication;
- localhost is not hospital deployment;
- optional Groq use creates an external service boundary;
- retraining data mapping is incomplete;
- the system lacks formal monitoring and alerting.

## Safety statement

The proper claim is:

> Carepath is a research decision-support prototype that helps a clinician
> review complication risk and short-horizon glucose trajectory.

The improper claim is:

> Carepath diagnoses a patient or predicts their future with certainty.

---

# 16. Recommended improvements

## Immediate

1. Run a fresh training cycle before presenting.
2. Use `/api/health` and `/api/models` as the authoritative live status.
3. Reconcile all older documentation with the active artifact names.
4. Add calibration metrics and reliability diagrams.
5. Add formal prediction intervals for glucose forecasts.
6. Add explicit out-of-distribution warnings.

## Research

1. Obtain a larger longitudinal outpatient cohort.
2. Perform temporal validation and external-site validation.
3. Report bootstrap confidence intervals.
4. Evaluate age, sex, ethnicity, and socioeconomic subgroup performance.
5. Compare calibration and precision-recall performance.
6. Quantify explanation stability.
7. Compare tuned Random Forest and Gradient Boosting under a pre-declared
   clinical selection objective.

## Production

1. Integrate an approved identity provider.
2. Add role-based access control.
3. Encrypt stored patient data.
4. Use secrets management for any external LLM service.
5. Add immutable audit export.
6. Add model approval and rollback workflow.
7. Monitor drift, latency, missingness, calibration, and subgroup behaviour.
8. Conduct clinician usability and safety testing.
9. Complete regulatory and institutional review.

---

# 17. Panel defence questions and answers

## Why did you use two datasets?

NHANES provides broad cross-sectional health features for complication-risk
estimation, while MIMIC-IV Demo provides repeated measurements for sequence
forecasting. The tasks require different data structures.

## Why is this not just a diabetes classifier?

The primary output is not diabetes diagnosis. Track A estimates related
complication risks, and Track B forecasts a future glucose reading from a
trajectory. The project therefore addresses management and progression support.

## Why Random Forest?

It captures nonlinear interactions in tabular data, is practical for the sample
size, performs competitively in the recorded run, and supports TreeSHAP.

## Why did Gradient Boosting have better F1?

The selected model criterion was mean AUC, while Gradient Boosting achieved
better mean F1. AUC and F1 measure different properties. A production model
choice should be based on a pre-defined clinical objective, calibration, and
subgroup performance.

## Why not use deep learning for Track A?

The Track A data is structured tabular data with a moderate sample size. Tree
ensembles are strong, interpretable, and less data-hungry for this setting.
Deep learning is used where the problem has sequential structure.

## Why LSTM for Track B?

An LSTM can represent order and temporal context. It is compared with naive and
Ridge baselines, and it is evaluated with patient-grouped validation rather
than assumed to be better because it is deep learning.

## Why use augmentation?

The real cohort has few patients but many readings. Window slicing and small
jitter create additional training examples while retaining a source-patient
trace. They do not create new real patients.

## How did you prevent leakage?

Track A uses a held-out test partition and tunes thresholds only within the
training data. Track B keeps every real and augmented window from one patient
in one leave-one-patient-out fold.

## What does the 0.8179 AUC mean?

It means the selected Track A model ranked positive cases above negative cases
reasonably well on the held-out test data, averaged across the three labels. It
does not mean 81.79% diagnostic accuracy or 81.79% clinical certainty.

## What does 45.45 mg/dL MAE mean?

Across pooled held-out Track B windows in the recorded run, the absolute
difference between the predicted and observed next glucose was 45.45 mg/dL on
average. This supports trend prioritisation but is not suitable for autonomous
treatment.

## Why use SHAP?

SHAP supplies patient-specific contributions and directions. A global feature
importance chart would not explain why this particular patient received this
particular estimate.

## Can the LLM change the model prediction?

No. The risk is calculated before the LLM call. The LLM receives selected SHAP
values only and is used as a readability layer. If it is unavailable, a local
template is used.

## Is the system production ready?

No. It is a research prototype. It needs external validation, calibration,
larger longitudinal data, security controls, clinical usability evaluation,
governance, and regulatory review.

## How would you improve it first?

First freeze and verify the active artifacts, then perform external validation
and calibration. For Track B, the most important improvement is a larger,
representative longitudinal outpatient dataset.

---

# 18. Demonstration and verification checklist

Before the panel:

1. Open a terminal in `t2d_project`.
2. Run the tests:

   ```text
   pytest tests/ -v
   ```

3. Start the service:

   ```text
   python -m src.api_server
   ```

4. Open `http://127.0.0.1:5001/`.
5. Confirm `/api/health` reports a loaded Track A model.
6. Confirm the active model and metrics in Model Transparency.
7. Sign in using the demonstration flow.
8. Load the example patient.
9. Run Current Risk and show all three risks.
10. Open the SHAP drivers and point out an assumed value if one exists.
11. Save the visit.
12. Change one measurement and show the trend response.
13. Open Patient History.
14. Open Progression Forecast and run a de-identified case.
15. Explain the forecast error bar.
16. If Groq is unavailable, demonstrate the deterministic narrative fallback
    instead of treating the failure as a defect.

The latest live health response should be treated as authoritative if it
differs from an older report. Do not present a stale model name or metric as
the currently serving artifact.

---

# Conclusion

Carepath demonstrates an end-to-end approach to explainable clinical machine
learning for T2D management. The project combines a multi-label tabular risk
model, a longitudinal glucose forecasting model, grouped validation,
patient-level explanations, local persistence, model versioning, and a usable
interface.

Its strongest engineering qualities are separation of training and serving,
explicit leakage prevention, visible uncertainty, grounded explanations, and
honest documentation of limitations. Its main scientific limitations are the
small and ICU-specific longitudinal cohort, the absence of external validation,
and the need for probability calibration and subgroup analysis.

The correct final interpretation is therefore balanced: the project is a
strong research prototype and a credible demonstration of an explainable
clinical decision-support workflow, but more data, validation, governance, and
clinical evaluation are required before deployment in real patient care.
