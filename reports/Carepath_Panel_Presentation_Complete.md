# Carepath: Explainable Type-2 Diabetes Risk and Progression Support

## Complete panel presentation script and technical defence

**Project type:** clinician-facing research prototype  
**Primary interface:** local browser application served by `src.api_server`  
**Evidence baseline used in this document:** the recorded training run
`reports/training_runs/20261006_213220.*`  
**Important scope statement:** Carepath is decision support, not a diagnostic
medical device and not a replacement for clinical judgement.

> **How to use this document:** Each `---` section is one presentation page or
> slide. The paragraphs under “Speaker notes” are the words to use if the
> panel asks for detail. The final pages are a question bank and a demo
> checklist.

---

# Page 1 — Title and one-minute introduction

## Carepath

### Explainable Type-2 Diabetes complication-risk assessment and glucose-progression forecasting

**Core idea:** combine a broad population-health dataset for complication risk
with a longitudinal clinical dataset for short-horizon glucose forecasting,
then expose both outputs through a local, explainable clinician workflow.

### Opening script

“Carepath addresses a practical clinical problem: diabetes management is not
only about identifying diabetes; it is about anticipating complications and
recognising deterioration early. The system has two complementary tracks.
Track A estimates the risk of hypertension, nephropathy, and cardiovascular
complications from NHANES-style patient measurements. Track B forecasts the next
glucose reading from a patient’s recent trajectory using MIMIC-IV Demo data.
The system shows the drivers behind risk using SHAP, stores visits and model
versions locally, and presents the result as decision support rather than an
automated diagnosis.”

### Project identity

- **Name:** Carepath
- **Application:** local clinical decision-support prototype
- **Backend:** Python standard-library HTTP server plus modular ML code
- **Frontend:** browser SPA and an optional Streamlit interface
- **Persistence:** SQLite for patients, visits, model registry, and prediction log
- **Privacy design:** loopback server at `127.0.0.1`; no required external
  patient-data service

---

# Page 2 — Problem statement and motivation

## The problem

Traditional risk scores and one-time statistical models are useful but limited
when a patient has interacting physiological, behavioural, and social factors.
They often provide a single population-level estimate, do not describe the
main drivers for an individual, and do not use repeated measurements to show
trajectory.

### Why this matters

- Type-2 diabetes is chronic and changes over time.
- Poor control is associated with hypertension, kidney damage, cardiovascular
  disease, and other complications.
- A clinician has limited time to manually combine HbA1c, glucose, BMI, blood
  pressure, lipids, kidney markers, smoking, activity, and socioeconomic
  variables.
- A probability without an explanation is difficult to trust or act on.
- A forecast trained on one row per patient cannot represent progression.

### Precise problem formulation

1. **Multi-label risk task:** given current patient features, estimate a
   probability for each of three related complications:
   hypertension, nephropathy, and cardiovascular disease.
2. **Time-series regression task:** given the most recent glucose readings,
   forecast the next glucose value.
3. **Human factors task:** provide an explanation, a trend view, an audit trail,
   and a clear statement of uncertainty.

### What the project does not claim

- It does not diagnose diabetes.
- It does not prescribe medication.
- It does not estimate a guaranteed future event date.
- It is not externally validated or approved for clinical deployment.

---

# Page 3 — Users, beneficiaries, and value

## Intended users

### Primary user: endocrinologist or clinician

The clinician enters or reviews routine measurements, inspects risk estimates
and SHAP drivers, saves the visit, and compares later visits.

### Beneficiary: patient

The patient benefits indirectly through earlier review, better prioritisation,
and more consistent follow-up. The current product is not a patient self-care
application because raw risk estimates could be misunderstood without clinical
context.

### Secondary users

- **Researcher:** compares models, features, and validation strategies.
- **ML engineer:** inspects model registry, artifacts, and reproducible runs.
- **Supervisor or auditor:** checks which model version produced a prediction.

## Workflow improvement

Without Carepath, the clinician must manually:

1. find measurements in the record,
2. remember or calculate risk relationships,
3. compare with previous visits,
4. decide which factors deserve attention,
5. document the result.

With Carepath, the same workflow becomes:

1. enter current values,
2. receive three risk probabilities,
3. see ranked positive and negative drivers,
4. compare risk with the previous saved visit,
5. save the snapshot and note,
6. forecast the next glucose value when a trajectory exists.

The system accelerates information synthesis; the clinician still owns the
interpretation and decision.

---

# Page 4 — Solution overview and system architecture

```text
Clinician browser
  login.html -> index.html / app.js / styles.css
                    |
                    | HTTP on loopback: 127.0.0.1:5001
                    v
              src/api_server.py
          /predict  /explain  /forecast
          /visits  /patients /models /health
             |          |         |
             v          v         v
       Track A      SHAP +       SQLite
       RF models    narrative    app_state.db
             |
             v
       Track B LSTM/Ridge artifacts

Training inputs:
  NHANES -> clean -> engineered features -> targets -> Track A models
  MIMIC-IV Demo -> trajectories -> windows/gap filter/augmentation -> Track B
```

## Architectural decisions

- **Local HTTP server:** easy to run, easy to test with real HTTP requests, and
  avoids exposing a development database to a network.
- **Modular pipeline:** data preparation, features, models, explainability,
  metrics, storage, and web layers have separate responsibilities.
- **Lazy model loading and cache invalidation:** model artifacts are loaded on
  demand and reloaded when their modification time changes.
- **Training/inference separation:** training scripts produce versioned
  artifacts; the API performs inference only.
- **SQLite auditability:** every visit and prediction can be traced to a model
  version.
- **Static frontend serving:** the backend serves the browser assets, so the
  demo does not require a second frontend server.

---

# Page 5 — Data sources and data governance

## Track A: NHANES

NHANES is a CDC population-health survey distributed in SAS transport files.
The project merges demographic, examination, laboratory, questionnaire, and
medical-condition variables into a model-ready table.

### Strengths

- broad population diversity,
- physiological, behavioural, and socioeconomic variables,
- suitable for cross-sectional complication-risk modelling,
- much larger sample than the longitudinal demonstration cohort.

### Caveats

- survey measurements are not a continuous patient timeline,
- some outcomes are self-reported,
- missingness can reflect survey skip patterns,
- associations must not be interpreted as causal effects.

## Track B: MIMIC-IV Demo

MIMIC-IV Demo supplies de-identified clinical encounter data. The project
extracts repeated glucose observations and constructs patient trajectories.

### Strengths

- repeated measurements,
- genuine patient grouping,
- appropriate for patient-held-out validation,
- supports a next-reading forecasting task.

### Caveats

- the Demo cohort is small and ICU-oriented,
- it is not representative of routine outpatient diabetes care,
- irregular measurement gaps make long-horizon forecasting difficult,
- the recorded run used 25 real patients and 661 filtered real windows.

## Privacy and ethics

The data used for development is de-identified or public/research data. The
application is designed to keep entered values on the local machine, but a
production deployment would still require access control, encryption, audit
policy, consent/legal review, and institutional governance.

---

# Page 6 — Data preprocessing and quality controls

## NHANES preprocessing

1. Load merged CSV data.
2. Drop `DBQ360`, which is dominated by a survey skip-pattern missingness
   problem.
3. Recode `PAD680` sentinel values `7777` and `9999` as missing.
4. Median-impute numeric fields.
5. Most-frequent-impute categorical fields such as sex and race/ethnicity.
6. Add physiologically motivated features.
7. Construct labels separately from feature cleaning.
8. Exclude target and leakage columns from model inputs.

## Engineered features

- `TG_HDL_RATIO = triglycerides / HDL`
- `LDL_HDL_RATIO = LDL / HDL`
- `TC_HDL_RATIO = total cholesterol / HDL`
- `OBESITY_FLAG = BMI >= 30`
- `PREDIABETES_FLAG = HbA1c in 5.7–6.4 or glucose in 100–125`
- `LAB_HYPERTENSION_FLAG = systolic >= 130 or diastolic >= 80`

Ratio calculations guard against missing or zero HDL values.

## MIMIC preprocessing

- filter to glucose observations,
- order each patient’s readings by chart time,
- create fixed sliding windows,
- retain a patient identifier for grouped validation,
- remove target gaps over 24 hours in the v4 run,
- add time-gap and patient-context information for the LSTM,
- jitter real windows conservatively for training only.

## Quality principle

Missing optional clinical inputs are filled with training medians at inference
and visibly labelled as “typical value assumed” in the explanation workflow.
This is safer than silently pretending that a value was observed.

---

# Page 7 — Target definitions and leakage prevention

## Track A labels

The target-building module defines:

- **Hypertension:** `BPQ020 == 1`
- **Nephropathy:** `KIQ022 == 1` or urine albumin `URXUMA >= 30`
- **Cardiovascular:** any of the heart disease, heart attack, heart failure,
  or stroke indicators is positive

The project also creates obesity and diabetes labels for analysis, but the
served Track A complication output is the three-label risk vector.

## Leakage prevention

The training script excludes label-source variables such as `BPQ020`, `KIQ022`,
`URXUMA`, cardiovascular diagnosis columns, and target columns from model
features. This prevents the model from simply copying the answer.

## Track B labels and grouping

For forecasting, the target is the fifth glucose value in a window; the first
four or seven values, depending on the active training path, form the recent
history. All windows from a patient remain in the same validation group.

## Why this matters

Randomly splitting windows would allow near-duplicate windows or jittered
copies from the same person to appear in both training and testing. That would
inflate performance and would not answer the real question: “does this work on
a patient the model has not seen?”

---

# Page 8 — Track A model methodology

## Compared models

1. **Logistic regression:** transparent traditional baseline; scaled features
   and class balancing.
2. **Random Forest:** nonlinear ensemble; 300 trees for the comparison model,
   with separate 500-tree per-label forests for SHAP and tuned thresholds.
3. **Gradient Boosting:** nonlinear boosting comparison.

Each complication is evaluated separately and the results are averaged across
labels.

## Training protocol

- 80/20 train-test split,
- `random_state=42` for reproducibility,
- median fill based on training data,
- class balancing for imbalanced outcomes,
- per-label threshold tuning using five-fold cross-validation inside the
  training partition,
- test set used only for final reported performance.

## Inference

The API:

1. validates numeric inputs and clinically plausible ranges,
2. derives the same engineered features as training,
3. aligns columns to the saved feature list,
4. imputes missing values with cached training medians,
5. returns a probability for each target,
6. logs the model version and prediction.

The classification threshold is not assumed to be 0.5; tuned thresholds are
stored in `nhanes_thresholds.json`. Probabilities remain continuous for SHAP.

---

# Page 9 — Track B forecasting methodology

## Forecasting objective

Predict the next glucose reading from a patient’s recent glucose history. This
is a regression task, so the principal error metric is mean absolute error
(MAE) in mg/dL.

## Baselines and models

- **Naive baseline:** repeat the last observed glucose value.
- **Ridge regression:** engineered lag, mean, standard deviation, and trend
  features.
- **LSTM v4:** two-layer sequence model with hidden size 64, dropout, SmoothL1
  loss, AdamW, gradient clipping, early stopping, time-gap information,
  horizon buckets, and patient context.
- **LSTM + Ridge blend:** combines complementary sequence and linear estimates.

## Data expansion without inventing patients

- window slicing creates several examples from long real trajectories,
- jittering adds small multiplicative perturbations,
- patient IDs are retained,
- all augmented examples from one real patient stay in that patient’s fold,
- augmented data is used for training, while evaluation emphasises real
  held-out windows.

## Why not a GAN?

Thirty-five-ish real patients is too small for a reliable generative model.
Window slicing and small perturbations are easier to inspect and preserve a
trace back to a real source patient.

---

# Page 10 — Explainability and clinical communication

## SHAP is the ground-truth explanation

For each complication, Carepath uses a separate tree model and TreeSHAP to
calculate how each feature moves the prediction relative to the model’s
baseline. The UI ranks the top factors:

- positive SHAP value: pushes the estimate upward,
- negative SHAP value: pushes the estimate downward,
- magnitude: relative contribution for this prediction.

Correlated raw/ratio features are deduplicated so a clinician does not see the
same cholesterol signal twice.

## Natural-language layer

An optional Groq Llama model converts only the risk, feature values, and SHAP
contributions into a short narrative. It is explicitly instructed:

- use only supplied numbers,
- do not diagnose,
- do not recommend medication,
- end with a decision-support disclaimer.

If the API key is unavailable or the call fails, a deterministic local
template is used. The model output is never the source of the risk score.

## Safety boundary

The bar chart and numeric probability remain available even when the LLM is
offline. This is an important separation: language generation improves
readability but does not control prediction or clinical logic.

---

# Page 11 — Application workflow and user interface

## 1. Login

The demo login is local and session-based. The sample credentials are for
demonstration only and are not production authentication.

## 2. Current Risk

The clinician enters required age, HbA1c, and fasting glucose, plus optional BMI,
blood pressure, lipids, sex, smoking, sedentary time, and income ratio. The
screen shows three risks, top drivers, imputation indicators, recommendations
for clinician review, and a save-visit action.

## 3. Progression Forecast

The clinician chooses a tracked patient or de-identified example trajectory.
The chart displays recent glucose history, the next forecast, and an error bar
based on held-out validation MAE.

## 4. Patient History

The clinician can search saved patients and inspect visit dates, measurements,
risk snapshots, and notes.

## 5. Model Transparency

The registry shows model name, training date, sample counts, metrics, active
status, and notes. This makes model changes visible rather than silent.

---

# Page 12 — API, storage, and reliability design

## API surface

- `GET /api/health` — artifact and service status
- `POST /api/predict` — Track A risk probabilities
- `POST /api/explain` — risk plus SHAP drivers and narrative
- `POST /api/forecast` — Track B next-glucose prediction
- `GET/POST /api/visits` — save and retrieve visit records
- `GET /api/patients` and `/api/patients/{id}/visits` — history
- `GET /api/models` — model registry

## Input and security controls

- JSON content type required,
- request body capped at 64 KB,
- finite numeric validation,
- clinical range validation,
- allowed-origin checks,
- loopback binding,
- path traversal protection for static files,
- `nosniff` and no-store response headers.

## SQLite tables

- `patients`
- `visits`
- `model_registry`
- `prediction_log`

Each saved visit can include risk JSON, model version ID, timestamp, and a
clinician note. This gives basic reproducibility and auditability.

## Reliability behaviour

- model artifacts are cached for speed,
- modification time triggers reload after retraining,
- optional LLM failure falls back to a local template,
- the API returns explicit validation errors rather than accepting invalid
  values silently.

---

# Page 13 — Performance results

## Track A: latest recorded run

Dataset: NHANES; 6,522 train rows, 1,631 test rows, 36 features.

| Model | Mean AUC | Mean F1 |
|---|---:|---:|
| Logistic regression | 0.817 | 0.508 |
| Random Forest | **0.818** | 0.459 |
| Gradient Boosting | 0.813 | **0.521** |

The selected active model in the recorded run is Random Forest because it has
the highest mean AUC and supports the per-label SHAP workflow. F1 is lower than
the gradient-boosting baseline, which should be acknowledged rather than
hidden.

### Random Forest per-target results

| Target | AUC | F1 |
|---|---:|---:|
| Hypertension | 0.822 | 0.682 |
| Nephropathy | 0.791 | 0.314 |
| Cardiovascular | 0.841 | 0.381 |

## Track B: grouped validation

Dataset: MIMIC-IV Demo; 25 real patients, 661 real filtered windows, 3 LSTM
seeds, 24-hour target-gap filter.

| Model | Pooled MAE (mg/dL) |
|---|---:|
| Naive | 54.51 |
| Ridge | 46.26 |
| LSTM v4 | 45.80 |
| **LSTM v4 + Ridge** | **45.45** |

The blend improves over the naive baseline by 9.06 mg/dL, or 16.6%.

### Correct interpretation

These are internal research results, not evidence of clinical effectiveness.
The Track B error is large enough that the forecast should be shown with an
error bar and used for prioritisation, not treated as an exact measurement.

---

# Page 14 — Validation, testing, and reproducibility

## Validation strategy

- Track A: held-out test set after training-only threshold selection.
- Track B: leave-one-patient-out cross-validation.
- Augmented windows stay grouped by source patient.
- Metrics are recorded in both JSON and Markdown training reports.
- Model registry stores active/inactive versions.

## Automated tests

The test suite covers:

- data preparation,
- augmentation behaviour,
- API requests and response contracts,
- prediction, explanation, forecast, and visit-saving flows.

The API tests start the server and issue real HTTP requests rather than only
testing isolated helper functions.

## Reproducibility controls

- central paths in `config.py`,
- fixed random seeds in training and threshold tuning,
- saved feature-column order,
- serialized model artifacts,
- timestamped training logs,
- model version registry,
- explicit training commands in `README.md`.

## Demonstration command

```text
cd t2d_project
python -m src.api_server
```

Then open `http://127.0.0.1:5001/`.

---

# Page 15 — Limitations and responsible interpretation

## Data limitations

- NHANES labels include self-report and survey-specific missingness.
- MIMIC-IV Demo is small, ICU-oriented, and not an outpatient population.
- There is no external validation in the current project.
- The cohort does not establish causal relationships.
- App-entered data is not yet automatically mapped into the original training
  schema for production retraining.

## Model limitations

- AUC can look acceptable while minority-class F1 is modest.
- Risk probabilities are not calibrated clinical probabilities unless a
  calibration study is added.
- Track B uncertainty is represented by a validation MAE/error bar, not a
  formally calibrated prediction interval.
- Dataset shift may be severe across hospitals, devices, and care settings.
- Missing-value imputation can influence explanations.

## Product limitations

- Demo login is not production-grade identity management.
- Localhost deployment is not a hospital integration.
- The optional LLM introduces an external API boundary if enabled; sensitive
  deployment should disable it or use an approved local model.
- Retraining logic is validation-gated by design, but the app UI does not yet
  provide a complete operational retraining workflow.

## Ethical position

The safe claim is: “Carepath helps a clinician review risk and trajectory.”
The unsafe claim is: “Carepath diagnoses or predicts an individual’s future
with certainty.”

---

# Page 16 — Improvements and production roadmap

## Immediate improvements

1. Run a fresh training cycle and make `/api/health` the single source of truth
   for the demo’s active artifacts.
2. Reconcile documentation and artifact names so Ridge-only, LSTM, and blended
   descriptions cannot conflict.
3. Add probability calibration and calibration plots.
4. Report confidence intervals across patient-level folds.
5. Add a formal prediction interval for Track B.
6. Add data-quality warnings for out-of-distribution input.

## Research improvements

- obtain full credentialed MIMIC-IV or a larger longitudinal outpatient cohort,
- perform temporal and external-site validation,
- compare subgroup performance by age, sex, ethnicity, and socioeconomic status,
- evaluate fairness and calibration, not only discrimination,
- study explanation stability under clinically plausible perturbations,
- compare the random forest against a calibrated gradient-boosting model,
- test whether repeated clinician visits improve performance without leakage.

## Production improvements

- hospital identity provider and role-based access,
- encrypted database and secrets management,
- structured audit exports,
- containerised deployment,
- monitoring for drift, missingness, latency, and calibration,
- model approval workflow and rollback,
- clinician usability testing,
- formal clinical safety and regulatory review.

---

# Page 17 — File-by-file implementation map

## Configuration and entry points

- `config.py`: central project/data/model paths and LLM model setting.
- `requirements.txt`: pandas, NumPy, scikit-learn, boosting libraries, PyTorch,
  SHAP, Groq, Streamlit, and pytest.
- `README.md`: run commands, demo walkthrough, layout, and honest limits.

## Data preparation and features

- `src/data_prep/merge_nhanes_core.py`: merges NHANES component files.
- `src/data_prep/build_mimic_trajectories.py`: builds longitudinal MIMIC rows.
- `src/data_prep/generate_synthetic_mimic.py`: synthetic/pretraining support.
- `src/data_prep/inspect_xpt.py`: inspects SAS transport files.
- `src/features/preprocess_nhanes.py`: missingness, sentinel, and imputation.
- `src/features/preprocess_mimic.py`: sequence-pipeline scaffold.
- `src/features/feature_engineering.py`: ratios and clinical flags.
- `src/features/build_targets.py`: complication labels.
- `src/features/augmentation.py`: window slicing, jitter, and time warping.

## Models and metrics

- `src/models/train_all.py`: unified Track A/Track B training and promotion.
- `src/models/train.py`: Track A training support.
- `src/models/train_track_b.py`: Ridge/naive/gradient-boosting Track B path.
- `src/models/train_lstm.py`: v4 LSTM, gap filtering, context, and blend.
- `src/models/predict.py`: served Track A inference and risk deltas.
- `src/models/evaluate.py`: multi-label metrics and train/validation gap.
- `src/models/lopo_cv.py`: patient-grouped cross-validation harness.
- `src/models/mimic_static_baseline.py`: static MIMIC baseline.
- `src/models/lstm_trajectory.py`: trajectory model support.
- `src/models/stacking_ensemble.py`: ensemble experimentation.
- `src/models/pretrain_finetune.py`: pretraining/finetuning support.
- `src/models/retrain_pipeline.py`: validation-gated retraining design.

## Explainability and persistence

- `src/explainability/explain.py`: SHAP factors and narrative orchestration.
- `src/explainability/shap_explainer.py`: SHAP helper.
- `src/explainability/llm_interpreter.py`: grounded optional narrative.
- `src/explainability/stability_check.py`: explanation stability analysis.
- `src/storage/db.py`: SQLite schema, migrations, visits, registry, audit log.
- `src/metrics/training_log.py`: timestamped JSON/Markdown metrics.

## Application and tests

- `src/api_server.py`: local HTTP API and static frontend server.
- `frontend/login.html`, `login.js`: demo login.
- `frontend/index.html`, `app.js`, `styles.css`: browser application.
- `app/Home.py` and `app/pages/*`: optional Streamlit interface.
- `tests/test_api.py`: endpoint-level tests.
- `tests/test_data_prep.py`: preparation tests.
- `tests/test_augmentation.py`: augmentation tests.

---

# Page 18 — Panel question bank: problem and users

## Q1. Why are there two datasets?

NHANES provides breadth and population diversity for cross-sectional risk
assessment. MIMIC-IV Demo provides repeated measurements for trajectory
forecasting. Neither dataset alone supports both goals well.

## Q2. Who is the user?

The primary user is a clinician. Patients are beneficiaries, not direct users,
because clinical context is required to interpret risk safely.

## Q3. Is this diagnosis?

No. The output is a model-generated risk estimate and a forecast. The tool
does not confirm disease, prescribe therapy, or replace a clinician.

## Q4. What is the practical benefit?

It reduces manual calculation, makes drivers visible, helps compare visits,
and prioritises which patient information deserves clinical attention.

## Q5. Why not make it patient-facing?

A patient interface would need different language, health-literacy testing,
emotional safeguards, and action guidance. That is a separate product scope.

---

# Page 19 — Panel question bank: data and ML

## Q6. Why Random Forest?

It captures nonlinear interactions, works well on tabular mixed clinical data,
handles feature interactions without requiring a neural network, and supports
TreeSHAP. It was selected by highest mean AUC in the recorded run, while the
lower F1 is disclosed.

## Q7. Why does gradient boosting have higher F1 but not win?

The selection criterion was mean AUC, which measures ranking quality across
thresholds. F1 depends on a particular classification threshold and reflects
the precision/recall trade-off. In a future release, model choice should use a
pre-declared clinical objective, calibration, and subgroup performance rather
than AUC alone.

## Q8. Why is Track B evaluated with LOPO-CV?

There are few patients, and row-level random splitting would leak patient
identity and repeated-window information. LOPO-CV tests generalisation to an
unseen patient and uses every patient as a test group once.

## Q9. Why use augmentation?

The cohort has few patients but long trajectories. Window slicing and small
jitter variations increase training examples without claiming that augmented
rows are new patients. The patient grouping rule prevents leakage.

## Q10. What does MAE 45.45 mean?

On the pooled held-out windows in the recorded run, the forecast differed from
the actual next glucose value by about 45.45 mg/dL on average in absolute
terms. It is useful for trend support, but not precise enough for autonomous
treatment.

---

# Page 20 — Panel question bank: explainability, security, and deployment

## Q11. Why SHAP instead of only feature importance?

Global feature importance says what matters on average. SHAP gives a
per-patient direction and contribution, which is more useful for explaining a
specific risk estimate.

## Q12. Can the LLM hallucinate?

The LLM is not used for prediction. It receives only the risk and selected SHAP
numbers, is instructed not to introduce claims or treatment advice, and falls
back to a deterministic template. In a regulated deployment, an approved local
language layer or no LLM would be preferable.

## Q13. Is localhost secure?

It reduces network exposure and includes origin, body-size, range, and path
checks, but it is not a complete production security solution. Production
needs identity, authorisation, encryption, secret management, logging, and
threat modelling.

## Q14. What happens when a field is missing?

The inference pipeline fills it with a training median, and the explanation
marks it as assumed. This preserves the model input contract while exposing
the uncertainty to the user.

## Q15. How are models versioned?

Training writes artifacts and a timestamped report. SQLite records model name,
metrics, training counts, notes, and active status. Predictions store the
version ID that produced them.

---

# Page 21 — Panel question bank: limitations and future work

## Q16. What is the biggest limitation?

Generalisation. Track B is based on a small ICU Demo cohort and may not
represent outpatient care. Track A has survey/self-report limitations and no
external validation. These are reasons to call the project a research
prototype.

## Q17. Can it be deployed in a hospital today?

Not safely as-is. It needs external validation, calibration, security review,
clinical workflow evaluation, governance, and integration with approved EHR
systems.

## Q18. How would you improve it first?

First reconcile and freeze the active artifact/version, then add external
validation, calibration, subgroup analysis, and out-of-distribution warnings.
For Track B, obtain a larger longitudinal outpatient cohort.

## Q19. What if a newly trained model is worse?

The retraining design registers a candidate and promotes it only when its
validation AUC is within the configured regression tolerance of the active
model. Otherwise the current model remains active.

## Q20. What is the novel contribution?

The project integrates two complementary tasks in one clinician workflow:
multi-label complication risk and longitudinal next-glucose forecasting. It
adds patient-level explanations, grounded narrative, grouped validation,
versioned auditability, and a local product interface around the models.

---

# Page 22 — Five-minute live demo plan

## Before opening the browser

1. Activate the intended Python environment.
2. Run the test suite.
3. Start `python -m src.api_server`.
4. Open `http://127.0.0.1:5001/`.
5. Check the Home/health panel and record the active model names and metrics.
6. Confirm the Model Transparency page agrees with the training report.

## Demo sequence

1. Sign in with the demonstration credentials.
2. Open Current Risk.
3. Load the example patient.
4. Review the three risk probabilities.
5. Point out positive/negative SHAP drivers and imputed fields.
6. Save the visit with a note.
7. Repeat with a changed measurement to show the risk delta.
8. Open Patient History and show the saved visit.
9. Open Progression Forecast and select a de-identified trajectory.
10. Forecast the next glucose and explain the error bar.
11. Open Model Transparency and show registry/version information.

## If something fails

- If Groq is unavailable, explain that the deterministic narrative fallback is
  intentional and continue with SHAP.
- If the server is unavailable, do not improvise metrics; show the recorded
  training report and explain the architecture.
- If the active model differs from this document, state the live `/api/health`
  value and treat it as the authoritative demo state.

---

# Page 23 — Final closing statement

“Carepath is a transparent research prototype for supporting—not replacing—
clinical reasoning. Its main contribution is an end-to-end path from public
health data and longitudinal clinical trajectories to validated model outputs,
patient-level explanations, a usable local interface, and an audit trail. The
results are promising enough to demonstrate feasibility, but the limitations
are equally important: the next stage is external validation, calibration,
larger longitudinal data, fairness analysis, and clinical workflow testing
before any real-world deployment.”

