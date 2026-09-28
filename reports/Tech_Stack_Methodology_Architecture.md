# Tech Stack, Methodology, and System Architecture

### A consolidated technical reference for the project, before implementation begins

---

## Part 1: Tech Stack — what we use, and why each choice was made

### 1.1 Core language and environment
**Python 3.10+** — the standard for data science/ML tooling, and what every base paper's method (scikit-learn, XGBoost, LightGBM, CatBoost, SHAP) is built around. No alternative seriously considered.

### 1.2 Data handling
| Tool | Used for | Why this, not an alternative |
|---|---|---|
| **pandas** | Loading, merging, cleaning all tabular data (NHANES `.XPT`, MIMIC-IV `.csv.gz`) | Industry standard; `pandas.read_sas()` is the only practical way to read NHANES's SAS transport format |
| **numpy** | Numeric operations underlying pandas and every model library | Required dependency of everything else in the stack |

### 1.3 Modeling — Track A (NHANES, multi-label complication prediction)
| Tool | Used for | Why | Tied to which base paper |
|---|---|---|---|
| **scikit-learn** | Logistic regression (traditional baseline), Random Forest, Binary Relevance / Classifier Chain wrappers, LASSO | The base implementation every other library plugs into; also the only realistic way to build the traditional-model comparison our problem statement is centered on | Paper 1's BR/CC frameworks are literally scikit-learn's `MultiOutputClassifier`/`ClassifierChain` |
| **LightGBM** | One of the three stacking base learners | Directly replicates Paper 1's exact model choice | Paper 1 |
| **CatBoost** | One of the three stacking base learners | Same — replicates Paper 1's exact model choice; also handles categorical NHANES variables (ethnicity, etc.) natively without manual encoding | Paper 1 |
| **XGBoost** | Used in Track B's static baseline (not Track A's ensemble, to avoid redundancy with LightGBM/CatBoost already covering the gradient-boosting family) | Matches Paper 2's model comparison | Paper 2 |
| **imbalanced-learn** | SMOTE, for the class-imbalance problem present in both tracks | Standard, well-maintained, integrates directly with scikit-learn pipelines | General project need, not paper-specific |

### 1.4 Modeling — Track B (MIMIC-IV, longitudinal forecasting)
| Tool | Used for | Why |
|---|---|---|
| **PyTorch** | The LSTM/GRU sequence model that consumes `mimic_trajectories.csv` | This is the one component with no equivalent in either base paper — it's our specific contribution filling the gap both papers leave open (neither builds a genuine sequence model on repeated visits). PyTorch chosen over TensorFlow for its more transparent, debuggable training loop, which matters more than convenience here given the small dataset (35 patients) requires careful, inspectable training rather than a black-box `fit()` call |
| **scikit-learn / XGBoost / SVM** | The static (first-visit-only) baseline replicating Paper 2 exactly, before the LSTM extension | Directly matches Paper 2's four-model comparison |

### 1.5 Explainability
| Tool | Used for | Why |
|---|---|---|
| **SHAP** | Feature-attribution explanations for every model in both tracks | Used by both base papers; the field's de facto standard for tree-model explainability |
| **Groq API (`groq` Python package, free tier)** | Turning raw SHAP numbers into a plain-language clinical narrative for the demo app | This is the "LLM for interpretability" component — it does not replace SHAP (SHAP remains the ground-truth, reported-in-the-paper explanation); it's a readability layer on top, with a strict grounding rule (only reference numbers it's actually given) to prevent hallucinated clinical claims. Runs a general-purpose instruction-following model (Llama 3.3 70B) rather than a medical-domain fine-tune — deliberately, since the task is strict grounding to given numbers, not medical knowledge recall; a domain-heavy model risks over-elaborating beyond the provided SHAP values. Free tier, no credit card required. Already built and designed (`src/explainability/llm_interpreter.py`) |

### 1.6 Application — designed as a real product, not a single-page demo
| Tool | Used for | Why |
|---|---|---|
| **Streamlit (multi-page)** | The full application, structured as five pages (below), not one screen | Streamlit's multi-page app support gives real navigation, a persistent sidebar, and per-page state — the structural difference between "a demo" and "an app" — without needing a separate frontend framework, which would be disproportionate effort for a six-week solo build |

**The five pages, and why each earns its place:**
1. **Home / Overview** — what the tool does, which two tracks it's built on, and a plain-language summary of current model status (active version, last retrained, validation performance) — the kind of transparency page a real clinical tool is expected to have, not just a hidden implementation detail.
2. **Mode 1 — Current Risk** — the manual-entry, multi-complication risk check.
3. **Mode 2 — Progression Forecast** — trajectory view, now pulling from **two sources**: the pre-loaded MIMIC-IV example cases, and any real patient with 2+ logged visits from `src/storage/db.py` — a patient the clinician has actually tracked through repeated use.
4. **Patient History** — a searchable list of every patient logged in the local database, with their visit count and a link into their Mode 2 trajectory view. This is the page that makes the tool feel like something a clinician would actually keep open across a workday, not a one-shot calculator.
5. **Model Transparency** — shows the model registry: every trained version, its validation metrics, whether it's currently active, and how much real app-collected data (vs. original training data) it incorporated. This directly surfaces the retraining pipeline (below) to the user instead of hiding it — a real product earns trust by showing this, not by being a black box that quietly changes behavior.

### 1.7 Retraining pipeline — the feedback loop from stored data back into the model
| Tool | Used for | Why |
|---|---|---|
| **`src/models/retrain_pipeline.py`** (built now, wired to real training once Step 3 exists) | Combines original training data with real, app-collected patient check-ins, trains a candidate model, and only promotes it to "active" if it doesn't regress validation performance beyond a small tolerance | This is what makes stored patient data actually useful, not just archived. The design deliberately does **not** auto-deploy every retrain — a naive "always use the newest model" approach risks quietly degrading predictions on a small, unrepresentative batch of new data, which is a real and well-documented failure mode in clinical ML, not a hypothetical one |
| **Model registry** (`model_registry` table in `src/storage/db.py`) | Every trained version is recorded with its metrics; exactly one version per model is marked active at a time | Real ML systems version their models the same way code is versioned — being able to say "this prediction came from version 7, trained on this data, with this validation score" is what an audit-ready system looks like, and it's directly surfaced on the Model Transparency page above |
| **Prediction log** (`prediction_log` table) | Every prediction the app serves is recorded: which model version, when, for which patient | The audit trail a real decision-support tool needs — if a prediction is ever questioned, there's a record of exactly what produced it |

### 1.8 Persistence — corrected
An earlier version of this document said "no database," reasoning only about the training data (NHANES/MIMIC-IV), which is correct — those stay as flat CSVs, read-only. What was missed: the **app itself** needs to remember real patients across separate sessions, or Mode 2 could only ever replay pre-loaded MIMIC-IV example cases, never show an actual patient's accumulated history from repeated real use.

| Tool | Used for | Why |
|---|---|---|
| **SQLite** (`sqlite3`, Python standard library) | Storing every Mode-1 check-in per patient, so a patient with 2+ logged visits has a real trajectory Mode 2 can plot; also backs the model registry and prediction log above | Single file, no server, no extra install — appropriate for a prototype's actual scale (one clinician's local use), while still solving the real problem. A production deployment at hospital scale would need a proper multi-user database; that's explicitly out of scope per the project's own honest scoping |

### 1.9 What's deliberately NOT in the stack, and why
- **No production-grade database** (PostgreSQL, MongoDB, etc.) — SQLite covers the prototype's actual persistence need (single-user, local, small volume); a production system would need one, but that's out of scope here.
- **No cloud deployment** (AWS/GCP/Azure) — out of scope per the project's own honest scoping (Section 6 of the Product Story document) — this is a research prototype, not a deployed service.
- **No TensorFlow alongside PyTorch** — redundant; one deep learning framework is enough, and PyTorch was chosen for the one place deep learning is actually used.
- **No full hypergraph/Neural ODE libraries** — deliberately excluded per the gap analysis; that level of architecture was judged out of scope for a six-week, non-GPU-cluster project (see `Base_Papers_Gap_Analysis.md`).

---

## Part 2: Methodology

### 2.1 Overall research design
The project follows a **replicate-then-extend** methodology against two specific, recent, journal-published base papers (see `Base_Papers_Deep_Dive_Blueprint.md` for full detail):
1. **Replicate** each base paper's method as closely as our data allows, producing a directly comparable number.
2. **Extend** by filling the specific, disclosed gap each paper leaves open.
3. **Report both**, side by side, so the contribution is precise and checkable rather than a general novelty claim.

This applies independently to each of the two tracks, since they map to two different base papers and two different datasets.

### 2.2 Data methodology
- **Two datasets, kept structurally separate** (no row-level merging, since they represent different populations) — NHANES for cross-sectional breadth, MIMIC-IV for longitudinal depth. Already acquired, cleaned, and verified (see `Progress_Report_Data_Foundation.md`).
- **Patient-level splitting only** — train/validation/test splits are always done by patient ID, never by row, to prevent data leakage (a specific risk already flagged in the project's domain-knowledge documentation).
- **Honest small-sample handling for Track B** — with only 35 diabetic patients (13 with computable multi-reading trends) in the MIMIC-IV Demo, the methodology explicitly treats Track B results as a small-scale proof-of-concept, not a generalizable claim, pending full MIMIC-IV access.

### 2.3 Modeling methodology
**Track A:**
1. Traditional baseline: single-label logistic regression per complication (the number our problem statement is actually about).
2. Base-paper replication: stacking ensemble (RF + LightGBM + CatBoost) inside Binary Relevance and Classifier Chain wrappers, class-weighted, with a multi-output logistic regression meta-learner — matching Paper 1 exactly.
3. Evaluation: 5-fold cross-validation, reporting Hamming Loss, F1, and AUC — same protocol as Paper 1, for direct numeric comparison.

**Track B:**
1. Base-paper replication: static (first-visit-only) baseline using RF, XGBoost, SVM, and logistic regression, with LASSO feature selection reducing available variables to a final predictor set — matching Paper 2 exactly.
2. Extension: LSTM/GRU model trained on the full multi-visit trajectory (`mimic_trajectories.csv`), testing whether the longitudinal signal Paper 2 never used actually improves prediction.
3. Evaluation: train-vs-validation AUC gap is explicitly reported for every model as a required overfitting check, directly benchmarked against Paper 2's disclosed 0.219 gap, given our smaller cohort.

### 2.4 Explainability methodology
1. SHAP computed for every reported model (both tracks) — the ground-truth, paper-reported explanation.
2. **Explanation-stability check** — small random perturbations applied to inputs, measuring how much the top-5 SHAP features change — a check neither base paper performs, and one of the project's stated differentiators (from the earlier general literature survey, Paper 1 of that survey specifically).
3. **LLM narrative layer** — SHAP output for one patient is passed to the Groq API (free tier, Llama 3.3 70B) under a strict grounding constraint, producing the plain-language explanation shown in the demo app. This is presentation, not a new source of evidence — the paper reports SHAP numbers directly, and the LLM layer is described as a usability feature for the demo, not a modeling contribution.

### 2.5 Evaluation and reporting methodology
- Every results table includes the corresponding base-paper number alongside our own, so comparisons are explicit rather than left to the reader.
- Every model's train/validation gap is reported, not just its best score.
- Limitations (small Track B sample, single-region data, no external validation, no clinician-defined pathways) are stated directly in the paper rather than omitted — consistent with how both base papers, and every paper in the earlier general literature survey, handle their own limitations.

---

## Part 3: System Architecture

### 3.1 The seven layers, with concrete tech stack (see diagram above)

| Layer | Track A implementation | Track B implementation | Tech |
|---|---|---|---|
| **Data sources** | NHANES merged CSV | MIMIC-IV Demo trajectories + summary CSVs | pandas |
| **Preprocessing** | Imputation, sentinel-value cleanup (`PAD680`), drop `DBQ360` | Sequence formatting, LOCF for gaps | pandas |
| **Feature engineering** | Lipid ratios, obesity/prediabetes flags | LASSO-selected static features (replicating Paper 2) | pandas, scikit-learn |
| **Class balancing** | SMOTE, class-weighting | Class-weighting (small-n makes SMOTE riskier here) | imbalanced-learn |
| **Model ensemble** | Stacking (RF+LightGBM+CatBoost) in BR/CC, + LR baseline | Static baseline (RF/XGBoost/SVM/LR) + LSTM extension | scikit-learn, LightGBM, CatBoost, XGBoost, PyTorch |
| **Explainability** | SHAP + stability check | SHAP + stability check | SHAP |
| **Persistence** | Real patient check-ins from Mode 1 | Real patient trajectories built from repeated check-ins | SQLite |
| **Serving/demo app** | Mode 1 (current risk) | Mode 2 (progression forecast — pre-loaded MIMIC-IV cases + real app-tracked patients) | Streamlit, + Groq API (free tier) for narrative layer |

### 3.2 How this maps to the project directory (already built)
```
t2d_project/
├── data/processed/          → Data sources layer (done)
├── src/features/            → Preprocessing + Feature engineering layers (next to build)
├── src/models/               → Class balancing + Model ensemble layers (after features)
├── src/storage/               → Persistence layer (db.py — done)
├── src/explainability/       → Explainability layer (llm_interpreter.py done; shap_explainer.py, stability_check.py next)
├── app/                      → Serving/demo app layer (last)
```

### 3.3 Why this architecture, restated concisely
- **Two tracks, not one**, because no single available dataset satisfies the full problem statement (breadth + true temporal depth) — established back in the original problem analysis.
- **Seven layers**, because each is a genuinely separate concern with its own failure modes (a bug in feature engineering is a different kind of bug than a bug in class balancing), and separating them makes the codebase debuggable and the paper's methodology section easy to structure.
- **Explainability as its own layer, not a step inside modeling**, because it has its own dedicated evaluation (the stability check) and its own dedicated output consumer (the demo app's narrative panel) — it's architecturally load-bearing, not an afterthought.

---

## Part 4: What this document changes vs. what stays the same

**Stays the same:** the directory structure, the two-track split, the seven-layer architecture, the demo app's two modes — all already built and unchanged.

**Now made concrete:** every layer has a named, justified tool; every modeling choice is tied to a specific base-paper replication or a specific, stated extension; the methodology section above is close to directly usable as the paper's own Methodology section with light editing.

---

*Companion to: Base_Papers_Deep_Dive_Blueprint.md, Base_Papers_Gap_Analysis.md, Progress_Report_Data_Foundation.md, README.md (in t2d_project/)*
