# Base Papers — Deep Dive and Implementation Blueprint

### Exactly how each paper works, and exactly how our code and paper build on top of them

---

## PAPER 1 (Track A): Zamani et al. (2026) — full methodology breakdown

### The pipeline, stage by stage

**1. Cohort:** 965 retrospective T2DM patients (single-center, Hamadan, Iran).

**2. Target construction:** complications aggregated into two groups —
   - **Microvascular:** retinopathy, nephropathy, neuropathy
   - **Macrovascular:** cardiovascular, cerebrovascular
   
   Each patient gets a **binary vector** (has/doesn't have each of the 5 complications) — this is the multi-label target.

**3. Model architecture — this is the core technical contribution to replicate:**
   - **Three base learners**, each trained independently: Random Forest, LightGBM, CatBoost.
   - **Two multi-label "wrapper" strategies**, each tested with the same three base learners:
     - **Binary Relevance (BR):** train one independent model per complication (5 separate models per base learner) — simple, but ignores that complications co-occur.
     - **Classifier Chain (CC):** train one model per complication *in sequence*, where each later model gets the earlier models' predictions as extra input features — lets the model learn "if this patient already has nephropathy, that changes their cardiovascular risk."
   - **Class-weighting** applied during training to counteract the fact that most complications are minority classes (this is the same imbalance problem our own MIMIC-IV cohort has).
   - **Stacking meta-learner:** a multi-output logistic regression sits on top, combining the three base learners' outputs into a final prediction — this is what makes it a "stacked ensemble," not just three separate models being compared.

**4. Evaluation:** 5-fold cross-validation, reporting Hamming Loss (how many individual complication labels are wrong, averaged), F1-score, and AUC.

**5. Explainability:** SHAP applied to the best-performing configuration.

### Full results, precisely
- **Best configuration: Stacking + Classifier Chain.** F1 = 0.752 ± 0.049, AUC = 0.857 ± 0.032.
- **Complications correlate with each other at r = 0.35** — this number is their empirical justification for multi-label modeling over separate single-label models, and it's a number we can compute and report on our own NHANES cohort too, as a direct comparison point.
- **SHAP findings, specific and useful for us:** macrovascular complications were driven mainly by **LDL cholesterol and diastolic blood pressure**; microvascular complications were driven by **drug addiction history, fasting blood sugar, and HDL**. Notably, **age and BMI showed minimal importance** — a genuinely counterintuitive, specific, checkable finding we can directly test against on our own NHANES SHAP output.

---

## PAPER 2 (Track B): Huang et al. (2025) — full methodology breakdown

### The pipeline, stage by stage

**1. Cohort construction (from MIMIC-IV, same database we're using):**
   - ICD-9 codes `24960`–`24963`, ICD-10 codes `E1040`–`E1149` → diabetic neuropathy (DN) diagnosis
   - Restricted to: first ICU admission only, length of stay > 48 hours, age > 18, complete blood-count data available
   - Result: **1,313 patients** (812 survived, 501 did not — their outcome is in-hospital mortality)

**2. Feature extraction:** **56 variables** total, from four categories —
   - Demographics (age, sex, etc.)
   - Comorbidities (Charlson Comorbidity Index and components)
   - **First 24 hours** of ICU labs
   - **First 24 hours** of vitals
   - Severity scores: SAPS-II, APS-III, CCI
   - Plus **6 engineered inflammatory ratios** (e.g., neutrophil-to-lymphocyte ratio, or NLR) — computed from raw labs, not present in the raw data directly. This is directly analogous to the lipid ratios (TG/HDL etc.) from our earlier literature survey — the same "engineer a clinically meaningful ratio, don't just feed raw values" principle.

**3. Feature selection:** **LASSO regression** narrows the 56 variables down to **12 final predictors**.

**4. Models compared:** Random Forest, XGBoost, SVM, Logistic Regression — four distinct model families, not variations of one architecture (contrast with Paper 1, which varies the *wrapper strategy* around similar tree-based learners).

**5. Split:** 80/20 train/validation (1,050 / 263 patients).

**6. Explainability:** SHAP on the best model (Random Forest).

### Full results, precisely
- **Random Forest best: training AUC 0.999, validation AUC 0.780.** That 0.219 gap is the single most important number from this paper for our own work — it's a textbook overfitting signature, openly disclosed by the authors, on a cohort (1,313) not dramatically larger than ours.
- **Top 5 SHAP-ranked predictors:** RDW (red cell distribution width), neutrophil count, Charlson Comorbidity Index, maximum chloride, minimum prothrombin time — all lab/comorbidity-derived, notably **no lab trend or repeated-measurement feature anywhere**, since the whole design uses only the first 24 hours of the first admission.

---

## Implementation Blueprint — Track A (built directly on Paper 1)

| Their step | Our equivalent, on NHANES | File it lives in |
|---|---|---|
| 965-patient cohort, 5-label target | 8,153-patient NHANES cohort; our targets: hypertension (`BPQ020`), nephropathy (`KIQ022`/`URXUMA`), 4 cardiovascular flags (`MCQ160B/C/E/F`) collapsed into one CVD label, obesity (derived from `BMXBMI`) | `src/features/build_targets.py` *(next to write)* |
| Base learners: RF, LightGBM, CatBoost | Same three, same libraries | `src/models/stacking_ensemble.py` *(to write)* |
| BR and CC wrappers | Replicate both, compare directly | same file |
| Class-weighting for imbalance | Apply `class_weight='balanced'` (RF/LR) and `scale_pos_weight` (LightGBM/CatBoost equivalent), same principle | same file |
| Multi-output logistic regression meta-learner | Same | same file |
| 5-fold CV, Hamming Loss / F1 / AUC | Identical protocol, for direct comparability | `src/models/evaluate.py` *(to write)* |
| **Gap we fill:** no traditional-model baseline | We add plain logistic regression (single-label, no ensembling) as an explicit comparison point — this produces the actual number our problem statement is about | same file |
| **Gap we fill:** no explanation-stability test | We add the SHAP perturbation-stability check on our best model | `src/explainability/stability_check.py` *(to write)* |
| SHAP: LDL + diastolic BP → macrovascular; drug use + FBS + HDL → microvascular; age/BMI minimal | We report our own SHAP top-features per complication and explicitly compare against these findings — do our behavioral/environmental NHANES variables (income, smoking, sedentary time) show up where their hospital-chart data couldn't have such variables at all? | Results section of the paper |

---

## Implementation Blueprint — Track B (built directly on Paper 2)

| Their step | Our equivalent, on MIMIC-IV Demo | File it lives in |
|---|---|---|
| ICD-9/10 cohort extraction | Already done — `build_mimic_trajectories.py`, 35 diabetic patients | already built |
| First-24-hours-only features | **We deliberately do NOT do this** — this is the gap we're filling | — |
| 56 raw variables → LASSO → 12 predictors | We replicate this reduction step on our own available MIMIC-IV variables (labs + comorbidity flags from `diagnoses_icd`), to keep the comparison fair at the "static/single-snapshot" baseline level | `src/features/lasso_select_mimic.py` *(to write)* |
| RF / XGBoost / SVM / LR comparison | Same four, first as a **static baseline** using only first-visit values (replicating their exact setup on our data — this is our direct benchmark number) | `src/models/mimic_static_baseline.py` *(to write)* |
| **Then, our actual contribution:** add the LSTM/GRU sequence model using the *full* trajectory (`mimic_trajectories.csv`) as input, not just first-visit values | This is the model their paper never builds, despite having the same data available to build it | `src/models/lstm_trajectory.py` *(to write)* |
| Disclosed overfitting (train 0.999 / val 0.780) | We explicitly report train-vs-validation gap for every model we train, flagging anything approaching their gap as a warning sign given our smaller cohort (35 vs. 1,313) | `src/models/evaluate.py` |
| SHAP top features: RDW, neutrophils, CCI, chloride, PT | We don't have identical lab panels available in the Demo subset, but we replicate the *method* (SHAP on the static baseline) and compare feature-importance patterns qualitatively | Results section |

---

## How this shapes the paper itself

**Related Work section:** each paper gets its own paragraph structured exactly as the summaries above — methodology, results, and the specific gap, ending with "we address this by..."

**Methodology section:** structured as two subsections (Track A, Track B), each explicitly opening with "Following [Paper], we implement..." and then stating the specific extension in a clearly marked sentence — this is what makes the contribution look precise rather than vague to a reviewer.

**Results section:** every results table should include, as a row or footnote, **the corresponding number from the base paper** — our Stacking-CC F1 vs. their 0.752; our static-baseline AUC vs. their 0.780 validation AUC — so the comparison is explicit, not left for the reader to infer.

**Discussion/Limitations section:** the honest scale differences (965 vs. 8,153 for Track A; 1,313 vs. 35 for Track B) are stated directly, alongside what that does and doesn't change about the validity of the comparison.

---

## Immediate next step

Per the step-by-step approach: the next concrete file to write is **`src/features/build_targets.py`** — the NHANES target-construction and feature-engineering script for Track A, since Track A's data is fully ready and this blueprint now fully specifies what that script needs to produce. Track B's static-baseline replication (`lasso_select_mimic.py`) is the next step after that.

---

*Companion to: Base_Papers_Gap_Analysis.md, Progress_Report_Data_Foundation.md*
