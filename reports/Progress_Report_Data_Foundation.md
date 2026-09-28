# Project Progress Report — Data Foundation Complete

### Type-2 Diabetes Progression Prediction | Status as of Data Preparation Phase

---

## 1. Where this report picks up

The project began with a college-assigned problem statement asking for prediction of T2D *progression* — not just diagnosis — using nonlinear, multi-factor, dynamic forecasting, to address the shortcomings of traditional statistical models. That problem statement was broken down clause by clause, a seven-layer system architecture was designed around it, and two real, complementary datasets were identified, acquired, verified, and processed. This report documents exactly what has been done, with the real numbers produced at each step — not projected numbers, actual output.

---

## 2. Problem framing and architecture (recap)

The problem statement was diagnosed as requiring **two things most student projects skip**: genuine multi-factor (physiological + behavioral + environmental) modeling, and genuine *temporal* forecasting, not a single-snapshot diagnosis. This led to a **two-track design**, since no single public dataset offers both population-scale breadth and real per-patient longitudinal depth at once:

- **Track A (cross-sectional, multi-label complication risk)** — built on NHANES.
- **Track B (longitudinal, progression forecasting)** — built on MIMIC-IV.

A seven-layer system architecture was designed to carry both tracks through one pipeline: Data Sources → Preprocessing → Feature Engineering → Class Balancing → Model Ensemble → Explainability Layer → Serving/Demo App. The demo app itself was designed around two matching interactive modes ("Current Risk" and "Progression Forecast"), so that every claim in the paper has a directly clickable counterpart in the demo — this was a deliberate response to the requirement that both the paper and the demo be genuinely demonstrable, not just described.

---

## 3. Track A — NHANES: what was actually done

### 3.1 Acquisition and the real problems hit along the way
NHANES files were downloaded directly (no credentialing required) across four separate CDC sections — Demographics, Examination, Laboratory, Questionnaire. Three real issues came up during merging, each diagnosed and fixed rather than guessed around:

1. **`DEMO_L` (age/sex/ethnicity/income) was initially missing entirely** from the first merge attempt — traced to the Demographics section being a separate page from the other three sections, easy to miss. Fixed by locating and downloading it directly.
2. **Triglyceride column naming ambiguity** — the script initially looked for `LBXTR`, which didn't exist in this NHANES cycle; the actual column was `LBXTLG`. Fixed by making the merge script check both possible names rather than guessing a single one.
3. **Diet/physical-activity variable names guessed incorrectly** — `DBQ700` and `PAQ650`/`PAQ665` (used in older cycles) don't exist in this cycle at all. The actual files were inspected directly (rather than guessed from memory) and the real available variables were substituted: `DBQ360` (fast food frequency), `DBQ930` (food security — repurposed as an environmental/SES variable), and `PAD680` (sedentary minutes/day, a reliable long-standing NHANES variable).

### 3.2 Final verified dataset
- **8,153 adult participants (18+), 39 usable features** across four categories: physiological (HbA1c, fasting glucose, lipid panel, BMI, blood pressure, kidney markers), behavioral (smoking, alcohol, sedentary time), environmental (income-to-poverty ratio, food security), and complication labels (self-reported hypertension, kidney disease, four cardiovascular conditions).
- **Diabetes target distribution:** 1,073 diagnosed diabetic, 272 prediabetic/borderline, 6,803 non-diabetic — a healthy, usable class split.
- **One feature flagged for removal before modeling:** `DBQ360` is 96% missing (only 342 of 8,153 respondents were actually asked the question due to a NHANES skip pattern) and should be dropped rather than imputed.
- **One data-quality note carried forward:** `PAD680` contains sentinel values (9999/7777 meaning "don't know"/refused) that need recoding to missing during preprocessing, not treated as literal minute counts.

**Files produced:** `nhanes_merged.csv` (final verified version), `merge_nhanes_core.py`, `inspect_xpt.py`.

---

## 4. Track B — MIMIC-IV: what was actually done

### 4.1 The access-strategy decision
Full MIMIC-IV access was applied for (PhysioNet credentialing + CITI training submitted) but approval was still pending. Rather than block progress on that, two alternatives were evaluated honestly:

- **UCI "Diabetes 130-US Hospitals" dataset** was seriously considered, then **rejected** after real scrutiny — verification showed only a minority of its ~70,000 patients have repeat encounters, it has no real time-interval information between encounters, and its lab values are coarse categorical bins rather than continuous numbers. It was judged too weak a substitute for the "dynamic forecasting" claim in the problem statement and set aside in favor of a stronger option.
- **The MIMIC-IV Clinical Database Demo** (an open-access, no-credentialing-required, 100-patient subset of the real MIMIC-IV database) was chosen instead — it has genuine timestamps (with per-patient-consistent relative time gaps preserved, even though absolute calendar dates are shifted for de-identification) and continuous lab values, which the UCI dataset lacked.

### 4.2 Cohort extraction — real, verified results
Six `hosp/` module files were obtained and inspected directly (`patients`, `admissions`, `diagnoses_icd`, `d_icd_diagnoses`, `labevents`, `d_labitems`). A cohort-extraction script was built and run, producing:

- **35 diabetic patients** identified out of the 100-patient demo, via ICD-9 (`250.x`) and ICD-10 (`E08`–`E13`) diagnosis codes.
- **18 of those 35 (51%) have 2 or more separate hospital admissions** — one patient has 20 admissions.
- **20 patients have real HbA1c readings**, several with many repeated measurements.
- **13 patients have 2+ HbA1c readings**, enough to compute a genuine trend — this is the actual usable sample size for trajectory-based modeling, and it is being stated honestly rather than inflated.
- A real, verified example trajectory: patient `10014354` has **18 HbA1c readings over 4.5 years**, climbing from 7.1% to a peak near 10.2% before settling near 9.5% — genuine disease-progression signal, not synthetic.
- **Complication prevalence across the 35-patient diabetic cohort:** nephropathy 16/35, cardiovascular disease 20/35, neuropathy 3/35, retinopathy 1/35. Neuropathy and retinopathy are too sparse at this scale to model reliably and are being scoped as future work pending full-dataset access; nephropathy and cardiovascular disease have enough cases to proceed with now.

### 4.3 Two output datasets and why both exist
- **`mimic_trajectories.csv`** (long format, 1,288 rows) — one row per glucose/HbA1c reading, in time order. This is the raw sequential input the LSTM/CNN-LSTM model will train on.
- **`mimic_patient_summary.csv`** (wide format, 35 rows) — one row per diabetic patient, with computed trend features (first/last HbA1c, slope, follow-up days) and complication flags. This is what the traditional baseline (logistic regression) and the tree-based model (XGBoost) will train on, since those models need one fixed-length row per patient, not a variable-length sequence.
- Having both, in both shapes, is what makes it possible to directly compare a traditional/flattened-feature model against a true sequence model on the *same underlying data* — this comparison is one of the strongest pieces of evidence the project can offer for its central claim that nonlinear/temporal models outperform traditional ones.

**Files produced:** `mimic_trajectories.csv`, `mimic_patient_summary.csv`, `build_mimic_trajectories.py`.

---

## 5. Honest limitations logged so far (carried into the paper's limitations section, not hidden)

- Track B's genuinely usable trajectory sample is **n = 13** at the current demo-data scale — framed explicitly as a small-scale proof-of-concept, with full MIMIC-IV access (still pending) expected to expand this substantially.
- Neuropathy and retinopathy have too few cases in the current MIMIC-IV cohort to model; only nephropathy and cardiovascular disease are being carried forward as Track B complication targets for now.
- NHANES is cross-sectional (one row per person) and cannot, by itself, demonstrate real temporal forecasting — this is precisely why Track B exists as a separate, necessary component rather than an optional add-on.
- `DBQ360` (NHANES diet variable) is being dropped due to a 96% missingness rate caused by a survey skip pattern, not a data error.

---

## 6. What's ready right now vs. what's next

| Status | Item |
|---|---|
| ✅ Done | Problem statement analysis, domain research, literature survey (3 papers, 2025–2026) |
| ✅ Done | Seven-layer system architecture designed |
| ✅ Done | Two-mode demo app concept designed, with full user-journey mapping |
| ✅ Done | NHANES acquired, cleaned, merged, verified (8,153 × 39) |
| ✅ Done | MIMIC-IV Demo acquired, diabetic cohort extracted, trajectories + summary built and verified |
| ⏳ Pending (not blocking) | Full MIMIC-IV credentialed access — will be substituted in if it arrives, current demo-based results stand on their own regardless |
| ⬜ Not started | Preprocessing/feature engineering (imputation, lipid ratios, obesity/prediabetes flags, sentinel-value cleanup) |
| ⬜ Not started | Class balancing (SMOTE/CTGAN) |
| ⬜ Not started | Model training (baseline vs. XGBoost vs. LSTM) |
| ⬜ Not started | Explainability layer (SHAP, Integrated Gradients, stability testing) |
| ⬜ Not started | Demo app build |
| ⬜ Not started | Paper writing |

---

## 7. Immediate next step

With both datasets now verified and in usable shape, the logical next action is the **preprocessing and feature-engineering stage** — applying imputation, engineering the lipid ratios and clinical flags for NHANES, cleaning the sentinel values, and preparing both the NHANES table and both MIMIC-IV tables into fully model-ready form. This is the next unstarted item in the table above and the direct predecessor to model training.

---

*Companion to: T2D_Progression_Project_Report.md, Dataset_Access_Guide.md, Product_Story_Build_to_Use.md, merge_nhanes_core.py, build_mimic_trajectories.py*
