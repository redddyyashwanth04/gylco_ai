# Base Papers and Gap Analysis — What We Build On Top Of

### Two 2025–2026 papers selected because they map directly onto our two tracks, not just the general topic

---

## Why "base papers" and not just "related work"

Everything in the earlier literature survey was context — showing the field's general direction. These two are different: each one **uses almost the exact same data-and-task setup we've already built** (NHANES-style multi-label complication prediction for Track A; MIMIC-IV diabetic-cohort longitudinal complication modeling for Track B). That means we can compare directly against their reported numbers, adopt their evaluation protocol, and — most usefully — extend past the specific limitations they disclose, on data we already have in hand.

---

## Base Paper 1 (Track A): Zamani, Farhadian, Piran & Borzouei (2026)

**"A Stacked Ensemble Multi-Label Model for Predicting Co-Occurring Microvascular and Macrovascular Complications in Type 2 Diabetes."** *Chronic Diseases and Translational Medicine*, 12: 151–160. Hamadan University of Medical Sciences, Iran.

### What they did
A retrospective cohort of **965 T2DM patients**, with complications grouped into **microvascular** (retinopathy, nephropathy, neuropathy) and **macrovascular** (cardiovascular, cerebrovascular). They built a **class-weighted stacking ensemble** combining three base learners — Random Forest, LightGBM, CatBoost — inside two multi-label frameworks (Binary Relevance, which predicts each complication independently, and Classifier Chain, which lets earlier predictions inform later ones), with a logistic regression meta-learner combining them. Evaluated with 5-fold cross-validation on Hamming Loss, F1, and AUC. SHAP was used for interpretation.

### Results
The Stacking–Classifier Chain configuration was their best: **F1 = 0.752 ± 0.049, AUC = 0.857 ± 0.032**. They also report a moderate correlation (r = 0.35) between co-occurring complications, which they use to justify the multi-label framing itself — complications aren't independent events, so modeling them jointly is claimed to add real value over separate single-label models.

### Gaps (real, stated or clearly implied by the paper's own design choices)
- **Single-center, single-country retrospective cohort (965 patients)** — no population-scale, no demographic/behavioral/environmental breadth. It cannot speak to how income, diet, or activity level relate to complication risk, because those variables simply aren't in a hospital's retrospective chart data.
- **No explanation-robustness testing** — SHAP is computed once and reported; whether those explanations are stable under small input perturbations (the stability check from the MDPI/Joseph et al. paper in our earlier survey) is not examined here at all.
- **Binary Relevance vs. Classifier Chain is the only multi-label architecture comparison** — a gradient-boosted stacking ensemble is compared against itself in two wrapper strategies, not against a genuinely different model family.
- **No traditional-statistical baseline reported** — there's no logistic regression or simple additive model shown losing to the ensemble; the "traditional models underperform" claim (central to our own problem statement) is asserted by framing, not demonstrated with a number.

### What we implement on top of it
1. **Reproduce their exact multi-label architecture** — stacking ensemble (RF + LightGBM + CatBoost) inside Binary Relevance and Classifier Chain wrappers — on our NHANES data. This gives us a direct, apples-to-apples benchmark: if our NHANES-trained version lands in a broadly similar F1/AUC range, that's real evidence our data and pipeline are sound.
2. **Add the traditional-baseline comparison they skip.** We run logistic regression (single-label, one per complication) alongside the ensemble on the same NHANES split, and report the gap explicitly — this is the one number their paper never gives, and it's the number our own problem statement is actually about.
3. **Add the explanation-stability check they don't do**, exactly as designed for our architecture's Explainability layer — perturb inputs slightly, measure SHAP top-feature drift, report it as a number, not just a plot.
4. **Extend their two complication categories with our behavioral/environmental variables** — since NHANES gives us income, diet, smoking, and sedentary time, which their hospital-chart dataset structurally cannot include, we can test whether adding these measurably changes SHAP-ranked feature importance or predictive performance — a genuinely new empirical question their setup couldn't ask.

---

## Base Paper 2 (Track B): Huang, Mo, Liu, Wang, Zeng, Jiang, Wang & Bi (2025)

**"Machine learning models predict mortality risk in diabetic neuropathy patients using MIMIC-IV data."** *Scientific Reports* (Nature Portfolio), 15, 38702. DOI: 10.1038/s41598-025-22363-x. Guangzhou University of Chinese Medicine / Guangdong Second Hospital of Traditional Chinese Medicine.

*(Note: an earlier draft of this document cited an arXiv preprint here. That paper was not published in a journal or confirmed conference proceedings and has been replaced with this genuine peer-reviewed journal article, per department guidance that base papers must be journal-published.)*

### What they did
A diabetic neuropathy (DN) cohort was built from **MIMIC-IV** using ICD-9 codes (`24960`–`24963`) and ICD-10 codes (`E1040`–`E1149`) — the same ICD-code cohort-extraction technique used in our own `build_mimic_trajectories.py`. Patients were restricted to first ICU admission, length of stay >48 hours, age >18, with complete blood-count data. **56 variables** were extracted: demographics, comorbidities, first-day ICU labs, first-day vitals, and severity scores (SAPS-II, APS-III, CCI, etc.), plus six engineered inflammatory ratios (e.g., neutrophil-lymphocyte ratio). LASSO regression selected the final 12 predictors. Four models — Random Forest, XGBoost, SVM, logistic regression — were compared, with SHAP used to interpret the best model.

### Results
**1,313 DN patients** (812 survived, 501 did not), split 80/20 train/validation. Random Forest was best: **training AUC 0.999, validation AUC 0.780**. Top SHAP-ranked predictors: red blood cell distribution width (RDW), neutrophil count, Charlson Comorbidity Index, maximum chloride, minimum prothrombin time.

### Gaps (their own stated limitations, directly usable)
- **Severe overfitting, explicitly disclosed by the authors**: a 0.219 gap between training AUC (0.999) and validation AUC (0.780). They attribute this to the limited dataset size (1,313) and MIMIC-IV's ICU-specific population not generalizing well — a very direct, honest warning for our own similarly small MIMIC-IV Demo cohort (35 patients) to take seriously and guard against (regularization, cross-validation, always reporting the train/validation gap rather than just the best number).
- **No external validation** — the authors state this explicitly and list it as planned future work (they mention wanting to validate on eICU).
- **Single-snapshot design, not longitudinal.** Despite using MIMIC-IV — a dataset with genuinely repeated visits per patient — this paper only uses each patient's **first 24 hours of their first ICU admission**. It never looks at how a patient's values change across multiple admissions. This is the same structural gap our project keeps finding in the literature: even a paper studying a diabetes *complication* on the *exact dataset* we're using treats it as a one-time classification problem, not a trajectory.
- **Outcome is mortality, not complication progression or onset.** It answers "will this DN patient die in-hospital," not "how is this patient's diabetes progressing" or "when will a new complication appear" — the actual question our problem statement asks.

### What we implement on top of it
1. **Reuse their exact cohort-construction and comparison protocol** (RF vs. XGBoost vs. SVM vs. LR, LASSO feature selection, SHAP interpretation) as a direct, replicable benchmark on our own MIMIC-IV Demo complication cohort (nephropathy and cardiovascular, per our earlier prevalence check) — same method family, our data.
2. **Fill the exact structural gap they leave open**: where they use one snapshot per patient, we use the full multi-visit sequence already extracted in `mimic_trajectories.csv` — testing whether tracking a patient's lab trend across admissions (something this published paper never attempts, despite having access to do so) improves prediction over their single-snapshot approach.
3. **Adopt their overfitting warning as a required check in our own evaluation.** Given our even smaller cohort (35 vs. their 1,313), we explicitly report train-vs-validation gap for every model, and treat any gap approaching their 0.219 as a signal to simplify the model or gather more data before trusting the result — this becomes a stated methodological safeguard in our paper, directly citing their disclosed failure mode as the reason we check for it.
4. **Reframe the target from mortality to progression/complication-onset**, which is the actual gap in the literature: nobody in either base paper is predicting *when a new complication appears* using a patient's own trend — that remains our project's real, literature-grounded contribution.

---

## How this changes (sharpens) the project's actual contribution claim

Before this detour, the project's novelty claim was somewhat general ("nonlinear models + explainability + two datasets"). After finding these two base papers, the claim becomes specific and defensible:

> *"We benchmark against Zamani et al. (2026)'s stacking multi-label framework on NHANES, adding the traditional-baseline comparison and explanation-stability testing their study omits. For the longitudinal track, we build on Huang et al. (2025)'s MIMIC-IV diabetic-neuropathy cohort methodology, replacing their single-snapshot (first-24-hours) design with a genuine multi-visit trajectory approach on the same underlying database — directly testing whether the longitudinal signal their published, peer-reviewed study never used actually improves prediction."*

That's a real, checkable, citation-anchored contribution — not just "we used explainable AI on diabetes data," which is now a crowded claim (as the earlier general literature search made clear).

---

## Immediate next step this unlocks

Both base papers converge on the same practical instruction for **Track B specifically**: implement complication markers as irreversible first-occurrence events (already done in our extraction script) feeding a standard sequence model (LSTM/GRU), with the real HbA1c/glucose values as inputs rather than markers alone. Concretely, the next step is: (1) set up the project directory and code structure that will hold the preprocessing, training, and explainability code; (2) build the NHANES preprocessing/feature-engineering pipeline (imputation, lipid ratios, obesity/prediabetes flags, sentinel-value cleanup); (3) build the equivalent MIMIC-IV preprocessing step (sequence padding/truncation, last-observation-carried-forward on risk factors); (4) train the baseline, tree-ensemble, and LSTM models on each track; (5) attach the SHAP/stability explainability layer; (6) build the demo app. These are being tackled in that order, one at a time, starting with the project structure.

---

*Companion to: T2D_Progression_Project_Report.md, Progress_Report_Data_Foundation.md, build_mimic_trajectories.py*
