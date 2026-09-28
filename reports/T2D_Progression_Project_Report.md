# Predicting Type-2 Diabetes Progression Using Machine Learning
### A Foundational Project Report — Problem Analysis, Domain Knowledge, Literature Survey, Scope, and Datasets

*Prepared as a working reference document for project execution and research paper drafting.*

---

## How to use this document

This report is written assuming **zero prior background** in diabetes medicine or in the specific ML sub-fields involved. Every section builds on the last. Read it in order once; after that, use it as a reference while you build the project. Sections 6–8 are the ones you'll return to most often while coding.

**Table of Contents**
1. Problem Analysis (deconstructing the problem statement line by line)
2. Domain Knowledge — Diabetes, from zero
3. Domain Knowledge — The relevant ML/AI concepts, from zero
4. Domain Research — how the field currently approaches this problem
5. Literature Survey — three 2025–2026 papers explained in depth, with gaps
6. Synthesis — what the literature collectively tells you to do
7. Scope of the Project (in-scope vs out-of-scope, and why)
8. Publicly Available Datasets (with realistic access notes)
9. Everything Else the Project Needs (evaluation design, ethics, tooling, deliverables)
10. Consolidated Reading List

---

## 1. Problem Analysis

Let's take the problem statement your department gave you and break it into atomic claims, because each claim implies a different technical requirement.

> "Diabetic patients and endocrinologists struggle with inaccurate prediction of Type-2 Diabetes progression using traditional statistical models."

**What this means technically:** Someone has already tried logistic regression, Cox proportional-hazards regression, or simple risk scores (like the Finnish Diabetes Risk Score, or QDiabetes) and found them inaccurate for *tracking how the disease evolves in an individual over time* — not just whether someone has diabetes or not. This is a **critical distinction** you must hold onto for the whole project: *diagnosis* (do you have diabetes?) is a different, much easier problem than *progression* (how will your disease evolve, and when will complications appear?). Most beginner projects accidentally solve the easy problem while claiming to solve the hard one. Don't let that happen to you.

> "These models fail to capture complex, nonlinear relationships among physiological, behavioral, and environmental factors."

**Technical implication:** Your project needs to explicitly bring together at least three categories of variables:
- **Physiological** — HbA1c, fasting glucose, BMI, blood pressure, lipid panel, kidney function markers (creatinine, eGFR), etc.
- **Behavioral** — diet, physical activity, smoking, alcohol use, medication adherence.
- **Environmental** — socioeconomic status, access to healthcare, urban/rural residence, pollution exposure (yes, this is a real, published risk factor).

And "nonlinear" is your justification for using tree ensembles (XGBoost, LightGBM, Random Forest) or deep learning (LSTM, Transformer) instead of plain linear/logistic regression — because these methods can learn interactions between variables (e.g., "high BMI + low physical activity + family history" combining multiplicatively, not additively) that linear models structurally cannot.

> "Poor disease management leads to severe complications including neuropathy, cardiovascular diseases, and kidney failure."

**Technical implication:** Your model's *output* should ideally relate to these complications — either predicting their onset (classification/survival) or predicting the physiological trajectory that leads to them (regression/time-series forecasting of HbA1c, eGFR, etc.). This sentence is effectively telling you what your target variable(s) should be about.

> "Rising global diabetes prevalence demands more effective predictive tools that leverage real-time and personalized data."

**Technical implication:** "Real-time" suggests continuous/streaming data — wearables, continuous glucose monitors (CGMs), or at minimum, repeated (longitudinal) clinical visits rather than a single snapshot. "Personalized" means your model shouldn't just output a population-average risk; it should account for individual variation (this is a good justification for deep learning approaches or hierarchical/mixed-effects models over one-size-fits-all linear regression).

> "The absence of dynamic, data-driven forecasting limits proactive clinical decision-making."

**Technical implication:** "Forecasting" is a temporal word — you are being asked to predict a *future state*, not just classify a *current state*. This is the strongest textual signal in the whole problem statement that you should NOT build a simple binary "diabetic vs. non-diabetic" classifier, because that isn't forecasting anything — it's diagnosing something that's already true today.

### The single most important takeaway from this analysis

Your problem statement, read literally, is asking for **temporal/longitudinal modeling of disease trajectory and complication risk**, not a one-shot diabetes classifier. This should shape every decision from here on: what data you look for, what target variable you pick, what models you compare, and what you claim as your contribution in the paper.

---

## 2. Domain Knowledge — Diabetes, From Zero

You said to assume you're absolute beginners, so here is the medicine, compressed to what you actually need.

### 2.1 What Type 2 Diabetes (T2D) is

Your body needs glucose (sugar) in the blood for energy. Insulin, a hormone made by the pancreas, is the "key" that lets glucose move from the blood into your cells. In T2D, two things go wrong, usually gradually over years:
- **Insulin resistance** — your cells stop responding properly to insulin, so glucose builds up in the blood.
- **Beta-cell dysfunction** — the pancreas cells that make insulin gradually wear out trying to compensate, and insulin production itself declines.

Unlike Type 1 diabetes (an autoimmune condition, usually diagnosed young, where the pancreas stops making insulin almost entirely), T2D develops slowly, is strongly linked to lifestyle and genetics, and is initially "silent" — many people have it for years before diagnosis.

### 2.2 The key numbers you will see in every dataset

| Marker | What it measures | Normal | Prediabetes | Diabetes |
|---|---|---|---|---|
| **Fasting Plasma Glucose (FPG)** | Blood sugar after ≥8 hrs fasting | <100 mg/dL | 100–125 mg/dL | ≥126 mg/dL |
| **HbA1c (glycated hemoglobin)** | Average blood sugar over ~3 months | <5.7% | 5.7–6.4% | ≥6.5% |
| **Oral Glucose Tolerance Test (OGTT, 2-hr)** | Blood sugar 2 hrs after a sugar drink | <140 mg/dL | 140–199 mg/dL | ≥200 mg/dL |
| **Random Plasma Glucose** | Blood sugar at any time + symptoms | — | — | ≥200 mg/dL |

**HbA1c is the single most important variable in almost every diabetes ML paper** — it is the de facto gold-standard measure of long-term glycemic control, and it is very frequently the SHAP-identified top predictor across the literature you'll read below. If your dataset has it, treat it as a first-class feature.

### 2.3 What "progression" actually looks like clinically

Progression isn't one thing — it has several dimensions researchers model separately:
1. **Glycemic progression** — HbA1c/FPG trending upward over years, insulin requirement increasing, oral medications failing and insulin therapy becoming necessary.
2. **Complication progression** — the appearance, over years, of:
   - **Microvascular complications** (small blood vessels): retinopathy (eye damage, can cause blindness), nephropathy (kidney damage, can progress to kidney failure/dialysis), neuropathy (nerve damage, especially in feet — a leading cause of amputation).
   - **Macrovascular complications** (large blood vessels): cardiovascular disease (heart attack, stroke), peripheral vascular disease.
3. **Multimorbidity progression** — diabetes rarely travels alone; hypertension, obesity, dyslipidemia (abnormal cholesterol), and metabolic syndrome frequently co-occur and interact.

This is why the strongest recent papers (see Section 5) frame the problem as **multi-label** (predicting several complications at once, because they co-occur and share risk factors) rather than single-label classification.

### 2.4 Why traditional statistical models actually fall short (the real reason, not just the marketing reason)

- **Cox proportional-hazards models** (the classic survival-analysis tool used for "time to complication") assume the *effect* of each risk factor on hazard is constant over time and combines additively on the log-hazard scale. In reality, e.g., the effect of high BMI on cardiovascular risk is not the same at age 30 as at age 70, and BMI interacts with HbA1c and blood pressure in ways a linear combination cannot represent.
- **Logistic/linear regression** similarly assumes additive, linear effects and requires you to manually specify any interaction terms you think matter — in a domain with 50+ candidate variables, you cannot manually try every interaction.
- **Cross-sectional design** — most classical risk scores (Finnish Diabetes Risk Score/FINDRISC, QDiabetes) use a single snapshot of the patient, and thus structurally cannot represent *trends* (is HbA1c rising fast or slowly? Was it always high, or did it just start rising?), which is precisely the kind of signal that predicts near-term progression.

---

## 3. Domain Knowledge — The Relevant ML/AI Concepts, From Zero

### 3.1 The three modeling framings you can choose between

| Framing | Question it answers | Example target variable | Typical models |
|---|---|---|---|
| **Classification** | Will complication X occur (yes/no), possibly within N years? | Binary/multi-label: has retinopathy in next 5 yrs? | Logistic regression, Random Forest, XGBoost, MLP |
| **Regression / time-series forecasting** | What will the patient's HbA1c / eGFR / weight be at the next visit or in N months? | Continuous value at future time point | Linear regression, LSTM, Transformer, Gradient Boosting on lag features |
| **Survival analysis (time-to-event)** | *When* will complication X occur, accounting for patients who haven't experienced it yet during the study (censoring)? | Time until event + event indicator | Cox Proportional Hazards, Random Survival Forest, DeepSurv, discrete-time survival neural nets |

As established in Section 1, **survival analysis or longitudinal forecasting are the framings that actually match your problem statement's language of "progression" and "forecasting."** Plain classification of "diabetic vs not" does not.

### 3.2 Key model families you'll be comparing

- **Traditional statistical baseline (your "straw man"):** Logistic regression, Cox regression. You need these in your paper specifically because your problem statement claims traditional models are inaccurate — you must *demonstrate* this on your own data, not just assert it.
- **Tree ensembles (the strong tabular baseline in almost every 2025–2026 paper you'll find):** Random Forest, XGBoost, LightGBM, CatBoost. These consistently outperform both linear models and often deep learning on structured/tabular clinical data with a few thousand patients — this is a well-established empirical pattern, not a shortcut. Expect these to be your strongest single-model baseline.
- **Recurrent/sequence models:** LSTM, Bi-LSTM, GRU — designed for sequences (a patient's visits over time). Good when you have genuinely longitudinal (multi-visit) data.
- **Convolutional + recurrent hybrids (CNN-LSTM):** CNN layers extract local temporal patterns (e.g., short-term glucose spikes) at multiple resolutions; LSTM layers model longer dependencies. See Paper 2 in Section 5 — this is currently one of the strongest published architectures for this exact problem.
- **Transformers / EHR foundation models:** Pretrained on large volumes of EHR data, then fine-tuned (often with parameter-efficient methods like LoRA — Low-Rank Adaptation, which updates only a small number of extra parameters instead of the whole model) on your specific task. See Paper 3 in Section 5.
- **Explainable AI (XAI) layer, applied on top of any of the above:**
  - **SHAP (SHapley Additive exPlanations)** — based on game theory; assigns each feature a contribution value for each individual prediction. Works best with tree models.
  - **Integrated Gradients** — an attribution method for neural networks, computed by integrating gradients along a path from a baseline input to the actual input.
  - You will use SHAP for your tree-based models and, if you build a neural model, Integrated Gradients for that model — this pairing is exactly what the strongest 2026 papers do.

### 3.3 Concepts you'll need for handling real clinical data (all datasets have these problems)

- **Class imbalance** — most complications are rare in any given dataset (e.g., only 10–15% of patients might have diagnosed nephropathy). Plain accuracy is misleading here (a model that always predicts "no complication" can still score 85%+ accuracy while being clinically useless). You need to report **sensitivity/recall, F1-score, AUC-PR (area under precision-recall curve), and per-class metrics**, not just accuracy.
- **Class-balancing techniques** — SMOTE (Synthetic Minority Oversampling), or more advanced generative approaches like CTGAN (Conditional Tabular GAN) which synthesizes realistic synthetic patient records for underrepresented outcomes. Applied *only to the training set*, never the test set (this is a common student mistake that invalidates results).
- **Missing data** — real clinical data is full of gaps (a lab test wasn't ordered, a field wasn't filled in). Standard approaches: mean/median imputation (simple), iterative/multivariate imputation (e.g., Random-Forest-based, more accurate), or last-observation-carried-forward (LOCF) for longitudinal data.
- **Data leakage** — the single most common way beginner medical ML projects produce fake "amazing" results. Examples: including a lab test in your features that is only ever ordered *because* a doctor already suspects the complication; splitting train/test randomly on a longitudinal dataset such that the same patient appears in both sets; or, in a time-to-event setup, using information recorded *after* the prediction point. You must split by patient and, for longitudinal data, split by time.
- **Calibration** — beyond just "is the ranking of risk correct" (AUC), for clinical use you also care whether "a predicted 20% risk" really does correspond to roughly 20% of such patients experiencing the outcome. Reported via calibration curves and the **Brier score**.
- **Explanation stability** — a newer and increasingly important evaluation criterion (see Paper 3): do your SHAP/explanation outputs stay consistent if you perturb the input slightly? An explanation that flips its top reasons under tiny noise cannot be trusted by a clinician.

---

## 4. Domain Research — How the Field Currently Approaches This Problem

Having read broadly across 2025–2026 publications (PubMed, Frontiers, MDPI, Nature Scientific Reports, medRxiv preprints, Oxford Bioinformatics) to prepare this report, a few consistent patterns emerge about where the field currently is:

1. **Most published work is still diagnostic (T2D onset prediction), not progression modeling.** The overwhelming majority of papers predict "will this person develop T2D in N years" from a single baseline snapshot. Genuine multi-step or multi-year *progression/complication* modeling is a smaller, more specialized, and more novel sub-field — which is good news for you, since it means your unique angle has real room to be genuinely novel rather than a rehash.
2. **Gradient-boosted tree ensembles (especially XGBoost/CatBoost) remain the strongest single-model baseline on structured/tabular clinical data**, consistently outperforming plain logistic regression and often matching or beating early deep-learning attempts when the dataset has a few thousand patients or fewer. Deep learning tends to win specifically when (a) the data is genuinely longitudinal/sequential and (b) there are tens of thousands of patient-visits, not just a few hundred patients.
3. **Explainability is treated as a first-class requirement, not an afterthought**, in essentially every 2025–2026 paper on clinical complication prediction — SHAP for tree models and Integrated Gradients/attention for neural models are near-universal. Several 2026 papers go further and treat *explanation stability under perturbation* as a quantitative metric to report (not just "here is a SHAP plot"), which is currently a genuinely cutting-edge practice worth adopting.
4. **Multi-label / multi-complication prediction (predicting several complications jointly)** is emerging as a more clinically realistic and more novel framing than single-complication prediction, because complications co-occur and share risk pathways — but it is still relatively rare, which again signals a good area for differentiation.
5. **Class imbalance and small regional datasets are the field's biggest practical bottleneck** — almost every non-Western-population paper explicitly reports working with a few hundred to a few thousand patients and having to apply synthetic data generation (CTGAN, SMOTE) to make minority complications learnable. This is *normal* for a student project too — don't be discouraged if your dataset is small; document how you handled it instead.
6. **External validation is rare and explicitly flagged as a limitation** in nearly every paper reviewed — models are validated on one hospital/region's data and not tested elsewhere. If your project at least *discusses* this limitation honestly (and ideally tests on 2 datasets, even small ones), you are already ahead of much published work in terms of rigor.
7. **EHR foundation models** (large pretrained models like CLMBR, EHRMamba, Hyena-based sequence models) are the current frontier, adapted via parameter-efficient fine-tuning (LoRA) — but they are complex, computationally heavy, and mostly evaluated on broad endpoints (mortality, readmission) rather than diabetes complications specifically, which is exactly the gap Paper 3 below addresses.

---

## 5. Literature Survey — Three 2025–2026 Papers in Depth

You asked for two or three very recent papers (2025–2026) explained in detail including gaps. Below are three that are directly relevant to your project's actual framing (progression / complications / explainability), each from a distinct angle so that together they justify a well-rounded methodology.

---

### Paper 1: Joseph, Dhaouadi, Ramesh, Sagahyroon & Aloul (2026), *"Adapting EHR Foundational Models to Predict Diabetes Complications with Precision Explainability,"* Machine Learning and Knowledge Extraction, 8(4), 89. American University of Sharjah, UAE.

**What they did:** This is arguably the single most relevant paper to your project. <cite index="20-1">The authors present a data-driven framework for predicting multiple diabetes-related complications using structured electronic health record data while ensuring clinically meaningful explainability, adapting a pretrained EHR foundation model to operate on static patient data and integrating it with classical machine learning baselines to address class imbalance, feature sparsity, and interpretability challenges.</cite>

**Data:** <cite index="20-1">A multi-label prediction setting covering eight common diabetes complications was evaluated using a real-world dataset from a regional diabetes center in the United Arab Emirates.</cite> Specifically, the dataset came from a real diabetes clinic with under a thousand patients and roughly 80 raw features — this is a genuinely realistic scale for a student project to relate to, unlike papers that use half-a-million-patient biobanks.

**Method, step by step (this is a genuinely good template to follow):**
1. Cleaned the data down to complete-enough records, handled missing values with iterative imputation.
2. Because complications were rare/imbalanced, used **CTGAN** (a GAN that generates synthetic tabular data) combined with **PiShield**, a tool that enforces clinically valid value ranges on the synthetic data so it doesn't generate physiologically impossible patients — this "clinically constrained synthetic data" idea is a nice, defensible technique you could replicate.
3. Performed two-stage feature selection: first by XGBoost feature importance, then by SHAP value ranking, keeping only features appearing in both top-25 lists — reducing ~55 candidate variables down to 18 clinically meaningful ones.
4. Compared classical ML (XGBoost, TabNet) against several EHR foundation model variants (CLMBR, GPT/Mamba/Hyena/Llama-based sequence encoders), each adapted to static (non-longitudinal) data and fine-tuned efficiently using **LoRA**.
5. Built a **weighted ensemble** of the best foundation model and XGBoost.
6. Crucially, evaluated not just predictive accuracy but **explanation robustness** — how much do the SHAP/Integrated-Gradients explanations change when you add small random noise to the input?

**Results:** <cite index="20-1">The best-performing configuration, a weighted ensemble combining a low-rank adapted Hyena-based foundation model with a tree-based predictor, achieved an average F1-score of 0.77, an average recall of 0.85, and an example-based F1-score of 0.71, outperforming all individual models, and this ensemble produced the most stable explanations under input perturbations.</cite>

**Gaps this paper itself identifies (directly usable as your paper's "gap" justification):**
- Their study used <cite index="20-1">a single electronic health record dataset derived from a specific clinical setting, which may limit generalizability, and external validation on independent cohorts was not performed.</cite>
- Most EHR foundation models are pretrained on Western populations, limiting applicability to other regions — the authors explicitly call for future work fine-tuning on non-Western populations.
- Robustness/stability evaluation was done on only 20 test patients due to computational cost — a genuinely small evaluation that a more resource-constrained but careful project could actually match or beat in rigor with a bigger perturbation test set.
- No clinician user study confirming that the explanations are actually usable in practice — evaluation stayed purely computational (Jaccard/cosine distance of attribution vectors), not validated against real clinician judgment.
- Complications were treated as static labels ("currently has vs. doesn't have"), not true future-onset prediction, because the dataset lacked longitudinal follow-up — the authors flag this explicitly as a reason they call it "complication identification" rather than "temporal prediction."

**Why this matters for you:** This paper gives you a nearly complete, replicable pipeline (preprocessing → constrained synthetic balancing → dual feature selection → baseline vs. foundation model comparison → ensembling → explanation-stability evaluation) that you can adapt at a smaller, more computationally feasible scale using classical ML + one deep model, instead of multiple foundation model variants, while still directly addressing the same gaps (true longitudinal onset prediction, larger explanation-robustness evaluation, non-Western population if your dataset allows).

---

### Paper 2: Naveed, Noaeen, AboArab, Kaleem, Keshavjee & Guergachi (2025), *"Temporal Deep Learning with Clinically Engineered Biomarkers for the Early Prediction of Type 2 Diabetes,"* medRxiv preprint, posted Dec 2025 (University of Toronto / University of Management and Technology, Lahore / Tanta University).

**What they did:** <cite index="21-1">This study introduces a hybrid deep learning framework that integrates hierarchical temporal modeling with clinically engineered predictors for early T2D risk estimation, including data preprocessing, temporal sequencing, and derived biomarkers such as triglyceride-to-HDL cholesterol ratio, LDL/HDL ratio, TC/HDL ratio, VLDL, obesity status, and prediabetes indicators.</cite>

**Data:** <cite index="21-1">Experiments were conducted on 19,218 patients and 368,790 clinical visits from the Canadian Primary Care Sentinel Surveillance Network (CPCSSN)</cite> — a genuinely large, genuinely longitudinal (multi-visit) primary-care dataset spanning 1998–2015, which is what made real temporal deep learning possible here (contrast with Paper 1's smaller, static dataset).

**Method, step by step:**
1. Arranged each patient's repeated visits chronologically to form a real time-series per patient, imputed missing values with cohort means, applied last-observation-carried-forward for outliers.
2. **Feature engineering grounded in physiology** — instead of just feeding raw lipid values, they computed *ratios* known from clinical literature to be more predictive: TG/HDL, LDL/HDL, TC/HDL, VLDL (estimated as triglycerides ÷ 2.2), plus binary prediabetes and obesity flags. This is an important, easily reusable idea: **engineered clinical ratios frequently outperform raw values** because they encode known physiological relationships (e.g., TG/HDL as a marker of insulin resistance) that a model would otherwise have to "discover" from scratch with more data than you likely have.
3. Built a **multilevel CNN–LSTM architecture**: convolutional pooling layers extract low-, mid-, and high-level temporal features (i.e., patterns over short, medium, and long time windows) from the visit sequence; each level is fed to its *own separate* LSTM module rather than being collapsed into one stream; the LSTM outputs are then fused ("late fusion") into a single patient representation before final prediction.
4. Benchmarked exhaustively against KNN, SVM, LSTM, Bi-LSTM, and standard (single-stream) CNN-LSTM, at two different training-set sizes (3,892 vs. 6,281 patients) and two feature configurations (raw variables only vs. raw + engineered ratios) — a genuinely thorough 24-experiment factorial design.

**Results:** <cite index="21-1">For the larger cohort with the integrated feature set, the model achieved 93.2% accuracy, 75.7% sensitivity, 98.8% specificity, and an 84.4% F1 score, outperforming Bi-LSTM, SVM, KNN, and baseline CNN-LSTM models.</cite> Feature importance analysis confirmed <cite index="21-1">fasting blood sugar, HbA1c, and lipid ratios as the strongest predictors</cite> — directly corroborating the domain knowledge in Section 2.2 of this report.

**Gaps this paper itself identifies:**
- Trained and validated only within the Canadian CPCSSN network; the authors explicitly state <cite index="21-1">transportability to other EHR networks, hospital or specialty settings, and health systems outside Canada remains untested</cite>, and that site differences in coding practices and population characteristics can affect performance.
- The dataset lacked family history of diabetes, polycystic ovary syndrome status, and skinfold thickness — established risk markers that were simply unavailable, a very common and honest data-availability limitation you will likely also face.
- No unstructured data (clinical notes) was used — purely structured/tabular longitudinal features.
- The architecture assumes a *uniform* time structure after preprocessing, but real EHR visit intervals are irregular; the authors suggest time-aware attention or continuous-time neural models as future work — a genuinely open, fairly advanced research direction if you want extra novelty.
- Despite good predictive metrics, the fusion and recurrent layers <cite index="21-1">remain partially opaque, and the authors suggest applying explainability methods such as attention weight visualization or counterfactual reasoning</cite> — meaning this paper, unlike Paper 1, does *not* have a rigorous explainability component. This is precisely the gap Paper 1 fills, and it's why reading both together gives you a complete methodological picture.
- No prospective (real-world, forward-looking) clinical evaluation — purely retrospective.

**Why this matters for you:** This paper demonstrates, very concretely, that (a) physiologically-grounded feature engineering measurably improves performance across every model type tested (not just the proposed one), and (b) multi-scale temporal modeling (separate short/medium/long-term LSTM streams) beats single-stream sequence models. If your dataset has any longitudinal structure at all (even just 2–3 visits per patient), this architecture and feature-engineering approach is directly reusable at a smaller scale, and pairing it with Paper 1's explainability layer covers Paper 2's own stated weakness.

---

### Paper 3: Majyambere, Lindgren, Twizere & Ntakirutimana (2026), *"Early Type 2 Diabetes Risk Prediction Using Explainable Machine Learning in a Two-Stage Approach,"* Frontiers in Digital Health, 8:1743619. Stockholm University / University of Rwanda.

**What they did:** <cite index="1-1">This study presents a two-stage, explainable machine learning framework for systematic screening for type 2 diabetes: the first stage evaluates risk based on reported symptoms, while the second stage incorporates demographic, anthropometric, and clinical/laboratory data.</cite> <cite index="4-1">The paper explicitly foregrounds diabetes management, prediction, explainable machine learning, interpretability, a Multi-Layer Perceptron, and SHAP as its core themes.</cite>

**Why this one is included as your third paper:** Papers 1 and 2 both come from resource-rich research settings (UAE regional diabetes center with an 80-variable EHR system; a Canadian national primary-care surveillance network). Paper 3 is valuable specifically *because* it comes from a lower-resource healthcare context (Rwanda), and it tackles a real-world deployment constraint your own project may face: **not every patient has expensive lab tests available.** Its two-stage design — first screen cheaply using only symptoms/history (things a patient can self-report with no blood draw), then only run the more expensive clinical/lab-based model on people flagged as at-risk — is a genuinely practical, deployable pattern for "industry grade" thinking, especially relevant if your dataset (or your target use-case) includes patients without complete lab panels.

**Method (at the structural level, based on the published framework):**
1. **Stage 1 — symptom-based screening model:** trained only on self-reportable variables (e.g., excessive thirst, frequent urination, fatigue, family history) — no blood test required. This model's job is high *sensitivity*: don't miss possible cases, even at the cost of some false positives, because it's just a triage step.
2. **Stage 2 — clinical confirmation model:** for patients flagged by Stage 1, a second model incorporating demographic, anthropometric (BMI, waist circumference), and laboratory data (glucose, HbA1c) makes the final, more confident risk assessment.
3. Multi-Layer Perceptron (MLP) is used as the underlying model family, with **SHAP** applied for explainability at both stages, so that both the cheap screening decision and the expensive confirmation decision can be justified to a clinician.

**Gaps and relevance:**
- This kind of staged, cost-aware architecture is **almost never discussed** in the higher-resource papers (1 and 2), yet it is exactly the kind of practical, deployment-realistic design that separates an "industry grade" project from an academic-only demonstration — it directly acknowledges that real clinics have resource constraints, unlike papers that assume a full 60–80 variable EHR is always available.
- Two-stage/cascade architectures like this introduce their own methodological subtlety that most papers (including this one, based on available details) don't fully address: **error compounding** — if Stage 1 has poor recall, Stage 2 never even sees those missed patients, and the *reported* performance of Stage 2 alone can look artificially good because it's only evaluated on the subset Stage 1 already flagged. Any project replicating this design should explicitly report **end-to-end pipeline performance** (Stage 1 → Stage 2 combined), not just Stage 2's isolated numbers — a genuinely useful methodological point you can add to your own project's rigor.
- Like the other two papers, this is fundamentally a **diagnostic/onset** framing (does this person have or is at risk of T2D), not a true multi-year progression/complication framing — reinforcing the earlier point that genuine progression modeling remains comparatively underexplored, and is where your project's novelty can live.

---

## 6. Synthesis — What the Literature Collectively Tells You to Do

Putting the three papers together with the domain research in Section 4, here is the concrete methodological recipe they jointly point to, which should become the backbone of your project:

1. **Frame the problem as multi-label complication prediction or short-horizon progression forecasting**, not simple binary diagnosis — this is both more clinically meaningful (Paper 1) and the area with the most room for genuine novelty (Section 4).
2. **Engineer physiologically grounded features** (lipid ratios, rate-of-change of HbA1c between visits, etc.) before throwing raw variables at any model — Paper 2 shows this measurably helps every model type, not just the fancy one.
3. **Use tree ensembles (XGBoost/LightGBM) as your strong, fast, and defensible primary baseline**, and only add a deep sequence model (LSTM/CNN-LSTM, or a lightweight Transformer) if your data actually has meaningful longitudinal structure (≥2–3 visits per patient) — don't force a complex architecture onto essentially cross-sectional data.
4. **Address class imbalance explicitly**, ideally with a clinically constrained synthetic balancing approach (Paper 1's CTGAN + range-constraint idea, or simpler SMOTE if compute/time is limited) — and report the ablation (with vs. without balancing), because reviewers/evaluators will want to see that you understand *why* it matters, not just that you did it.
5. **Make explainability a first-class, quantitatively evaluated component**, not a single SHAP plot at the end — compute SHAP for your tree model, and if you build a neural model, Integrated Gradients for it, then (following Paper 1) test explanation **stability under small input perturbations** as an actual reported metric. This single addition is currently one of the most differentiating things you can do relative to typical student projects.
6. **If resource/cost realism matters for your use-case, consider a staged/cascade design** (Paper 3) — cheap symptom-based triage feeding into a fuller clinical model — and if you do, explicitly report combined end-to-end performance, not just the final stage in isolation.
7. **Be honest about generalizability limits** — single-institution/single-region data, lack of external validation, and lack of prospective testing are limitations that *every one of these three 2025–2026 papers* explicitly discloses. Doing the same in your own report is not a weakness; it is what separates rigorous work from overclaiming, and evaluators consistently reward this honesty.

---

## 7. Scope of the Project

Given a genuinely available six-week window, here is a scope that is ambitious but achievable, split clearly into what's in and what's out.

### 7.1 In scope

- **Problem framing:** Multi-label prediction of diabetes-related complications (pick 3–5 of: hypertension, obesity/metabolic syndrome, dyslipidemia, nephropathy, neuropathy, retinopathy, cardiovascular disease) OR short-horizon (e.g., 1-year-ahead) forecasting of HbA1c/glycemic trajectory — pick **one** as primary, the other can be a secondary experiment if time allows.
- **Baselines:** Logistic/linear regression, Cox proportional hazards (if doing time-to-event) — these exist specifically to let you demonstrate, on your own data, the "traditional models are inaccurate" claim from the problem statement.
- **Primary model:** Gradient-boosted trees (XGBoost or LightGBM) as your strongest, most defensible model.
- **Secondary/comparison model:** One sequence model (LSTM or a small Transformer) IF your chosen dataset has real longitudinal structure; otherwise, a well-tuned MLP as your "deep learning" comparison point.
- **Class imbalance handling:** SMOTE at minimum; CTGAN-based synthetic balancing as a stretch goal if time allows (Week 4–5).
- **Explainability:** SHAP for tree models (mandatory); Integrated Gradients for the neural model (if built); basic explanation-stability check (perturb inputs slightly, measure how much the top-5 SHAP features change) as your differentiating contribution.
- **Evaluation:** AUC-ROC, AUC-PR, F1, sensitivity/specificity, calibration (Brier score) — reported per-label if multi-label.
- **End-to-end demo:** A simple Streamlit or Gradio app where a user inputs patient variables and receives a risk prediction + SHAP explanation + confidence/uncertainty indication.
- **Subgroup/fairness check:** Performance broken down by at least age group and sex, to show awareness of equity — increasingly expected in 2025–2026 clinical ML papers and easy to add.
- **Paper:** A structured research report/preprint targeting arXiv and/or a student conference track, following the standard IMRaD-plus structure.

### 7.2 Explicitly out of scope (and why — state this in your paper's limitations, don't hide it)

- **Prospective/real clinical deployment or trial** — no time, no ethics approval pathway in six weeks; state this as future work, as every paper you read also does.
- **Full EHR foundation model pretraining from scratch** (like CLMBR/EHRMamba) — Paper 1 shows even a well-resourced research group used pretrained models and LoRA fine-tuning, not from-scratch training; you should do the same *if* you use a foundation model at all, or skip this entirely and rely on the tree ensemble + sequence model comparison, which is still a fully legitimate and current approach.
- **Genomic/imaging data integration** — interesting but a separate, much larger project; stick to structured clinical + behavioral + demographic tabular/longitudinal data.
- **Multi-institution external validation** — use train/validation/test splits within your chosen dataset(s); if you can get access to a *second*, smaller public dataset for even a partial validation check, that's a bonus, not a requirement.
- **Real-time streaming/wearable integration** — unless you specifically find and use a CGM (continuous glucose monitor) dataset; otherwise, treat "real-time" in the problem statement as "the model should be usable at each new clinical visit," not literally live streaming data.

---

## 8. Publicly Available Datasets

Ranked roughly by realistic accessibility for a six-week academic project.

### 8.1 Immediately accessible, no application needed

| Dataset | Scale | Structure | Best for | Notes |
|---|---|---|---|---|
| **NHANES** (National Health and Nutrition Examination Survey, CDC) | Tens of thousands of participants across multiple survey cycles | Structured tabular; repeated cross-sectional cycles (not per-patient longitudinal, but multiple *years* of survey data exist) | Population-level risk factor analysis, prediabetes progression proxies, subgroup/fairness analysis | <cite index="23-1">A cornerstone population-representative dataset for AI models in healthcare, with decades of anthropometric, laboratory, and lifestyle data.</cite> Free, no application, direct CDC download. |
| **Pima Indians Diabetes Dataset** | 768 records, 8 features | Simple structured tabular | Quick baseline sanity-check only — NOT your main dataset | Extremely overused; reviewers will penalize a project that relies on this alone. Use only to validate your pipeline works before moving to a real dataset. |
| **sklearn `diabetes` (Regression) dataset** | 442 patients, 10 features, 1 continuous progression target | Structured tabular, one-year progression score | Good for demonstrating you understand *regression-based progression* framing on a toy scale | Built into scikit-learn (`sklearn.datasets.load_diabetes`); genuinely designed around a "disease progression one year after baseline" target, which matches your problem statement's language well as a small demonstration. |
| **BRFSS (Behavioral Risk Factor Surveillance System)** | Hundreds of thousands of respondents/year, CDC | Structured tabular, self-reported behavioral + health data | Behavioral/environmental variable-rich modeling; multi-complication prediction (as used in Paper 1's related work — Dzakiyullah et al.) | Free, direct download from CDC; commonly used for multi-label complication studies. |
| **CPCSSN-style / other national primary care networks** | Varies | Longitudinal, multi-visit | True temporal modeling | CPCSSN (used in Paper 2) requires a formal data-access application, not instantly downloadable — check if a public sample/subset exists, or use as an architecture inspiration and instead find an open longitudinal alternative below. |

### 8.2 Requires free registration/credentialing (start this in Week 1 if you want it)

| Dataset | Scale | Structure | Notes |
|---|---|---|---|
| **MIMIC-IV / MIMIC-III** (PhysioNet, MIT) | Tens of thousands of ICU/hospital patients | Rich longitudinal EHR: labs, vitals, diagnoses, medications, time-stamped | Free but requires completing a short human-subjects-research training course and signing a data use agreement — typically approved within a few days. Good for a genuinely longitudinal, realistic EHR structure; note it's ICU/hospital-focused, so diabetes is a comorbidity within a broader critical-care population, not the primary cohort — filter carefully. |
| **UK Biobank** | ~500,000 participants | Genomic, clinical, imaging, longitudinal linked health records | Extremely rich but access approval process is typically too slow for a 6-week project unless your institution already has an approved access agreement — check with your department first. |

### 8.3 If none of the above give you genuine multi-visit longitudinal structure in time

- **Simulate realistic longitudinal data**: take a real cross-sectional dataset (e.g., NHANES or BRFSS) and generate multiple synthetic follow-up visits per patient using clinically grounded progression rules (e.g., HbA1c drifts upward with a rate correlated to baseline BMI, age, and lifestyle factors, plus noise). This is a legitimate, commonly used technique **as long as you are fully transparent about it** in your methodology and limitations sections — reviewers respect disclosed simulation far more than an undisclosed static-data-treated-as-longitudinal shortcut (which is exactly the kind of flaw Paper 1 had to explicitly caveat).

### 8.4 Recommended path given your timeline

1. **Week 1:** Apply for MIMIC-IV/PhysioNet access immediately (fast, free, and gives you real longitudinal EHR structure) while simultaneously downloading NHANES and/or BRFSS as your guaranteed fallback/primary dataset.
2. Use **NHANES or BRFSS** as your primary dataset for the multi-label complication / cross-sectional risk modeling track (guaranteed to work, no access delay).
3. If MIMIC-IV access comes through in time, use it for a genuine longitudinal/temporal modeling track (LSTM/CNN-LSTM comparison) as a secondary experiment — this is exactly the kind of "two datasets, two framings" design that meaningfully strengthens a paper's credibility per Section 6, point 7.

---

## 9. Everything Else the Project Needs

### 9.1 Tooling and environment
- **Python** (3.10+), with `scikit-learn`, `xgboost`/`lightgbm`, `pandas`, `numpy`, `shap`, `imbalanced-learn` (for SMOTE) as your core stack.
- **PyTorch** (or TensorFlow/Keras) only if you build the LSTM/Transformer component.
- **Streamlit** or **Gradio** for the end-to-end demo app — both let you build a usable clinician-facing interface in under a day.
- **Weights & Biases or simple CSV logging** to track experiment results systematically — you will run many model/feature/balancing combinations, and you need a reproducible record of each, not just the final numbers.

### 9.2 Evaluation design checklist
- Patient-level train/validation/test split (never row-level, if you have multiple rows per patient).
- Time-aware split if longitudinal (train on earlier years, test on later years, or hold out each patient's most recent visit).
- Report per-label metrics if multi-label, not just averaged/aggregate metrics.
- Include at least one calibration metric (Brier score) alongside discrimination metrics (AUC).
- Include a subgroup breakdown (age, sex, and any other demographic your data supports) — this is now close to a standard expectation, not a nice-to-have.
- Ablation table: baseline → + feature engineering → + class balancing → + best model → + ensemble (mirrors the structure both Paper 1 and Paper 2 use, and makes your results section far more convincing than a single final-number table).

### 9.3 Ethics and responsible-use notes to include in your report
- State clearly that the tool is a **decision-support aid, not a diagnostic replacement** — this is standard, expected phrasing in every paper you read.
- If using a public dataset with real (even if de-identified) patient data, note the dataset's own consent/ethics provenance (NHANES, BRFSS, and MIMIC all have this documented on their respective sites — cite it).
- Note fairness/subgroup performance explicitly, even if simple, per Section 9.2.

### 9.4 Paper structure (recap, now dataset- and gap-informed)
Abstract → Introduction (motivate using this problem statement, cite the prevalence/burden statistics from Papers 2 and 3's introductions) → Related Work (structure it exactly like Section 5 above — three or more recent works, each with a stated gap) → Dataset & Preprocessing → Methodology (baseline vs. proposed pipeline, explicitly diagrammed) → Experimental Setup → Results (with ablation table, subgroup analysis, explainability results, explanation-stability results) → Discussion (clinical implications; compare your numbers honestly against the three papers above where the framing is comparable) → Limitations (state the out-of-scope items from Section 7.2 explicitly here) → Conclusion & Future Work.

### 9.5 Six-week working timeline (skeleton — adjust to your actual calendar)

- **Weeks 1–2:** Literature finalization, dataset access/download, EDA, preprocessing pipeline, baseline models (logistic regression, Cox if applicable) running end-to-end on real data.
- **Weeks 3–4:** Primary model (XGBoost/LightGBM) fully tuned; class imbalance handling with ablation; secondary/deep model if data supports it; SHAP integration.
- **Week 5:** Explanation-stability evaluation; subgroup analysis; Streamlit/Gradio demo app; full ablation table finalized.
- **Week 6:** Paper writing, figure generation, internal review/proofreading, arXiv/preprint submission, final report and presentation preparation.

---

## 10. Consolidated Reading List

**Primary papers (read in full, in this order):**
1. Joseph, T., Dhaouadi, A., Ramesh, J., Sagahyroon, A., & Aloul, F. (2026). *Adapting EHR Foundational Models to Predict Diabetes Complications with Precision Explainability.* Machine Learning and Knowledge Extraction, 8(4), 89. https://www.mdpi.com/2504-4990/8/4/89
2. Naveed, I., Noaeen, M., AboArab, M. A., Kaleem, M. F., Keshavjee, K., & Guergachi, A. (2025). *Temporal Deep Learning with Clinically Engineered Biomarkers for the Early Prediction of Type 2 Diabetes.* medRxiv preprint. https://www.medrxiv.org/content/10.1101/2025.11.26.25341040
3. Majyambere, S., Lindgren, T., Twizere, C., & Ntakirutimana, I. (2026). *Early Type 2 Diabetes Risk Prediction Using Explainable Machine Learning in a Two-Stage Approach.* Frontiers in Digital Health, 8:1743619. https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2026.1743619/full

**Supporting/background reading (skim for context and citation material):**
4. Kuri, K. A. C., Viola, V., & Pozzilli, P. (2026). *Integrating artificial intelligence and machine learning into risk prediction for type 2 diabetes.* Diabetes Research and Clinical Practice, 238:113389. — good general framing of nonlinear risk-factor interactions.
5. Oxford Bioinformatics narrative review (2026). *Artificial intelligence-powered prediction of diabetic complications: from clinical data to molecular omics.* Briefings in Bioinformatics. — a 58-study systematic review of AI for retinopathy/nephropathy/CVD, useful for your Related Work section's breadth.
6. Pan, J. (2026). *Temporal Signals of Disease: A Transformer Approach for Predicting Diabetes from Longitudinal EHRs.* International Journal of Population Data Science, 11(5). — an alternative Transformer-on-longitudinal-labs approach if you want a third architectural comparison point.
7. Xu, Q. et al. (2026). *Machine Learning–Based Prediction Model Construction for Type 2 Diabetes Mellitus: A Comparison of Algorithms and Multilevel Risk Factor Analysis.* Journal of Diabetes Research, 4525736. — uses NHANES directly; useful as a methodological reference if you choose NHANES as your primary dataset.

**Dataset documentation to read before use:**
- NHANES: https://wwwn.cdc.gov/nchs/nhanes/Default.aspx
- PhysioNet / MIMIC-IV: https://physionet.org/content/mimiciv/
- BRFSS: https://www.cdc.gov/brfss/

---

*End of report. Next practical step: lock in your dataset choice (Section 8.4) and your primary framing (Section 7.1) this week — everything else in the six-week plan depends on those two decisions being made first.*
