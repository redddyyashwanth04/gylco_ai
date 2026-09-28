# Dataset Access Guide — Type 2 Diabetes Progression Project

Companion document to the System Architecture. Covers every dataset referenced in the project report, with direct links and exact access steps, ordered from "download right now" to "requires approval time."

---

## Tier 1 — No registration, download today

### 1. NHANES (National Health and Nutrition Examination Survey)
**Best for:** primary dataset — population-scale, rich lab + behavioral + demographic variables, free, instant access.

- **Main portal:** https://wwwn.cdc.gov/nchs/nhanes/Default.aspx
- **Direct data + documentation index:** https://wwwn.cdc.gov/nchs/nhanes/
- **Latest continuous cycle:** https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?Cycle=2021-2023 (use the most recent cycle with complete lab data)
- **Access steps:**
  1. Go to the main portal link above → click **"Questionnaires, Datasets, and Related Documentation."**
  2. Pick a two-year cycle (e.g., 2021–2023). Diabetes-relevant components are under **Laboratory Data** (glucose, HbA1c/glycohemoglobin, lipid panel), **Examination Data** (BMI, blood pressure), and **Questionnaire Data** (diabetes questionnaire, diet, physical activity).
  3. Each component downloads as a `.XPT` (SAS transport) file — read directly in Python with `pandas.read_sas('file.XPT', format='xport')`, or use the R package `nhanesA` for programmatic pulls (`install.packages("nhanesA")`).
  4. Merge components on the shared `SEQN` (respondent sequence number) column.
  5. Read the **Analytic Guidelines** page first (linked on the main portal) — NHANES uses complex survey weights; if you want population-representative statistics you need `nhanesA`'s or `survey`/`srvyr` weighting helpers, though for a pure ML prediction task you can often work with the raw merged records and just document that weighting wasn't applied.
- **No account, no application, no waiting.**

### 2. BRFSS (Behavioral Risk Factor Surveillance System)
**Best for:** the largest source of behavioral/lifestyle variables (diet, exercise, smoking, healthcare access) tied to diabetes status — directly matches your problem statement's "behavioral and environmental factors" language.

- **Main annual data page:** https://www.cdc.gov/brfss/annual_data/annual_data.htm
- **Access steps:**
  1. Open the link above, pick a year (most recent fully released year).
  2. Download the ASCII/SAS/XPT data file plus the **codebook** for that year (codebooks change slightly year to year — always download the matching one).
  3. Filter to the diabetes-relevant questions: `DIABETE3`/`DIABETE4` (diagnosed diabetes status), `_BMI5` (computed BMI), `_RFHYPE5` (hypertension), `EXERANY2` (physical activity), `SMOKE100` (smoking), `_INCOMG` (income bracket, useful as a socioeconomic/environmental proxy).
  4. Alternative, faster route: a pre-cleaned **"Diabetes Health Indicators"** version of a past BRFSS year is hosted on Kaggle and UCI's ML repository — search "Diabetes Health Indicators Dataset BRFSS" on Kaggle if you want a ready-to-use CSV instead of raw survey files (useful for Week 1 pipeline testing, but cite it as a derived/secondary source, not primary BRFSS, in your paper).
- **No account, no application, no waiting** for the CDC raw files.

### 3. UCI/Kaggle "Pima Indians Diabetes Dataset"
**Best for:** pipeline sanity-checking only — do not use as your main dataset (Section 8.1 of the project report explains why).

- **UCI source:** https://archive.ics.uci.edu/dataset/34/diabetes (search "Pima Indians Diabetes" if the ID changes)
- **Kaggle mirror:** search "Pima Indians Diabetes Database" on kaggle.com/datasets — instant CSV download with a free Kaggle account.
- **Access steps:** Kaggle → create a free account → click **Download** on the dataset page, or use the Kaggle API: `kaggle datasets download -d uciml/pima-indians-diabetes-database`.

### 4. scikit-learn built-in `diabetes` regression dataset
**Best for:** a quick demonstration of the "one-year disease progression score" framing that matches your problem statement's language almost exactly — good for a small proof-of-concept before scaling to a real dataset.

- **No download needed at all** — it ships inside scikit-learn:
  ```python
  from sklearn.datasets import load_diabetes
  data = load_diabetes(as_frame=True)
  df = data.frame  # 442 patients, 10 features, target = progression score
  ```
- **Documentation:** https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset

---

## Tier 2 — Free, but requires registration + short training (start in Week 1)

### 5. MIMIC-IV (via PhysioNet)
**Best for:** genuine longitudinal/multi-visit EHR structure — the dataset that enables real temporal modeling (LSTM/CNN-LSTM) rather than cross-sectional-only modeling.

- **Dataset page:** https://physionet.org/content/mimiciv/3.1/
- **Access steps (typically approved within 24–48 hours, per PhysioNet's own stated turnaround):**
  1. Create a free account at https://physionet.org
  2. Complete your profile under **Credentialing**: https://physionet.org/settings/credentialing/
  3. Complete the required training course — PhysioNet requires the CITI Program's **"Data or Specimens Only Research"** course: https://physionet.org/about/citi-course/ (create a free CITI account first, complete the short online modules, then submit the completion certificate through your PhysioNet profile's Training page).
  4. Once your credentialing and training are approved, go to the MIMIC-IV project page and sign the **PhysioNet Credentialed Health Data Use Agreement 1.5.0** under the project's **Files** section.
  5. Download via the site directly, or via the PhysioNet command-line tool (`wget -r -N -c -np --user <your-username> --ask-password https://physionet.org/files/mimiciv/3.1/`).
  6. **Filter to diabetic patients**: MIMIC-IV is an ICU/hospital-wide dataset, not diabetes-specific — after download, filter on diabetes-related ICD-10 codes (E10–E14) in the `diagnoses_icd` table, then pull each filtered patient's longitudinal labs (`labevents`, especially glucose and HbA1c loinc/itemid codes) and vitals across their admissions.
  7. **Important restriction to note in your methodology:** individual MIMIC records/notes cannot be published verbatim in your paper without consent — report only aggregated statistics and model results, never row-level patient data, which is standard practice anyway.

---

## Tier 3 — Free but slow, or requires institutional backing (don't count on this for a 6-week project unless already underway)

### 6. UK Biobank
- **Access portal:** https://www.ukbiobank.ac.uk/enable-your-research/apply-for-access
- **Reality check:** application review and approval routinely takes weeks to months, and typically requires your institution to already hold an approved access framework agreement. Check with your department first; if they already have access, ask your supervisor about getting added to an existing approved project rather than applying fresh. Otherwise, treat this as out of reach for a 6-week timeline and don't plan around it.

---

## Recommended download order for your timeline

| Week | Action |
|---|---|
| Day 1 | Download NHANES (latest cycle) and BRFSS (latest year) — your guaranteed datasets, zero wait. |
| Day 1 | Create PhysioNet account, start CITI training immediately (takes a few hours to complete the modules) — this is the only item on this list with a waiting period, so start it in parallel with everything else, not after. |
| Day 2–3 | Pull the sklearn `diabetes` dataset and Pima (Kaggle) just to build and test your preprocessing/modeling pipeline end-to-end on something small and clean before pointing it at the real, messier NHANES/BRFSS data. |
| Day 3–5 | Once MIMIC-IV access is approved (check daily), download and filter to your diabetic cohort as a secondary longitudinal dataset. |

---

*Companion to: T2D_Progression_Project_Report.md*
