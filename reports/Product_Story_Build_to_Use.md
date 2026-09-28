# The Complete Story — From Build to Bedside

### How this system gets made, and who actually uses it

---

## Part 1: The Origin — why this exists

Every year, more people are diagnosed with Type 2 diabetes, and the tools doctors currently use to predict how a patient's disease will evolve are built on decades-old statistical methods that can't see the complicated, tangled-up ways a person's weight, blood sugar, diet, income, and genetics interact with each other. A doctor sees a patient today and has to guess: will this person's kidneys start failing in three years, or ten? Will they need insulin next year, or never? Traditional tools give a rough, population-average answer. This project's job is to give a sharper, personalized, explainable one.

That's the "why." Everything below is the "how" — how it gets built, and by whom, and then how it gets used, and by whom.

---

## Part 2: The Build Story — six weeks, told as a journey

This is the story of how the raw ingredients turn into a working system, in the order it actually happens.

**Weeks 1: Gathering the raw material.** This is where you are right now. Two datasets are pulled in — NHANES, a large population health survey (your breadth dataset), and a genuinely longitudinal hospital-encounter dataset (your depth dataset, giving you real repeated visits per patient). Every file gets inspected, cleaned, and merged into one usable table per dataset. This stage is unglamorous — column names don't match documentation, files go missing, tools fight each other — but it's the stage every later stage depends on, so it's worth the friction you've already been through.

**Weeks 2–3: Teaching the models to see patterns.** The cleaned data is split into training and testing sets, always by patient, never by row, so the model is never accidentally tested on a person it already "met" during training. Three kinds of models get trained side by side: a simple traditional model (logistic regression) that plays the role of "what doctors already have," a strong tree-based model (XGBoost) that finds nonlinear interactions the simple model can't, and — on the longitudinal dataset only — a sequence model that can look at a patient's history of visits and project forward. Each one gets scored honestly, on data it's never seen.

**Week 4: Making the models explain themselves.** A model that just outputs "72% risk" with no reason is not something a doctor will trust or a reviewer will accept. This stage attaches SHAP — a method that, for each individual prediction, shows exactly which factors pushed the risk up or down and by how much. The team also runs a small stress-test: does the explanation stay stable if the input is nudged slightly? A model whose "reasons" flip around under tiny noise isn't trustworthy, so this check matters.

**Week 5: Building the thing people actually touch.** Everything so far has lived in notebooks and scripts — useful for you, invisible to anyone else. This week turns it into an app: two connected modes, one for the cross-sectional model, one for the longitudinal model, each showing a prediction and its explanation.

**Week 6: Writing it down and presenting it.** The paper is written using exactly the results generated in weeks 2–5 — no result appears in the paper that isn't reproducible by opening the app or rerunning the notebook. This is what makes the project "trustworthy" rather than "a nice demo": every claim has a receipt.

---

## Part 3: Who Builds It

Just you (and any teammates), across the six weeks above, playing several different roles in sequence:

- **Data engineer** (Weeks 1) — sourcing, cleaning, and merging.
- **ML researcher** (Weeks 2–4) — training, comparing, and explaining models.
- **Product builder** (Week 5) — turning results into something usable.
- **Author** (Week 6) — turning results into something readable and defensible.

This matters to name explicitly because it's the honest answer to "who creates it" — not a company, not a team of specialists, but one person moving through four different jobs on a tight deadline. That's worth being upfront about in your own presentation, because it's exactly what a project evaluator expects from a student capstone, and pretending otherwise would look worse, not better.

---

## Part 4: Who Uses It — the people on the other side

This is the part your build story has been aiming at the whole time. Four distinct groups interact with the finished product, each in a different way.

### Persona 1: The endocrinologist — the primary intended user, and the only direct user of the app

**Scope decision, locked in, with the full reasoning behind it:**

The problem statement names both patients and endocrinologists as affected by inaccurate progression prediction — but a closer read shows it narrows the actual ask: *"the absence of dynamic, data-driven forecasting limits proactive **clinical** decision-making."* Decision-making, in a clinical context, is something a clinician does. Patients appear in the statement as the people harmed by the problem, not as the intended operators of the solution — and conflating "who is affected" with "who should use the tool" is a common, avoidable mistake in health-tech design.

Four reasons this project draws the line at clinician-only, all worth keeping on record:

1. **Precedent.** Both of this project's own base papers (Zamani et al., 2026; Huang et al., 2025) build clinician-facing risk-stratification tools, not patient apps — as does essentially every real clinical risk calculator in active use (Framingham, QDiabetes, and so on). This isn't a historical accident; a raw risk number needs to be integrated with exam findings, history, and clinical judgment to be used safely, which is precisely the clinician's job, not the patient's.
2. **Safety.** A patient seeing "64% nephropathy risk" with no clinician present to contextualize it is a real risk of causing unwarranted distress or unsupervised self-directed decisions — not just a UX inconvenience. The explanation layer (SHAP plus the LLM narrative) is written in clinical register for exactly this reason, and that only holds together if the actual user is a clinician.
3. **Scope honesty.** A genuinely patient-safe interface isn't a simpler version of the clinician app — it's a second product, with its own health-literacy language design, emotional-tone considerations, and safety guardrails. Trying to serve both half-well in six weeks would weaken both.
4. **User vs. beneficiary is a standard, defensible distinction.** The patient is the intended *beneficiary* of better clinical decisions; the clinician is the intended *user* of the tool that informs those decisions. This is exactly how the finished product's five pages are all clinician-only in language and function — the app assumes clinical literacy throughout, and there is no patient-facing view, login, or simplified summary mode in scope.

**The paragraph this becomes in the paper** (Introduction or Methodology, addressing the problem statement's patient mention directly rather than leaving it looking overlooked):

> *"While the problem statement identifies both patients and clinicians as affected by inaccurate progression prediction, this system is designed as a clinician-facing decision-support tool, consistent with standard clinical risk-stratification practice. Patients are the intended beneficiaries of improved care decisions, accessed through their treating clinician, rather than direct users of the system."*

A doctor managing a full clinic of diabetic patients, each with limited consultation time. They already know the medicine; what they don't have is a fast way to see *this specific patient's* nonlinear risk picture and a forecast of where things are heading. They open the demo app, and this is where the two modes from the diagram above come in:

- If they want a **snapshot of current risk** — "how likely is this patient to already be dealing with early kidney or cardiovascular involvement, given what I know about them today?" — they use **Mode 1**, typing in the patient's current labs and vitals, and get back a multi-condition risk breakdown with the top contributing factors shown plainly (e.g. "elevated urine albumin and long-standing hypertension are the two biggest drivers of this patient's nephropathy risk").
- If they want to understand **where the patient is heading** — "this person's HbA1c has been creeping up over their last three visits, should I intervene now or wait?" — they use **Mode 2**, selecting a matching real patient case from the trajectory model's test set, and see the historical trend plotted alongside the model's forecast for the next measurement, with an explanation of what's driving that trajectory.

They never touch a model, a script, or a line of code. They see a number, a chart, and a plain-language reason — and they still make the actual medical decision themselves. The tool supports; it does not replace.

### Persona 2: The patient — a beneficiary, not a user

Worth being precise about the distinction: the patient is not a secondary *user* of this tool in any sense — they never open the app, never see a risk score directly, never interact with a mode, a login, or a screen. They are the intended **beneficiary** of the better clinical decision the tool helps produce. The only place they appear in the actual flow is secondhand: the *output* of Mode 1, shown to them by their doctor during a consultation, is arguably the moment the whole project matters most in human terms — a doctor turning the screen around and saying "here's what's driving your risk, and here's what we can work on together." The patient's story ends with a clearer conversation, not with them operating software.

### Persona 3: Your project evaluator / department committee — the academic user

This person interacts with the *paper* and the *demo* together, not the raw code. They will open the demo, likely try both modes with a sample input, and cross-check that the numbers and explanations shown match what's claimed in your results section. Their core question is simply: "does this actually do what the problem statement asked for, and can I verify that myself in under five minutes?" This is exactly why the two-mode demo design matters — it gives them something to click, not just something to read.

### Persona 4: A future researcher — the paper's real audience

Once published (arXiv or a student/conference track), someone outside your department — another student, a researcher in health informatics — might read the paper to see how you approached multi-label complication prediction and longitudinal forecasting together. Their interaction is entirely with the writing: the related-work section, the ablation tables, the honestly-stated limitations. They never touch the demo at all. Writing for this reader is what pushes you to be precise about method and honest about what didn't work, not just what did.

---

## Part 5: A Day in the Life — walking through an actual use

Concretely, here's what happens end to end, step by step, the way it would really unfold:

1. A clinician has a diabetic patient in front of them and opens the demo app on a laptop or tablet.
2. They choose **Mode 1** and enter the patient's most recent lab values, BMI, blood pressure, and a couple of lifestyle answers.
3. Behind the scenes, the app sends these values to the trained XGBoost model, which returns a risk score for each of the three or four tracked complications (hypertension, nephropathy, cardiovascular).
4. Alongside the numeric risk, the SHAP explainer runs automatically and returns the top 3–5 factors pushing that risk up or down for this specific patient — this is shown as a simple horizontal bar chart, not a dump of numbers.
5. The clinician reads this in seconds, forms a clinical impression, and — if the patient has a known multi-visit history — switches to **Mode 2**, picks the closest-matching case from the pre-loaded trajectory set, and sees a chart of past HbA1c readings with the model's forecasted next value plotted just past the last real data point.
6. The clinician now has two pieces of model-generated evidence, both explained in plain terms, to weigh alongside their own judgment — and they decide the actual next step (lifestyle counseling, medication adjustment, more frequent monitoring) themselves.

Nothing in this flow makes a decision *for* anyone. It compresses fifteen minutes of manual chart-review and mental risk-calculation into thirty seconds of interpretable model output — that's the entire value proposition, stated honestly.

---

## Part 6: What This Project Honestly Is — and Isn't

Worth stating plainly, because it protects your credibility more than it costs you:

- **It is:** a research prototype and decision-support demonstration, built on real (if secondary/public) clinical data, with a working interactive interface and a defensible, explainable methodology.
- **It is not:** a clinically validated, regulator-approved, or deployed medical device. No claim in your paper or demo should imply otherwise. This is the same honest limitation every paper in your literature survey (Section 5 of the earlier project report) explicitly states about its own work — you're in good company saying it about yours too.

---

## Part 7: The Full Lifecycle, End to End

| Stage | What happens | Who's involved |
|---|---|---|
| 1. Problem framing | Department gives the problem statement | Department, you |
| 2. Data acquisition | NHANES + longitudinal dataset sourced and merged | You (data engineer role) |
| 3. Modeling | Baseline, primary, and sequence models trained and compared | You (ML researcher role) |
| 4. Explainability | SHAP + stability testing layered on top | You (ML researcher role) |
| 5. Product build | Two-mode demo app built | You (product builder role) |
| 6. Writing | Paper drafted from real, reproducible results | You (author role) |
| 7. Evaluation | Demo + paper reviewed | Department committee |
| 8. Publication | Preprint/conference submission | Research community |
| 9. Real use (in the story, not literally deployed) | A clinician uses the demo during a consultation | Endocrinologist, patient |

Every earlier document in this project (the architecture diagram, the dataset guide, the merge scripts) is a piece of Stages 2–5 above. This document is the thread that connects all of them into one coherent story — the one you'll actually tell in your presentation.

---

*Companion to: T2D_Progression_Project_Report.md, Dataset_Access_Guide.md, merge_nhanes_core.py*
