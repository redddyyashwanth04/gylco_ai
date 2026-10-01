const navButtons = [...document.querySelectorAll(".nav-item[data-view]")];
const views = [...document.querySelectorAll(".view")];
const pageNames = {
  overview: "Overview",
  assessment: "Current assessment",
  trajectory: "Progression forecast",
  history: "Patient history",
  models: "Model transparency",
};
const TARGETS = ["hypertension", "nephropathy", "cardiovascular"];
const API_BASE = `/api`;
const DEMO_SESSION_KEY = "carepath_demo_user";

const visits = [];        // saved visits from the local database
const tempVisits = [];    // visits without a patient ID, browser memory only
let mimicCases = [];
let pendingVisit = null;
let forecastState = null;
let explainResult = null;
let explainTarget = "hypertension";
let toastTimer;
let backendConnected = false;
let forecastAvailable = false;

const byId = (id) => document.getElementById(id);
const valueOf = (id) => {
  const value = byId(id).value.trim();
  return value === "" ? null : Number(value);
};
const allVisits = () => [...tempVisits, ...visits];

function escapeHTML(text) {
  return String(text).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[c]);
}

function showToast(message) {
  const toast = byId("toast");
  toast.textContent = message;
  toast.classList.add("show");
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => toast.classList.remove("show"), 2800);
}

async function apiRequest(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  let result = {};
  try { result = await response.json(); } catch { /* non-JSON error body */ }
  if (!response.ok) throw new Error(result.error || `Service request failed (${response.status}).`);
  return result;
}

function formatDate(date) {
  if (Number.isNaN(date.getTime())) return "—";
  return new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric", year: "numeric" }).format(date);
}

function metricText(model) {
  // Track B: both LSTM and Ridge notes contain "mean abs error N"
  const maeMatch = /mean abs error ([\d.]+)/.exec(model.notes || "");
  if (maeMatch) return `MAE ${maeMatch[1]} mg/dL`;
  // Track A: AUC stored as a numeric column
  return model.val_auc ? `AUC ${Number(model.val_auc).toFixed(3)}` : "Not recorded";
}

/* ---------- connection ---------- */

function setConnectionState(connected, message, modelName = "") {
  backendConnected = connected;
  const banner = byId("connection-banner");
  banner.classList.toggle("connected", connected);
  banner.classList.toggle("disconnected", !connected);
  banner.hidden = connected;
  byId("connection-title").textContent = connected ? "Model service connected." : "Model service not connected.";
  byId("connection-message").textContent = message;
  byId("backend-pill-label").textContent = connected ? "API CONNECTED" : "API OFFLINE";
  byId("backend-pill").classList.toggle("connected", connected);
  byId("sidebar-status").textContent = connected ? "Local model connected" : "Model service offline";
  byId("sidebar-detail").textContent = connected ? modelName : "Start the local Python API";
  byId("help-button").title = connected
    ? `Connected to the local Python service using ${modelName}.`
    : "The local Python model service is not responding.";
  ["sidebar-status-dot", "backend-status-dot"].forEach((id) => byId(id).classList.toggle("connected", connected));
}

function normalizeVisit(row) {
  let risks = null;
  try { risks = row.risks_json ? JSON.parse(row.risks_json) : null; } catch { risks = null; }
  const num = (v) => (v === null || v === undefined ? null : Number(v));
  return {
    patientId: row.patient_id || "",
    date: new Date(row.visit_date),
    hba1c: num(row.hba1c),
    glucose: num(row.glucose),
    bmi: num(row.bmi),
    systolic: num(row.systolic_bp),
    diastolic: num(row.diastolic_bp),
    risks,
  };
}

async function loadVisitsFromBackend() {
  const result = await apiRequest("/visits");
  visits.splice(0, visits.length, ...result.visits.map(normalizeVisit));
  updateTrajectoryOptions();
  updateVisitCounts();
  renderHistory();
  drawTrajectory();
}

function updateTrajectoryOptions() {
  const select = byId("trajectory-patient");
  const selected = select.value;
  const patients = [...new Set(visits.map((v) => v.patientId).filter(Boolean))];
  select.replaceChildren(new Option("Illustrative example (synthetic)", "illustrative"));
  if (mimicCases.length) {
    const group = document.createElement("optgroup");
    group.label = "MIMIC-IV demo cases (ICU)";
    mimicCases.forEach((c) => group.append(new Option(`${c.label} · ${c.totalReadings} readings`, `case:${c.id}`)));
    select.append(group);
  }
  if (patients.length) {
    const group = document.createElement("optgroup");
    group.label = "Tracked patients";
    patients.forEach((id) => group.append(new Option(id, `patient:${id}`)));
    select.append(group);
  }
  if ([...select.options].some((o) => o.value === selected)) select.value = selected;
}

async function loadModelsFromBackend(health) {
  const { models } = await apiRequest("/models");
  const current = models.find((m) => m.model_name === "nhanes_active" && m.is_active);
  const trajectory = models.find((m) => m.model_name === "mimic_glucose_forecaster_lstm" && m.is_active)
    || models.find((m) => m.model_name === "mimic_glucose_forecaster" && m.is_active);

  byId("registry-status").textContent = models.length
    ? `${models.length} registered model version${models.length === 1 ? "" : "s"}`
    : "Registry connected; no versions registered";
  byId("registry-detail").textContent = `Active risk artifact: ${health.modelName}`;
  byId("registry-badge").textContent = "Local registry connected";
  byId("track-a-status").textContent = current ? "Active" : "Artifact loaded";
  byId("track-a-status").classList.toggle("unavailable", !current);
  byId("track-a-version").textContent = current ? `Version ${current.version_id}` : health.modelName;
  byId("track-a-validation").textContent = current ? metricText(current) : "Not recorded";
  byId("track-b-status").textContent = trajectory ? "Active" : (health.forecastAvailable ? "Artifact available" : "Artifact unavailable");
  byId("track-b-status").classList.toggle("unavailable", !health.forecastAvailable);
  byId("track-b-version").textContent = trajectory ? `Version ${trajectory.version_id}` : (health.forecastModel || "No active registry version");
  byId("track-b-validation").textContent = trajectory ? metricText(trajectory) : "Not recorded";

  byId("registry-rows").innerHTML = models.map((m) => `<tr>
    <td>${escapeHTML(m.model_name)}</td>
    <td>v${m.version_id}</td>
    <td>${formatDate(new Date(m.trained_at))}</td>
    <td>${m.n_original_rows} / ${m.n_app_rows}</td>
    <td>${escapeHTML(metricText(m))}</td>
    <td>${m.is_active ? "Active" : "Archived"}</td>
  </tr>`).join("") || '<tr><td colspan="6">No versions registered.</td></tr>';
  return models;
}

function fillOverview(health, models, info) {
  const a = models.find((m) => m.model_name === "nhanes_active" && m.is_active);
  const b = models.find((m) => m.model_name === "mimic_glucose_forecaster_lstm" && m.is_active)
    || models.find((m) => m.model_name === "mimic_glucose_forecaster" && m.is_active);
  byId("ov-a-model").textContent = health.modelName.replace(/^nhanes_/, "").replace(/_/g, " ");
  byId("ov-a-detail").textContent = a ? `Registry v${a.version_id} · ${metricText(a)}` : "Not in registry";
  byId("ov-b-model").textContent = health.forecastAvailable ? (health.forecastModel || "Ridge forecaster") : "Unavailable";
  byId("ov-b-detail").textContent = b ? `Registry v${b.version_id} · ${metricText(b)}` : "No active registry version";
  byId("ov-patients").textContent = info ? String(info.patients.length) : "—";
  byId("ov-predictions").textContent = info ? String(info.predictionCount) : "—";
}

function markOverviewOffline() {
  ["ov-a-model", "ov-b-model", "ov-patients", "ov-predictions"].forEach((id) => { byId(id).textContent = "—"; });
  byId("ov-a-detail").textContent = "Service offline";
  byId("ov-b-detail").textContent = "Service offline";
}

async function connectBackend() {
  try {
    const health = await apiRequest("/health");
    forecastAvailable = Boolean(health.forecastAvailable);
    setConnectionState(true, `Using ${health.modelName} (${health.featureCount} features). Predictions run locally; this remains a research prototype.`, health.modelName);
    byId("forecast-badge").textContent = forecastAvailable ? "Forecast model available" : "Forecast artifact unavailable";

    try { mimicCases = (await apiRequest("/cases")).cases; } catch { mimicCases = []; }
    let info = null;
    try { info = await apiRequest("/patients"); } catch { info = null; }

    const [, models] = await Promise.all([loadVisitsFromBackend(), loadModelsFromBackend(health)]);
    fillOverview(health, models, info);
    byId("history-description").textContent = "Saved visits from the local patient database. Select a patient to open their trajectory.";
    updateForecastControls();
  } catch (error) {
    forecastAvailable = false;
    setConnectionState(false, "Start the local API with: python -m src.api_server. The UI retries when reloaded.");
    byId("registry-status").textContent = "Registry unavailable";
    byId("registry-detail").textContent = "Start the local API to read model versions.";
    byId("track-a-status").textContent = "Service offline";
    byId("track-b-status").textContent = "Service offline";
    byId("registry-badge").textContent = "Registry unavailable";
    byId("forecast-badge").textContent = "Forecast unavailable";
    byId("forecast-status").textContent = "Service offline";
    markOverviewOffline();
  }
}

/* ---------- navigation ---------- */

function navigate(viewName) {
  if (!pageNames[viewName]) return;
  navButtons.forEach((button) => {
    const selected = button.dataset.view === viewName;
    button.classList.toggle("active", selected);
    button.setAttribute("aria-current", selected ? "page" : "false");
  });
  views.forEach((view) => view.classList.toggle("active", view.id === `view-${viewName}`));
  byId("current-section").textContent = pageNames[viewName];
  if (viewName === "trajectory") drawTrajectory();
  if (viewName === "history") renderHistory();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

navButtons.forEach((button) => button.addEventListener("click", () => navigate(button.dataset.view)));
document.querySelector(".brand").addEventListener("click", (event) => {
  event.preventDefault();
  navigate("overview");
});
document.querySelectorAll("[data-goto]").forEach((button) => {
  button.addEventListener("click", () => navigate(button.dataset.goto));
});

byId("today-label").textContent = new Intl.DateTimeFormat(undefined, {
  weekday: "short", month: "short", day: "numeric", year: "numeric",
}).format(new Date());

byId("help-button").addEventListener("click", () => {
  showToast(backendConnected ? "Connected to the local model API. Results are research estimates." : "Start the local API server to enable model predictions.");
});
byId("logout-button").addEventListener("click", () => {
  sessionStorage.removeItem(DEMO_SESSION_KEY);
  window.location.replace("/login.html");
});
byId("connection-dismiss").addEventListener("click", () => {
  byId("connection-banner").hidden = true;
});

/* ---------- assessment ---------- */

byId("example-button").addEventListener("click", () => {
  const example = {
    "patient-id": "DEMO-1042", age: "58", hba1c: "7.4", glucose: "150", bmi: "31.0",
    systolic: "142", diastolic: "88", ldl: "130", hdl: "42",
    sedentary: "9", income: "1.8",
  };
  Object.entries(example).forEach(([id, value]) => {
    const el = byId(id);
    if (el) el.value = value;
  });
  // dropdowns
  byId("sex").value = "1";        // Male
  byId("smoking").value = "2";    // Former smoker
  pendingVisit = null;
  clearModelResult();
  byId("model-empty-title").textContent = "Ready for assessment";
  byId("model-empty-copy").textContent = "Enter the current visit values and select Review visit to request estimates from the Python model.";
  byId("save-visit-button").disabled = true;
  byId("review-values").hidden = true;
  byId("model-empty-state").hidden = false;
  byId("form-feedback").textContent = "Example values loaded. Review to request a real estimate from the local model.";
  byId("visit-note").value = "";
  showToast("Example values loaded. Review to run the local model.");
});

function readVisit() {
  const sexEl = byId("sex");
  const smokingEl = byId("smoking");
  return {
    patientId: byId("patient-id").value.trim(),
    age: valueOf("age"),
    hba1c: valueOf("hba1c"),
    glucose: valueOf("glucose"),
    bmi: valueOf("bmi"),
    systolic: valueOf("systolic"),
    diastolic: valueOf("diastolic"),
    ldl: valueOf("ldl"),
    hdl: valueOf("hdl"),
    sex: sexEl && sexEl.value ? Number(sexEl.value) : null,
    smoking: smokingEl && smokingEl.value ? Number(smokingEl.value) : null,
    sedentary: valueOf("sedentary"),
    income: valueOf("income"),
    note: byId("visit-note").value.trim(),
    date: new Date(),
  };
}

function setSnapshot(visit) {
  byId("review-patient-name").textContent = visit.patientId || "Unidentified patient";
  byId("review-patient-meta").textContent = `${visit.age} years · Current visit`;
  const snapshots = [
    ["HbA1c", visit.hba1c === null ? "Not entered" : `${visit.hba1c.toFixed(1)}%`],
    ["Glucose", visit.glucose === null ? "Not entered" : `${visit.glucose} mg/dL`],
    ["BMI", visit.bmi === null ? "Not entered" : `${visit.bmi.toFixed(1)}`],
    ["Blood pressure", visit.systolic === null || visit.diastolic === null ? "Not entered" : `${visit.systolic}/${visit.diastolic}`],
  ];
  byId("snapshot-list").innerHTML = snapshots.map(([label, value]) =>
    `<div class="snapshot-item"><span>${label}</span><strong>${value}</strong></div>`).join("");
  byId("review-values").hidden = false;
  byId("model-empty-state").hidden = true;
}

function showModelResult(result) {
  byId("model-empty-state").hidden = true;
  byId("model-badge").innerHTML = `<span></span>${escapeHTML(result.modelName)}`;
  byId("model-slots").setAttribute("aria-label", `Risk estimates from ${result.modelName}`);
  TARGETS.forEach((target) => {
    const risk = Number(result.risks[target]);
    const trend = result.trends?.[target];
    const trendText = trend ? ` · ${trend.direction} ${Math.abs(trend.delta * 100).toFixed(1)}%` : "";
    byId(`risk-${target}`).textContent = `${(risk * 100).toFixed(0)}%${trendText}`;
    byId(`risk-${target}`).classList.add("loaded");
  });
  byId("recommendation-empty").hidden = true;
  byId("recommendation-list").innerHTML = result.recommendations.map((r) => `<li>${escapeHTML(r)}</li>`).join("");
  byId("recommendation-list").hidden = false;
}

function clearModelResult() {
  TARGETS.forEach((target) => {
    const value = byId(`risk-${target}`);
    value.textContent = backendConnected ? "Not run" : "Service offline";
    value.classList.remove("loaded");
  });
  byId("model-badge").innerHTML = "<span></span>Awaiting model";
  byId("model-slots").setAttribute("aria-label", "Model estimates not run");
  byId("recommendation-list").innerHTML = "";
  byId("recommendation-list").hidden = true;
  byId("recommendation-empty").hidden = false;
  byId("explain-panel").hidden = true;
  explainResult = null;
}

/* ---------- explanation (SHAP drivers) ---------- */

async function loadExplanation(values) {
  byId("explain-panel").hidden = false;
  byId("explain-tabs").innerHTML = "";
  byId("explain-bars").innerHTML = "";
  byId("explain-source").textContent = "";
  byId("explain-narrative").textContent = "Computing drivers…";
  try {
    explainResult = await apiRequest("/explain", { method: "POST", body: JSON.stringify({ values }) });
    renderExplanation();
  } catch (error) {
    explainResult = null;
    byId("explain-narrative").textContent = `Explanation unavailable: ${error.message}`;
  }
}

function renderExplanation() {
  if (!explainResult) return;
  byId("explain-tabs").innerHTML = TARGETS.map((t) =>
    `<button type="button" class="chip${t === explainTarget ? " active" : ""}" data-target="${t}">${t}</button>`).join("");
  const factors = explainResult.factors[explainTarget] || [];
  const max = Math.max(...factors.map((f) => Math.abs(f.shap)), 1e-9);
  byId("explain-bars").innerHTML = factors.map((f) => `<div class="bar-row">
    <span>${escapeHTML(f.label)}${f.entered ? "" : " <em>(typical value assumed)</em>"}</span>
    <span class="bar-track"><i class="${f.shap > 0 ? "up" : "down"}" style="width:${(Math.abs(f.shap) / max) * 100}%"></i></span>
    <span class="bar-val">${f.shap > 0 ? "+" : ""}${f.shap.toFixed(2)}</span>
  </div>`).join("");
  byId("explain-narrative").textContent = explainResult.narrative[explainTarget] || "";
  const source = explainResult.narrativeSource[explainTarget];
  byId("explain-source").textContent = source === "llm"
    ? "Narrative written by an LLM from the SHAP values above only."
    : source === "template" ? "Template summary of the SHAP values above." : "";
}

byId("explain-tabs").addEventListener("click", (event) => {
  const button = event.target.closest("[data-target]");
  if (button && explainResult) {
    explainTarget = button.dataset.target;
    renderExplanation();
  }
});

byId("assessment-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = event.currentTarget;
  const feedback = byId("form-feedback");
  if (!form.reportValidity()) {
    feedback.textContent = "Enter valid age, HbA1c, and fasting glucose values before reviewing.";
    return;
  }
  if (!backendConnected) {
    feedback.textContent = "The local model service is offline. Start it and reload before requesting an estimate.";
    return;
  }

  const visit = readVisit();
  const values = {
    RIDAGEYR: visit.age, LBXGH: visit.hba1c, LBXGLU: visit.glucose, BMXBMI: visit.bmi,
    BPXOSY1: visit.systolic, BPXODI1: visit.diastolic, LBDLDL: visit.ldl, LBDHDD: visit.hdl,
    RIAGENDR: visit.sex,           // 1=Male 2=Female (NHANES code)
    SMQ020: visit.smoking,         // 1=current 2=former 3=never (NHANES SMQ020)
    PAD680: visit.sedentary,       // sedentary mins/day → hours sent as-is; backend scales if needed
    INDFMPIR: visit.income,        // income-to-poverty ratio (NHANES INDFMPIR)
  };
  const reviewButton = form.querySelector("button[type='submit']");
  clearModelResult();
  byId("model-empty-title").textContent = "Running assessment";
  byId("model-empty-copy").textContent = "Waiting for the local model response.";
  byId("model-empty-state").hidden = false;
  byId("review-values").hidden = true;
  reviewButton.disabled = true;
  reviewButton.textContent = "Running model…";
  feedback.textContent = "Requesting an estimate from the local model.";
  byId("save-visit-button").disabled = true;
  try {
    const result = await apiRequest("/predict", {
      method: "POST",
      body: JSON.stringify({ patientId: visit.patientId, values }),
    });
    pendingVisit = { ...visit, values, result };
    setSnapshot(visit);
    showModelResult(result);
    byId("save-visit-button").disabled = false;
    byId("save-visit-button").textContent = visit.patientId ? "Save visit to patient record" : "Add to temporary history";
    byId("save-visit-note").textContent = visit.patientId
      ? "Measurements, risk snapshot, and visit note will be saved in the local SQLite database."
      : "Enter a patient ID to save to SQLite; without one, this visit stays in browser memory only.";
    feedback.textContent = `Estimate returned by ${result.modelName}. This is research decision support, not a diagnosis.`;
    loadExplanation(values); // non-blocking: risks show immediately, drivers follow
  } catch (error) {
    pendingVisit = null;
    clearModelResult();
    byId("model-empty-title").textContent = "Assessment unavailable";
    byId("model-empty-copy").textContent = error.message;
    byId("model-empty-state").hidden = false;
    byId("review-values").hidden = true;
    byId("save-visit-button").disabled = true;
    feedback.textContent = error.message;
  } finally {
    reviewButton.disabled = false;
    reviewButton.innerHTML = 'Review visit <span aria-hidden="true">→</span>';
  }
});

byId("save-visit-button").addEventListener("click", async () => {
  if (!pendingVisit) return;
  const saved = { ...pendingVisit };
  if (saved.patientId) {
    try {
      await apiRequest("/visits", {
        method: "POST",
        body: JSON.stringify({ patientId: saved.patientId, values: saved.values, note: saved.note }),
      });
      await loadVisitsFromBackend();
      byId("form-feedback").textContent = `Visit saved to the local patient record for ${saved.patientId}.`;
      showToast("Visit and risk snapshot saved to the local SQLite database.");
    } catch (error) {
      byId("form-feedback").textContent = `Visit was not saved: ${error.message}`;
      return;
    }
  } else {
    tempVisits.unshift({
      patientId: "", date: saved.date, hba1c: saved.hba1c, glucose: saved.glucose, bmi: saved.bmi,
      systolic: saved.systolic, diastolic: saved.diastolic, risks: saved.result.risks,
    });
    updateVisitCounts();
    renderHistory();
    byId("form-feedback").textContent = "Visit added to temporary browser history only.";
    showToast("No patient ID provided; visit remains in browser memory only.");
  }
  pendingVisit = null;
  byId("save-visit-button").disabled = true;
});

function updateVisitCounts() {
  const total = allVisits().length;
  byId("visit-count").textContent = String(total);
  byId("history-total").textContent = String(total);
}

/* ---------- history ---------- */

function renderHistory() {
  const query = byId("history-search").value.trim().toLowerCase();
  const list = allVisits();
  const filtered = list.filter((v) => (v.patientId || "unidentified patient").toLowerCase().includes(query));
  const rows = byId("history-rows");
  byId("history-empty").hidden = list.length > 0;
  if (list.length > 0 && filtered.length === 0) {
    rows.innerHTML = '<tr><td colspan="7">No saved visits match this search.</td></tr>';
    return;
  }
  const counts = {};
  list.forEach((v) => { counts[v.patientId] = (counts[v.patientId] || 0) + 1; });
  rows.innerHTML = filtered.map((v) => {
    const pressure = v.systolic === null || v.diastolic === null ? "—" : `${v.systolic}/${v.diastolic}`;
    const patientCell = v.patientId
      ? `<button type="button" class="text-button" data-open-patient="${escapeHTML(v.patientId)}">${escapeHTML(v.patientId)}</button> <small>${counts[v.patientId]} visit${counts[v.patientId] === 1 ? "" : "s"}</small>`
      : "Unidentified";
    const riskCell = v.risks
      ? `H ${Math.round(v.risks.hypertension * 100)}% · N ${Math.round(v.risks.nephropathy * 100)}% · C ${Math.round(v.risks.cardiovascular * 100)}%`
      : "—";
    return `<tr>
      <td>${patientCell}</td>
      <td>${formatDate(v.date)}</td>
      <td>${v.hba1c === null ? "—" : `${v.hba1c.toFixed(1)}%`}</td>
      <td>${v.glucose === null ? "—" : `${v.glucose} mg/dL`}</td>
      <td>${v.bmi === null ? "—" : v.bmi.toFixed(1)}</td>
      <td>${pressure}</td>
      <td>${riskCell}</td>
    </tr>`;
  }).join("");
}

byId("history-search").addEventListener("input", renderHistory);
byId("history-rows").addEventListener("click", (event) => {
  const button = event.target.closest("[data-open-patient]");
  if (!button) return;
  navigate("trajectory");
  byId("trajectory-patient").value = `patient:${button.dataset.openPatient}`;
  resetForecast();
  drawTrajectory();
});

/* ---------- progression forecast ---------- */

const illustrativeReadings = [
  { label: "#1", glucose: 126 }, { label: "#2", glucose: 139 }, { label: "#3", glucose: 132 },
  { label: "#4", glucose: 158 }, { label: "#5", glucose: 149 }, { label: "#6", glucose: 171 },
];

function selectedPatientReadings() {
  const selected = byId("trajectory-patient").value;
  if (!selected.startsWith("patient:")) return [];
  const patientId = selected.slice("patient:".length);
  return visits
    .filter((v) => v.patientId === patientId && v.glucose !== null)
    .slice()
    .sort((a, b) => a.date - b.date)
    .map((v, i) => ({ label: `#${i + 1}`, glucose: v.glucose }));
}

function selectedReadings() {
  const selected = byId("trajectory-patient").value;
  if (selected.startsWith("case:")) {
    const c = mimicCases.find((x) => `case:${x.id}` === selected);
    return c ? c.readings.map((g, i) => ({ label: `#${i + 1}`, glucose: g })) : [];
  }
  return selectedPatientReadings();
}

function resetForecast() {
  forecastState = null;
  byId("forecast-result").hidden = true;
}

function updateForecastControls() {
  const selected = byId("trajectory-patient").value;
  const real = selected !== "illustrative";
  const enough = selectedReadings().length >= 4;
  byId("forecast-button").disabled = !(backendConnected && forecastAvailable && real && enough);
  const status = byId("forecast-status");
  if (!backendConnected) status.textContent = "Service offline";
  else if (!forecastAvailable) status.textContent = "Artifact unavailable";
  else if (!real) status.textContent = "Select a case or patient";
  else if (!enough) status.textContent = "Need 4 readings";
  else status.textContent = byId("forecast-result").hidden ? "Ready to run" : "Forecast ready";
}

byId("forecast-button").addEventListener("click", async () => {
  if (byId("forecast-button").disabled) return;
  const readings = selectedReadings().map((r) => r.glucose);
  const resultPanel = byId("forecast-result");
  byId("forecast-button").disabled = true;
  byId("forecast-status").textContent = "Forecasting…";
  resultPanel.hidden = true;
  try {
    const result = await apiRequest("/forecast", { method: "POST", body: JSON.stringify({ readings }) });
    const change = result.forecast - result.lastObserved;
    const errText = result.expectedError ? ` · typical error ±${Number(result.expectedError).toFixed(0)} mg/dL` : "";
    const modelText = result.forecasterName ? ` · ${result.forecasterName}` : "";
    resultPanel.innerHTML = `<span>Forecasted next glucose reading</span><strong>${Number(result.forecast).toFixed(0)} mg/dL</strong><small>${change >= 0 ? "+" : ""}${change.toFixed(0)} mg/dL vs last reading · ${result.readingsUsed} recent readings${errText}${modelText}</small>`;
    resultPanel.hidden = false;
    forecastState = {
      selection: byId("trajectory-patient").value,
      value: result.forecast,
      expectedError: result.expectedError || 0,
    };
    byId("forecast-status").textContent = "Forecast ready";
    drawTrajectory();
  } catch (error) {
    resultPanel.textContent = error.message;
    resultPanel.hidden = false;
    byId("forecast-status").textContent = "Forecast unavailable";
  } finally {
    updateForecastControls();
  }
});

function drawTrajectory() {
  const canvas = byId("glucose-chart");
  const context = canvas.getContext("2d");
  const selected = byId("trajectory-patient").value;
  const isCase = selected.startsWith("case:");
  const isPatient = selected.startsWith("patient:");
  const readings = isCase || isPatient ? selectedReadings() : illustrativeReadings;

  byId("reading-count").textContent = String(readings.length);
  byId("chart-empty").hidden = readings.length > 0;
  canvas.hidden = readings.length === 0;

  if (isCase) {
    const c = mimicCases.find((x) => `case:${x.id}` === selected);
    byId("chart-caption").textContent = c
      ? `${c.label} · ICU point-of-care glucose (de-identified) · showing last ${readings.length} of ${c.totalReadings}`
      : "MIMIC-IV demo case";
  } else if (isPatient) {
    byId("chart-caption").textContent = `${selected.slice("patient:".length)} · saved patient glucose measurements`;
  } else {
    byId("chart-caption").textContent = "Illustrative values only · synthetic, not patient data";
  }
  updateForecastControls();
  if (!readings.length) return;

  const box = canvas.getBoundingClientRect();
  if (!box.width || !box.height) return;
  const ratio = Math.max(window.devicePixelRatio || 1, 1);
  canvas.width = Math.round(box.width * ratio);
  canvas.height = Math.round(box.height * ratio);
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  const width = box.width;
  const height = box.height;
  const pad = { top: 16, right: 16, bottom: 32, left: 43 };
  const chartWidth = width - pad.left - pad.right;
  const chartHeight = height - pad.top - pad.bottom;

  const fc = forecastState && forecastState.selection === selected ? forecastState : null;
  const err = fc ? fc.expectedError : 0;
  const vals = readings.map((r) => r.glucose);
  if (fc) vals.push(fc.value + err, Math.max(fc.value - err, 0));
  const minValue = Math.max(0, Math.floor((Math.min(...vals, 100) - 10) / 50) * 50);
  const maxValue = Math.ceil((Math.max(...vals, 180) + 10) / 50) * 50;
  const step = maxValue - minValue > 300 ? 100 : 50;
  const n = readings.length + (fc ? 1 : 0);
  const xAt = (i) => pad.left + (n === 1 ? chartWidth / 2 : (i * chartWidth) / (n - 1));
  const yAt = (v) => pad.top + ((maxValue - v) * chartHeight) / (maxValue - minValue);

  context.clearRect(0, 0, width, height);
  context.font = '10px "DM Sans", sans-serif';
  context.textAlign = "right";
  context.textBaseline = "middle";
  for (let tick = minValue; tick <= maxValue; tick += step) {
    const y = yAt(tick);
    context.beginPath();
    context.strokeStyle = "#e9efec";
    context.lineWidth = 1;
    context.moveTo(pad.left, y);
    context.lineTo(width - pad.right, y);
    context.stroke();
    context.fillStyle = "#98a59f";
    context.fillText(String(tick), pad.left - 9, y);
  }

  if (140 >= minValue && 140 <= maxValue) {
    context.save();
    context.setLineDash([4, 4]);
    context.strokeStyle = "#e4b9a9";
    context.beginPath();
    context.moveTo(pad.left, yAt(140));
    context.lineTo(width - pad.right, yAt(140));
    context.stroke();
    context.restore();
  }

  context.beginPath();
  readings.forEach((r, i) => {
    const x = xAt(i);
    const y = yAt(r.glucose);
    if (i === 0) context.moveTo(x, y);
    else context.lineTo(x, y);
  });
  context.strokeStyle = "#178775";
  context.lineWidth = 2.5;
  context.lineJoin = "round";
  context.lineCap = "round";
  context.stroke();

  const dense = readings.length > 12;
  readings.forEach((r, i) => {
    const x = xAt(i);
    const y = yAt(r.glucose);
    context.beginPath();
    context.arc(x, y, dense ? 2.5 : 4, 0, Math.PI * 2);
    context.fillStyle = "#fff";
    context.fill();
    context.strokeStyle = "#178775";
    context.lineWidth = 2;
    context.stroke();
    if (!dense || i % 5 === 0) {
      context.fillStyle = "#83918b";
      context.font = '9px "DM Sans", sans-serif';
      context.textAlign = "center";
      context.textBaseline = "top";
      context.fillText(r.label, x, height - 20);
    }
  });

  if (fc) {
    const last = readings.length - 1;
    const x = xAt(readings.length);
    const y = yAt(fc.value);
    context.save();
    context.strokeStyle = "#d7785c";
    context.lineWidth = 2;
    context.setLineDash([5, 4]);
    context.beginPath();
    context.moveTo(xAt(last), yAt(readings[last].glucose));
    context.lineTo(x, y);
    context.stroke();
    if (err) {
      context.setLineDash([]);
      context.beginPath();
      context.moveTo(x, yAt(fc.value + err));
      context.lineTo(x, yAt(Math.max(fc.value - err, 0)));
      context.stroke();
    }
    context.restore();
    context.beginPath();
    context.arc(x, y, 5, 0, Math.PI * 2);
    context.fillStyle = "#d7785c";
    context.fill();
    context.fillStyle = "#a45e47";
    context.font = '9px "DM Sans", sans-serif';
    context.textAlign = "center";
    context.textBaseline = "top";
    context.fillText("Forecast", x, height - 20);
  }
}

byId("trajectory-patient").addEventListener("change", () => {
  resetForecast();
  drawTrajectory();
});
window.addEventListener("resize", () => {
  if (byId("view-trajectory").classList.contains("active")) drawTrajectory();
});

/* ---------- init ---------- */

updateVisitCounts();
renderHistory();

function initializeApp() {
  const username = sessionStorage.getItem(DEMO_SESSION_KEY);
  if (!username) {
    window.location.replace("/login.html");
    return;
  }
  byId("current-username").textContent = username;
  connectBackend();
}

initializeApp();