const navButtons = [...document.querySelectorAll(".nav-item[data-view]")];
const views = [...document.querySelectorAll(".view")];
const pageNames = {
  assessment: "Current assessment",
  trajectory: "Progression forecast",
  history: "Patient history",
  models: "Model transparency",
};
const visits = [];
let pendingVisit = null;
let toastTimer;
const API_BASE = `${window.location.protocol}//${window.location.hostname}:5001/api`;
const DEMO_SESSION_KEY = "carepath_demo_user";
let backendConnected = false;
let forecastAvailable = false;

const byId = (id) => document.getElementById(id);
const valueOf = (id) => {
  const value = byId(id).value.trim();
  return value === "" ? null : Number(value);
};

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
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || `Service request failed (${response.status}).`);
  return result;
}

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
  ["sidebar-status-dot", "backend-status-dot"].forEach((id) => {
    byId(id).classList.toggle("connected", connected);
  });
}

function normalizeVisit(row) {
  return {
    patientId: row.patient_id || row.patientId || "",
    date: new Date(row.visit_date || row.date),
    hba1c: row.hba1c === null || row.hba1c === undefined ? null : Number(row.hba1c),
    glucose: row.glucose === null || row.glucose === undefined ? null : Number(row.glucose),
    bmi: row.bmi === null || row.bmi === undefined ? null : Number(row.bmi),
    systolic: row.systolic_bp ?? row.systolic ?? null,
    diastolic: row.diastolic_bp ?? row.diastolic ?? null,
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
  const patients = [...new Set(visits.map((visit) => visit.patientId).filter(Boolean))];
  select.replaceChildren(new Option("Illustrative example (synthetic)", "illustrative"));
  patients.forEach((patientId) => {
    select.add(new Option(patientId, `patient:${patientId}`));
  });
  if (patients.includes(selected.slice("patient:".length)) && selected.startsWith("patient:")) {
    select.value = selected;
  }
}

async function loadModelsFromBackend(health) {
  const { models } = await apiRequest("/models");
  const current = models.find((model) => model.model_name === "nhanes_active" && model.is_active);
  const trajectory = models.find((model) =>
    ["mimic_glucose_forecaster", "mimic_glucose_forecaster_lstm"].includes(model.model_name) && model.is_active
  );
  byId("registry-status").textContent = models.length
    ? `${models.length} registered model version${models.length === 1 ? "" : "s"}`
    : "Registry connected; no versions registered";
  byId("registry-detail").textContent = `Active risk artifact: ${health.modelName}`;
  byId("registry-badge").textContent = "Local registry connected";
  byId("track-a-status").textContent = current ? "Active" : "Artifact loaded";
  byId("track-a-status").classList.toggle("unavailable", !current);
  byId("track-a-version").textContent = current ? `Version ${current.version_id}` : health.modelName;
  byId("track-a-validation").textContent = current?.val_auc == null ? "Not recorded" : `AUC ${Number(current.val_auc).toFixed(3)}`;
  byId("track-b-status").textContent = health.forecastAvailable ? "Artifact available" : "Artifact unavailable";
  byId("track-b-status").classList.toggle("unavailable", !health.forecastAvailable);
  byId("track-b-version").textContent = trajectory ? `Version ${trajectory.version_id}` : "No active registry version";
  byId("track-b-validation").textContent = trajectory?.val_auc == null ? "Not recorded" : `AUC ${Number(trajectory.val_auc).toFixed(3)}`;
}

async function connectBackend() {
  try {
    const health = await apiRequest("/health");
    forecastAvailable = Boolean(health.forecastAvailable);
    setConnectionState(true, `Using ${health.modelName} (${health.featureCount} features). Predictions run locally; this remains a research prototype.`, health.modelName);
    byId("forecast-badge").textContent = forecastAvailable ? "Forecast model available" : "Forecast artifact unavailable";
    await Promise.all([loadVisitsFromBackend(), loadModelsFromBackend(health)]);
    byId("history-description").textContent = "Saved visits from the local patient database.";
    byId("history-state").lastChild.textContent = " Local database";
    updateForecastControls();
  } catch (error) {
    forecastAvailable = false;
    setConnectionState(false, "Start the local API with .venv/Scripts/python.exe -m src.api_server. The UI will retry when reloaded.");
    byId("registry-status").textContent = "Registry unavailable";
    byId("registry-detail").textContent = "Start the local API to read model versions.";
    byId("track-a-status").textContent = "Service offline";
    byId("track-b-status").textContent = "Service offline";
    byId("registry-badge").textContent = "Registry unavailable";
    byId("forecast-badge").textContent = "Forecast unavailable";
    byId("forecast-status").textContent = "Service offline";
  }
}

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
  navigate("assessment");
});
document.querySelectorAll("[data-goto]").forEach((button) => {
  button.addEventListener("click", () => navigate(button.dataset.goto));
});

byId("today-label").textContent = new Intl.DateTimeFormat(undefined, {
  weekday: "short",
  month: "short",
  day: "numeric",
  year: "numeric",
}).format(new Date());

byId("help-button").addEventListener("click", () => {
  showToast(backendConnected ? "Connected to the local model API. Results are research estimates." : "Start the local API server to enable model predictions.");
});
byId("logout-button").addEventListener("click", () => {
  sessionStorage.removeItem(DEMO_SESSION_KEY);
  window.location.replace("/login.html");
});
document.querySelector(".banner-dismiss").addEventListener("click", () => {
  document.querySelector(".connection-banner").hidden = true;
});

byId("example-button").addEventListener("click", () => {
  const example = {
    "patient-id": "DEMO-1042",
    age: "58",
    hba1c: "7.4",
    glucose: "150",
    bmi: "31.0",
    systolic: "142",
    diastolic: "88",
    ldl: "130",
    hdl: "42",
  };
  Object.entries(example).forEach(([id, value]) => { byId(id).value = value; });
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
    `<div class="snapshot-item"><span>${label}</span><strong>${value}</strong></div>`
  ).join("");
  byId("review-values").hidden = false;
  byId("model-empty-state").hidden = true;
}

function showModelResult(result) {
  byId("model-empty-state").hidden = true;
  byId("model-badge").innerHTML = `<span></span>${escapeHTML(result.modelName)}`;
  byId("model-slots").setAttribute("aria-label", `Risk estimates from ${result.modelName}`);
  ["hypertension", "nephropathy", "cardiovascular"].forEach((target) => {
    const risk = Number(result.risks[target]);
    const trend = result.trends?.[target];
    const trendText = trend ? ` · ${trend.direction} ${Math.abs(trend.delta * 100).toFixed(1)}%` : "";
    byId(`risk-${target}`).textContent = `${(risk * 100).toFixed(0)}%${trendText}`;
    byId(`risk-${target}`).classList.add("loaded");
  });
  byId("recommendation-empty").hidden = true;
  byId("recommendation-list").innerHTML = result.recommendations
    .map((recommendation) => `<li>${escapeHTML(recommendation)}</li>`)
    .join("");
  byId("recommendation-list").hidden = false;
}

function clearModelResult() {
  ["hypertension", "nephropathy", "cardiovascular"].forEach((target) => {
    const value = byId(`risk-${target}`);
    value.textContent = backendConnected ? "Not run" : "Service offline";
    value.classList.remove("loaded");
  });
  byId("model-badge").innerHTML = "<span></span>Awaiting model";
  byId("model-slots").setAttribute("aria-label", "Model estimates not run");
  byId("recommendation-list").innerHTML = "";
  byId("recommendation-list").hidden = true;
  byId("recommendation-empty").hidden = false;
}

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
    RIDAGEYR: visit.age,
    LBXGH: visit.hba1c,
    LBXGLU: visit.glucose,
    BMXBMI: visit.bmi,
    BPXOSY1: visit.systolic,
    BPXODI1: visit.diastolic,
    LBDLDL: visit.ldl,
    LBDHDD: visit.hdl,
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
      ? "Patient measurements will be saved in the local SQLite database. The visit note is not persisted."
      : "Enter a patient ID to save to SQLite; without one, this visit stays in browser memory only.";
    feedback.textContent = `Estimate returned by ${result.modelName}. This is research decision support, not a diagnosis.`;
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
  const savedVisit = { ...pendingVisit };
  if (savedVisit.patientId) {
    try {
      await apiRequest("/visits", {
        method: "POST",
        body: JSON.stringify({ patientId: savedVisit.patientId, values: savedVisit.values }),
      });
      await loadVisitsFromBackend();
      byId("form-feedback").textContent = `Visit saved to the local patient record for ${savedVisit.patientId}.`;
      showToast("Visit measurements saved to the local SQLite database.");
    } catch (error) {
      byId("form-feedback").textContent = `Visit was not saved: ${error.message}`;
      return;
    }
  } else {
    visits.unshift(savedVisit);
    updateVisitCounts();
    renderHistory();
    drawTrajectory();
    byId("form-feedback").textContent = "Visit added to temporary browser history only.";
    showToast("No patient ID provided; visit remains in browser memory only.");
  }
  pendingVisit = null;
  byId("save-visit-button").disabled = true;
});

function updateVisitCounts() {
  byId("visit-count").textContent = String(visits.length);
  byId("history-total").textContent = String(visits.length);
}

function formatDate(date) {
  return new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric", year: "numeric" }).format(date);
}

function selectedPatientReadings() {
  const selected = byId("trajectory-patient").value;
  if (!selected.startsWith("patient:")) return [];
  const patientId = selected.slice("patient:".length);
  return visits.filter((visit) => visit.patientId === patientId && visit.glucose !== null)
    .slice()
    .sort((left, right) => left.date - right.date)
    .map((visit, index) => ({ label: `Visit ${index + 1}`, glucose: visit.glucose }));
}

function updateForecastControls() {
  const selectedPatient = byId("trajectory-patient").value.startsWith("patient:");
  const enoughReadings = selectedPatientReadings().length >= 4;
  const canForecast = backendConnected && forecastAvailable && selectedPatient && enoughReadings;
  byId("forecast-button").disabled = !canForecast;
  if (!backendConnected) byId("forecast-status").textContent = "Service offline";
  else if (!forecastAvailable) byId("forecast-status").textContent = "Artifact unavailable";
  else if (!selectedPatient) byId("forecast-status").textContent = "Select tracked patient";
  else if (!enoughReadings) byId("forecast-status").textContent = "Need 4 readings";
  else byId("forecast-status").textContent = byId("forecast-result").hidden ? "Ready to run" : "Forecast ready";
}

byId("forecast-button").addEventListener("click", async () => {
  const readings = selectedPatientReadings().map((reading) => reading.glucose);
  if (byId("forecast-button").disabled) return;
  const button = byId("forecast-button");
  const resultPanel = byId("forecast-result");
  button.disabled = true;
  byId("forecast-status").textContent = "Forecasting…";
  resultPanel.hidden = true;
  try {
    const result = await apiRequest("/forecast", {
      method: "POST",
      body: JSON.stringify({ readings }),
    });
    const change = result.forecast - result.lastObserved;
    resultPanel.innerHTML = `<span>Forecasted next glucose reading</span><strong>${Number(result.forecast).toFixed(0)} mg/dL</strong><small>${change >= 0 ? "+" : ""}${change.toFixed(0)} mg/dL vs last reading · ${result.readingsUsed} recent readings</small>`;
    resultPanel.hidden = false;
    byId("forecast-status").textContent = "Forecast ready";
  } catch (error) {
    resultPanel.textContent = error.message;
    resultPanel.hidden = false;
    byId("forecast-status").textContent = "Forecast unavailable";
  } finally {
    updateForecastControls();
  }
});

function escapeHTML(text) {
  return String(text).replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[character]);
}

function renderHistory() {
  const query = byId("history-search").value.trim().toLowerCase();
  const filtered = visits.filter((visit) => (visit.patientId || "unidentified patient").toLowerCase().includes(query));
  const rows = byId("history-rows");
  byId("history-empty").hidden = visits.length > 0;
  if (visits.length > 0 && filtered.length === 0) {
    rows.innerHTML = '<tr><td colspan="7">No saved visits match this search.</td></tr>';
    return;
  }
  rows.innerHTML = filtered.map((visit) => {
    const pressure = visit.systolic === null || visit.diastolic === null ? "—" : `${visit.systolic}/${visit.diastolic}`;
    return `<tr>
      <td>${escapeHTML(visit.patientId || "Unidentified")}</td>
      <td>${formatDate(visit.date)}</td>
      <td>${visit.hba1c === null ? "—" : `${visit.hba1c.toFixed(1)}%`}</td>
      <td>${visit.glucose === null ? "—" : `${visit.glucose} mg/dL`}</td>
      <td>${visit.bmi === null ? "—" : visit.bmi.toFixed(1)}</td>
      <td>${pressure}</td>
      <td><span class="table-state">Not attached to visit</span></td>
    </tr>`;
  }).join("");
}

byId("history-search").addEventListener("input", renderHistory);

const illustrativeReadings = [
  { label: "Visit 1", glucose: 126 },
  { label: "Visit 2", glucose: 139 },
  { label: "Visit 3", glucose: 132 },
  { label: "Visit 4", glucose: 158 },
  { label: "Visit 5", glucose: 149 },
  { label: "Visit 6", glucose: 171 },
];

function drawTrajectory() {
  const canvas = byId("glucose-chart");
  const context = canvas.getContext("2d");
  const selected = byId("trajectory-patient").value;
  const usePatient = selected.startsWith("patient:");
  const readings = usePatient ? selectedPatientReadings() : illustrativeReadings;
  byId("reading-count").textContent = String(readings.length);
  byId("chart-empty").hidden = readings.length > 0;
  canvas.hidden = readings.length === 0;
  byId("chart-caption").textContent = usePatient
    ? `${selected.slice("patient:".length)} · saved patient glucose measurements`
    : "Illustrative values only · synthetic, not patient data";
  byId("chart-empty").textContent = "No saved glucose measurements for this patient.";
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
  const minValue = 50;
  const maxValue = 250;
  const xAt = (index) => pad.left + (readings.length === 1 ? chartWidth / 2 : index * chartWidth / (readings.length - 1));
  const yAt = (value) => pad.top + (maxValue - value) * chartHeight / (maxValue - minValue);

  context.clearRect(0, 0, width, height);
  context.font = '10px "DM Sans", sans-serif';
  context.textAlign = "right";
  context.textBaseline = "middle";
  [50, 100, 150, 200, 250].forEach((tick) => {
    const y = yAt(tick);
    context.beginPath();
    context.strokeStyle = "#e9efec";
    context.lineWidth = 1;
    context.moveTo(pad.left, y);
    context.lineTo(width - pad.right, y);
    context.stroke();
    context.fillStyle = "#98a59f";
    context.fillText(String(tick), pad.left - 9, y);
  });

  context.save();
  context.setLineDash([4, 4]);
  context.strokeStyle = "#e4b9a9";
  context.beginPath();
  context.moveTo(pad.left, yAt(140));
  context.lineTo(width - pad.right, yAt(140));
  context.stroke();
  context.restore();

  context.beginPath();
  readings.forEach((reading, index) => {
    const x = xAt(index);
    const y = yAt(reading.glucose);
    if (index === 0) context.moveTo(x, y);
    else context.lineTo(x, y);
  });
  context.strokeStyle = "#178775";
  context.lineWidth = 2.5;
  context.lineJoin = "round";
  context.lineCap = "round";
  context.stroke();

  readings.forEach((reading, index) => {
    const x = xAt(index);
    const y = yAt(reading.glucose);
    context.beginPath();
    context.arc(x, y, 4, 0, Math.PI * 2);
    context.fillStyle = "#fff";
    context.fill();
    context.strokeStyle = "#178775";
    context.lineWidth = 2;
    context.stroke();
    context.fillStyle = "#83918b";
    context.font = '9px "DM Sans", sans-serif';
    context.textAlign = "center";
    context.textBaseline = "top";
    context.fillText(reading.label, x, height - 20);
  });
}

byId("trajectory-patient").addEventListener("change", drawTrajectory);
window.addEventListener("resize", () => {
  if (byId("view-trajectory").classList.contains("active")) drawTrajectory();
});

updateVisitCounts();
renderHistory();
window.requestAnimationFrame(drawTrajectory);

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