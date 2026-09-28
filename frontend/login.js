const DEMO_SESSION_KEY = "carepath_demo_user";
const DEMO_ACCOUNTS = ["dr.smith", "dr.lee", "dr.patel"];
const DEMO_PASSWORD = "clinic123";
const byId = (id) => document.getElementById(id);

byId("fill-demo-button").addEventListener("click", () => {
  byId("username").value = "dr.smith";
  byId("password").value = DEMO_PASSWORD;
  byId("auth-error").textContent = "";
});

byId("auth-form").addEventListener("submit", (event) => {
  event.preventDefault();
  if (!event.currentTarget.reportValidity()) return;

  const username = byId("username").value.trim().toLowerCase();
  const password = byId("password").value;
  if (!["dr.smith", "dr.lee", "dr.patel"].includes(username) || password !== "clinic123") {
    byId("auth-error").textContent = "Use one of the listed demo accounts and its shared password.";
    return;
  }

  sessionStorage.setItem(DEMO_SESSION_KEY, username);
  window.location.replace("/");
});

if (sessionStorage.getItem(DEMO_SESSION_KEY)) window.location.replace("/");
