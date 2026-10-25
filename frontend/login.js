const DEMO_SESSION_KEY = "carepath_demo_user";
const DEMO_ACCOUNTS = ["dr.smith", "dr.lee", "dr.patel"];
const DEMO_PASSWORD = "clinic123";
const byId = (id) => document.getElementById(id);

byId("auth-form").addEventListener("submit", (event) => {
  event.preventDefault();
  if (!event.currentTarget.reportValidity()) return;

  const username = byId("username").value.trim().toLowerCase();
  const password = byId("password").value;
  if (!DEMO_ACCOUNTS.includes(username) || password !== DEMO_PASSWORD) {
    byId("auth-error").textContent = "Incorrect username or password.";
    return;
  }

  sessionStorage.setItem(DEMO_SESSION_KEY, username);
  window.location.replace("/");
});

if (sessionStorage.getItem(DEMO_SESSION_KEY)) window.location.replace("/");
