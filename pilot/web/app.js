// Pilot web UI — vanilla JS. Talks to the FastAPI backend, streams run events
// over SSE, and drives the start/pause/resume/kill + approval controls.

const $ = (id) => document.getElementById(id);
const api = (path, opts) => fetch(path, opts).then((r) => r.json());

let evtSource = null;

// ---------------------------------------------------------------- first-run ack
async function checkAck() {
  const { acknowledged } = await api("/api/ack");
  if (!acknowledged) $("ack-overlay").classList.remove("hidden");
}
$("ack-check").addEventListener("change", (e) => {
  $("ack-accept").disabled = !e.target.checked;
});
$("ack-accept").addEventListener("click", async () => {
  await api("/api/ack", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ accept: true }),
  });
  $("ack-overlay").classList.add("hidden");
});

// ----------------------------------------------------------------- task list
async function loadTasks() {
  const { tasks } = await api("/api/tasks");
  const sel = $("task-select");
  for (const t of tasks) {
    const o = document.createElement("option");
    o.value = t.file;
    o.textContent = `${t.name} (${t.file})`;
    sel.appendChild(o);
  }
}
$("task-select").addEventListener("change", (e) => {
  const adhoc = !e.target.value;
  $("goal-field").classList.toggle("hidden", !adhoc);
  $("url-field").classList.toggle("hidden", !adhoc);
});

// ----------------------------------------------------------------- UI helpers
function setStatus(label, cls) {
  const pill = $("status-pill");
  pill.textContent = label;
  pill.className = "pill " + cls;
}
function logEntry(text, cls) {
  const div = document.createElement("div");
  div.className = "entry " + (cls || "");
  div.textContent = text;
  const log = $("log");
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}
function setRunning(running) {
  $("btn-start").disabled = running;
  $("btn-pause").disabled = !running;
  $("btn-resume").disabled = true;
  $("btn-stop").disabled = !running;
}

// ----------------------------------------------------------------- SSE events
function connectEvents() {
  if (evtSource) evtSource.close();
  evtSource = new EventSource("/api/events");
  evtSource.onmessage = (m) => handleEvent(JSON.parse(m.data));
}

function handleEvent(ev) {
  switch (ev.event) {
    case "perception":
      logEntry(`· step ${ev.step}: perceived ${ev.elements} elements [${ev.mode}] ${ev.url}`, "e-perception");
      if (ev.screenshot) {
        $("screenshot").src = `/api/screenshot?path=${encodeURIComponent(ev.screenshot)}&t=${Date.now()}`;
      }
      break;
    case "decision":
      logEntry(`→ step ${ev.step}: ${ev.summary}  [${ev.risk}]  ${ev.reasoning || ""}`, "e-decision");
      break;
    case "awaiting_approval":
      showApproval(ev.summary, ev.risk);
      setStatus("waiting for approval", "paused");
      break;
    case "approval_result":
      hideApproval();
      break;
    case "executed":
      logEntry(`✓ step ${ev.step}: ${ev.ok ? "ok" : "ERROR " + ev.error}`, ev.ok ? "e-executed" : "e-error");
      break;
    case "replay_step":
      logEntry(`▷ replay ${ev.step}: ${ev.summary} [${ev.risk}]`, "e-perception");
      break;
    case "replay_fallback":
      logEntry(`! replay step ${ev.step} failed — falling back to model`, "e-approval");
      break;
    case "report":
      showReport(ev);
      break;
    case "error":
      logEntry(`✗ ${ev.error}`, "e-error");
      setStatus("error", "error");
      break;
    case "finished":
      logEntry(`— finished: ${ev.message}`, ev.ok ? "e-executed" : "e-error");
      setStatus(ev.ok ? "done" : "stopped", ev.ok ? "done" : "error");
      break;
    case "closed":
      setRunning(false);
      if (evtSource) evtSource.close();
      break;
  }
}

function showApproval(summary, risk) {
  $("approval-panel").classList.remove("hidden");
  $("approval-text").textContent = `${summary}  —  classified "${risk}". Approve to proceed.`;
}
function hideApproval() {
  $("approval-panel").classList.add("hidden");
}
function showReport(ev) {
  const r = $("report");
  r.classList.remove("hidden");
  const links = Object.entries(ev.paths || {})
    .map(([k, v]) => `<li>${k}: <code>${v}</code></li>`)
    .join("");
  r.innerHTML = `<strong>${ev.ok ? "✅" : "⚠️"} ${ev.message || ""}</strong>
    <ul>${links}</ul>
    <div>Artifacts saved under <code>${ev.run_dir}</code></div>`;
}

// ----------------------------------------------------------------- controls
$("btn-start").addEventListener("click", async () => {
  $("log").innerHTML = "";
  $("report").classList.add("hidden");
  const body = {
    task_file: $("task-select").value || null,
    goal: $("goal").value || null,
    start_url: $("start-url").value || null,
    provider: $("provider").value,
    approval_mode: $("approval").value,
    headed: $("headed").checked,
    action_delay: parseFloat($("delay").value) || 0,
  };
  const res = await fetch("/api/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json();
    logEntry(`✗ could not start: ${err.detail}`, "e-error");
    return;
  }
  setRunning(true);
  setStatus("running", "running");
  connectEvents();
});

$("btn-pause").addEventListener("click", async () => {
  await api("/api/pause", { method: "POST" });
  setStatus("paused", "paused");
  $("btn-pause").disabled = true;
  $("btn-resume").disabled = false;
});
$("btn-resume").addEventListener("click", async () => {
  await api("/api/resume", { method: "POST" });
  setStatus("running", "running");
  $("btn-pause").disabled = false;
  $("btn-resume").disabled = true;
});
$("btn-stop").addEventListener("click", async () => {
  await api("/api/stop", { method: "POST" });
  setStatus("stopping", "error");
});
$("btn-approve").addEventListener("click", () => resolveApproval(true));
$("btn-decline").addEventListener("click", () => resolveApproval(false));
async function resolveApproval(approved) {
  await api("/api/approve", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ approved }),
  });
  hideApproval();
}

// ----------------------------------------------------------------- boot
checkAck();
loadTasks();
