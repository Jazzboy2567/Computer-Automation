// Pilot web UI — vanilla JS. Talks to the FastAPI backend, streams run events
// over SSE, and drives the start/pause/resume/kill + approval controls.

const $ = (id) => document.getElementById(id);
const api = (path, opts) => fetch(path, opts).then((r) => r.json());
const esc = (s) =>
  String(s == null ? "" : s).replace(/[&<>"]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])
  );

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
  pill.className = "pill " + cls;
  pill.querySelector(".label").textContent = label;
}
function setMeta(text) {
  $("activity-meta").textContent = text || "";
}
function logEntry(text, cls, icon) {
  const log = $("log");
  const empty = log.querySelector(".log-empty");
  if (empty) empty.remove();
  const row = document.createElement("div");
  row.className = "entry " + (cls || "");
  const i = document.createElement("span");
  i.className = "entry-icon";
  i.textContent = icon || "·";
  const t = document.createElement("span");
  t.className = "entry-text";
  t.textContent = text;
  row.append(i, t);
  log.appendChild(row);
  log.scrollTop = log.scrollHeight;
}
function setScreenshot(url, src) {
  if (src) {
    const img = $("screenshot");
    img.src = src;
    img.classList.add("show");
    $("shot-empty").classList.add("hidden");
  }
  if (url) $("shot-url").textContent = url;
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
      logEntry(`step ${ev.step}: perceived ${ev.elements} elements [${ev.mode}]`, "e-perception", "◍");
      setMeta(`step ${ev.step} · ${ev.mode}`);
      if (ev.screenshot)
        setScreenshot(ev.url, `/api/screenshot?path=${encodeURIComponent(ev.screenshot)}&t=${Date.now()}`);
      else setScreenshot(ev.url);
      break;
    case "decision":
      logEntry(`${ev.summary}  [${ev.risk}]  ${ev.reasoning || ""}`, "e-decision", "→");
      break;
    case "awaiting_approval":
      showApproval(ev.summary, ev.risk);
      setStatus("awaiting approval", "paused");
      break;
    case "approval_result":
      hideApproval();
      break;
    case "executed":
      logEntry(`step ${ev.step}: ${ev.ok ? "ok" : "ERROR " + ev.error}`,
               ev.ok ? "e-executed" : "e-error", ev.ok ? "✓" : "✕");
      break;
    case "replay_step":
      logEntry(`replay ${ev.step}: ${ev.summary} [${ev.risk}]`, "e-perception", "▷");
      break;
    case "replay_fallback":
      logEntry(`replay step ${ev.step} failed — falling back to model`, "e-approval", "!");
      break;
    case "report":
      showReport(ev);
      break;
    case "error":
      logEntry(`${ev.error}`, "e-error", "✕");
      setStatus("error", "error");
      break;
    case "finished":
      logEntry(`finished: ${ev.message}`, ev.ok ? "e-finish" : "e-error", ev.ok ? "✓" : "—");
      setStatus(ev.ok ? "done" : "stopped", ev.ok ? "done" : "error");
      setMeta("");
      break;
    case "closed":
      setRunning(false);
      if (evtSource) evtSource.close();
      break;
  }
}

function showApproval(summary, risk) {
  $("approval-panel").classList.remove("hidden");
  $("approval-text").textContent = `${summary} — classified "${risk}". Approve to proceed.`;
}
function hideApproval() {
  $("approval-panel").classList.add("hidden");
}
function showReport(ev) {
  const r = $("report");
  r.classList.remove("hidden");
  const cls = ev.ok ? "ok" : "warn";
  const head = ev.ok ? "✓" : "⚠";
  const items = Object.entries(ev.paths || {})
    .map(([k, v]) => `<li><span class="ftype">${esc(k)}</span><code>${esc(v)}</code></li>`)
    .join("");
  r.innerHTML = `<div class="report-head ${cls}">${head} ${esc(ev.message)}</div>
    <ul>${items}</ul>
    <div class="run-dir">Artifacts saved under <code>${esc(ev.run_dir)}</code></div>`;
}

// ----------------------------------------------------------------- controls
$("btn-start").addEventListener("click", async () => {
  $("log").innerHTML = "";
  $("report").classList.add("hidden");
  $("screenshot").classList.remove("show");
  $("shot-empty").classList.remove("hidden");
  $("shot-url").textContent = "about:blank";
  const body = {
    task_file: $("task-select").value || null,
    goal: $("goal").value || null,
    start_url: $("start-url").value || null,
    provider: $("provider").value,
    model: $("model").value || null,
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
    logEntry(`could not start: ${err.detail}`, "e-error", "✕");
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
