"use strict";

const $ = (id) => document.getElementById(id);
const canvases = {
  track: $("track-canvas"),
  network: $("network-canvas"),
  history: $("history-canvas"),
};
const contexts = Object.fromEntries(Object.entries(canvases).map(([key, canvas]) => [key, canvas.getContext("2d")]));
let state = null;
let trackImage = null;
let imageUrl = "";
let imageToken = 0;
let pending = false;
let connected = false;
let socket = null;
let reconnectTimer = null;
let lastSequence = -1;
let revision = null;
let instance = null;
const retiredInstances = new Set();
let generationLimit = 100;
const fitnessDomain = Object.freeze({ min: -5, max: 1100 });
const TIME_LIMIT = 1400 / 30;
let networkNodes = [];
let hoveredNode = null;
let dragDepth = 0;
let circuits = [];
let catalogLoaded = false;
let catalogVersion = 0;
let circuitOptionsKey = "";
let architectures = [];
let architecturesLoaded = false;
let architectureCatalogVersion = 0;
let architectureOptionsKey = "";
let comparisonRowsKey = "";
let comparisonLegendKey = "";
const networkColors = Object.freeze({ compact: "#2563ff", wide: "#ff8500", deep: "#00ad65", split: "#d629f5" });

function setStatus(message = "", error = false) {
  if ($("status").textContent !== message) $("status").textContent = message;
  $("status").parentElement.classList.toggle("error", error);
}

function scalar(value, digits = 2) { return Number.isFinite(value) ? value.toFixed(digits) : "—"; }
function signed(value) { return Number.isFinite(value) ? (value >= 0 ? "+" : "") + value.toFixed(2) : "—"; }
function pedal(value) { return Number.isFinite(value) ? `${Math.round(Math.max(0, Math.min(1, value)) * 100)}%` : "—"; }
function timeLabel(value) { return Number.isFinite(value) ? `${value.toFixed(2)} s` : "—"; }
function leader() { return state?.comparison?.entries?.find((entry) => entry.id === state.network?.id)?.leader || state?.cars?.[state.best_index] || state?.cars?.[0] || null; }
function carStatus(car) { return car?.finished ? `finished in ${timeLabel(car.lap_time)}` : `${car?.crashed ? "crashed" : car?.alive ? "driving" : "stopped"}, elapsed time ${timeLabel(car?.elapsed_time)}`; }

function updateControls() {
  $("upload-button").disabled = pending;
  $("circuit-select").disabled = pending || !state || !catalogLoaded || !circuits.length;
  $("architecture-select").disabled = pending || !state || !architecturesLoaded || !architectures.length;
  $("train-button").disabled = pending || !state || !connected || state.complete;
  $("train-button").textContent = state?.complete ? "Complete" : state?.running ? "Pause" : "Train";
  $("compare-button").disabled = pending || !state || !connected;
  $("compare-button").textContent = state?.comparison ? "Restart comparison" : "Compare networks";
  $("single-button").hidden = !state?.comparison;
  $("single-button").disabled = pending || !state || !connected;
  $("reset-button").disabled = pending || !state;
  $("speed-select").disabled = pending || !state;
  $("save-button").disabled = pending || !state?.history?.length;
  $("comparison-rows").querySelectorAll("button").forEach((button) => { button.disabled = pending; });
  if (state) $("speed-select").value = String(state.speed);
  renderCircuitSelect();
  renderArchitectureSelect();
  $("workspace").setAttribute("aria-busy", String(pending));
}

function renderCircuitSelect() {
  const select = $("circuit-select");
  const currentId = state?.track?.circuit_id;
  const custom = !!state && currentId == null;
  const key = `${catalogVersion}/${custom}/${circuits.length ? "" : currentId || ""}`;
  if (key !== circuitOptionsKey) {
    circuitOptionsKey = key;
    const fragment = document.createDocumentFragment();
    if (custom) {
      const option = document.createElement("option");
      option.value = "__custom__";
      option.textContent = "Custom";
      option.disabled = true;
      fragment.append(option);
    }
    for (const circuit of circuits) {
      const option = document.createElement("option");
      option.value = circuit.id;
      option.textContent = circuit.name + (circuit.difficulty ? ` / ${circuit.difficulty}` : "");
      option.title = circuit.description || "";
      fragment.append(option);
    }
    if (!custom && !circuits.length) {
      const option = document.createElement("option");
      option.value = currentId || "";
      option.textContent = currentId ? state.track.name : catalogLoaded ? "Circuits unavailable" : "Loading circuits";
      option.disabled = true;
      fragment.append(option);
    }
    select.replaceChildren(fragment);
  }
  if (custom) select.value = "__custom__";
  else if (currentId) select.value = currentId;
  const current = circuits.find((circuit) => circuit.id === currentId);
  select.title = current?.description || (custom ? "Custom uploaded track. Choose a circuit to replace it." : "Choose a circuit");
}

async function loadCircuits() {
  try {
    const result = await request("/api/circuits");
    circuits = Array.isArray(result.circuits) ? result.circuits.filter((circuit) => typeof circuit.id === "string" && typeof circuit.name === "string") : [];
  } catch (error) { setStatus(`Circuit list unavailable. ${error.message}`, true); }
  finally { catalogLoaded = true; ++catalogVersion; updateControls(); }
}

function renderArchitectureSelect() {
  const select = $("architecture-select");
  const currentId = state?.network?.id;
  const key = `${architectureCatalogVersion}/${architectures.length ? "" : currentId || ""}`;
  if (key !== architectureOptionsKey) {
    architectureOptionsKey = key;
    const fragment = document.createDocumentFragment();
    for (const architecture of architectures) {
      const option = document.createElement("option");
      option.value = architecture.id;
      option.textContent = `${architecture.name} / ${architecture.sizes.join("-")}`;
      option.title = architecture.description || "";
      fragment.append(option);
    }
    if (!architectures.length) {
      const option = document.createElement("option");
      option.value = currentId || "";
      option.textContent = state?.network?.name || (architecturesLoaded ? "Networks unavailable" : "Loading networks");
      option.disabled = true;
      fragment.append(option);
    }
    select.replaceChildren(fragment);
  }
  if (currentId) select.value = currentId;
  const current = architectures.find((architecture) => architecture.id === currentId);
  select.title = `${current?.description || "Choose a neural network."} ${state?.comparison ? "Inspect this network without restarting the comparison." : "Changing networks starts a new population."}`;
}

async function loadArchitectures() {
  try {
    const result = await request("/api/architectures");
    architectures = Array.isArray(result.architectures) ? result.architectures.filter((architecture) => typeof architecture.id === "string" && typeof architecture.name === "string" && Array.isArray(architecture.sizes)) : [];
  } catch (error) { setStatus(`Network list unavailable. ${error.message}`, true); }
  finally { architecturesLoaded = true; ++architectureCatalogVersion; updateControls(); }
}

async function request(path, options = {}) {
  const headers = new Headers(options.headers || {});
  if (options.body !== undefined && !(options.body instanceof FormData)) headers.set("Content-Type", "application/json");
  let response;
  try { response = await fetch(path, { ...options, headers }); }
  catch { throw new Error("Cannot reach the server. Reconnecting…"); }
  if (options.raw && response.ok) return response;
  let result;
  try { result = await response.json(); } catch { throw new Error("The server returned an invalid response."); }
  if (!response.ok) throw new Error(result.error?.message || `Request failed (${response.status}).`);
  return result;
}

async function action(work) {
  if (pending) return;
  pending = true;
  updateControls();
  setStatus("");
  try { await work(); } catch (error) { setStatus(error.message, true); }
  finally { pending = false; updateControls(); }
}

function acceptState(snapshot) {
  if (!snapshot || !snapshot.track || !Array.isArray(snapshot.cars)) return;
  if (typeof snapshot.instance === "string" && snapshot.instance !== instance) {
    if (retiredInstances.has(snapshot.instance)) return;
    if (instance !== null) retiredInstances.add(instance);
    instance = snapshot.instance;
    lastSequence = -1;
    revision = null;
    imageUrl = "";
    trackImage = null;
    ++imageToken;
  }
  if (Number.isFinite(snapshot.sequence) && snapshot.sequence < lastSequence) return;
  const changedTrack = revision !== snapshot.revision;
  if (changedTrack) {
    revision = snapshot.revision;
    lastSequence = -1;
    generationLimit = 100;
  }
  if (snapshot.network?.id !== state?.network?.id) hoveredNode = null;
  state = snapshot;
  lastSequence = Number.isFinite(snapshot.sequence) ? snapshot.sequence : lastSequence;
  generationLimit = Math.max(generationLimit, state.generation || 0);
  const car = leader();
  $("track-name").textContent = state.track.name || "Track";
  $("track-name").title = state.track.name || "Track";
  $("generation").textContent = String(state.generation);
  $("laps").textContent = `${car?.laps || 0} / 1`;
  $("leader-time").textContent = timeLabel(car?.finished ? car.lap_time : car?.elapsed_time);
  $("best-time").textContent = timeLabel(state.metrics?.best_ever_time);
  $("alive").textContent = `${state.cars.filter((item) => item.alive).length} / ${state.cars.length}`;
  $("run-state").textContent = state.complete ? "Complete" : state.running ? "Learning" : "Paused";
  $("leader-label").textContent = state.comparison ? `${state.comparison.entries.length} network leaders` : `Leader ${String((state.best_index || 0) + 1).padStart(2, "0")}`;
  $("leader-state").textContent = car?.finished ? "Finished" : car?.crashed ? "Crashed" : car?.alive ? "Driving" : "Stopped";
  $("leader-state").className = car?.finished ? "finished" : car?.crashed ? "crashed" : "";
  $("network-leader").textContent = `Car ${String((state.best_index || 0) + 1).padStart(2, "0")}`;
  $("step-label").textContent = `Step ${state.step_count}`;
  $("track-size").textContent = `${state.track.width} x ${state.track.height}`;
  $("steer").textContent = signed(car?.steering);
  $("target-speed").textContent = pedal(car?.target_speed);
  $("brake").textContent = pedal(car?.brake);
  $("fitness").textContent = scalar(car?.fitness, 1);
  $("car-speed").textContent = scalar(car?.speed, 1);
  renderComparison();
  if (snapshot.error) setStatus(typeof snapshot.error === "string" ? snapshot.error : snapshot.error.message || "Training stopped because of an error.", true);
  if (state.track.image_url && state.track.image_url !== imageUrl) loadTrackImage(state.track.image_url);
  updateControls();
  drawAll();
}

function loadTrackImage(url) {
  const token = ++imageToken;
  imageUrl = url;
  trackImage = null;
  const image = new Image();
  image.onload = () => { if (token === imageToken) { trackImage = image; drawTrack(); } };
  image.onerror = () => { if (token === imageToken) setStatus("The track image could not be loaded.", true); };
  image.src = url;
}

function canvasSize(key) {
  const canvas = canvases[key];
  const ctx = contexts[key];
  const bounds = canvas.getBoundingClientRect();
  const width = Math.max(1, bounds.width);
  const height = Math.max(1, bounds.height);
  const ratio = window.devicePixelRatio || 1;
  const pixelWidth = Math.round(width * ratio);
  const pixelHeight = Math.round(height * ratio);
  if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) { canvas.width = pixelWidth; canvas.height = pixelHeight; }
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  return { width, height, ctx };
}

function drawCar(ctx, car, transform, selected, color = null) {
  const x = transform.x + car.x * transform.scale;
  const y = transform.y + car.y * transform.scale;
  const scale = transform.scale;
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(car.heading);
  ctx.globalAlpha = color ? 1 : selected ? 1 : car.finished ? 0.46 : car.alive ? 0.26 : 0.1;
  // the whole population stays visible at the same marker size as its leader.
  const markerScale = Math.max(1, 14 / (state.track.car_length * scale), 5 / (state.track.car_width * scale));
  // comparison glyphs have fixed screen dimensions independent of the inspected model.
  const length = color ? 16 : state.track.car_length * scale * markerScale;
  const width = color ? 6 : state.track.car_width * scale * markerScale;
  if (color) {
    // comparison cars share the same outline and dimensions, including the inspected leader.
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2;
    ctx.strokeRect(-length / 2, -width / 2, length, width);
  }
  ctx.fillStyle = color || (selected ? "#3e7fbd" : "#8f9da8");
  ctx.fillRect(-length / 2, -width / 2, length, width);
  if (!color && (selected || car.finished)) {
    ctx.strokeStyle = car.finished ? "#64a587" : "#e6f0fa"; ctx.lineWidth = scale;
    ctx.strokeRect(-length / 2 + scale / 2, -width / 2 + scale / 2, length - scale, width - scale);
  }
  ctx.fillStyle = color ? "#15222f" : selected ? "#263c51" : "#4d5d6c";
  for (const dx of [-length * 0.29, length * 0.29]) for (const dy of [-width * 0.42, width * 0.42]) ctx.fillRect(dx - length * 0.115, dy - width * 0.08, length * 0.23, width * 0.16);
  ctx.fillStyle = color ? "#e4f4ff" : selected ? "#bad5eb" : "#ccd5dc";
  ctx.fillRect(length * 0.05, -width * 0.35, length * 0.2, width * 0.7);
  // rear lamps use the applied brake pedal, not a diagnostic network output.
  if (Number.isFinite(car.brake) && car.brake > 0) {
    ctx.fillStyle = `rgba(223,64,67,${0.4 + Math.min(1, car.brake) * 0.6})`;
    for (const dy of [-width * 0.36, width * 0.36]) ctx.fillRect(-length / 2, dy - width * 0.09, length * 0.12, width * 0.18);
  }
  ctx.restore();
}

function drawTrack() {
  const { width, height, ctx } = canvasSize("track");
  ctx.fillStyle = "#edf0f3";
  ctx.fillRect(0, 0, width, height);
  if (!state) {
    ctx.fillStyle = "#9aa5b0"; ctx.font = "11px ui-monospace, monospace"; ctx.textAlign = "center";
    ctx.fillText("Loading track…", width / 2, height / 2);
    return;
  }
  const scale = Math.min((width - 24) / state.track.width, (height - 24) / state.track.height);
  const transform = { scale, x: (width - state.track.width * scale) / 2, y: (height - state.track.height * scale) / 2 };
  if (trackImage) ctx.drawImage(trackImage, transform.x, transform.y, state.track.width * scale, state.track.height * scale);
  const car = leader();
  if (car && $("show-rays").checked && Array.isArray(state.sensors?.endpoints)) {
    ctx.strokeStyle = state.comparison ? `${networkColors[state.network?.id] || "#2563ff"}88` : "#3d82b66b";
    ctx.fillStyle = state.comparison ? networkColors[state.network?.id] || "#2563ff" : "#5998c6"; ctx.lineWidth = 1;
    for (const point of state.sensors.endpoints) {
      const x = transform.x + point[0] * scale;
      const y = transform.y + point[1] * scale;
      ctx.beginPath(); ctx.moveTo(transform.x + car.x * scale, transform.y + car.y * scale); ctx.lineTo(x, y); ctx.stroke();
      ctx.beginPath(); ctx.arc(x, y, 1.5, 0, Math.PI * 2); ctx.fill();
    }
  }
  if (state.comparison) {
    const entries = state.comparison.entries.filter((entry) => entry.leader);
    for (const entry of entries) drawCar(ctx, entry.leader, transform, false, networkColors[entry.id] || "#2563ff");
    canvases.track.setAttribute("aria-label", `${entries.length} network leaders on the same track. ${entries.map((entry) => `${entry.name}: ${carStatus(entry.leader)}`).join(". ")}. Inspecting ${state.network?.name || "selected network"}.`);
  } else {
    state.cars.forEach((item, index) => { if (index !== state.best_index) drawCar(ctx, item, transform, false); });
    if (car) drawCar(ctx, car, transform, true);
    canvases.track.setAttribute("aria-label", `Generation ${state.generation}. ${state.cars.filter((item) => item.alive).length} cars active. Leader ${carStatus(car)}. Best lap ${timeLabel(state.metrics?.best_ever_time)}.`);
  }
}

function drawNetwork() {
  // the hidden dialog has no canvas bounds until it is opened.
  if (!$("network-dialog").open) return;
  const { width, height, ctx } = canvasSize("network");
  if (width < 150 || height < 80) return;
  const network = state?.network;
  const brain = state?.brain;
  networkNodes = [];
  const counts = network?.sizes;
  if (!Array.isArray(counts) || counts.length < 2) {
    ctx.fillStyle = "#9aa5b0"; ctx.font = "10px ui-monospace, monospace"; ctx.textAlign = "center";
    ctx.fillText("Loading network…", width / 2, height / 2);
    $("network-info").textContent = "";
    return;
  }
  const lastLayer = counts.length - 1;
  const left = 72;
  const right = width - 37;
  const top = 34;
  const bottom = height - 32;
  const labelNames = {
    wall_left_60: "Ray -60", wall_left_30: "Ray -30", wall_forward: "Ahead",
    wall_right_30: "Ray +30", wall_right_60: "Ray +60",
    steering: "Steer", steering_left: "Steer L", steering_right: "Steer R", target_speed: "Target speed", brake: "Brake",
  };
  const humanName = (name) => labelNames[name] || String(name || "Neuron").replaceAll("_", " ");
  const activations = [brain?.inputs || [], ...(brain?.hidden_layers || []), brain?.outputs || []];
  const layerNodes = counts.map((count, layer) => Array.from({ length: count }, (_, index) => ({
    layer, index,
    x: left + layer / lastLayer * (right - left),
    y: count === 1 ? (top + bottom) / 2 : top + index / (count - 1) * (bottom - top),
  })));
  networkNodes = layerNodes.flat();
  const focused = hoveredNode && layerNodes[hoveredNode.layer]?.[hoveredNode.index];
  // each edge and activation comes from the selected leader's real layers.
  for (let layer = 0; layer < lastLayer; layer++) {
    const matrix = network.layers?.[layer]?.weights;
    if (!Array.isArray(matrix)) continue;
    for (let from = 0; from < matrix.length; from++) for (let to = 0; to < matrix[from].length; to++) {
      const start = layerNodes[layer]?.[from];
      const end = layerNodes[layer + 1]?.[to];
      const weight = matrix[from][to];
      if (!start || !end || !Number.isFinite(weight)) continue;
      const active = !focused || (focused.layer === layer && focused.index === from) || (focused.layer === layer + 1 && focused.index === to);
      const strength = Math.tanh(Math.abs(weight) / 2);
      ctx.strokeStyle = weight >= 0 ? `rgba(71,126,166,${active ? 0.1 + strength * 0.3 : 0.015})` : `rgba(174,103,117,${active ? 0.1 + strength * 0.3 : 0.015})`;
      ctx.lineWidth = 0.4 + strength * 1.1;
      ctx.beginPath(); ctx.moveTo(start.x + 4, start.y); ctx.lineTo(end.x - 4, end.y); ctx.stroke();
    }
  }
  ctx.textAlign = "center"; ctx.font = "8px ui-monospace, monospace"; ctx.fillStyle = "#9ca7b2";
  layerNodes.forEach((nodes, layer) => {
    if (nodes.length) ctx.fillText(layer === 0 ? "Inputs" : layer === lastLayer ? "Outputs" : `Hidden ${layer}`, nodes[0].x, 14);
  });
  for (const node of networkNodes) {
    const value = activations[node.layer]?.[node.index];
    const magnitude = Number.isFinite(value) ? Math.min(1, Math.abs(value)) : 0;
    const active = focused && node.layer === focused.layer && node.index === focused.index;
    const base = value < 0 ? [181, 111, 127] : [86, 140, 179];
    const fill = [244, 246, 248].map((part, index) => Math.round(part + (base[index] - part) * magnitude));
    ctx.fillStyle = `rgb(${fill.join(",")})`;
    ctx.strokeStyle = active ? "#4f789b" : "#abbac7";
    ctx.lineWidth = active ? 1.8 : 0.8;
    const hidden = node.layer > 0 && node.layer < lastLayer;
    const radius = hidden ? Math.max(2.5, Math.min(4, (bottom - top) / Math.max(1, counts[node.layer] - 1) * 0.3)) : 5;
    ctx.beginPath(); ctx.arc(node.x, node.y, radius, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    ctx.font = "8px ui-monospace, monospace"; ctx.fillStyle = "#8c9aa8";
    if (node.layer === 0) { ctx.textAlign = "left"; ctx.fillText(humanName(network.inputs?.[node.index]), 8, node.y - 3); ctx.fillText(scalar(value), 8, node.y + 9); }
    if (node.layer === lastLayer) { ctx.textAlign = "center"; ctx.fillText(humanName(network.outputs?.[node.index]), node.x, node.y + 15); }
  }
  const hiddenSummary = counts.slice(1, -1).join(" + ");
  const summary = `${counts[0]} inputs / ${hiddenSummary} hidden / ${counts[lastLayer]} outputs`;
  $("network-summary").textContent = summary;
  if (focused) {
    const output = focused.layer === lastLayer;
    const outputName = output ? network.outputs?.[focused.index] : null;
    const name = focused.layer === 0 ? humanName(network.inputs?.[focused.index]) : output ? humanName(outputName) : `Hidden ${focused.layer}, neuron ${focused.index + 1}`;
    const bias = focused.layer > 0 ? network.layers?.[focused.layer - 1]?.bias?.[focused.index] : null;
    const value = activations[focused.layer]?.[focused.index];
    const command = output ? ` / command ${outputName === "steering" ? signed(value) : pedal(value)}` : "";
    $("network-info").textContent = `${leader()?.finished ? "Finished. " : ""}${name}: ${signed(value)}${command}${Number.isFinite(bias) ? ` / bias ${signed(bias)}` : ""}`;
  } else {
    const status = leader() && !leader().alive ? `${leader().finished ? "Finished" : leader().crashed ? "Crashed" : "Stopped"}. Diagnostic outputs` : "Live leader";
    $("network-info").textContent = `${network.parameters} parameters / ${status}`;
  }
  const commands = (network.outputs || []).map((name, index) => `${humanName(name)} command ${name === "steering" ? signed(brain?.outputs?.[index]) : pedal(brain?.outputs?.[index])}`).join(", ");
  canvases.network.setAttribute("aria-label", `${network.name} network: ${summary}. ${network.parameters} parameters. ${commands}. Requested target speed ${pedal(leader()?.target_speed)} of maximum. Actual speed ${scalar(leader()?.speed, 1)} pixels per simulation step. Applied steering ${signed(leader()?.steering)}, throttle ${pedal(leader()?.throttle)}, brake ${pedal(leader()?.brake)}. The car accelerates or brakes toward its target speed. Brake takes priority over throttle.`);
}

function drawHistory() {
  if (state?.comparison) { drawComparisonHistory(); return; }
  const { width, height, ctx } = canvasSize("history");
  const history = state?.history || [];
  const area = { left: 48, right: width - 42, top: 15, bottom: height - 22 };
  if (area.right <= area.left || area.bottom <= area.top) return;
  const x = (generation) => area.left + generation / generationLimit * (area.right - area.left);
  const fitnessY = (value) => area.bottom - (value - fitnessDomain.min) / (fitnessDomain.max - fitnessDomain.min) * (area.bottom - area.top);
  const timeY = (value) => area.bottom - value / TIME_LIMIT * (area.bottom - area.top);
  ctx.font = "8px ui-monospace, monospace";
  ctx.fillStyle = "#76a28a"; ctx.textAlign = "right"; ctx.fillText("Time(s)", area.left - 7, 8);
  ctx.fillStyle = "#648cb4"; ctx.textAlign = "left"; ctx.fillText("Fitness", area.right + 7, 8);
  for (const ratio of [0, 0.5, 1]) {
    const y = area.bottom - ratio * (area.bottom - area.top);
    ctx.strokeStyle = "#eef1f4"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(area.left, y); ctx.lineTo(area.right, y); ctx.stroke();
    ctx.fillStyle = "#9ea9b5"; ctx.textAlign = "right";
    ctx.fillText(scalar(ratio * TIME_LIMIT, ratio === 0 ? 0 : 2), area.left - 7, y + 3);
    ctx.textAlign = "left"; ctx.fillText(String(Math.round(fitnessDomain.min + ratio * (fitnessDomain.max - fitnessDomain.min))), area.right + 7, y + 3);
  }
  for (const generation of [0, generationLimit / 2, generationLimit]) { ctx.textAlign = "center"; ctx.fillText(String(generation), x(generation), area.bottom + 15); }
  for (const [field, color, scale] of [["best_fitness", "#648cb4", fitnessY], ["lap_time", "#76a28a", timeY]]) {
    if (!history.length) continue;
    ctx.strokeStyle = color; ctx.lineWidth = 1.6; ctx.beginPath();
    let connected = false;
    for (const entry of history) {
      const value = entry[field];
      // unfinished generations leave real gaps in the lap-time trace.
      if (!Number.isFinite(value)) { connected = false; continue; }
      if (connected) ctx.lineTo(x(entry.generation), scale(value)); else ctx.moveTo(x(entry.generation), scale(value));
      connected = true;
    }
    ctx.stroke();
    const points = field === "lap_time" ? history.filter((entry) => Number.isFinite(entry[field])) : [history[history.length - 1]];
    ctx.fillStyle = color;
    for (const entry of points) if (Number.isFinite(entry[field])) { ctx.beginPath(); ctx.arc(x(entry.generation), scale(entry[field]), 2, 0, Math.PI * 2); ctx.fill(); }
  }
  canvases.history.setAttribute("aria-label", history.length ? `Completed ${history.length} generations. Latest lap time ${timeLabel(history[history.length - 1].lap_time)}. Best lap ${timeLabel(state.metrics?.best_ever_time)}. Lower times are better. Fitness ${scalar(history[history.length - 1].best_fitness, 1)}.` : "No completed generations yet.");
}

function inspectNetwork(id) {
  if (id === state?.network?.id) return;
  action(async () => { acceptState(await request("/api/architecture", { method: "POST", body: JSON.stringify({ id }) })); });
}

function renderComparison() {
  const comparison = state?.comparison;
  const entries = comparison?.entries || [];
  $("workspace").classList.toggle("comparing", !!comparison);
  $("comparison-note").hidden = !comparison;
  $("comparison-results").hidden = !comparison;
  $("history-title").textContent = comparison ? "Comparison" : "Learning";
  $("history-label").textContent = comparison ? `${comparison.completed_generations} / ${comparison.generation_limit} generations` : `Generation ${state?.generation || 0}`;
  const legendKey = comparison ? entries.map((entry) => `${entry.id}/${entry.name}`).join(",") : "normal";
  if (comparisonLegendKey !== legendKey) {
    comparisonLegendKey = legendKey;
    const legend = $("history-legend");
    legend.replaceChildren();
    const items = comparison ? entries.map((entry) => [entry.name, networkColors[entry.id] || "#748492"]) : [["Lap time / lower is better", "#76a28a"], ["Best fitness", "#648cb4"]];
    for (const [label, color] of items) {
      const item = document.createElement("span");
      const key = document.createElement("i");
      key.style.background = color;
      item.append(key, document.createTextNode(label));
      legend.append(item);
    }
  }
  if (!comparison) { comparisonRowsKey = ""; return; }
  $("comparison-note").textContent = `Same track / seed ${comparison.seed} / ${comparison.population} cars per network / ${comparison.generation_limit} generations each. Lower lap times are better.`;
  const ranked = [...entries].sort((a, b) => {
    const aTime = Number.isFinite(a.best_time) ? a.best_time : Infinity;
    const bTime = Number.isFinite(b.best_time) ? b.best_time : Infinity;
    return (aTime - bTime) || a.id.localeCompare(b.id);
  });
  const rowsKey = JSON.stringify([ranked.map((entry) => [entry.id, entry.name, entry.completed_generations, entry.first_lap_generation, entry.best_time]), state.network?.id, comparison.generation_limit]);
  if (comparisonRowsKey !== rowsKey) {
    comparisonRowsKey = rowsKey;
    const fragment = document.createDocumentFragment();
    for (const entry of ranked) {
      const row = document.createElement("tr");
      const selected = entry.id === state.network?.id;
      row.classList.toggle("selected", selected);
      const nameCell = document.createElement("th");
      nameCell.scope = "row";
      const button = document.createElement("button");
      button.type = "button";
      button.disabled = pending;
      button.textContent = entry.name;
      button.title = `Inspect ${entry.name} leader and network`;
      button.setAttribute("aria-pressed", String(selected));
      button.style.setProperty("--trace-color", networkColors[entry.id] || "#748492");
      button.addEventListener("click", () => inspectNetwork(entry.id));
      nameCell.append(button);
      row.append(nameCell);
      for (const text of [`${entry.completed_generations} / ${comparison.generation_limit}`, Number.isInteger(entry.first_lap_generation) ? `Gen ${entry.first_lap_generation}` : "—", timeLabel(entry.best_time)]) {
        const cell = document.createElement("td");
        cell.textContent = text;
        row.append(cell);
      }
      fragment.append(row);
    }
    $("comparison-rows").replaceChildren(fragment);
  }
  const fastest = ranked.find((entry) => Number.isFinite(entry.best_time));
  const tiedNames = fastest ? ranked.filter((entry) => Number.isFinite(entry.best_time) && Math.abs(entry.best_time - fastest.best_time) <= 1e-9).map((entry) => entry.name).join(", ") : "";
  $("comparison-ranking").textContent = fastest ? `${comparison.status === "complete" ? "Fastest lap" : "Best so far"}: ${tiedNames}, ${timeLabel(fastest.best_time)}` : "No completed laps yet.";
}

function drawComparisonHistory() {
  const { width, height, ctx } = canvasSize("history");
  const comparison = state.comparison;
  const area = { left: 48, right: width - 20, top: 20, bottom: height - 27 };
  if (area.right <= area.left || area.bottom <= area.top) return;
  const lastGeneration = Math.max(1, comparison.generation_limit - 1);
  const times = comparison.entries.flatMap((entry) => (entry.history || []).map((point) => point.best_time)).filter(Number.isFinite);
  const fastest = times.length ? Math.min(...times) : 0;
  const slowest = times.length ? Math.max(...times) : TIME_LIMIT;
  const padding = Math.max(0.25, (slowest - fastest) * 0.15);
  // all traces share a fitted scale so small lap-time improvements stay visible.
  const minimum = times.length ? Math.max(0, fastest - padding) : 0;
  const maximum = times.length ? Math.min(TIME_LIMIT, slowest + padding) : TIME_LIMIT;
  const x = (generation) => area.left + generation / lastGeneration * (area.right - area.left);
  const y = (time) => area.bottom - (time - minimum) / (maximum - minimum) * (area.bottom - area.top);
  ctx.font = "8px ui-monospace, monospace";
  ctx.fillStyle = "#8997a5"; ctx.textAlign = "left";
  ctx.fillText("Best lap (s)", area.left, 10);
  for (const ratio of [0, 0.5, 1]) {
    const value = minimum + ratio * (maximum - minimum);
    const lineY = y(value);
    ctx.strokeStyle = "#eef1f4"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(area.left, lineY); ctx.lineTo(area.right, lineY); ctx.stroke();
    ctx.fillStyle = "#8997a5"; ctx.textAlign = "right";
    ctx.fillText(scalar(value, value === 0 ? 0 : 2), area.left - 7, lineY + 3);
  }
  for (const generation of [0, Math.floor(lastGeneration / 2), lastGeneration]) {
    ctx.textAlign = "center"; ctx.fillText(String(generation), x(generation), area.bottom + 14);
  }
  ctx.textAlign = "right"; ctx.fillText("Generation", area.right, height - 2);
  let hasLap = false;
  // best-time traces begin only after a real completed lap exists.
  for (const entry of comparison.entries) {
    ctx.strokeStyle = networkColors[entry.id] || "#748492";
    ctx.fillStyle = ctx.strokeStyle;
    ctx.lineWidth = entry.id === state.network?.id ? 2.3 : 1.5;
    ctx.beginPath();
    let started = false;
    let latest = null;
    for (const point of entry.history || []) {
      if (!Number.isFinite(point.best_time)) continue;
      const pointX = x(point.generation);
      const pointY = y(point.best_time);
      if (started) ctx.lineTo(pointX, pointY); else ctx.moveTo(pointX, pointY);
      started = true;
      hasLap = true;
      latest = { x: pointX, y: pointY };
    }
    ctx.stroke();
    if (latest) { ctx.beginPath(); ctx.arc(latest.x, latest.y, 2.5, 0, Math.PI * 2); ctx.fill(); }
  }
  if (!hasLap) {
    ctx.fillStyle = "#9aa5b0"; ctx.textAlign = "center";
    ctx.fillText("Waiting for the first completed lap", (area.left + area.right) / 2, (area.top + area.bottom) / 2);
  }
  canvases.history.setAttribute("aria-label", `Cumulative best lap times by generation, from 0 to ${lastGeneration}, on the same track. ${comparison.entries.map((entry) => `${entry.name}: ${Number.isFinite(entry.best_time) ? timeLabel(entry.best_time) : "no completed lap"}`).join(". ")}. Lower times are better.`);
}

function drawAll() { drawTrack(); drawNetwork(); drawHistory(); }

async function uploadTrack(file) {
  if (!file) return;
  if (!/\.(png|jpe?g)$/i.test(file.name)) { setStatus("Choose a PNG or JPEG track image.", true); return; }
  if (file.size > 8 * 1024 * 1024) { setStatus("Track images must be 8 MB or smaller.", true); return; }
  await action(async () => {
    setStatus("Reading track…");
    const form = new FormData();
    form.append("file", file);
    const result = await request("/api/track", { method: "POST", body: form });
    acceptState(result);
    setStatus("");
  });
}

function connect() {
  if (reconnectTimer) clearTimeout(reconnectTimer);
  socket = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  socket.addEventListener("open", () => { connected = true; setStatus(""); updateControls(); });
  socket.addEventListener("message", (event) => {
    try { acceptState(JSON.parse(event.data)); }
    catch { setStatus("A live update could not be read.", true); }
  });
  socket.addEventListener("close", () => {
    connected = false;
    updateControls();
    setStatus("Connection lost. Reconnecting…", true);
    reconnectTimer = setTimeout(connect, 1200);
  });
  socket.addEventListener("error", () => { connected = false; updateControls(); });
}

$("upload-button").addEventListener("click", () => $("track-file").click());
$("track-file").addEventListener("change", () => { const file = $("track-file").files[0]; $("track-file").value = ""; uploadTrack(file); });
$("circuit-select").addEventListener("change", () => {
  const id = $("circuit-select").value;
  if (!circuits.some((circuit) => circuit.id === id)) { renderCircuitSelect(); return; }
  action(async () => { acceptState(await request("/api/circuit", { method: "POST", body: JSON.stringify({ id }) })); });
});
$("architecture-select").addEventListener("change", () => {
  const id = $("architecture-select").value;
  if (!architectures.some((architecture) => architecture.id === id)) { renderArchitectureSelect(); return; }
  action(async () => { acceptState(await request("/api/architecture", { method: "POST", body: JSON.stringify({ id }) })); });
});
$("compare-button").addEventListener("click", () => action(async () => { acceptState(await request("/api/comparison", { method: "POST" })); }));
$("single-button").addEventListener("click", () => action(async () => {
  acceptState(await request("/api/reset", { method: "POST" }));
  acceptState(await request("/api/train", { method: "POST", body: JSON.stringify({ running: true }) }));
}));
$("train-button").addEventListener("click", () => action(async () => { acceptState(await request("/api/train", { method: "POST", body: JSON.stringify({ running: !state.running }) })); }));
$("reset-button").addEventListener("click", () => action(async () => { acceptState(await request("/api/reset", { method: "POST" })); }));
$("speed-select").addEventListener("change", () => {
  const speed = Number($("speed-select").value);
  action(async () => { acceptState(await request("/api/speed", { method: "POST", body: JSON.stringify({ speed }) })); });
});
$("show-rays").addEventListener("change", drawTrack);
$("view-network-button").addEventListener("click", () => {
  if (!$("network-dialog").open) $("network-dialog").showModal();
  drawNetwork();
});
$("close-network-button").addEventListener("click", () => $("network-dialog").close());
$("network-dialog").addEventListener("close", () => {
  hoveredNode = null;
  networkNodes = [];
});
$("save-button").addEventListener("click", () => action(async () => {
  const response = await request("/api/model", { raw: true });
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "neural-racer-brain.json";
  document.body.append(anchor); anchor.click(); anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  setStatus("Brain saved.");
}));
canvases.network.addEventListener("pointermove", (event) => {
  const bounds = canvases.network.getBoundingClientRect();
  const x = event.clientX - bounds.left;
  const y = event.clientY - bounds.top;
  hoveredNode = networkNodes.find((node) => Math.hypot(x - node.x, y - node.y) < 9) || null;
  drawNetwork();
});
canvases.network.addEventListener("pointerleave", () => { hoveredNode = null; drawNetwork(); });
window.addEventListener("dragenter", (event) => { event.preventDefault(); if (event.dataTransfer?.types.includes("Files")) { dragDepth++; $("drop-label").hidden = false; } });
window.addEventListener("dragover", (event) => { event.preventDefault(); });
window.addEventListener("dragleave", (event) => { event.preventDefault(); dragDepth = Math.max(0, dragDepth - 1); if (!dragDepth) $("drop-label").hidden = true; });
window.addEventListener("drop", (event) => { event.preventDefault(); dragDepth = 0; $("drop-label").hidden = true; uploadTrack(event.dataTransfer?.files[0]); });
const observer = new ResizeObserver(drawAll);
Object.values(canvases).forEach((canvas) => observer.observe(canvas));
request("/api/state").then(acceptState).catch((error) => setStatus(error.message, true));
loadCircuits();
loadArchitectures();
connect();
