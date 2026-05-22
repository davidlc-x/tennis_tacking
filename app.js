const COURTS = {
  singles: { width: 8.23, length: 23.77 },
  doubles: { width: 10.97, length: 23.77 },
};

const CALIBRATION_LABELS = [
  "Tap near-left corner",
  "Tap near-right corner",
  "Tap far-right corner",
  "Tap far-left corner",
];

const BALL_MODEL = {
  path: "./models/tennis-ball-yolov8n.onnx",
  inputSize: 640,
  confidenceFloor: 0.22,
  minRunGap: 0.18,
};

const state = {
  stream: null,
  streamKind: "none",
  isTracking: false,
  calibrationMode: false,
  calibrationPoints: [],
  homography: null,
  inverseHomography: null,
  tracks: [],
  courtTrail: [],
  shots: [],
  rallies: [],
  activeRally: null,
  rallyCount: 0,
  shotInRally: 0,
  lastShotTime: 0,
  lastFrameTime: 0,
  fpsSamples: [],
  lastDetection: null,
  lastPrediction: null,
  autoCalibration: null,
  model: {
    session: null,
    status: "idle",
    loading: false,
    pending: false,
    lastRunAt: 0,
    lastDetection: null,
    inputName: null,
    outputName: null,
    canvas: document.createElement("canvas"),
    context: null,
  },
  sim: {
    active: false,
    canvas: document.createElement("canvas"),
    context: null,
    stream: null,
    frameRequest: 0,
    startMs: 0,
    currentTruth: null,
    metrics: {
      frames: 0,
      misses: 0,
      detections: 0,
      positionErrors: [],
      speedErrors: [],
      lastPositionError: 0,
      lastSpeedError: 0,
    },
  },
  replay: {
    rally: null,
    playing: false,
    startedAt: 0,
    pausedAt: 0,
    durationMs: 0,
    frameRequest: 0,
  },
  videoRect: { x: 0, y: 0, width: 0, height: 0 },
  stats: {
    peak: 0,
    serve: 0,
    return: 0,
    current: 0,
    confidence: 0,
  },
};

const $ = (id) => document.getElementById(id);

const els = {
  video: $("camera"),
  overlay: $("overlay"),
  processor: $("processor"),
  courtMap: $("courtMap"),
  cameraStatus: $("cameraStatus"),
  calibrationStatus: $("calibrationStatus"),
  calibrationGuide: $("calibrationGuide"),
  calibrationStep: $("calibrationStep"),
  serveSpeed: $("serveSpeed"),
  returnSpeed: $("returnSpeed"),
  currentSpeed: $("currentSpeed"),
  peakSpeed: $("peakSpeed"),
  confidence: $("confidence"),
  positionError: $("positionError"),
  speedError: $("speedError"),
  shotCount: $("shotCount"),
  rallyCount: $("rallyCount"),
  shotList: $("shotList"),
  fpsBadge: $("fpsBadge"),
  replayModal: $("replayModal"),
  replayCanvas: $("replayCanvas"),
  replayTitle: $("replayTitle"),
  replayMeta: $("replayMeta"),
  replayBtn: $("replayBtn"),
  replayPlayBtn: $("replayPlayBtn"),
  replayRestartBtn: $("replayRestartBtn"),
  closeReplayBtn: $("closeReplayBtn"),
  startCameraBtn: $("startCameraBtn"),
  recordBtn: $("recordBtn"),
  autoCalibrateBtn: $("autoCalibrateBtn"),
  calibrateBtn: $("calibrateBtn"),
  simBtn: $("simBtn"),
  resetBtn: $("resetBtn"),
  exportBtn: $("exportBtn"),
  courtMode: $("courtMode"),
  units: $("units"),
  detectorMode: $("detectorMode"),
  detectorStatus: $("detectorStatus"),
  sensitivity: $("sensitivity"),
  minShotSpeed: $("minShotSpeed"),
};

const overlayCtx = els.overlay.getContext("2d", { alpha: true });
const processorCtx = els.processor.getContext("2d", { willReadFrequently: true });
const courtCtx = els.courtMap.getContext("2d");
const replayCtx = els.replayCanvas.getContext("2d");
state.model.context = state.model.canvas.getContext("2d", { willReadFrequently: true });
state.sim.context = state.sim.canvas.getContext("2d");

function setAppHeight() {
  document.documentElement.style.setProperty("--app-height", `${window.innerHeight}px`);
}

function setStatus(el, text, tone = "muted") {
  el.textContent = text;
  el.className = `status-pill ${tone}`;
}

function setDetectorStatus(text, tone = "muted") {
  els.detectorStatus.textContent = text;
  els.detectorStatus.className = tone === "muted" ? "field-note" : `field-note ${tone}`;
}

function courtSize() {
  return COURTS[els.courtMode.value] ?? COURTS.singles;
}

function preferredUnitLabel() {
  return els.units.value === "mph" ? "mph" : "km/h";
}

function convertSpeed(kmh) {
  return els.units.value === "mph" ? kmh * 0.621371 : kmh;
}

function formatSpeed(kmh) {
  if (!Number.isFinite(kmh) || kmh <= 0) return "--";
  return Math.round(convertSpeed(kmh)).toString();
}

function resizeCanvases() {
  const rect = els.overlay.getBoundingClientRect();
  const dpr = Math.max(1, Math.min(window.devicePixelRatio || 1, 2));
  els.overlay.width = Math.max(1, Math.round(rect.width * dpr));
  els.overlay.height = Math.max(1, Math.round(rect.height * dpr));
  overlayCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  updateVideoContentRect();
  drawOverlay();
}

function updateVideoContentRect() {
  const box = els.overlay.getBoundingClientRect();
  const videoW = els.video.videoWidth || 16;
  const videoH = els.video.videoHeight || 9;
  const scale = Math.min(box.width / videoW, box.height / videoH);
  const width = videoW * scale;
  const height = videoH * scale;
  state.videoRect = {
    x: (box.width - width) / 2,
    y: (box.height - height) / 2,
    width,
    height,
  };
}

function normToScreen(point) {
  const rect = state.videoRect;
  return {
    x: rect.x + point.x * rect.width,
    y: rect.y + point.y * rect.height,
  };
}

function screenToNorm(clientX, clientY) {
  const bounds = els.overlay.getBoundingClientRect();
  const x = clientX - bounds.left;
  const y = clientY - bounds.top;
  const rect = state.videoRect;
  return {
    x: clamp((x - rect.x) / rect.width, 0, 1),
    y: clamp((y - rect.y) / rect.height, 0, 1),
  };
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

async function startCamera() {
  if (state.stream) {
    stopCamera();
    return;
  }

  const localHostnames = ["localhost", "127.0.0.1", "::1"];
  if (!window.isSecureContext && !localHostnames.includes(window.location.hostname)) {
    setStatus(els.cameraStatus, "HTTPS needed", "warn");
    return;
  }

  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus(els.cameraStatus, "Camera unavailable", "warn");
    return;
  }

  try {
    const constraints = {
      audio: false,
      video: {
        facingMode: { ideal: "environment" },
        width: { ideal: 1920 },
        height: { ideal: 1080 },
        frameRate: { ideal: 60, max: 60 },
      },
    };
    state.stream = await navigator.mediaDevices.getUserMedia(constraints);
    state.streamKind = "camera";
    els.video.srcObject = state.stream;
    await els.video.play();
    setStatus(els.cameraStatus, "Camera ready", "good");
    els.startCameraBtn.querySelector("span:last-child").textContent = "Stop";
    els.recordBtn.disabled = false;
    els.autoCalibrateBtn.disabled = false;
    els.calibrateBtn.disabled = false;
    resizeCanvases();
    if (els.detectorMode.value !== "color") {
      void ensureBallModel();
    }
    scheduleFrame();
  } catch (error) {
    console.error(error);
    setStatus(els.cameraStatus, "Camera blocked", "warn");
  }
}

function stopCamera() {
  stopSimulation();
  state.isTracking = false;
  state.stream?.getTracks().forEach((track) => track.stop());
  state.stream = null;
  state.streamKind = "none";
  els.video.srcObject = null;
  els.startCameraBtn.querySelector("span:last-child").textContent = "Camera";
  els.recordBtn.disabled = true;
  els.autoCalibrateBtn.disabled = true;
  els.calibrateBtn.disabled = true;
  els.recordBtn.classList.remove("recording");
  els.recordBtn.querySelector("span:last-child").textContent = "Track";
  setStatus(els.cameraStatus, "Camera idle", "muted");
  state.autoCalibration = null;
  renderStats();
  drawOverlay();
}

function toggleTracking() {
  if (!state.stream) return;
  state.isTracking = !state.isTracking;
  els.recordBtn.classList.toggle("recording", state.isTracking);
  els.recordBtn.querySelector("span:last-child").textContent = state.isTracking ? "Pause" : "Track";
  setStatus(els.cameraStatus, state.isTracking ? "Tracking" : "Camera ready", state.isTracking ? "good" : "good");
}

async function startSimulation() {
  if (state.sim.active) {
    stopCamera();
    return;
  }

  if (!state.sim.canvas.captureStream) {
    setStatus(els.cameraStatus, "Sim unavailable", "warn");
    return;
  }

  if (state.stream) stopCamera();

  resetSession();
  resetSimulationMetrics();
  els.detectorMode.value = "color";
  setDetectorStatus("Color tracker", "muted");

  state.sim.canvas.width = 1280;
  state.sim.canvas.height = 720;
  state.sim.active = true;
  state.sim.startMs = performance.now();
  setSyntheticCalibration();
  drawSyntheticFrame(state.sim.startMs);

  state.sim.stream = state.sim.canvas.captureStream(60);
  state.stream = state.sim.stream;
  state.streamKind = "simulation";
  els.video.srcObject = state.stream;
  await els.video.play();

  els.startCameraBtn.querySelector("span:last-child").textContent = "Stop";
  els.simBtn.querySelector("span:last-child").textContent = "Stop";
  els.recordBtn.disabled = false;
  els.autoCalibrateBtn.disabled = false;
  els.calibrateBtn.disabled = false;
  els.recordBtn.classList.add("recording");
  els.recordBtn.querySelector("span:last-child").textContent = "Pause";
  state.isTracking = true;

  setStatus(els.cameraStatus, "Virtual feed", "good");
  resizeCanvases();
  scheduleFrame();
  state.sim.frameRequest = requestAnimationFrame(drawSyntheticFrame);
}

function stopSimulation() {
  if (!state.sim.active) return;
  cancelAnimationFrame(state.sim.frameRequest);
  state.sim.frameRequest = 0;
  state.sim.active = false;
  state.sim.currentTruth = null;
  state.sim.stream = null;
  els.simBtn.querySelector("span:last-child").textContent = "Sim";
}

function setSyntheticCalibration() {
  state.calibrationMode = false;
  state.autoCalibration = null;
  state.calibrationPoints = [
    { x: 0.18, y: 0.91 },
    { x: 0.82, y: 0.91 },
    { x: 0.63, y: 0.25 },
    { x: 0.37, y: 0.25 },
  ];
  finishCalibration("Sim calibrated");
}

function autoCalibrate() {
  if (!state.stream || !els.video.videoWidth || !els.video.videoHeight) return;

  state.calibrationMode = false;
  state.calibrationPoints = [];
  state.autoCalibration = null;
  els.calibrationGuide.hidden = true;
  setStatus(els.calibrationStatus, "Auto scanning", "warn");

  const result = detectCourtCalibrationFromFrame();
  if (!result) {
    setStatus(els.calibrationStatus, "Auto failed", "warn");
    drawOverlay();
    return;
  }

  state.calibrationPoints = result.points;
  state.autoCalibration = result.debug;
  finishCalibration(`Auto ${Math.round(result.confidence * 100)}%`);
}

function beginCalibration() {
  if (!state.stream) return;
  state.calibrationMode = true;
  state.calibrationPoints = [];
  state.autoCalibration = null;
  state.homography = null;
  state.inverseHomography = null;
  els.calibrationGuide.hidden = false;
  els.calibrationStep.textContent = CALIBRATION_LABELS[0];
  setStatus(els.calibrationStatus, "Tap 4 corners", "warn");
  drawOverlay();
}

function handleCalibrationTap(event) {
  if (!state.calibrationMode) return;
  const point = screenToNorm(event.clientX, event.clientY);
  state.calibrationPoints.push(point);

  if (state.calibrationPoints.length < 4) {
    els.calibrationStep.textContent = CALIBRATION_LABELS[state.calibrationPoints.length];
  } else {
    finishCalibration();
  }
  drawOverlay();
}

function finishCalibration(statusText = "Calibrated") {
  const size = courtSize();
  const src = state.calibrationPoints.map((point) => [point.x, point.y]);
  const dst = [
    [0, size.length],
    [size.width, size.length],
    [size.width, 0],
    [0, 0],
  ];
  state.homography = computeHomography(src, dst);
  state.inverseHomography = computeHomography(dst, src);
  if (!state.homography || !state.inverseHomography) {
    state.calibrationMode = false;
    els.calibrationGuide.hidden = true;
    setStatus(els.calibrationStatus, "Retry calibration", "warn");
    drawOverlay();
    return;
  }
  state.calibrationMode = false;
  els.calibrationGuide.hidden = true;
  setStatus(els.calibrationStatus, statusText, "good");
  drawOverlay();
  drawCourtMap();
}

function detectCourtCalibrationFromFrame() {
  const videoW = els.video.videoWidth;
  const videoH = els.video.videoHeight;
  const targetW = 480;
  const targetH = Math.max(1, Math.round((videoH / videoW) * targetW));

  els.processor.width = targetW;
  els.processor.height = targetH;
  processorCtx.drawImage(els.video, 0, 0, targetW, targetH);

  const image = processorCtx.getImageData(0, 0, targetW, targetH);
  const mask = buildCourtLineMask(image.data, targetW, targetH);
  const lines = findHoughLines(mask, targetW, targetH);
  const candidate = chooseCourtCalibration(lines, targetW, targetH);

  if (!candidate) return null;

  return {
    points: candidate.pointsPx.map((point) => ({
      x: clamp(point.x / targetW, 0, 1),
      y: clamp(point.y / targetH, 0, 1),
    })),
    confidence: candidate.confidence,
    debug: {
      confidence: candidate.confidence,
      lines: candidate.lines
        .map((line) => lineSegmentInBounds(line, targetW, targetH))
        .filter(Boolean)
        .map((segment) => segment.map((point) => ({ x: point.x / targetW, y: point.y / targetH }))),
    },
  };
}

function buildCourtLineMask(data, width, height) {
  const mask = new Uint8Array(width * height);

  for (let y = Math.floor(height * 0.06); y < Math.floor(height * 0.98); y += 1) {
    for (let x = 0; x < width; x += 1) {
      const index = (y * width + x) * 4;
      const r = data[index];
      const g = data[index + 1];
      const b = data[index + 2];
      const max = Math.max(r, g, b);
      const min = Math.min(r, g, b);
      const saturation = max === 0 ? 0 : (max - min) / max;
      const luminance = r * 0.299 + g * 0.587 + b * 0.114;
      const brightLine = luminance > 145 && saturation < 0.38;
      const shadowLine = luminance > 118 && min > 82 && max - min < 74;

      if (brightLine || shadowLine) {
        mask[y * width + x] = 1;
      }
    }
  }

  return mask;
}

function findHoughLines(mask, width, height) {
  const points = [];
  const sampleStep = 2;

  for (let y = 2; y < height - 2; y += sampleStep) {
    for (let x = 2; x < width - 2; x += sampleStep) {
      const index = y * width + x;
      if (!mask[index]) continue;
      const edge =
        !mask[index - sampleStep] ||
        !mask[index + sampleStep] ||
        !mask[index - width * sampleStep] ||
        !mask[index + width * sampleStep];
      if (edge) points.push({ x, y });
    }
  }

  if (points.length < 80) return [];

  const maxPoints = 18000;
  const pointStride = Math.max(1, Math.ceil(points.length / maxPoints));
  const thetaStep = 2;
  const thetaCount = Math.floor(180 / thetaStep);
  const diagonal = Math.ceil(Math.hypot(width, height));
  const rhoBins = diagonal * 2 + 1;
  const accumulator = new Uint16Array(thetaCount * rhoBins);
  const trig = [];

  for (let thetaIndex = 0; thetaIndex < thetaCount; thetaIndex += 1) {
    const theta = (thetaIndex * thetaStep * Math.PI) / 180;
    trig.push({ cos: Math.cos(theta), sin: Math.sin(theta) });
  }

  let usedPoints = 0;
  for (let pointIndex = 0; pointIndex < points.length; pointIndex += pointStride) {
    const point = points[pointIndex];
    usedPoints += 1;
    for (let thetaIndex = 0; thetaIndex < thetaCount; thetaIndex += 1) {
      const { cos, sin } = trig[thetaIndex];
      const rho = Math.round(point.x * cos + point.y * sin) + diagonal;
      accumulator[thetaIndex * rhoBins + rho] += 1;
    }
  }

  const minVotes = Math.max(18, Math.round(usedPoints / 90));
  const rawLines = [];
  for (let thetaIndex = 0; thetaIndex < thetaCount; thetaIndex += 1) {
    for (let rhoIndex = 0; rhoIndex < rhoBins; rhoIndex += 1) {
      const votes = accumulator[thetaIndex * rhoBins + rhoIndex];
      if (votes >= minVotes) {
        const theta = thetaIndex * thetaStep;
        const radians = (theta * Math.PI) / 180;
        rawLines.push({
          theta,
          radians,
          cos: Math.cos(radians),
          sin: Math.sin(radians),
          rho: rhoIndex - diagonal,
          votes,
        });
      }
    }
  }

  rawLines.sort((a, b) => b.votes - a.votes);

  const lines = [];
  for (const line of rawLines) {
    const duplicate = lines.some((existing) => {
      const thetaDistance = Math.min(
        Math.abs(existing.theta - line.theta),
        180 - Math.abs(existing.theta - line.theta),
      );
      return thetaDistance < 6 && Math.abs(existing.rho - line.rho) < 18;
    });
    if (!duplicate) lines.push(line);
    if (lines.length >= 44) break;
  }

  return lines;
}

function chooseCourtCalibration(lines, width, height) {
  const centerX = width / 2;
  const nearY = height * 0.92;
  const farY = height * 0.16;
  const horizontals = [];
  const leftSides = [];
  const rightSides = [];

  for (const line of lines) {
    const horizontalAngle = angleFromHorizontal(line);
    const yCenter = yAtX(line, centerX);

    if (horizontalAngle < 20 && Number.isFinite(yCenter) && yCenter > height * 0.04 && yCenter < height * 1.02) {
      horizontals.push({ line, y: yCenter, score: line.votes });
      continue;
    }

    if (horizontalAngle < 28) continue;

    const xNear = xAtY(line, nearY);
    const xFar = xAtY(line, farY);
    if (!Number.isFinite(xNear) || !Number.isFinite(xFar)) continue;
    if (xNear < -width * 0.45 || xNear > width * 1.45) continue;
    if (xFar < -width * 0.55 || xFar > width * 1.55) continue;

    const candidate = {
      line,
      xNear,
      xFar,
      score: line.votes + Math.abs(xNear - xFar) * 0.12,
    };

    if (xNear < centerX) leftSides.push(candidate);
    if (xNear > centerX) rightSides.push(candidate);
  }

  horizontals.sort((a, b) => b.score - a.score);
  leftSides.sort((a, b) => b.score - a.score);
  rightSides.sort((a, b) => b.score - a.score);

  const hCandidates = horizontals.slice(0, 14);
  const lCandidates = leftSides.slice(0, 10);
  const rCandidates = rightSides.slice(0, 10);
  let best = null;

  for (const left of lCandidates) {
    for (const right of rCandidates) {
      if (Math.abs(left.xNear - right.xNear) < width * 0.24) continue;
      for (let i = 0; i < hCandidates.length; i += 1) {
        for (let j = i + 1; j < hCandidates.length; j += 1) {
          const a = hCandidates[i];
          const b = hCandidates[j];
          if (Math.abs(a.y - b.y) < height * 0.22) continue;

          const near = a.y > b.y ? a : b;
          const far = a.y > b.y ? b : a;
          const pointsPx = [
            intersectLines(left.line, near.line),
            intersectLines(right.line, near.line),
            intersectLines(right.line, far.line),
            intersectLines(left.line, far.line),
          ];

          if (pointsPx.some((point) => !point || !Number.isFinite(point.x) || !Number.isFinite(point.y))) continue;

          const score = scoreCourtCandidate(pointsPx, [left.line, right.line, near.line, far.line], width, height);
          if (!score.valid) continue;

          const totalVoteScore = left.line.votes + right.line.votes + near.line.votes + far.line.votes;
          const confidence = clamp(score.score * 0.68 + clamp(totalVoteScore / 700, 0, 1) * 0.32, 0, 1);
          const candidate = {
            pointsPx,
            lines: [left.line, right.line, near.line, far.line],
            confidence,
            score: confidence,
          };

          if (!best || candidate.score > best.score) best = candidate;
        }
      }
    }
  }

  return best && best.confidence > 0.34 ? best : null;
}

function scoreCourtCandidate(points, lines, width, height) {
  const [nearLeft, nearRight, farRight, farLeft] = points;
  const nearWidth = Math.hypot(nearRight.x - nearLeft.x, nearRight.y - nearLeft.y);
  const farWidth = Math.hypot(farRight.x - farLeft.x, farRight.y - farLeft.y);
  const nearMid = midpoint(nearLeft, nearRight);
  const farMid = midpoint(farLeft, farRight);
  const courtHeight = nearMid.y - farMid.y;
  const area = polygonArea(points);
  const inFrameCount = points.filter((point) => pointInsideExpanded(point, width, height, 0.16)).length;

  if (nearLeft.x >= nearRight.x || farLeft.x >= farRight.x) return { valid: false, score: 0 };
  if (courtHeight < height * 0.24) return { valid: false, score: 0 };
  if (nearWidth < width * 0.22 || farWidth < width * 0.06) return { valid: false, score: 0 };
  if (area < width * height * 0.055) return { valid: false, score: 0 };
  if (inFrameCount < 3) return { valid: false, score: 0 };

  const centerScore = 1 - clamp(Math.abs((nearMid.x + farMid.x) / 2 - width / 2) / (width * 0.55), 0, 1);
  const heightScore = clamp(courtHeight / (height * 0.64), 0, 1);
  const widthScore = clamp(nearWidth / (width * 0.82), 0, 1);
  const areaScore = clamp(area / (width * height * 0.38), 0, 1);
  const perspectiveScore = clamp(farWidth / nearWidth, 0.18, 1.25) / 1.25;
  const bottomScore = clamp((nearMid.y / height - 0.36) / 0.56, 0, 1);
  const parallelScore = lineParallelScore(lines[2], lines[3]);

  return {
    valid: true,
    score:
      centerScore * 0.16 +
      heightScore * 0.22 +
      widthScore * 0.12 +
      areaScore * 0.2 +
      perspectiveScore * 0.1 +
      bottomScore * 0.1 +
      parallelScore * 0.1,
  };
}

function resetSession() {
  closeReplay();
  state.tracks = [];
  state.courtTrail = [];
  state.shots = [];
  state.rallies = [];
  state.activeRally = null;
  state.rallyCount = 0;
  state.shotInRally = 0;
  state.lastShotTime = 0;
  state.lastDetection = null;
  state.lastPrediction = null;
  state.model.lastDetection = null;
  state.stats = { peak: 0, serve: 0, return: 0, current: 0, confidence: 0 };
  resetSimulationMetrics();
  renderStats();
  updateReplayButton();
  drawOverlay();
  drawCourtMap();
}

function resetSimulationMetrics() {
  state.sim.metrics = {
    frames: 0,
    misses: 0,
    detections: 0,
    positionErrors: [],
    speedErrors: [],
    lastPositionError: 0,
    lastSpeedError: 0,
  };
}

function drawSyntheticFrame(nowMs) {
  if (!state.sim.active && nowMs !== state.sim.startMs) return;

  const canvas = state.sim.canvas;
  const ctx = state.sim.context;
  const width = canvas.width;
  const height = canvas.height;
  const elapsed = Math.max(0, (nowMs - state.sim.startMs) / 1000);
  const truth = syntheticTruthAt(elapsed);
  state.sim.currentTruth = truth;

  ctx.fillStyle = "#202d2a";
  ctx.fillRect(0, 0, width, height);
  drawSyntheticBackdrop(ctx, width, height);
  drawSyntheticCourt(ctx, width, height);
  drawSyntheticPlayers(ctx, width, height, truth);
  drawSyntheticBall(ctx, width, height, truth);

  if (state.sim.active) {
    state.sim.frameRequest = requestAnimationFrame(drawSyntheticFrame);
  }
}

function syntheticTruthAt(elapsed) {
  const size = courtSize();
  const segments = [
    { duration: 0.5, from: { x: size.width * 0.72, y: size.length - 0.6 }, to: { x: size.width * 0.28, y: size.length * 0.32 } },
    { duration: 0.62, from: { x: size.width * 0.28, y: size.length * 0.32 }, to: { x: size.width * 0.82, y: size.length - 2.2 } },
    { duration: 0.58, from: { x: size.width * 0.82, y: size.length - 2.2 }, to: { x: size.width * 0.18, y: 2.8 } },
    { duration: 0.64, from: { x: size.width * 0.18, y: 2.8 }, to: { x: size.width * 0.55, y: size.length - 4.6 } },
    { duration: 0.55, from: { x: size.width * 0.55, y: size.length - 4.6 }, to: { x: size.width * 0.35, y: 1.9 } },
    { duration: 1.1, from: { x: size.width * 0.35, y: 1.9 }, to: { x: size.width * 0.35, y: 1.9 }, pause: true },
  ];

  const cycle = segments.reduce((sum, segment) => sum + segment.duration, 0);
  let t = elapsed % cycle;
  for (let index = 0; index < segments.length; index += 1) {
    const segment = segments[index];
    if (t <= segment.duration) {
      const amount = segment.pause ? 0 : clamp(t / segment.duration, 0, 1);
      const court = {
        x: segment.from.x + (segment.to.x - segment.from.x) * amount,
        y: segment.from.y + (segment.to.y - segment.from.y) * amount,
      };
      const distanceMeters = Math.hypot(segment.to.x - segment.from.x, segment.to.y - segment.from.y);
      const speedKmh = segment.pause ? 0 : (distanceMeters / segment.duration) * 3.6;
      return {
        court,
        speedKmh,
        segmentIndex: index + 1,
        progress: t / segment.duration,
      };
    }
    t -= segment.duration;
  }

  return { court: segments[0].from, speedKmh: 0, segmentIndex: 1, progress: 0 };
}

function drawSyntheticBackdrop(ctx, width, height) {
  const gradient = ctx.createLinearGradient(0, 0, 0, height);
  gradient.addColorStop(0, "#27312f");
  gradient.addColorStop(1, "#111615");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, width, height);

  ctx.fillStyle = "rgba(255, 255, 255, 0.08)";
  for (let i = 0; i < 10; i += 1) {
    const x = 80 + i * 130;
    ctx.fillRect(x, 0, 3, height);
  }
}

function drawSyntheticCourt(ctx, width, height) {
  const court = courtSize();
  const courtCorners = [
    { x: 0, y: 0 },
    { x: court.width, y: 0 },
    { x: court.width, y: court.length },
    { x: 0, y: court.length },
  ].map(courtPointToSyntheticPixel);

  ctx.beginPath();
  courtCorners.forEach((point, index) => {
    if (index === 0) ctx.moveTo(point.x, point.y);
    else ctx.lineTo(point.x, point.y);
  });
  ctx.closePath();
  ctx.fillStyle = "#2d7a62";
  ctx.fill();

  ctx.strokeStyle = "rgba(245, 244, 232, 0.94)";
  ctx.lineWidth = 5;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  drawSyntheticCourtLine(ctx, { x: 0, y: 0 }, { x: court.width, y: 0 });
  drawSyntheticCourtLine(ctx, { x: court.width, y: 0 }, { x: court.width, y: court.length });
  drawSyntheticCourtLine(ctx, { x: court.width, y: court.length }, { x: 0, y: court.length });
  drawSyntheticCourtLine(ctx, { x: 0, y: court.length }, { x: 0, y: 0 });

  const netY = court.length / 2;
  const service = 6.4;
  drawSyntheticCourtLine(ctx, { x: 0, y: netY }, { x: court.width, y: netY });
  drawSyntheticCourtLine(ctx, { x: 0, y: netY - service }, { x: court.width, y: netY - service });
  drawSyntheticCourtLine(ctx, { x: 0, y: netY + service }, { x: court.width, y: netY + service });
  drawSyntheticCourtLine(ctx, { x: court.width / 2, y: netY - service }, { x: court.width / 2, y: netY + service });

  ctx.strokeStyle = "rgba(10, 12, 12, 0.34)";
  ctx.lineWidth = 3;
  drawSyntheticCourtLine(ctx, { x: -0.4, y: netY }, { x: court.width + 0.4, y: netY });

  if (court.width > COURTS.singles.width) {
    const alley = (court.width - COURTS.singles.width) / 2;
    ctx.strokeStyle = "rgba(245, 244, 232, 0.82)";
    ctx.lineWidth = 4;
    drawSyntheticCourtLine(ctx, { x: alley, y: 0 }, { x: alley, y: court.length });
    drawSyntheticCourtLine(ctx, { x: court.width - alley, y: 0 }, { x: court.width - alley, y: court.length });
  }

  ctx.fillStyle = "rgba(0, 0, 0, 0.22)";
  ctx.fillRect(0, height * 0.86, width, height * 0.14);
}

function drawSyntheticCourtLine(ctx, a, b) {
  const pa = courtPointToSyntheticPixel(a);
  const pb = courtPointToSyntheticPixel(b);
  ctx.beginPath();
  ctx.moveTo(pa.x, pa.y);
  ctx.lineTo(pb.x, pb.y);
  ctx.stroke();
}

function drawSyntheticPlayers(ctx, width, height, truth) {
  const size = courtSize();
  const near = courtPointToSyntheticPixel({ x: size.width * 0.65, y: size.length - 1.2 });
  const far = courtPointToSyntheticPixel({ x: size.width * 0.36, y: 1.7 });
  drawSyntheticPlayer(ctx, near, 34, "#74b9ff");
  drawSyntheticPlayer(ctx, far, 18, "#ffbf55");

  const truthGround = courtPointToSyntheticPixel(truth.court);
  ctx.fillStyle = "rgba(0, 0, 0, 0.18)";
  ctx.beginPath();
  ctx.ellipse(truthGround.x, truthGround.y + 8, 16, 5, 0, 0, Math.PI * 2);
  ctx.fill();
}

function drawSyntheticPlayer(ctx, point, size, color) {
  ctx.fillStyle = "rgba(0, 0, 0, 0.22)";
  ctx.beginPath();
  ctx.ellipse(point.x, point.y + size * 0.8, size * 0.7, size * 0.22, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(point.x, point.y, size * 0.28, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillRect(point.x - size * 0.22, point.y + size * 0.18, size * 0.44, size * 0.9);
}

function drawSyntheticBall(ctx, width, height, truth) {
  const point = courtPointToSyntheticPixel(truth.court);
  const size = courtSize();
  const radius = 5 + (truth.court.y / size.length) * 8;
  ctx.fillStyle = "#d7fa5f";
  ctx.beginPath();
  ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
  ctx.fill();

  ctx.strokeStyle = "rgba(20, 26, 15, 0.64)";
  ctx.lineWidth = Math.max(1, radius * 0.18);
  ctx.beginPath();
  ctx.arc(point.x + radius * 0.15, point.y, radius * 0.62, -1.2, 1.2);
  ctx.stroke();

  ctx.fillStyle = "rgba(255, 255, 255, 0.65)";
  ctx.beginPath();
  ctx.arc(point.x - radius * 0.28, point.y - radius * 0.3, radius * 0.24, 0, Math.PI * 2);
  ctx.fill();
}

function courtPointToSyntheticPixel(point) {
  const norm = courtToNorm(point);
  return {
    x: norm.x * state.sim.canvas.width,
    y: norm.y * state.sim.canvas.height,
  };
}

function scheduleFrame() {
  if (!state.stream) return;
  if ("requestVideoFrameCallback" in HTMLVideoElement.prototype) {
    els.video.requestVideoFrameCallback((now, metadata) => {
      processFrame(metadata.mediaTime || now / 1000);
      scheduleFrame();
    });
  } else {
    requestAnimationFrame((now) => {
      processFrame(now / 1000);
      scheduleFrame();
    });
  }
}

function processFrame(timestamp) {
  if (!els.video.videoWidth || !els.video.videoHeight) return;

  updateFps(timestamp);

  if (!state.isTracking || !state.homography) {
    drawOverlay();
    return;
  }

  if (state.sim.active) {
    state.sim.metrics.frames += 1;
  }

  queueModelDetection(timestamp);
  const detection = detectBall(timestamp);
  if (detection) {
    acceptDetection(detection);
  } else {
    state.stats.confidence = Math.max(0, state.stats.confidence - 0.015);
    if (state.sim.active) {
      state.sim.metrics.misses += 1;
    }
  }

  drawOverlay();
  drawCourtMap();
  renderStats();
}

function updateFps(timestamp) {
  if (state.lastFrameTime) {
    const dt = timestamp - state.lastFrameTime;
    if (dt > 0) {
      state.fpsSamples.push(1 / dt);
      if (state.fpsSamples.length > 40) state.fpsSamples.shift();
      const fps = average(state.fpsSamples);
      els.fpsBadge.textContent = `${Math.round(fps)} fps`;
    }
  }
  state.lastFrameTime = timestamp;
}

function detectBall(timestamp) {
  const mode = els.detectorMode.value;
  const candidates = [];

  if (mode !== "model") {
    const colorDetection = detectBallByColor(timestamp);
    if (colorDetection) {
      candidates.push({ ...colorDetection, source: "color", score: colorDetection.score * 0.88 });
    }
  }

  if (mode !== "color") {
    const modelDetection = recentModelDetection(timestamp);
    if (modelDetection) {
      candidates.push({ ...modelDetection, source: "model", score: modelDetection.score * 1.12 });
    }
  }

  if (!candidates.length) return null;

  const predicted = predictNextNorm(timestamp);
  let best = null;
  let bestScore = -Infinity;

  for (const candidate of candidates) {
    const continuity = predicted ? 1 - clamp(distance(candidate.norm, predicted) / 0.18, 0, 1) : 0.55;
    const sourceBonus = candidate.source === "model" ? 0.16 : 0;
    const agePenalty = candidate.age ? clamp(candidate.age / 0.28, 0, 1) * 0.18 : 0;
    const score = candidate.score * 0.72 + continuity * 0.28 + sourceBonus - agePenalty;
    if (score > bestScore) {
      bestScore = score;
      best = { ...candidate, score: clamp(score, 0, 1) };
    }
  }

  return best;
}

function detectBallByColor(timestamp) {
  const videoW = els.video.videoWidth;
  const videoH = els.video.videoHeight;
  const targetW = 360;
  const targetH = Math.max(1, Math.round((videoH / videoW) * targetW));
  if (els.processor.width !== targetW || els.processor.height !== targetH) {
    els.processor.width = targetW;
    els.processor.height = targetH;
  }

  processorCtx.drawImage(els.video, 0, 0, targetW, targetH);
  const image = processorCtx.getImageData(0, 0, targetW, targetH);
  const data = image.data;
  const sensitivity = Number(els.sensitivity.value) / 100;
  const hueTolerance = 16 + sensitivity * 24;
  const minSat = 0.26 + (1 - sensitivity) * 0.22;
  const minVal = 0.34 + (1 - sensitivity) * 0.16;
  const components = findBallComponents(data, targetW, targetH, hueTolerance, minSat, minVal);
  if (!components.length) return null;

  const last = state.lastDetection;
  const predicted = predictNextNorm(timestamp);
  let best = null;
  let bestScore = -Infinity;

  for (const component of components) {
    const norm = {
      x: component.cx / targetW,
      y: component.cy / targetH,
    };
    const court = applyHomography(state.homography, [norm.x, norm.y]);
    const inCourtBonus = court && court.x >= -2 && court.x <= courtSize().width + 2 && court.y >= -4 && court.y <= courtSize().length + 4 ? 0.3 : -0.4;
    const continuity = predicted ? 1 - clamp(distance(norm, predicted) / 0.22, 0, 1) : last ? 1 - clamp(distance(norm, last.norm) / 0.28, 0, 1) : 0.3;
    const areaScore = 1 - clamp(Math.abs(component.area - 42) / 140, 0, 1);
    const roundness = 1 - clamp(Math.abs(component.aspect - 1), 0, 1);
    const score = component.colorScore * 0.45 + continuity * 0.32 + areaScore * 0.12 + roundness * 0.11 + inCourtBonus;

    if (score > bestScore) {
      bestScore = score;
      best = { ...component, norm, court, score: clamp(score / 1.5, 0, 1), timestamp };
    }
  }

  return best;
}

function recentModelDetection(timestamp) {
  const candidate = state.model.lastDetection;
  if (!candidate) return null;
  const age = Math.max(0, timestamp - candidate.timestamp);
  if (age > 0.32) return null;
  return { ...candidate, age, timestamp };
}

async function ensureBallModel() {
  if (state.model.session || state.model.loading) return state.model.session;
  if (!window.ort) {
    state.model.status = "unavailable";
    setDetectorStatus("Model unavailable", "warn");
    return null;
  }

  state.model.loading = true;
  state.model.status = "loading";
  setDetectorStatus("Loading model", "warn");

  try {
    window.ort.env.wasm.wasmPaths = new URL("./vendor/", window.location.href).href;
    window.ort.env.wasm.numThreads = 1;
    const session = await window.ort.InferenceSession.create(BALL_MODEL.path, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    state.model.session = session;
    state.model.inputName = session.inputNames[0];
    state.model.outputName = session.outputNames[0];
    state.model.status = "ready";
    setDetectorStatus("YOLO ready", "good");
    return session;
  } catch (error) {
    console.warn("Ball model failed to load", error);
    state.model.status = "failed";
    setDetectorStatus("Model failed", "warn");
    els.detectorStatus.title = String(error?.message || error);
    return null;
  } finally {
    state.model.loading = false;
  }
}

function queueModelDetection(timestamp) {
  const mode = els.detectorMode.value;
  if (mode === "color") return;

  if (!state.model.session) {
    void ensureBallModel();
    return;
  }

  if (state.model.pending) return;
  if (timestamp - state.model.lastRunAt < BALL_MODEL.minRunGap) return;

  state.model.pending = true;
  state.model.lastRunAt = timestamp;
  runModelDetection(timestamp).finally(() => {
    state.model.pending = false;
  });
}

async function runModelDetection(timestamp) {
  const prepared = prepareModelInput();
  if (!prepared) return;

  try {
    const tensor = new window.ort.Tensor("float32", prepared.input, [
      1,
      3,
      BALL_MODEL.inputSize,
      BALL_MODEL.inputSize,
    ]);
    const outputMap = await state.model.session.run({ [state.model.inputName]: tensor });
    const output = outputMap[state.model.outputName] ?? Object.values(outputMap)[0];
    const detection = decodeYoloOutput(output, prepared, timestamp);

    if (detection) {
      state.model.lastDetection = detection;
      if (state.model.status !== "ready") {
        state.model.status = "ready";
        setDetectorStatus("YOLO ready", "good");
      }
    }
  } catch (error) {
    console.warn("Ball model inference failed", error);
    state.model.status = "failed";
    setDetectorStatus("Model failed", "warn");
    els.detectorStatus.title = String(error?.message || error);
  }
}

function prepareModelInput() {
  const videoW = els.video.videoWidth;
  const videoH = els.video.videoHeight;
  if (!videoW || !videoH) return null;

  const size = BALL_MODEL.inputSize;
  const canvas = state.model.canvas;
  const ctx = state.model.context;
  if (canvas.width !== size || canvas.height !== size) {
    canvas.width = size;
    canvas.height = size;
  }

  const scale = Math.min(size / videoW, size / videoH);
  const drawW = videoW * scale;
  const drawH = videoH * scale;
  const padX = (size - drawW) / 2;
  const padY = (size - drawH) / 2;

  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, size, size);
  ctx.drawImage(els.video, padX, padY, drawW, drawH);

  const { data } = ctx.getImageData(0, 0, size, size);
  const planeSize = size * size;
  const input = new Float32Array(planeSize * 3);

  for (let i = 0, pixel = 0; i < data.length; i += 4, pixel += 1) {
    input[pixel] = data[i] / 255;
    input[planeSize + pixel] = data[i + 1] / 255;
    input[planeSize * 2 + pixel] = data[i + 2] / 255;
  }

  return { input, scale, padX, padY, videoW, videoH, size };
}

function decodeYoloOutput(output, prepared, timestamp) {
  if (!output?.data || !output?.dims?.length) return null;

  const detections = extractYoloDetections(output, prepared);
  if (!detections.length) return null;

  const predicted = predictNextNorm(timestamp);
  const last = state.lastDetection;
  let best = null;
  let bestScore = -Infinity;

  for (const detection of detections) {
    const court = applyHomography(state.homography, [detection.norm.x, detection.norm.y]);
    if (!court) continue;

    const inCourtBonus =
      court.x >= -2 && court.x <= courtSize().width + 2 && court.y >= -4 && court.y <= courtSize().length + 4
        ? 0.2
        : -0.35;
    const continuity = predicted
      ? 1 - clamp(distance(detection.norm, predicted) / 0.2, 0, 1)
      : last
        ? 1 - clamp(distance(detection.norm, last.norm) / 0.32, 0, 1)
        : 0.45;
    const sizeScore = 1 - clamp(Math.abs(detection.radius - 8) / 22, 0, 1);
    const score = detection.confidence * 0.68 + continuity * 0.2 + sizeScore * 0.12 + inCourtBonus;

    if (score > bestScore) {
      bestScore = score;
      best = {
        norm: detection.norm,
        court,
        timestamp,
        score: clamp(score, 0, 1),
        radius: detection.radius,
        bbox: detection.bbox,
      };
    }
  }

  return best;
}

function extractYoloDetections(output, prepared) {
  const dims = output.dims;
  const data = output.data;
  const detections = [];

  if (dims.length === 3) {
    const a = dims[1];
    const b = dims[2];
    const channelFirst = a <= 16 && b > a;
    const rows = channelFirst ? b : a;
    const channels = channelFirst ? a : b;

    for (let row = 0; row < rows; row += 1) {
      const values = [];
      for (let channel = 0; channel < channels; channel += 1) {
        values.push(channelFirst ? data[channel * rows + row] : data[row * channels + channel]);
      }
      const detection = yoloValuesToDetection(values, prepared);
      if (detection) detections.push(detection);
    }
  } else if (dims.length === 2) {
    const rows = dims[0];
    const channels = dims[1];
    for (let row = 0; row < rows; row += 1) {
      const values = Array.from(data.slice(row * channels, row * channels + channels));
      const detection = yoloValuesToDetection(values, prepared);
      if (detection) detections.push(detection);
    }
  }

  detections.sort((a, b) => b.confidence - a.confidence);
  return nonMaxSuppress(detections, 0.45).slice(0, 8);
}

function yoloValuesToDetection(values, prepared) {
  if (values.length < 5) return null;

  let confidence = values[4];
  if (values.length > 5) {
    confidence = Math.max(...values.slice(4));
  }
  if (!Number.isFinite(confidence) || confidence < BALL_MODEL.confidenceFloor) return null;

  const maxCoord = Math.max(Math.abs(values[0]), Math.abs(values[1]), Math.abs(values[2]), Math.abs(values[3]));
  const coordScale = maxCoord <= 2 ? prepared.size : 1;

  let x1;
  let y1;
  let x2;
  let y2;

  const maybeXyxy = values[2] > values[0] && values[3] > values[1] && values[2] - values[0] < prepared.size * 0.55;
  if (maybeXyxy && values.length >= 6) {
    x1 = values[0] * coordScale;
    y1 = values[1] * coordScale;
    x2 = values[2] * coordScale;
    y2 = values[3] * coordScale;
  } else {
    const cx = values[0] * coordScale;
    const cy = values[1] * coordScale;
    const w = Math.abs(values[2] * coordScale);
    const h = Math.abs(values[3] * coordScale);
    x1 = cx - w / 2;
    y1 = cy - h / 2;
    x2 = cx + w / 2;
    y2 = cy + h / 2;
  }

  const unboxed = unletterboxBox({ x1, y1, x2, y2 }, prepared);
  if (!unboxed) return null;

  const width = unboxed.x2 - unboxed.x1;
  const height = unboxed.y2 - unboxed.y1;
  if (width < 1 || height < 1 || width > prepared.videoW * 0.18 || height > prepared.videoH * 0.18) return null;

  return {
    confidence,
    bbox: unboxed,
    radius: Math.max(width, height) / 2,
    norm: {
      x: clamp((unboxed.x1 + unboxed.x2) / 2 / prepared.videoW, 0, 1),
      y: clamp((unboxed.y1 + unboxed.y2) / 2 / prepared.videoH, 0, 1),
    },
  };
}

function unletterboxBox(box, prepared) {
  const x1 = clamp((box.x1 - prepared.padX) / prepared.scale, 0, prepared.videoW);
  const y1 = clamp((box.y1 - prepared.padY) / prepared.scale, 0, prepared.videoH);
  const x2 = clamp((box.x2 - prepared.padX) / prepared.scale, 0, prepared.videoW);
  const y2 = clamp((box.y2 - prepared.padY) / prepared.scale, 0, prepared.videoH);

  if (x2 <= x1 || y2 <= y1) return null;
  return { x1, y1, x2, y2 };
}

function nonMaxSuppress(detections, threshold) {
  const kept = [];

  for (const detection of detections) {
    const overlaps = kept.some((item) => boxIou(item.bbox, detection.bbox) > threshold);
    if (!overlaps) kept.push(detection);
  }

  return kept;
}

function boxIou(a, b) {
  const x1 = Math.max(a.x1, b.x1);
  const y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2);
  const y2 = Math.min(a.y2, b.y2);
  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = Math.max(0, a.x2 - a.x1) * Math.max(0, a.y2 - a.y1);
  const areaB = Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1);
  const union = areaA + areaB - intersection;
  return union <= 0 ? 0 : intersection / union;
}

function findBallComponents(data, width, height, hueTolerance, minSat, minVal) {
  const mask = new Uint8Array(width * height);
  const visited = new Uint8Array(width * height);
  const centerHue = 70;
  const step = 2;

  for (let y = 0; y < height; y += step) {
    for (let x = 0; x < width; x += step) {
      const idx = (y * width + x) * 4;
      const r = data[idx];
      const g = data[idx + 1];
      const b = data[idx + 2];
      const hsv = rgbToHsv(r, g, b);
      const hueDelta = Math.min(Math.abs(hsv.h - centerHue), 360 - Math.abs(hsv.h - centerHue));
      const greenYellow = hueDelta <= hueTolerance && hsv.s >= minSat && hsv.v >= minVal;
      const brightBall = g > r * 0.86 && g > b * 1.08 && r > 95 && g > 110;
      if (greenYellow || brightBall) {
        mask[y * width + x] = 1;
      }
    }
  }

  const components = [];
  const queue = [];
  for (let y = 0; y < height; y += step) {
    for (let x = 0; x < width; x += step) {
      const start = y * width + x;
      if (!mask[start] || visited[start]) continue;
      let area = 0;
      let sumX = 0;
      let sumY = 0;
      let minX = x;
      let maxX = x;
      let minY = y;
      let maxY = y;
      let hueFit = 0;
      queue.length = 0;
      queue.push(start);
      visited[start] = 1;

      while (queue.length) {
        const current = queue.pop();
        const cx = current % width;
        const cy = Math.floor(current / width);
        const pixelIdx = current * 4;
        const hsv = rgbToHsv(data[pixelIdx], data[pixelIdx + 1], data[pixelIdx + 2]);
        const hueDelta = Math.min(Math.abs(hsv.h - centerHue), 360 - Math.abs(hsv.h - centerHue));
        hueFit += 1 - clamp(hueDelta / 60, 0, 1);
        area += 1;
        sumX += cx;
        sumY += cy;
        minX = Math.min(minX, cx);
        maxX = Math.max(maxX, cx);
        minY = Math.min(minY, cy);
        maxY = Math.max(maxY, cy);

        const neighbors = [current - step, current + step, current - width * step, current + width * step];
        for (const next of neighbors) {
          if (next < 0 || next >= mask.length || visited[next] || !mask[next]) continue;
          const nx = next % width;
          const ny = Math.floor(next / width);
          if (Math.abs(nx - cx) > step || Math.abs(ny - cy) > step) continue;
          visited[next] = 1;
          queue.push(next);
        }
      }

      if (area < 3 || area > 520) continue;
      const boxW = maxX - minX + step;
      const boxH = maxY - minY + step;
      if (boxW < 2 || boxH < 2 || boxW > 80 || boxH > 80) continue;
      const aspect = boxW / boxH;
      if (aspect < 0.25 || aspect > 4) continue;
      components.push({
        area,
        cx: sumX / area,
        cy: sumY / area,
        radius: Math.max(boxW, boxH) / 2,
        aspect,
        colorScore: clamp(hueFit / area, 0, 1),
      });
    }
  }

  return components.sort((a, b) => b.area - a.area).slice(0, 16);
}

function acceptDetection(detection) {
  const court = detection.court;
  if (!court) return;

  const last = state.tracks.at(-1);
  let speedKmh = 0;
  if (last) {
    const dt = detection.timestamp - last.timestamp;
    if (dt > 0.005 && dt < 0.28) {
      const meters = Math.hypot(court.x - last.court.x, court.y - last.court.y);
      const rawSpeed = (meters / dt) * 3.6;
      const plausible = rawSpeed < 260;
      speedKmh = plausible ? smoothSpeed(rawSpeed) : state.stats.current * 0.72;
    }
  }

  const sample = {
    norm: detection.norm,
    court,
    timestamp: detection.timestamp,
    speedKmh,
    confidence: detection.score,
    radius: detection.radius,
    source: detection.source ?? "color",
    bbox: detection.bbox ?? null,
  };

  state.tracks.push(sample);
  if (state.tracks.length > 180) state.tracks.shift();
  state.courtTrail.push(sample);
  if (state.courtTrail.length > 220) state.courtTrail.shift();
  state.lastDetection = sample;
  state.stats.current = speedKmh || state.stats.current * 0.9;
  state.stats.peak = Math.max(state.stats.peak, speedKmh);
  state.stats.confidence = state.stats.confidence * 0.82 + detection.score * 0.18;
  appendRallySample(sample);
  recordSimulationAccuracy(sample);
  maybeRegisterShot(sample);
}

function appendRallySample(sample) {
  const lastSampleAt = state.activeRally?.lastSampleAt ?? 0;
  const gap = sample.timestamp - lastSampleAt;
  if (!state.activeRally || gap > 5) {
    const nextId = (state.rallies.at(-1)?.id ?? 0) + 1;
    state.activeRally = {
      id: nextId,
      startedAt: sample.timestamp,
      lastSampleAt: sample.timestamp,
      samples: [],
      shots: [],
    };
    state.rallies.push(state.activeRally);
    if (state.rallies.length > 24) state.rallies.shift();
  }

  const rally = state.activeRally;
  sample.rally = rally.id;
  rally.lastSampleAt = sample.timestamp;
  rally.samples.push({
    court: { ...sample.court },
    timestamp: sample.timestamp,
    speedKmh: sample.speedKmh,
    confidence: sample.confidence,
    source: sample.source,
  });
  if (rally.samples.length > 900) rally.samples.shift();
  state.rallyCount = Math.max(state.rallyCount, rally.id);
  updateReplayButton();
}

function recordSimulationAccuracy(sample) {
  if (!state.sim.active || !state.sim.currentTruth) return;

  const truth = state.sim.currentTruth;
  const metrics = state.sim.metrics;
  const positionError = Math.hypot(sample.court.x - truth.court.x, sample.court.y - truth.court.y);
  metrics.detections += 1;
  metrics.positionErrors.push(positionError);
  if (metrics.positionErrors.length > 240) metrics.positionErrors.shift();
  metrics.lastPositionError = positionError;

  if (sample.speedKmh > 0 && truth.speedKmh > 0) {
    const speedError = Math.abs(sample.speedKmh - truth.speedKmh);
    metrics.speedErrors.push(speedError);
    if (metrics.speedErrors.length > 240) metrics.speedErrors.shift();
    metrics.lastSpeedError = speedError;
  }
}

function smoothSpeed(rawSpeed) {
  const recent = state.tracks.slice(-4).map((track) => track.speedKmh).filter((speed) => speed > 0);
  if (!recent.length) return rawSpeed;
  return rawSpeed * 0.45 + average(recent) * 0.55;
}

function predictNextNorm(timestamp) {
  const a = state.tracks.at(-1);
  const b = state.tracks.at(-2);
  if (!a || !b) return null;
  const dt = a.timestamp - b.timestamp;
  if (dt <= 0) return a.norm;
  const ahead = clamp((timestamp - a.timestamp) / dt, 0, 2.2);
  return {
    x: clamp(a.norm.x + (a.norm.x - b.norm.x) * ahead, 0, 1),
    y: clamp(a.norm.y + (a.norm.y - b.norm.y) * ahead, 0, 1),
  };
}

function maybeRegisterShot(sample) {
  const minSpeed = Number(els.minShotSpeed.value) || 35;
  if (sample.speedKmh < minSpeed) return;
  const lastShotGap = sample.timestamp - state.lastShotTime;
  if (state.lastShotTime && lastShotGap < 0.55) {
    const lastShot = state.shots.at(-1);
    if (lastShot && sample.speedKmh > lastShot.speedKmh) {
      lastShot.speedKmh = sample.speedKmh;
      lastShot.court = sample.court;
      lastShot.location = describeCourtLocation(sample.court);
      const rallyShot = state.activeRally?.shots.at(-1);
      if (rallyShot?.id === lastShot.id) {
        rallyShot.speedKmh = lastShot.speedKmh;
        rallyShot.court = lastShot.court;
        rallyShot.location = lastShot.location;
      }
      renderShotList();
    }
    return;
  }

  const activeRally = state.activeRally;
  const shotIndex = activeRally ? activeRally.shots.length + 1 : state.shotInRally + 1;
  state.shotInRally = shotIndex;
  state.rallyCount = Math.max(state.rallyCount, sample.rally ?? state.rallyCount);

  const kind = classifyShot(sample.speedKmh, shotIndex);
  const shot = {
    id: crypto.randomUUID?.() ?? `${Date.now()}-${state.shots.length}`,
    kind,
    speedKmh: sample.speedKmh,
    timestamp: sample.timestamp,
    court: sample.court,
    location: describeCourtLocation(sample.court),
    confidence: sample.confidence,
    rally: sample.rally ?? state.rallyCount,
    index: shotIndex,
  };
  state.shots.unshift(shot);
  state.shots = state.shots.slice(0, 80);
  state.lastShotTime = sample.timestamp;
  activeRally?.shots.push(shot);

  if (kind === "Serve") state.stats.serve = Math.max(state.stats.serve, shot.speedKmh);
  if (kind === "Return") state.stats.return = Math.max(state.stats.return, shot.speedKmh);
  renderShotList();
}

function classifyShot(speedKmh, shotIndex = state.shotInRally) {
  if (shotIndex === 1 || speedKmh >= 135) return "Serve";
  if (shotIndex === 2) return "Return";
  return "Rally";
}

function describeCourtLocation(court) {
  const size = courtSize();
  const side = court.y < size.length / 2 ? "far" : "near";
  let lane = "middle";
  if (court.x < size.width / 3) lane = "left";
  if (court.x > (size.width * 2) / 3) lane = "right";

  let depth = "midcourt";
  const distanceFromBaseline = side === "far" ? court.y : size.length - court.y;
  if (distanceFromBaseline < 3.2) depth = "deep";
  if (Math.abs(court.y - size.length / 2) < 3.2) depth = "short";

  return `${side} ${lane} ${depth}`;
}

function renderStats() {
  const unit = preferredUnitLabel();
  els.serveSpeed.textContent = formatSpeed(state.stats.serve);
  els.returnSpeed.textContent = formatSpeed(state.stats.return);
  els.currentSpeed.textContent = formatSpeed(state.stats.current);
  els.peakSpeed.textContent = state.stats.peak ? `${formatSpeed(state.stats.peak)} ${unit}` : "--";
  els.confidence.textContent = state.stats.confidence ? `${Math.round(state.stats.confidence * 100)}%` : "--";
  els.shotCount.textContent = state.shots.length.toString();
  els.rallyCount.textContent = state.rallyCount.toString();
  document.querySelectorAll(".metric-strip small").forEach((el) => {
    el.textContent = unit;
  });
  renderBenchmarkStats();
}

function renderBenchmarkStats() {
  if (!state.sim.active) {
    els.positionError.textContent = "--";
    els.speedError.textContent = "--";
    return;
  }

  const metrics = state.sim.metrics;
  const meanPosition = average(metrics.positionErrors);
  const p95Position = percentile(metrics.positionErrors, 0.95);
  const meanSpeed = average(metrics.speedErrors);
  const missRate = metrics.frames ? (metrics.misses / metrics.frames) * 100 : 0;

  els.positionError.textContent = metrics.positionErrors.length
    ? `${meanPosition.toFixed(2)} m`
    : "--";
  els.positionError.title = metrics.positionErrors.length
    ? `Last ${metrics.lastPositionError.toFixed(2)} m, p95 ${p95Position.toFixed(2)} m, miss ${Math.round(missRate)}%`
    : "";
  els.speedError.textContent = metrics.speedErrors.length
    ? `${Math.round(meanSpeed)} km/h`
    : "--";
  els.speedError.title = metrics.speedErrors.length
    ? `Last ${Math.round(metrics.lastSpeedError)} km/h`
    : "";
}

function renderShotList() {
  const unit = preferredUnitLabel();
  els.shotList.innerHTML = "";
  for (const shot of state.shots.slice(0, 30)) {
    const li = document.createElement("li");
    const left = document.createElement("div");
    const kind = document.createElement("div");
    kind.className = "shot-kind";
    kind.textContent = shot.kind;
    const meta = document.createElement("div");
    meta.className = "shot-meta";
    meta.textContent = `Rally ${shot.rally} | Shot ${shot.index} | ${shot.location} | ${Math.round(shot.confidence * 100)}%`;
    const speed = document.createElement("div");
    speed.className = "shot-speed";
    speed.textContent = `${formatSpeed(shot.speedKmh)} ${unit}`;
    left.append(kind, meta);
    li.append(left, speed);
    els.shotList.append(li);
  }
  renderStats();
}

function updateReplayButton() {
  const hasReplay = state.rallies.some((rally) => rally.samples.length > 6);
  els.replayBtn.disabled = !hasReplay;
}

function openReplay() {
  const source = [...state.rallies].reverse().find((rally) => rally.samples.length > 6);
  if (!source) return;

  const samples = source.samples.map((sample) => ({
    ...sample,
    court: { ...sample.court },
  }));
  const shots = source.shots.map((shot) => ({
    ...shot,
    court: { ...shot.court },
  }));
  const first = samples[0];
  const last = samples.at(-1);

  state.replay.rally = {
    id: source.id,
    samples,
    shots,
    startedAt: first.timestamp,
    endedAt: last.timestamp,
  };
  state.replay.durationMs = Math.max(1200, (last.timestamp - first.timestamp) * 1000);
  state.replay.startedAt = performance.now();
  state.replay.pausedAt = 0;
  state.replay.playing = true;

  els.replayTitle.textContent = `Point ${source.id}`;
  els.replayPlayBtn.textContent = "Pause";
  els.replayModal.hidden = false;
  drawReplayFrame();
}

function closeReplay() {
  cancelAnimationFrame(state.replay.frameRequest);
  state.replay.frameRequest = 0;
  state.replay.playing = false;
  state.replay.rally = null;
  if (els.replayModal) els.replayModal.hidden = true;
}

function toggleReplayPlayback() {
  if (!state.replay.rally) return;

  if (state.replay.playing) {
    state.replay.pausedAt = replayElapsedMs();
    state.replay.playing = false;
    els.replayPlayBtn.textContent = "Play";
    cancelAnimationFrame(state.replay.frameRequest);
    drawReplayFrame();
    return;
  }

  state.replay.startedAt = performance.now() - state.replay.pausedAt;
  state.replay.playing = true;
  els.replayPlayBtn.textContent = "Pause";
  drawReplayFrame();
}

function restartReplay() {
  if (!state.replay.rally) return;
  state.replay.startedAt = performance.now();
  state.replay.pausedAt = 0;
  state.replay.playing = true;
  els.replayPlayBtn.textContent = "Pause";
  drawReplayFrame();
}

function replayElapsedMs() {
  if (!state.replay.rally) return 0;
  if (!state.replay.playing) return state.replay.pausedAt;
  return clamp(performance.now() - state.replay.startedAt, 0, state.replay.durationMs);
}

function drawReplayFrame() {
  cancelAnimationFrame(state.replay.frameRequest);
  const rally = state.replay.rally;
  if (!rally) return;

  const canvas = els.replayCanvas;
  const ctx = replayCtx;
  const elapsedMs = replayElapsedMs();
  const progress = state.replay.durationMs ? elapsedMs / state.replay.durationMs : 0;
  const replayTime = rally.startedAt + elapsedMs / 1000;
  const visible = rally.samples.filter((sample) => sample.timestamp <= replayTime);
  const current = sampleAtReplayTime(rally.samples, replayTime);
  const court = courtSize();
  const transform = replayCourtTransform(canvas.width, canvas.height, court);

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawReplayCourt(ctx, transform, court);
  drawReplayTrail(ctx, transform, visible);
  drawReplayShots(ctx, transform, rally.shots.filter((shot) => shot.timestamp <= replayTime));

  if (current) {
    const point = transform(current.court);
    ctx.fillStyle = "#d7fa5f";
    ctx.strokeStyle = "rgba(18, 20, 23, 0.82)";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(point.x, point.y, 8, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }

  const speed = current?.speedKmh ? `${Math.round(convertSpeed(current.speedKmh))} ${preferredUnitLabel()}` : "--";
  const shots = rally.shots.length;
  els.replayMeta.textContent = `${Math.round(progress * 100)}% | ${speed} | ${shots} shots`;

  if (state.replay.playing && elapsedMs < state.replay.durationMs) {
    state.replay.frameRequest = requestAnimationFrame(drawReplayFrame);
  } else if (state.replay.playing) {
    state.replay.playing = false;
    state.replay.pausedAt = state.replay.durationMs;
    els.replayPlayBtn.textContent = "Play";
  }
}

function sampleAtReplayTime(samples, time) {
  if (!samples.length) return null;
  if (time <= samples[0].timestamp) return samples[0];
  if (time >= samples.at(-1).timestamp) return samples.at(-1);

  for (let index = 1; index < samples.length; index += 1) {
    const next = samples[index];
    if (next.timestamp < time) continue;
    const previous = samples[index - 1];
    const span = next.timestamp - previous.timestamp || 1;
    const amount = clamp((time - previous.timestamp) / span, 0, 1);
    return {
      timestamp: time,
      speedKmh: previous.speedKmh + (next.speedKmh - previous.speedKmh) * amount,
      court: {
        x: previous.court.x + (next.court.x - previous.court.x) * amount,
        y: previous.court.y + (next.court.y - previous.court.y) * amount,
      },
    };
  }

  return samples.at(-1);
}

function replayCourtTransform(width, height, court) {
  const pad = 42;
  const scale = Math.min((width - pad * 2) / court.width, (height - pad * 2) / court.length);
  const courtW = court.width * scale;
  const courtH = court.length * scale;
  const ox = (width - courtW) / 2;
  const oy = (height - courtH) / 2;
  return (point) => ({
    x: ox + point.x * scale,
    y: oy + point.y * scale,
  });
}

function drawReplayCourt(ctx, transform, court) {
  const tl = transform({ x: 0, y: 0 });
  const br = transform({ x: court.width, y: court.length });
  const width = br.x - tl.x;
  const height = br.y - tl.y;

  ctx.fillStyle = "#153328";
  ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.fillStyle = "#29745d";
  ctx.fillRect(tl.x, tl.y, width, height);
  ctx.strokeStyle = "rgba(244, 242, 234, 0.92)";
  ctx.lineWidth = 3;
  ctx.strokeRect(tl.x, tl.y, width, height);

  const line = (a, b) => {
    const pa = transform(a);
    const pb = transform(b);
    ctx.beginPath();
    ctx.moveTo(pa.x, pa.y);
    ctx.lineTo(pb.x, pb.y);
    ctx.stroke();
  };

  const netY = court.length / 2;
  const service = 6.4;
  line({ x: 0, y: netY }, { x: court.width, y: netY });
  line({ x: 0, y: netY - service }, { x: court.width, y: netY - service });
  line({ x: 0, y: netY + service }, { x: court.width, y: netY + service });
  line({ x: court.width / 2, y: netY - service }, { x: court.width / 2, y: netY + service });

  if (court.width > COURTS.singles.width) {
    const alley = (court.width - COURTS.singles.width) / 2;
    line({ x: alley, y: 0 }, { x: alley, y: court.length });
    line({ x: court.width - alley, y: 0 }, { x: court.width - alley, y: court.length });
  }
}

function drawReplayTrail(ctx, transform, samples) {
  if (samples.length < 2) return;
  ctx.save();
  ctx.lineWidth = 4;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  for (let index = 1; index < samples.length; index += 1) {
    const a = transform(samples[index - 1].court);
    const b = transform(samples[index].court);
    const alpha = index / samples.length;
    ctx.strokeStyle = `rgba(116, 185, 255, ${0.2 + alpha * 0.7})`;
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }

  ctx.restore();
}

function drawReplayShots(ctx, transform, shots) {
  ctx.save();
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.font = "900 12px system-ui";

  shots.forEach((shot) => {
    const point = transform(shot.court);
    ctx.fillStyle = shot.kind === "Serve" ? "#ffbf55" : "#d7fa5f";
    ctx.beginPath();
    ctx.arc(point.x, point.y, 11, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#121417";
    ctx.fillText(String(shot.index), point.x, point.y);
  });

  ctx.restore();
}

function drawOverlay() {
  const rect = els.overlay.getBoundingClientRect();
  overlayCtx.clearRect(0, 0, rect.width, rect.height);
  updateVideoContentRect();
  drawAutoCalibration();
  drawCalibration();
  drawTrackTrail();
  drawBall();
}

function drawAutoCalibration() {
  if (!state.autoCalibration?.lines?.length) return;

  overlayCtx.save();
  overlayCtx.lineWidth = 2;
  overlayCtx.setLineDash([8, 6]);
  overlayCtx.strokeStyle = "rgba(116, 185, 255, 0.86)";

  for (const segment of state.autoCalibration.lines) {
    const a = normToScreen(segment[0]);
    const b = normToScreen(segment[1]);
    overlayCtx.beginPath();
    overlayCtx.moveTo(a.x, a.y);
    overlayCtx.lineTo(b.x, b.y);
    overlayCtx.stroke();
  }

  overlayCtx.restore();
}

function drawCalibration() {
  if (!state.calibrationPoints.length) return;

  overlayCtx.save();
  overlayCtx.lineWidth = 2;
  overlayCtx.strokeStyle = "rgba(215, 250, 95, 0.95)";
  overlayCtx.fillStyle = "rgba(215, 250, 95, 0.95)";

  const points = state.calibrationPoints.map(normToScreen);
  overlayCtx.beginPath();
  points.forEach((point, index) => {
    if (index === 0) overlayCtx.moveTo(point.x, point.y);
    else overlayCtx.lineTo(point.x, point.y);
  });
  if (points.length === 4) overlayCtx.closePath();
  overlayCtx.stroke();

  points.forEach((point, index) => {
    overlayCtx.beginPath();
    overlayCtx.arc(point.x, point.y, 7, 0, Math.PI * 2);
    overlayCtx.fill();
    overlayCtx.fillStyle = "#151719";
    overlayCtx.font = "900 11px system-ui";
    overlayCtx.textAlign = "center";
    overlayCtx.textBaseline = "middle";
    overlayCtx.fillText(String(index + 1), point.x, point.y);
    overlayCtx.fillStyle = "rgba(215, 250, 95, 0.95)";
  });

  overlayCtx.restore();
}

function drawTrackTrail() {
  const visibleTracks = state.tracks.slice(-36);
  if (visibleTracks.length < 2) return;
  overlayCtx.save();
  overlayCtx.lineWidth = 3;
  overlayCtx.lineJoin = "round";
  overlayCtx.lineCap = "round";
  overlayCtx.beginPath();
  visibleTracks.forEach((track, index) => {
    const point = normToScreen(track.norm);
    if (index === 0) overlayCtx.moveTo(point.x, point.y);
    else overlayCtx.lineTo(point.x, point.y);
  });
  overlayCtx.strokeStyle = "rgba(116, 185, 255, 0.82)";
  overlayCtx.stroke();
  overlayCtx.restore();
}

function drawBall() {
  const last = state.lastDetection;
  if (!last) return;
  const age = state.lastFrameTime - last.timestamp;
  if (age > 0.6) return;
  const point = normToScreen(last.norm);
  const radius = clamp(last.radius * (state.videoRect.width / 360), 5, 18);
  const accent = last.source === "model" ? "116, 185, 255" : "215, 250, 95";
  overlayCtx.save();
  overlayCtx.beginPath();
  overlayCtx.arc(point.x, point.y, radius + 7, 0, Math.PI * 2);
  overlayCtx.fillStyle = `rgba(${accent}, 0.18)`;
  overlayCtx.fill();
  overlayCtx.lineWidth = 2;
  overlayCtx.strokeStyle = `rgba(${accent}, 0.95)`;
  overlayCtx.stroke();
  overlayCtx.beginPath();
  overlayCtx.arc(point.x, point.y, Math.max(4, radius * 0.55), 0, Math.PI * 2);
  overlayCtx.fillStyle = `rgba(${accent}, 0.92)`;
  overlayCtx.fill();

  if (last.bbox) {
    const topLeft = normToScreen({ x: last.bbox.x1 / els.video.videoWidth, y: last.bbox.y1 / els.video.videoHeight });
    const bottomRight = normToScreen({ x: last.bbox.x2 / els.video.videoWidth, y: last.bbox.y2 / els.video.videoHeight });
    overlayCtx.strokeStyle = `rgba(${accent}, 0.72)`;
    overlayCtx.lineWidth = 1.5;
    overlayCtx.strokeRect(topLeft.x, topLeft.y, bottomRight.x - topLeft.x, bottomRight.y - topLeft.y);
  }

  overlayCtx.restore();
}

function drawCourtMap() {
  const canvas = els.courtMap;
  const ctx = courtCtx;
  const width = canvas.width;
  const height = canvas.height;
  const pad = 30;
  const court = courtSize();
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#153328";
  ctx.fillRect(0, 0, width, height);

  const scale = Math.min((width - pad * 2) / court.width, (height - pad * 2) / court.length);
  const courtW = court.width * scale;
  const courtH = court.length * scale;
  const ox = (width - courtW) / 2;
  const oy = (height - courtH) / 2;

  ctx.fillStyle = "#29745d";
  ctx.fillRect(ox, oy, courtW, courtH);
  ctx.strokeStyle = "rgba(244, 242, 234, 0.9)";
  ctx.lineWidth = 2;
  ctx.strokeRect(ox, oy, courtW, courtH);

  const lineY = (meters) => oy + meters * scale;
  const lineX = (meters) => ox + meters * scale;
  const serviceFromNet = 6.4;
  const netY = lineY(court.length / 2);
  const nearServiceY = lineY(court.length / 2 + serviceFromNet);
  const farServiceY = lineY(court.length / 2 - serviceFromNet);
  const centerX = lineX(court.width / 2);

  ctx.beginPath();
  ctx.moveTo(ox, netY);
  ctx.lineTo(ox + courtW, netY);
  ctx.moveTo(ox, nearServiceY);
  ctx.lineTo(ox + courtW, nearServiceY);
  ctx.moveTo(ox, farServiceY);
  ctx.lineTo(ox + courtW, farServiceY);
  ctx.moveTo(centerX, farServiceY);
  ctx.lineTo(centerX, nearServiceY);
  ctx.stroke();

  if (court.width > COURTS.singles.width) {
    const alley = (court.width - COURTS.singles.width) / 2;
    ctx.beginPath();
    ctx.moveTo(lineX(alley), oy);
    ctx.lineTo(lineX(alley), oy + courtH);
    ctx.moveTo(lineX(court.width - alley), oy);
    ctx.lineTo(lineX(court.width - alley), oy + courtH);
    ctx.stroke();
  }

  drawCourtTrail(ctx, ox, oy, scale, court);
}

function drawCourtTrail(ctx, ox, oy, scale, court) {
  const points = state.courtTrail.slice(-90).filter((track) => {
    return track.court.x >= -1 && track.court.x <= court.width + 1 && track.court.y >= -1 && track.court.y <= court.length + 1;
  });
  if (!points.length) return;

  ctx.save();
  for (let i = 1; i < points.length; i += 1) {
    const a = points[i - 1];
    const b = points[i];
    const alpha = i / points.length;
    ctx.strokeStyle = `rgba(116, 185, 255, ${0.18 + alpha * 0.72})`;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(ox + a.court.x * scale, oy + a.court.y * scale);
    ctx.lineTo(ox + b.court.x * scale, oy + b.court.y * scale);
    ctx.stroke();
  }

  const latest = points.at(-1);
  ctx.beginPath();
  ctx.arc(ox + latest.court.x * scale, oy + latest.court.y * scale, 5, 0, Math.PI * 2);
  ctx.fillStyle = "#d7fa5f";
  ctx.fill();
  ctx.restore();
}

function exportSession() {
  const payload = {
    app: "CourtSpeed",
    createdAt: new Date().toISOString(),
    units: preferredUnitLabel(),
    court: els.courtMode.value,
    calibrationPoints: state.calibrationPoints,
    stats: {
      peak: convertSpeed(state.stats.peak),
      serve: convertSpeed(state.stats.serve),
      return: convertSpeed(state.stats.return),
      confidence: state.stats.confidence,
    },
    shots: state.shots.map((shot) => ({
      kind: shot.kind,
      speed: convertSpeed(shot.speedKmh),
      rally: shot.rally,
      index: shot.index,
      confidence: shot.confidence,
      court: shot.court,
      location: shot.location,
    })),
    rallies: state.rallies.map((rally) => ({
      id: rally.id,
      samples: rally.samples.map((sample) => ({
        t: sample.timestamp - rally.startedAt,
        court: sample.court,
        speed: convertSpeed(sample.speedKmh),
        confidence: sample.confidence,
        source: sample.source,
      })),
      shots: rally.shots.map((shot) => ({
        kind: shot.kind,
        speed: convertSpeed(shot.speedKmh),
        index: shot.index,
        location: shot.location,
        court: shot.court,
      })),
    })),
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `courtspeed-${new Date().toISOString().slice(0, 10)}.json`;
  link.click();
  URL.revokeObjectURL(url);
}

function computeHomography(src, dst) {
  const matrix = [];
  const vector = [];

  for (let i = 0; i < 4; i += 1) {
    const [x, y] = src[i];
    const [u, v] = dst[i];
    matrix.push([x, y, 1, 0, 0, 0, -u * x, -u * y]);
    vector.push(u);
    matrix.push([0, 0, 0, x, y, 1, -v * x, -v * y]);
    vector.push(v);
  }

  const solution = solveLinearSystem(matrix, vector);
  if (!solution) return null;
  return [
    [solution[0], solution[1], solution[2]],
    [solution[3], solution[4], solution[5]],
    [solution[6], solution[7], 1],
  ];
}

function solveLinearSystem(matrix, vector) {
  const n = vector.length;
  const a = matrix.map((row, index) => [...row, vector[index]]);

  for (let col = 0; col < n; col += 1) {
    let pivot = col;
    for (let row = col + 1; row < n; row += 1) {
      if (Math.abs(a[row][col]) > Math.abs(a[pivot][col])) pivot = row;
    }
    if (Math.abs(a[pivot][col]) < 1e-12) return null;
    [a[col], a[pivot]] = [a[pivot], a[col]];

    const divisor = a[col][col];
    for (let j = col; j <= n; j += 1) a[col][j] /= divisor;

    for (let row = 0; row < n; row += 1) {
      if (row === col) continue;
      const factor = a[row][col];
      for (let j = col; j <= n; j += 1) a[row][j] -= factor * a[col][j];
    }
  }

  return a.map((row) => row[n]);
}

function applyHomography(h, point) {
  if (!h) return null;
  const [x, y] = point;
  const denominator = h[2][0] * x + h[2][1] * y + h[2][2];
  if (Math.abs(denominator) < 1e-8) return null;
  return {
    x: (h[0][0] * x + h[0][1] * y + h[0][2]) / denominator,
    y: (h[1][0] * x + h[1][1] * y + h[1][2]) / denominator,
  };
}

function courtToNorm(point) {
  return applyHomography(state.inverseHomography, [point.x, point.y]) ?? { x: 0.5, y: 0.5 };
}

function angleFromHorizontal(line) {
  const direction = (line.theta + 90) % 180;
  return Math.min(direction, 180 - direction);
}

function yAtX(line, x) {
  if (Math.abs(line.sin) < 1e-6) return Number.NaN;
  return (line.rho - x * line.cos) / line.sin;
}

function xAtY(line, y) {
  if (Math.abs(line.cos) < 1e-6) return Number.NaN;
  return (line.rho - y * line.sin) / line.cos;
}

function intersectLines(a, b) {
  const determinant = a.cos * b.sin - b.cos * a.sin;
  if (Math.abs(determinant) < 1e-8) return null;
  return {
    x: (a.rho * b.sin - b.rho * a.sin) / determinant,
    y: (a.cos * b.rho - b.cos * a.rho) / determinant,
  };
}

function lineSegmentInBounds(line, width, height) {
  const points = [];
  const addPoint = (point) => {
    if (!Number.isFinite(point.x) || !Number.isFinite(point.y)) return;
    if (point.x < -1 || point.x > width + 1 || point.y < -1 || point.y > height + 1) return;
    const duplicate = points.some((existing) => Math.hypot(existing.x - point.x, existing.y - point.y) < 2);
    if (!duplicate) points.push(point);
  };

  addPoint({ x: 0, y: yAtX(line, 0) });
  addPoint({ x: width, y: yAtX(line, width) });
  addPoint({ x: xAtY(line, 0), y: 0 });
  addPoint({ x: xAtY(line, height), y: height });

  if (points.length < 2) return null;

  let bestPair = [points[0], points[1]];
  let bestDistance = 0;
  for (let i = 0; i < points.length; i += 1) {
    for (let j = i + 1; j < points.length; j += 1) {
      const pairDistance = Math.hypot(points[i].x - points[j].x, points[i].y - points[j].y);
      if (pairDistance > bestDistance) {
        bestDistance = pairDistance;
        bestPair = [points[i], points[j]];
      }
    }
  }

  return bestPair;
}

function lineParallelScore(a, b) {
  const diff = Math.min(Math.abs(a.theta - b.theta), 180 - Math.abs(a.theta - b.theta));
  return 1 - clamp(diff / 18, 0, 1);
}

function midpoint(a, b) {
  return {
    x: (a.x + b.x) / 2,
    y: (a.y + b.y) / 2,
  };
}

function polygonArea(points) {
  let area = 0;
  for (let i = 0; i < points.length; i += 1) {
    const next = points[(i + 1) % points.length];
    area += points[i].x * next.y - next.x * points[i].y;
  }
  return Math.abs(area) / 2;
}

function pointInsideExpanded(point, width, height, marginRatio) {
  const mx = width * marginRatio;
  const my = height * marginRatio;
  return point.x >= -mx && point.x <= width + mx && point.y >= -my && point.y <= height + my;
}

function rgbToHsv(r, g, b) {
  r /= 255;
  g /= 255;
  b /= 255;
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const delta = max - min;
  let h = 0;

  if (delta !== 0) {
    if (max === r) h = 60 * (((g - b) / delta) % 6);
    else if (max === g) h = 60 * ((b - r) / delta + 2);
    else h = 60 * ((r - g) / delta + 4);
  }
  if (h < 0) h += 360;

  return {
    h,
    s: max === 0 ? 0 : delta / max,
    v: max,
  };
}

function average(values) {
  if (!values.length) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function percentile(values, ratio) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const index = clamp(Math.ceil(sorted.length * ratio) - 1, 0, sorted.length - 1);
  return sorted[index];
}

function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

els.startCameraBtn.addEventListener("click", startCamera);
els.recordBtn.addEventListener("click", toggleTracking);
els.autoCalibrateBtn.addEventListener("click", autoCalibrate);
els.calibrateBtn.addEventListener("click", beginCalibration);
els.simBtn.addEventListener("click", startSimulation);
els.resetBtn.addEventListener("click", resetSession);
els.exportBtn.addEventListener("click", exportSession);
els.replayBtn.addEventListener("click", openReplay);
els.replayPlayBtn.addEventListener("click", toggleReplayPlayback);
els.replayRestartBtn.addEventListener("click", restartReplay);
els.closeReplayBtn.addEventListener("click", closeReplay);
els.replayModal.addEventListener("click", (event) => {
  if (event.target === els.replayModal) closeReplay();
});
window.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !els.replayModal.hidden) closeReplay();
});
els.overlay.addEventListener("pointerdown", handleCalibrationTap);
els.courtMode.addEventListener("change", () => {
  if (state.calibrationPoints.length === 4) finishCalibration();
  drawCourtMap();
});
els.detectorMode.addEventListener("change", () => {
  state.model.lastDetection = null;
  if (els.detectorMode.value === "color") {
    setDetectorStatus("Color tracker", "muted");
  } else {
    void ensureBallModel();
  }
});
els.units.addEventListener("change", () => {
  renderStats();
  renderShotList();
});
window.addEventListener("resize", () => {
  setAppHeight();
  resizeCanvases();
});
els.video.addEventListener("loadedmetadata", resizeCanvases);

if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("./sw.js").catch(() => {});
  });
}

setAppHeight();
setDetectorStatus(window.ort ? "Model idle" : "Model unavailable", window.ort ? "muted" : "warn");
renderStats();
updateReplayButton();
drawCourtMap();
