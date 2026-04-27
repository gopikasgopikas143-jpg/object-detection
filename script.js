// ================================================
//  VISION-AI Object Detection & Tracking
//  script.js — CodeAlpha Task 4
//  Uses: TensorFlow.js + COCO-SSD (YOLO-style)
//        Custom SORT-inspired tracker
//        Canvas bounding boxes + trail paths
// ================================================

// ───────────────────────────────────────────────
//  GLOBALS
// ───────────────────────────────────────────────
let model       = null;       // COCO-SSD model
let isRunning   = false;
let animId      = null;
let stream      = null;
let mode        = 'ALL';
let boxStyle    = 'corner';
let confThresh  = 0.50;
let showTrail   = true;

let frameCount  = 0;
let totalDetections = 0;
let fpsLastTime = performance.now();
let fpsValue    = 0;

// Stats
let classCounts = {};
let allConfs    = [];

// SORT-style tracker
let trackedObjects = [];   // { id, label, bbox, trail[], age, missed }
let nextId = 1;

// Colour palette per class
const CLASS_COLORS = {
  person:       '#00ff9d',
  car:          '#00d4ff',
  truck:        '#ffd700',
  bus:          '#ff9900',
  bicycle:      '#ff3a3a',
  motorbike:    '#ff6699',
  dog:          '#cc99ff',
  cat:          '#ff99cc',
  chair:        '#aaffdd',
  bottle:       '#99ccff',
  laptop:       '#ffddaa',
  phone:        '#ff99aa',
  default:      '#ffffff'
};

function classColor(label) {
  return CLASS_COLORS[label.toLowerCase()] || CLASS_COLORS.default;
}

// DOM refs
const videoEl    = document.getElementById('videoEl');
const canvas     = document.getElementById('detectCanvas');
const ctx        = canvas.getContext('2d');
const idleOverlay= document.getElementById('idleOverlay');
const logBox     = document.getElementById('logBox');
const scanSweep  = document.getElementById('scanSweep');
const recIndicator = document.getElementById('recIndicator');

// ───────────────────────────────────────────────
//  LOGGING
// ───────────────────────────────────────────────
function log(msg, type = '') {
  const line = document.createElement('div');
  line.className = `log-line ${type}`;
  const ts = new Date().toLocaleTimeString('en-GB', { hour12: false });
  line.textContent = `[${ts}] ${msg}`;
  logBox.appendChild(line);
  logBox.scrollTop = logBox.scrollHeight;
  // Keep max 40 lines
  while (logBox.children.length > 40) logBox.removeChild(logBox.firstChild);
}

// ───────────────────────────────────────────────
//  CLOCK
// ───────────────────────────────────────────────
(function updateClock() {
  const now  = new Date();
  document.getElementById('clockDisplay').textContent =
    now.toLocaleTimeString('en-GB', { hour12: false });
  document.getElementById('dateDisplay').textContent =
    now.toLocaleDateString('en-GB', { weekday:'short', day:'2-digit', month:'short', year:'numeric' }).toUpperCase();
  setTimeout(updateClock, 1000);
})();

// ───────────────────────────────────────────────
//  LOAD MODEL
// ───────────────────────────────────────────────
async function loadModel() {
  log('Loading COCO-SSD model...', 'info');
  document.getElementById('statusSystem').querySelector('span:last-child').textContent = 'LOADING MODEL...';
  try {
    model = await cocoSsd.load({ base: 'lite_mobilenet_v2' });
    log('COCO-SSD model loaded! ✓', 'init');
    document.getElementById('statusSystem').querySelector('span:last-child').textContent = 'MODEL READY';
    document.getElementById('statusSystem').querySelector('.dot').classList.remove('red');
  } catch(e) {
    log('Model load failed: ' + e.message, 'error');
  }
}

// ───────────────────────────────────────────────
//  WEBCAM
// ───────────────────────────────────────────────
async function startWebcam() {
  if (!model) { log('Model not ready yet!', 'warn'); return; }
  stopAll();
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width:1280, height:720 }, audio:false });
    videoEl.srcObject = stream;
    videoEl.style.display = 'block';
    await videoEl.play();
    log('Webcam activated ✓', 'init');
    setupCanvas();
    startDetection();
    document.getElementById('hudSource').textContent = 'SOURCE: WEBCAM';
  } catch(e) {
    log('Webcam error: ' + e.message, 'error');
    alert('Webcam access denied or not available. Please upload a video or image instead.');
  }
}

// ───────────────────────────────────────────────
//  VIDEO UPLOAD
// ───────────────────────────────────────────────
function loadVideo(input) {
  if (!model) { log('Model not ready!', 'warn'); return; }
  const file = input.files[0];
  if (!file) return;
  stopAll();
  const url = URL.createObjectURL(file);
  videoEl.srcObject = null;
  videoEl.src = url;
  videoEl.style.display = 'block';
  videoEl.play();
  log(`Video loaded: ${file.name}`, 'info');
  videoEl.onloadedmetadata = () => { setupCanvas(); startDetection(); };
  document.getElementById('hudSource').textContent = 'SOURCE: VIDEO';
}

// ───────────────────────────────────────────────
//  IMAGE UPLOAD
// ───────────────────────────────────────────────
function loadImage(input) {
  if (!model) { log('Model not ready!', 'warn'); return; }
  const file = input.files[0];
  if (!file) return;
  stopAll();
  const img = new Image();
  img.onload = async () => {
    idleOverlay.classList.add('hidden');
    canvas.style.display = 'block';
    document.querySelectorAll('.canvas-hud').forEach(h => h.classList.add('visible'));
    canvas.width  = img.width;
    canvas.height = img.height;
    ctx.drawImage(img, 0, 0);
    log(`Analysing image: ${file.name}`, 'info');
    const preds = await model.detect(img);
    drawDetections(preds, false);
    updateStats(preds);
    log(`Detection complete. Found ${preds.length} object(s).`, 'init');
    document.getElementById('hudSource').textContent = 'SOURCE: IMAGE';
    document.getElementById('hudRes').textContent =
      `RES: ${img.width}×${img.height}`;
    document.getElementById('snapBtn').disabled = false;
  };
  img.src = URL.createObjectURL(file);
}

// ───────────────────────────────────────────────
//  SETUP CANVAS
// ───────────────────────────────────────────────
function setupCanvas() {
  idleOverlay.classList.add('hidden');
  canvas.style.display = 'block';
  document.querySelectorAll('.canvas-hud').forEach(h => h.classList.add('visible'));
  recIndicator.classList.add('show');
  scanSweep.classList.add('active');
  document.getElementById('playPauseBtn').disabled = false;
  document.getElementById('stopBtn').disabled = false;
  document.getElementById('snapBtn').disabled = false;
}

// ───────────────────────────────────────────────
//  DETECTION LOOP
// ───────────────────────────────────────────────
function startDetection() {
  isRunning = true;
  document.getElementById('statusSystem').querySelector('span:last-child').textContent = 'SYSTEM ACTIVE';
  document.getElementById('statusSystem').querySelector('.dot').className = 'dot';
  log('Detection loop started.', 'info');
  detect();
}

async function detect() {
  if (!isRunning) return;

  canvas.width  = videoEl.videoWidth  || videoEl.offsetWidth;
  canvas.height = videoEl.videoHeight || videoEl.offsetHeight;

  try {
    const predictions = await model.detect(videoEl);
    const filtered = filterByMode(predictions);

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Update tracking
    updateTracking(filtered);

    // Draw trails
    if (showTrail) drawTrails();

    // Draw detections
    drawDetections(filtered, true);

    // Stats
    updateStats(filtered);
    frameCount++;

    // FPS
    const now = performance.now();
    const elapsed = now - fpsLastTime;
    if (elapsed >= 1000) {
      fpsValue = Math.round((frameCount / elapsed) * 1000);
      frameCount = 0;
      fpsLastTime = now;
      document.getElementById('statFps').textContent  = fpsValue;
      document.getElementById('hudObjects').textContent = `OBJECTS: ${filtered.length}`;
      document.getElementById('fpsLabel').textContent   = `${fpsValue} FPS`;
    }

    document.getElementById('hudRes').textContent =
      `RES: ${canvas.width}×${canvas.height}`;

  } catch(e) { /* silently skip frame */ }

  animId = requestAnimationFrame(detect);
}

// ───────────────────────────────────────────────
//  FILTER BY MODE
// ───────────────────────────────────────────────
const MODE_MAP = {
  PERSON:  ['person'],
  VEHICLE: ['car','truck','bus','motorbike','bicycle','boat','aeroplane','train'],
  OBJECT:  ['bottle','chair','laptop','keyboard','mouse','book','cup','phone','cell phone',
             'remote','scissors','clock','vase','teddy bear','handbag','backpack','umbrella']
};

function filterByMode(preds) {
  return preds.filter(p => {
    if (p.score < confThresh) return false;
    if (mode === 'ALL') return true;
    const allowed = MODE_MAP[mode] || [];
    return allowed.some(a => p.class.toLowerCase().includes(a));
  });
}

// ───────────────────────────────────────────────
//  SIMPLE SORT-STYLE TRACKER
// ───────────────────────────────────────────────
function iouScore(bA, bB) {
  const [ax,ay,aw,ah] = bA;
  const [bx,by,bw,bh] = bB;
  const ix = Math.max(ax, bx), iy = Math.max(ay, by);
  const ix2= Math.min(ax+aw, bx+bw), iy2 = Math.min(ay+ah, by+bh);
  const inter = Math.max(0, ix2-ix) * Math.max(0, iy2-iy);
  const union  = aw*ah + bw*bh - inter;
  return union > 0 ? inter/union : 0;
}

function updateTracking(detections) {
  const matched = new Set();

  // Match detections to tracked objects
  detections.forEach(det => {
    let best = null, bestScore = 0.3; // IoU threshold
    trackedObjects.forEach((t, idx) => {
      if (t.label !== det.class) return;
      const score = iouScore(det.bbox, t.bbox);
      if (score > bestScore) { bestScore = score; best = idx; }
    });

    if (best !== null) {
      const cx = det.bbox[0] + det.bbox[2]/2;
      const cy = det.bbox[1] + det.bbox[3]/2;
      trackedObjects[best].bbox  = det.bbox;
      trackedObjects[best].age++;
      trackedObjects[best].missed = 0;
      if (showTrail) {
        trackedObjects[best].trail.push({ x: cx, y: cy });
        if (trackedObjects[best].trail.length > 30) trackedObjects[best].trail.shift();
      }
      det._trackId = trackedObjects[best].id;
      matched.add(best);
    } else {
      // New object
      const cx = det.bbox[0] + det.bbox[2]/2;
      const cy = det.bbox[1] + det.bbox[3]/2;
      trackedObjects.push({
        id: nextId++,
        label: det.class,
        bbox: det.bbox,
        trail: [{ x: cx, y: cy }],
        age: 1,
        missed: 0
      });
      det._trackId = trackedObjects[trackedObjects.length-1].id;
      totalDetections++;
      document.getElementById('statTotal').textContent = totalDetections;
      document.getElementById('statFrames').textContent = frameCount + totalDetections;
    }
  });

  // Age out missing tracks
  trackedObjects.forEach((t, i) => {
    if (!matched.has(i)) t.missed++;
  });
  trackedObjects = trackedObjects.filter(t => t.missed < 5);
}

// ───────────────────────────────────────────────
//  DRAW TRAILS
// ───────────────────────────────────────────────
function drawTrails() {
  trackedObjects.forEach(t => {
    if (t.trail.length < 2) return;
    const color = classColor(t.label);
    ctx.save();
    for (let i = 1; i < t.trail.length; i++) {
      const alpha = i / t.trail.length;
      ctx.beginPath();
      ctx.moveTo(t.trail[i-1].x, t.trail[i-1].y);
      ctx.lineTo(t.trail[i].x,   t.trail[i].y);
      ctx.strokeStyle = color;
      ctx.globalAlpha = alpha * 0.6;
      ctx.lineWidth   = alpha * 2;
      ctx.stroke();
    }
    ctx.restore();
  });
}

// ───────────────────────────────────────────────
//  DRAW DETECTIONS
// ───────────────────────────────────────────────
function drawDetections(preds, hasId) {
  preds.forEach(pred => {
    const [x, y, w, h] = pred.bbox;
    const label  = pred.class;
    const conf   = Math.round(pred.score * 100);
    const color  = classColor(label);
    const trackId = pred._trackId || '?';

    ctx.save();

    if (boxStyle === 'full') {
      // Full rectangle
      ctx.strokeStyle = color;
      ctx.lineWidth   = 2;
      ctx.shadowColor = color;
      ctx.shadowBlur  = 8;
      ctx.strokeRect(x, y, w, h);
      ctx.fillStyle   = color + '18';
      ctx.fillRect(x, y, w, h);

    } else if (boxStyle === 'dot') {
      // Just center dot + crosshair
      const cx = x + w/2, cy = y + h/2;
      ctx.strokeStyle = color;
      ctx.shadowColor = color;
      ctx.shadowBlur  = 10;
      ctx.lineWidth   = 1.5;
      // Cross
      ctx.beginPath();
      ctx.moveTo(cx-20, cy); ctx.lineTo(cx+20, cy);
      ctx.moveTo(cx, cy-20); ctx.lineTo(cx, cy+20);
      ctx.stroke();
      // Circle
      ctx.beginPath();
      ctx.arc(cx, cy, 10, 0, Math.PI*2);
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.stroke();

    } else {
      // CORNER style (default)
      const cs = Math.min(w, h) * 0.25; // corner size
      ctx.strokeStyle = color;
      ctx.lineWidth   = 2.5;
      ctx.shadowColor = color;
      ctx.shadowBlur  = 10;

      // Top-left
      ctx.beginPath(); ctx.moveTo(x, y+cs); ctx.lineTo(x,y); ctx.lineTo(x+cs, y); ctx.stroke();
      // Top-right
      ctx.beginPath(); ctx.moveTo(x+w-cs, y); ctx.lineTo(x+w,y); ctx.lineTo(x+w, y+cs); ctx.stroke();
      // Bottom-left
      ctx.beginPath(); ctx.moveTo(x, y+h-cs); ctx.lineTo(x,y+h); ctx.lineTo(x+cs, y+h); ctx.stroke();
      // Bottom-right
      ctx.beginPath(); ctx.moveTo(x+w-cs, y+h); ctx.lineTo(x+w,y+h); ctx.lineTo(x+w, y+h-cs); ctx.stroke();

      // Center dot
      ctx.fillStyle   = color;
      ctx.shadowBlur  = 4;
      ctx.beginPath();
      ctx.arc(x+w/2, y+h/2, 3, 0, Math.PI*2);
      ctx.fill();
    }

    // Label background
    const labelText = `${label.toUpperCase()}  ${conf}%${hasId ? `  #${trackId}` : ''}`;
    ctx.font         = 'bold 11px "Share Tech Mono"';
    const tw         = ctx.measureText(labelText).width;
    const lh         = 18;
    const lx         = x;
    const ly         = y > lh + 4 ? y - lh - 4 : y + h + 4;

    ctx.shadowBlur   = 0;
    ctx.fillStyle    = '#000000cc';
    ctx.fillRect(lx, ly, tw + 12, lh);
    ctx.fillStyle    = color;
    ctx.shadowColor  = color;
    ctx.shadowBlur   = 6;
    ctx.fillText(labelText, lx + 6, ly + 13);

    ctx.restore();
  });
}

// ───────────────────────────────────────────────
//  UPDATE STATS PANEL
// ───────────────────────────────────────────────
function updateStats(preds) {
  // Class breakdown
  const counts = {};
  let confSum  = 0;
  preds.forEach(p => {
    const k = p.class.toLowerCase();
    counts[k]  = (counts[k] || 0) + 1;
    classCounts[k] = (classCounts[k] || 0) + 1;
    confSum += p.score;
  });

  // Avg confidence
  if (preds.length > 0) {
    const avg = Math.round((confSum / preds.length) * 100);
    document.getElementById('statConf').textContent = avg + '%';
    allConfs.push(avg);
  }

  // Object registry (current frame)
  const objList = document.getElementById('objList');
  objList.innerHTML = '';
  if (preds.length === 0) {
    objList.innerHTML = '<div class="obj-empty">No objects in current frame.</div>';
  } else {
    preds.forEach(p => {
      const item = document.createElement('div');
      item.className = 'obj-item';
      item.innerHTML = `
        <span class="obj-id">#${p._trackId || '?'}</span>
        <span class="obj-name">${p.class.toUpperCase()}</span>
        <span class="obj-conf">${Math.round(p.score*100)}%</span>
      `;
      item.style.borderLeftColor = classColor(p.class);
      objList.appendChild(item);
    });
  }

  // Class bars
  const barWrap = document.getElementById('classBars');
  barWrap.innerHTML = '';
  const total = Object.values(classCounts).reduce((a,b)=>a+b,0);
  if (total === 0) {
    barWrap.innerHTML = '<div class="obj-empty">Waiting for detections...</div>';
    return;
  }
  Object.entries(classCounts)
    .sort((a,b) => b[1]-a[1])
    .slice(0, 8)
    .forEach(([cls, cnt]) => {
      const pct = Math.round((cnt / total) * 100);
      const row = document.createElement('div');
      row.className = 'class-row';
      row.innerHTML = `
        <div class="class-header">
          <span class="class-name">${cls}</span>
          <span class="class-count">${cnt}</span>
        </div>
        <div class="class-bar-bg">
          <div class="class-bar-fill" style="width:${pct}%; background: linear-gradient(90deg, ${classColor(cls)}, ${classColor(cls)}88);"></div>
        </div>
      `;
      barWrap.appendChild(row);
    });

  // Update HUD labels
  document.getElementById('hudMode').textContent  = `MODE: ${mode}`;
  document.getElementById('hudConf').textContent  = `CONF: ${Math.round(confThresh*100)}%`;
}

// ───────────────────────────────────────────────
//  CONTROLS
// ───────────────────────────────────────────────
function updateConf(val) {
  confThresh = val / 100;
  document.getElementById('confVal').textContent = val + '%';
}

function setMode(btn, m) {
  mode = m;
  document.querySelectorAll('.mode-btns .mode-btn').forEach(b => {
    if (b.closest('.ctrl-group') === btn.closest('.ctrl-group'))
      b.classList.remove('active');
  });
  btn.classList.add('active');
  log(`Detection mode set to: ${m}`, 'info');
}

function setBoxStyle(btn, s) {
  boxStyle = s;
  document.querySelectorAll('.mode-btns .mode-btn').forEach(b => {
    if (b.closest('.ctrl-group') === btn.closest('.ctrl-group'))
      b.classList.remove('active');
  });
  btn.classList.add('active');
}

document.getElementById('trailToggle').addEventListener('change', function() {
  showTrail = this.checked;
  document.getElementById('trailLabel').textContent = showTrail ? 'ON' : 'OFF';
});

function togglePlayPause() {
  if (!videoEl.paused) {
    videoEl.pause();
    document.getElementById('playPauseBtn').textContent = '▶';
    isRunning = false;
    if (animId) cancelAnimationFrame(animId);
    log('Paused.', 'warn');
  } else {
    videoEl.play();
    document.getElementById('playPauseBtn').textContent = '⏸';
    isRunning = true;
    detect();
    log('Resumed.', 'info');
  }
}

function stopAll() {
  isRunning = false;
  if (animId) { cancelAnimationFrame(animId); animId = null; }
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  videoEl.pause();
  videoEl.src = '';
  videoEl.srcObject = null;
  videoEl.style.display = 'none';
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  canvas.style.display = 'none';
  idleOverlay.classList.remove('hidden');
  document.querySelectorAll('.canvas-hud').forEach(h => h.classList.remove('visible'));
  scanSweep.classList.remove('active');
  recIndicator.classList.remove('show');
  document.getElementById('playPauseBtn').disabled = true;
  document.getElementById('playPauseBtn').textContent = '⏸';
  document.getElementById('stopBtn').disabled = true;
  document.getElementById('snapBtn').disabled = true;
  trackedObjects = [];
  classCounts    = {};
  document.getElementById('statFps').textContent = '--';
  document.getElementById('statConf').textContent = '--';
  document.getElementById('objList').innerHTML = '<div class="obj-empty">No objects detected yet.</div>';
  document.getElementById('classBars').innerHTML = '<div class="obj-empty">Waiting for detections...</div>';
  document.getElementById('statusSystem').querySelector('span:last-child').textContent = 'SYSTEM STANDBY';
  document.getElementById('statusSystem').querySelector('.dot').classList.add('red');
  log('System stopped.', 'warn');
}

function takeSnapshot() {
  const link = document.createElement('a');
  const snap = document.createElement('canvas');
  snap.width  = canvas.width;
  snap.height = canvas.height;
  const sc = snap.getContext('2d');
  if (videoEl.style.display !== 'none') sc.drawImage(videoEl, 0, 0, snap.width, snap.height);
  sc.drawImage(canvas, 0, 0);
  link.download = `vision-ai-snapshot-${Date.now()}.png`;
  link.href = snap.toDataURL('image/png');
  link.click();
  const toast = document.getElementById('snapToast');
  toast.classList.add('show');
  setTimeout(() => toast.classList.remove('show'), 2500);
  log('Snapshot saved.', 'init');
}

// ───────────────────────────────────────────────
//  BOOT
// ───────────────────────────────────────────────
window.addEventListener('load', () => {
  log('VISION-AI booting...', 'init');
  log('Loading TensorFlow.js runtime...', 'info');
  loadModel();
});
