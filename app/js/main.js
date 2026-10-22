import * as THREE from 'three';
import { World } from './world.js';
import { Car } from './car.js';
import { CameraFeed } from './camera-feed.js';
import { Recorder, INTENTS } from './recorder.js';
import { Policy } from './policy.js';

const $ = (id) => document.getElementById(id);

// ---------- scene ----------

const renderer = new THREE.WebGLRenderer({ canvas: $('viewport'), antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);

const scene = new THREE.Scene();
const world = new World(scene);
const car = new Car(scene);
car.reset(world.start);

// Only the car uses lit materials; the world is unlit, so these don't change
// what the policy sees.
scene.add(new THREE.HemisphereLight(0xdff1ff, 0x4a7c4f, 1.6));
const sun = new THREE.DirectionalLight(0xffffff, 2.2);
sun.position.set(30, 60, 20);
scene.add(sun);

// The driver camera keeps the window's aspect ratio even though it renders into
// a square target. That squeeze is what the training frames looked like.
const driverCam = new THREE.PerspectiveCamera(70, innerWidth / innerHeight, 0.5, 200);
const chaseCam = new THREE.PerspectiveCamera(70, innerWidth / innerHeight, 0.5, 200);
let viewCam = chaseCam;

const feed = new CameraFeed(renderer, scene, driverCam, 128);
// The training car never appeared in the driver view; keep the new body out too.
feed.hidden.push(car.group);
const recorder = new Recorder(feed);
const fpvCtx = $('fpv').getContext('2d');

// ---------- state ----------

const state = {
  mode: 'drive',
  intent: 0,
  policy: null,
  loading: null,
  engaged: false,
  control: 'smooth',
  thresholds: { fwd: 0.3, left: 0.3, right: 0.3, diff: 0.05 },
  chunk: [],
  step: 0,
  run: { branch: null, intent: 0, still: 0 },
  results: [],
};

// ---------- small UI helpers ----------

let toastTimer;
function toast(msg) {
  const t = $('toast');
  t.textContent = msg;
  t.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove('show'), 2200);
}

function setActive(container, el) {
  for (const b of container.querySelectorAll('button')) b.classList.toggle('active', b === el);
}

function setIntent(i) {
  state.intent = i;
  setActive($('intent'), $('intent').querySelector(`[data-intent="${i}"]`));
}

function setMode(mode) {
  state.mode = mode;
  for (const t of document.querySelectorAll('.tab[data-mode]')) t.classList.toggle('active', t.dataset.mode === mode);
  $('panel-drive').classList.toggle('hidden', mode !== 'drive');
  $('panel-autopilot').classList.toggle('hidden', mode !== 'autopilot');
  $('card-policy').classList.toggle('hidden', mode !== 'autopilot');
  $('card-score').classList.toggle('hidden', mode !== 'autopilot');
  if (mode === 'drive') disengage();
  else {
    if (recorder.recording) toggleRecording();
    if (!state.policy) loadPolicy(() => Policy.fromURL('model.json'), 'model.json (bundled)');
  }
}

// ---------- recording ----------

function toggleRecording() {
  if (recorder.recording) recorder.stop(); else recorder.start();
  const on = recorder.recording;
  $('btn-record').classList.toggle('on', on);
  $('record-text').textContent = on ? 'Stop' : 'Record';
  $('btn-download').disabled = on || !recorder.count;
}

function downloadDataset() {
  if (!recorder.count) return;
  recorder.download();
  toast(`Saved ${recorder.count} samples`);
  recorder.clear();
  $('btn-download').disabled = true;
}

// ---------- model ----------

async function loadPolicy(load, name) {
  if (state.loading) return state.loading;
  disengage();
  $('model-dot').className = 'status-dot';
  $('model-name').textContent = 'Loading…';
  $('model-info').textContent = name;
  state.loading = (async () => {
    try {
      const p = await load();
      if (p.size !== feed.size) throw new Error(`model expects ${p.size}px frames, app renders ${feed.size}px`);
      state.policy = p;
      $('model-dot').className = 'status-dot ok';
      $('model-name').textContent = name;
      $('model-info').textContent = p.summary;
      $('btn-engage').disabled = false;
      renderArchMeta();
      toast('Model ready, press P to engage');
    } catch (err) {
      console.error(err);
      $('model-dot').className = 'status-dot err';
      $('model-name').textContent = state.policy ? 'Kept previous model' : 'No model loaded';
      $('model-info').textContent = err.message;
      $('btn-engage').disabled = !state.policy;
    } finally {
      state.loading = null;
    }
  })();
  return state.loading;
}

function engage() {
  if (!state.policy || state.engaged) return;
  if (state.mode !== 'autopilot') setMode('autopilot');
  state.engaged = true;
  state.chunk = [];
  car.reset(world.start);
  state.run = { branch: null, intent: state.intent, still: 0 };
  $('btn-engage').classList.add('on');
  $('btn-engage').firstChild.textContent = 'Disengage ';
}

function disengage() {
  if (!state.engaged) return;
  state.engaged = false;
  state.chunk = [];
  car.press(false, false, false, false);
  $('btn-engage').classList.remove('on');
  $('btn-engage').firstChild.textContent = 'Engage autopilot ';
}

// One policy query yields K actions; play them one per frame, then re-query.
function autopilotStep() {
  if (state.step >= state.chunk.length) {
    const t0 = performance.now();
    feed.grab().drawTo(fpvCtx);
    state.chunk = state.policy.predict(feed.toTensor(), state.intent);
    state.step = 0;
    $('latency').textContent = `${(performance.now() - t0).toFixed(0)} ms`;
  }
  const [pF, , pL, pR] = state.chunk[state.step];
  const th = state.thresholds;
  if (state.control === 'smooth') {
    car.drive(pF > th.fwd, pL - pR);
  } else {
    const turn = Math.abs(pL - pR) > th.diff;
    const left = turn && pL > pR && pL > th.left;
    const right = turn && pR > pL && pR > th.right;
    car.press(pF > th.fwd || (!left && !right), false, left, right);
  }
  showPolicy(pF, pL, pR);
  state.step++;
}

const chunkCells = Array.from({ length: 5 }, () => $('chunk').appendChild(document.createElement('div')));

function showPolicy(f, l, r) {
  for (const [k, v] of [['f', f], ['l', l], ['r', r]]) {
    $(`bar-${k}`).style.width = `${(v * 100).toFixed(1)}%`;
    $(`val-${k}`).textContent = v.toFixed(2);
  }
  state.chunk.forEach(([, , cl, cr], i) => {
    const s = cl - cr;
    const cell = chunkCells[i];
    cell.textContent = Math.abs(s) < 0.15 ? '↑' : s > 0 ? '↖' : '↗';
    cell.className = (i < state.step ? 'done ' : i === state.step ? 'now ' : '') + (Math.abs(s) < 0.15 ? '' : s > 0 ? 'l' : 'r');
  });
}

// ---------- route check ----------

function trackRun(dt) {
  const { x, z } = car.position;
  const run = state.run;
  // Mid-loop the two branches are ~40 units apart, so x alone tells them apart.
  if (!run.branch && z < 15 && z > -20 && Math.abs(x) > 10) {
    run.branch = x < 0 ? 'left' : 'right';
    run.intent = state.intent;
  }
  run.still = Math.abs(car.speed) < 0.5 ? run.still + dt : 0;
  if (run.still > 3) return 'stuck';
  return null;
}

function finishRun(outcome) {
  if (!state.engaged) return;
  const run = state.run;
  const wanted = INTENTS[run.branch ? run.intent : state.intent].label.endsWith('left') ? 'left' : 'right';
  state.results.unshift({ outcome, branch: run.branch, wanted });
  renderScore();
  state.run = { branch: null, intent: state.intent, still: 0 };
}

function renderScore() {
  const r = state.results;
  const done = r.filter((x) => x.outcome === 'finished');
  const ok = done.filter((x) => x.branch === x.wanted).length;
  $('sc-runs').textContent = r.length;
  $('sc-ok').textContent = done.length ? `${ok}/${done.length}` : '–';
  $('sc-lr').textContent = `${r.filter((x) => x.branch === 'left').length} / ${r.filter((x) => x.branch === 'right').length}`;
  $('sc-crash').textContent = r.filter((x) => x.outcome !== 'finished').length;
  $('run-log').innerHTML = r.slice(0, 12).map((x, i) => {
    const n = r.length - i;
    if (x.outcome !== 'finished') return `<li><span>#${n} wanted ${x.wanted}</span><b class="bad">${x.outcome}</b></li>`;
    const good = x.branch === x.wanted;
    return `<li><span>#${n} wanted ${x.wanted}</span><b class="${good ? 'ok' : 'warn'}">took ${x.branch ?? '?'}</b></li>`;
  }).join('');
}

// ---------- architecture dialog ----------

function renderArchMeta() {
  const m = state.policy?.meta;
  if (!m) return;
  const rows = [
    ['Parameters', m.num_params.toLocaleString()], ['Input', `${m.img_size}×${m.img_size} RGB`],
    ['d_model', m.d_model], ['Heads', m.n_heads], ['Enc / Dec layers', `${m.n_enc_layers} / ${m.n_dec_layers}`],
    ['Chunk K', m.chunk_size], ['Latent z', m.latent_dim], ['Intents', m.intent_labels.join(', ')],
  ];
  $('arch-meta').innerHTML = rows.map(([k, v]) => `<div><span>${k}</span><b>${v}</b></div>`).join('');
}

// ---------- events ----------

for (const t of document.querySelectorAll('.tab[data-mode]')) t.addEventListener('click', () => setMode(t.dataset.mode));
$('intent').addEventListener('click', (e) => {
  const b = e.target.closest('button');
  if (b) setIntent(+b.dataset.intent);
});
$('control-mode').addEventListener('click', (e) => {
  const b = e.target.closest('button');
  if (!b) return;
  state.control = b.dataset.mode;
  setActive($('control-mode'), b);
});
for (const [key, id, scale] of [['fwd', 'th-fwd', 100], ['left', 'th-left', 100], ['right', 'th-right', 100], ['diff', 'th-diff', 100]]) {
  const input = $(id);
  input.addEventListener('input', () => {
    state.thresholds[key] = input.value / scale;
    input.nextElementSibling.textContent = state.thresholds[key].toFixed(2);
  });
}

$('btn-record').addEventListener('click', toggleRecording);
$('btn-download').addEventListener('click', downloadDataset);
$('btn-clear').addEventListener('click', () => { recorder.clear(); $('btn-download').disabled = true; });
$('btn-engage').addEventListener('click', () => (state.engaged ? disengage() : engage()));
$('btn-reset-score').addEventListener('click', () => { state.results = []; renderScore(); });
$('model-file').addEventListener('change', (e) => {
  const f = e.target.files[0];
  if (f) loadPolicy(() => Policy.fromFile(f), f.name);
  e.target.value = '';
});

$('btn-arch').addEventListener('click', () => {
  if (!state.policy) loadPolicy(() => Policy.fromURL('model.json'), 'model.json (bundled)');
  $('arch').showModal();
});
$('arch-close').addEventListener('click', () => $('arch').close());

// Drop a model.json anywhere.
let dragDepth = 0;
addEventListener('dragenter', (e) => { e.preventDefault(); dragDepth++; $('dropveil').classList.add('show'); });
addEventListener('dragleave', () => { if (--dragDepth <= 0) { dragDepth = 0; $('dropveil').classList.remove('show'); } });
addEventListener('dragover', (e) => e.preventDefault());
addEventListener('drop', (e) => {
  e.preventDefault();
  dragDepth = 0;
  $('dropveil').classList.remove('show');
  const f = e.dataTransfer.files[0];
  if (!f) return;
  if (!f.name.endsWith('.json')) return toast('Drop a model.json file');
  setMode('autopilot');
  loadPolicy(() => Policy.fromFile(f), f.name);
});

const DRIVE_KEYS = ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Space'];
addEventListener('keydown', (e) => {
  if (e.target.closest?.('input, select, textarea') || $('arch').open) return;
  if (DRIVE_KEYS.includes(e.code)) {
    e.preventDefault();
    if (state.engaged) { disengage(); toast('Manual control'); }
    car.setKey(e.code, true);
    return;
  }
  if (e.repeat) return;
  switch (e.code) {
    case 'Digit1': setIntent(0); break;
    case 'Digit2': setIntent(1); break;
    case 'KeyC':
      viewCam = viewCam === chaseCam ? driverCam : chaseCam;
      $('camera-label').firstChild.textContent = viewCam === chaseCam ? 'Chase cam · ' : 'Driver cam · ';
      break;
    case 'KeyR': if (state.mode === 'drive') toggleRecording(); break;
    case 'KeyD': if (state.mode === 'drive' && !recorder.recording) downloadDataset(); break;
    case 'KeyP': state.engaged ? disengage() : (state.policy ? engage() : setMode('autopilot')); break;
  }
});
addEventListener('keyup', (e) => car.setKey(e.code, false));
addEventListener('blur', () => DRIVE_KEYS.forEach((k) => car.setKey(k, false)));

addEventListener('resize', () => {
  renderer.setSize(innerWidth, innerHeight);
  for (const c of [driverCam, chaseCam]) { c.aspect = innerWidth / innerHeight; c.updateProjectionMatrix(); }
});

// ---------- loop ----------

const clock = new THREE.Clock();
let frames = 0, fpsClock = 0, previewTick = 0;

function frame() {
  requestAnimationFrame(frame);
  const dt = clock.getDelta();
  if ((fpsClock += dt) >= 1) { $('stat-fps').textContent = frames; frames = 0; fpsClock = 0; }
  frames++;

  car.update(dt, world.colliders);
  car.placeDriverCamera(driverCam);
  car.placeChaseCamera(chaseCam);

  let outcome = car.collided ? 'crash' : car.position.z < -63 ? 'finished' : null;
  // car.update skips frames longer than 0.1 s, so only judge progress on real steps.
  if (!outcome && state.engaged && dt <= 0.1) outcome = trackRun(dt);
  if (outcome) {
    finishRun(outcome);
    car.reset(world.start);
    state.chunk = [];
  } else if (state.engaged) {
    autopilotStep();
  }

  if (state.mode === 'drive') recorder.tick(car, state.intent);
  if (state.mode === 'drive') $('stat-samples').textContent = recorder.count;

  // Refresh the preview every 4th frame when the autopilot isn't already grabbing.
  if (!state.engaged && ++previewTick % 4 === 0) feed.grab().drawTo(fpvCtx);

  $('stat-speed').textContent = Math.abs(car.speed).toFixed(1);
  $('stat-steer').textContent = car.steering.toFixed(2);

  renderer.render(scene, viewCam);
}

renderScore();
frame();
