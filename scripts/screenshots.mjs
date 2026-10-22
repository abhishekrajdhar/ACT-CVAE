// Regenerates docs/screenshots/*.png with headless Chrome (GPU) over the DevTools protocol.
// Serve the app first:  python3 -m http.server 5174 --directory app
// Then:                 node scripts/screenshots.mjs [outDir] [appUrl]
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const OUT = process.argv[2] ?? 'docs/screenshots';
const URL = (process.argv[3] ?? 'http://127.0.0.1:5174/') + '?debug';
const CHROME = process.env.CHROME ?? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const port = 9333;
const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'shots-'));
const chrome = spawn(CHROME, [
  '--headless=new', `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`,
  '--use-angle=metal', '--enable-gpu', '--ignore-gpu-blocklist',
  '--hide-scrollbars', '--no-first-run', 'about:blank',
], { stdio: 'ignore' });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let targets;
for (let i = 0; i < 50; i++) {
  try { targets = await (await fetch(`http://127.0.0.1:${port}/json`)).json(); break; } catch { await sleep(200); }
}
const page = targets.find((t) => t.type === 'page');
const ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((r) => (ws.onopen = r));
let id = 0;
const pending = new Map();
ws.onmessage = (m) => {
  const d = JSON.parse(m.data);
  if (d.id && pending.has(d.id)) { pending.get(d.id)(d); pending.delete(d.id); }
};
const send = (method, params = {}) => new Promise((res, rej) => {
  const i = ++id;
  pending.set(i, (d) => (d.error ? rej(new Error(`${method}: ${d.error.message}`)) : res(d.result)));
  ws.send(JSON.stringify({ id: i, method, params }));
});
const js = async (expr) => {
  const r = await send('Runtime.evaluate', { expression: `(async () => { ${expr} })()`, awaitPromise: true, returnByValue: true });
  if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description || 'eval failed');
  return r.result.value;
};
const shot = async (name, clip) => {
  await js(`document.getElementById('toast').classList.remove('show'); await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));`);
  await sleep(300);
  const { data } = await send('Page.captureScreenshot', { format: 'png', ...(clip ? { clip: { ...clip, scale: 1 } } : {}) });
  fs.writeFileSync(path.join(OUT, `${name}.png`), Buffer.from(data, 'base64'));
  console.log('saved', name);
};

await send('Page.enable');
await send('Emulation.setDeviceMetricsOverride', { width: 1440, height: 900, deviceScaleFactor: 2, mobile: false });
await send('Page.navigate', { url: URL });
await sleep(2500);
await js(`await document.fonts.ready; fork.pause();`);

const key = (code) => `dispatchEvent(new KeyboardEvent('keydown', { code: '${code}' })); dispatchEvent(new KeyboardEvent('keyup', { code: '${code}' }));`;

// 1. Hero: chase cam rolling up to the fork
await js(`fork.car.press(true,false,false,false); fork.run(40); fork.car.press(false,false,false,false); fork.run(2);`);
await shot('hero');

// 2. Recording: left instruction, steering into the left branch
await js(`fork.car.reset({ position: fork.car.position.clone().set(0,0,34), heading: Math.PI });
  ${key('KeyR')}
  fork.car.press(true,false,false,false); fork.run(30);
  fork.car.press(true,false,false,true); fork.run(14);
  fork.car.press(true,false,false,false); fork.run(10);
  fork.car.press(true,false,false,true); fork.run(10);`);
await shot('record');
await js(`${key('KeyR')} document.getElementById('btn-clear').click(); fork.car.press(false,false,false,false);`);

// 3. Autopilot: one finished "left" run, one "right" run, then mid-way through a left run
await js(`fork.setMode('autopilot'); while (!fork.state.policy) await new Promise(r => setTimeout(r, 100));
  const until = (n) => { for (let i = 0; i < 3000 && fork.state.results.length < n; i++) fork.run(1); };
  fork.setIntent(0); fork.engage(); until(1);
  fork.setIntent(1); until(2);
  fork.setIntent(0); until(3);
  fork.run(150);`);
await shot('autopilot');

// 4. Driver camera, right instruction on the right branch
await js(`const n = fork.state.results.length; for (let i = 0; i < 3000 && fork.state.results.length === n; i++) fork.run(1);
  fork.setIntent(1); fork.run(170); ${key('KeyC')} fork.run(1);`);
await shot('driver-cam');
await js(`${key('KeyC')} ${key('KeyP')} fork.run(1);`);

// 5. Architecture dialog
await js(`document.getElementById('btn-arch').click();`);
await sleep(300);
await shot('architecture');
await js(`document.getElementById('arch').close();`);

// 6. F1 car close-up
await js(`
  const THREE = await import('three'); const { Car } = await import('/js/car.js');
  const r = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  r.setPixelRatio(2); r.setSize(1200, 560);
  const s = new THREE.Scene(); s.background = new THREE.Color(0x87ceeb);
  s.add(new THREE.HemisphereLight(0xdff1ff, 0x4a7c4f, 1.6));
  const sun = new THREE.DirectionalLight(0xffffff, 2.2); sun.position.set(30, 60, 20); s.add(sun);
  const road = new THREE.Mesh(new THREE.PlaneGeometry(80, 80), new THREE.MeshBasicMaterial({ color: 0x3a3a3a }));
  road.rotation.x = -Math.PI / 2; s.add(road);
  const car = new Car(s); car.reset({ position: new THREE.Vector3(), heading: 0 });
  const cam = new THREE.PerspectiveCamera(30, 1200 / 560, 0.1, 100);
  cam.position.set(6.2, 2.4, 6.0); cam.lookAt(0, 0.45, 0.1); r.render(s, cam);
  const el = r.domElement; el.id = 'closeup';
  el.style.cssText = 'position:fixed;left:0;top:0;width:1200px;height:560px;z-index:999';
  document.body.appendChild(el);`);
await shot('f1-car', { x: 0, y: 0, width: 1200, height: 560 });

ws.close();
chrome.kill();
fs.rmSync(profile, { recursive: true, force: true });
