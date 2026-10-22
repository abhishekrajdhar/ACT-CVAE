import * as THREE from 'three';

// Fork-road world. The policy's CNN was trained on frames rendered from this
// exact scene, so layout, colors and the seeded scenery must stay unchanged or
// the model will see out-of-distribution images.

const ROAD_WIDTH = 10;
const GROUND_SIZE = 200;
const V = (x, z) => new THREE.Vector3(x, 0, z);

// Seeded PRNG (mulberry32) so scenery is identical on every load.
function seeded(seed) {
  return () => {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const basic = (color, opts = {}) => new THREE.MeshBasicMaterial({ color, ...opts });

export class World {
  constructor(scene) {
    this.scene = scene;
    this.colliders = [];

    const curve = (pts) => new THREE.CatmullRomCurve3(pts, false, 'catmullrom', 0.5);
    this.approachCurve = curve([V(0, 60), V(0, 45), V(0, 30), V(0, 24)]);
    this.leftCurve = curve([V(0, 30), V(-8, 22), V(-18, 10), V(-22, -5), V(-18, -20), V(-8, -32), V(0, -40)]);
    this.rightCurve = curve([V(0, 30), V(8, 22), V(18, 10), V(22, -5), V(18, -20), V(8, -32), V(0, -40)]);
    this.exitCurve = curve([V(0, -34), V(0, -40), V(0, -55), V(0, -65)]);

    this.approachPts = this.approachCurve.getPoints(60);
    this.exitPts = this.exitCurve.getPoints(60);
    this.leftPts = this.leftCurve.getPoints(120);
    this.rightPts = this.rightCurve.getPoints(120);

    // Branch road surfaces skip their ends so they don't overlap the shared road.
    const trim = (pts) => pts.slice(Math.round(pts.length * 0.06), Math.round(pts.length * 0.94));

    scene.background = new THREE.Color(0x87ceeb);
    scene.fog = new THREE.Fog(0x87ceeb, 100, 220);
    this._ground();

    this._ribbon(this.approachPts, 0, ROAD_WIDTH, 0.01, 0x3a3a3a);
    this._ribbon(this.exitPts, 0, ROAD_WIDTH, 0.01, 0x3a3a3a);
    this._ribbon(trim(this.leftPts), 0, ROAD_WIDTH, 0.01, 0x2a4a5a);
    this._ribbon(trim(this.rightPts), 0, ROAD_WIDTH, 0.01, 0x5a3a2a);
    this._edges(this.approachPts, 0xffffff);
    this._edges(this.leftPts, 0x66bbff);
    this._edges(this.rightPts, 0xffaa44);
    this._edges(this.exitPts, 0xffffff);
    this._dashes(this.approachPts);
    this._dashes(this.exitPts);

    this._centerObstacles();
    this._endMarkers();
    this._leftScenery();
    this._rightScenery();
    this._sharedScenery();
    this._boundary();
  }

  get start() { return { position: V(0, 34), heading: Math.PI }; }

  _add(geo, mat, x, y, z) {
    const m = new THREE.Mesh(geo, mat);
    m.position.set(x, y, z);
    this.scene.add(m);
    return m;
  }

  _solid(mesh) {
    this.colliders.push(new THREE.Box3().setFromObject(mesh));
  }

  _ground() {
    const g = this._add(new THREE.PlaneGeometry(GROUND_SIZE, GROUND_SIZE), basic(0x4a7c4f), 0, -0.05, 0);
    g.rotation.x = -Math.PI / 2;
  }

  // Flat strip following `pts`, centred `offset` units to the side, `width` wide.
  _ribbon(pts, offset, width, y, color) {
    const hw = width / 2;
    const pos = [];
    const perpAt = (a, b) => {
      const t = new THREE.Vector3().subVectors(b, a).normalize();
      return new THREE.Vector3(-t.z, 0, t.x);
    };
    for (let i = 0; i < pts.length - 1; i++) {
      const cur = pts[i], nxt = pts[i + 1];
      const p0 = perpAt(pts[Math.max(0, i - 1)], nxt);
      const p1 = perpAt(cur, pts[Math.min(pts.length - 1, i + 2)]);
      const c = cur.clone().addScaledVector(p0, offset);
      const n = nxt.clone().addScaledVector(p1, offset);
      const cL = c.clone().addScaledVector(p0, hw), cR = c.clone().addScaledVector(p0, -hw);
      const nL = n.clone().addScaledVector(p1, hw), nR = n.clone().addScaledVector(p1, -hw);
      pos.push(cL.x, y, cL.z, cR.x, y, cR.z, nR.x, y, nR.z, cL.x, y, cL.z, nR.x, y, nR.z, nL.x, y, nL.z);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    this.scene.add(new THREE.Mesh(geo, basic(color, { side: THREE.DoubleSide })));
  }

  _edges(pts, color) {
    for (const side of [1, -1]) this._ribbon(pts, (ROAD_WIDTH / 2 - 0.15) * side, 0.3, 0.025, color);
  }

  _dashes(pts) {
    const hw = 0.1, y = 0.03, pos = [];
    let run = 0, drawing = true;
    for (let i = 0; i < pts.length - 1; i++) {
      const cur = pts[i], nxt = pts[i + 1];
      if (drawing) {
        const t = new THREE.Vector3().subVectors(nxt, pts[Math.max(0, i - 1)]).normalize();
        const px = -t.z * hw, pz = t.x * hw;
        pos.push(
          cur.x + px, y, cur.z + pz, cur.x - px, y, cur.z - pz, nxt.x - px, y, nxt.z - pz,
          cur.x + px, y, cur.z + pz, nxt.x - px, y, nxt.z - pz, nxt.x + px, y, nxt.z + pz,
        );
      }
      run += cur.distanceTo(nxt);
      if (run >= 3) { run = 0; drawing = !drawing; }
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    this.scene.add(new THREE.Mesh(geo, basic(0xf5c542, { side: THREE.DoubleSide })));
  }

  _centerObstacles() {
    const bark = basic(0x5c3a1e);
    // Big tree at the fork that forces a left/right decision.
    this._add(new THREE.CylinderGeometry(0.8, 1.0, 4, 8), bark, 0, 2, 5);
    this._add(new THREE.SphereGeometry(5, 12, 8), basic(0x1a6b2e), 0, 4 + 5 * 0.6, 5);
    this.colliders.push(new THREE.Box3(new THREE.Vector3(-2, 0, 3), new THREE.Vector3(2, 10, 7)));

    // Smaller trees inside the loop to block shortcuts.
    const cone = basic(0x2d8a4e);
    for (const [x, z] of [[0, -5], [0, -15], [-3, 15], [3, 15]]) {
      this._add(new THREE.CylinderGeometry(0.3, 0.4, 2, 6), bark, x, 1, z);
      this._add(new THREE.ConeGeometry(2, 3, 8), cone, x, 3, z);
      this.colliders.push(new THREE.Box3(new THREE.Vector3(x - 1, 0, z - 1), new THREE.Vector3(x + 1, 5, z + 1)));
    }
  }

  _endMarkers() {
    this._add(new THREE.BoxGeometry(8, 3, 0.5), basic(0x2563eb), 0, 1.5, 63);
    this._add(new THREE.BoxGeometry(8, 3, 0.5), basic(0xdc2626), 0, 1.5, -68);
  }

  // Left branch: white birches with blue/purple canopies, blue lamps, blue building.
  _leftScenery() {
    const rand = seeded(77);
    const bark = basic(0xd4cfc4);
    const leaves = [basic(0x3366aa), basic(0x5544aa), basic(0x4477bb)];
    for (let i = 0; i < 25; i++) {
      const x = -8 - rand() * 40;
      const z = (rand() - 0.5) * 120;
      if (this._distToRoad(x, z) < ROAD_WIDTH / 2 + 3) continue;
      const trunkH = 4 + rand() * 4;
      const r = 1.2 + rand();
      rand(); // keeps the seeded sequence aligned with the training scene
      this._add(new THREE.CylinderGeometry(0.15, 0.25, trunkH, 6), bark, x, trunkH / 2, z);
      this._add(new THREE.SphereGeometry(r, 8, 6), leaves[Math.floor(rand() * leaves.length)], x, trunkH + r * 0.5, z);
    }

    const pole = basic(0x334488), bulb = basic(0x88bbff);
    for (let t = 0.15; t < 0.85; t += 0.2) {
      const p = this.leftCurve.getPoint(t);
      const d = this.leftCurve.getTangent(t).normalize();
      const x = p.x + d.z * 7, z = p.z - d.x * 7;
      if (this._distToRoad(x, z) < ROAD_WIDTH / 2 + 1) continue;
      this._add(new THREE.CylinderGeometry(0.15, 0.15, 4, 6), pole, x, 2, z);
      this._add(new THREE.SphereGeometry(0.5, 6, 4), bulb, x, 4.3, z);
    }

    this._solid(this._add(new THREE.BoxGeometry(10, 14, 8), basic(0x4466aa), -38, 7, 0));
  }

  // Right branch: low orange/red/yellow bushes, wooden fence, warm building.
  _rightScenery() {
    const rand = seeded(99);
    const bushes = [basic(0xcc6633), basic(0xbb4422), basic(0xddaa33), basic(0xcc8844)];
    for (let i = 0; i < 30; i++) {
      const x = 8 + rand() * 40;
      const z = (rand() - 0.5) * 120;
      if (this._distToRoad(x, z) < ROAD_WIDTH / 2 + 3) continue;
      const r = 0.8 + rand() * 1.2;
      this._add(new THREE.SphereGeometry(r, 8, 6), bushes[Math.floor(rand() * bushes.length)], x, r * 0.7, z);
    }

    const postMat = basic(0x8b6914), railMat = basic(0xa07828);
    const posts = [];
    for (let t = 0.1; t < 0.9; t += 0.12) {
      const p = this.rightCurve.getPoint(t);
      const d = this.rightCurve.getTangent(t).normalize();
      const x = p.x - d.z * 7, z = p.z + d.x * 7;
      if (this._distToRoad(x, z) < ROAD_WIDTH / 2 + 1) continue;
      this._add(new THREE.BoxGeometry(0.3, 2, 0.3), postMat, x, 1, z);
      posts.push(new THREE.Vector3(x, 1.5, z));
    }
    for (let i = 0; i < posts.length - 1; i++) {
      const a = posts[i], b = posts[i + 1];
      const mid = a.clone().add(b).multiplyScalar(0.5);
      const rail = this._add(new THREE.BoxGeometry(0.15, 0.15, a.distanceTo(b)), railMat, mid.x, mid.y, mid.z);
      rail.rotation.y = Math.atan2(b.x - a.x, b.z - a.z);
    }

    this._solid(this._add(new THREE.BoxGeometry(8, 10, 10), basic(0xbb6633), 38, 5, 0));
  }

  _sharedScenery() {
    const rand = seeded(55);
    const bark = basic(0x5c3a1e), leaf = basic(0x2d8a4e);
    for (let i = 0; i < 15; i++) {
      const x = (rand() - 0.5) * 160;
      const z = (rand() - 0.5) * 160;
      if (Math.abs(x) < 45 && Math.abs(z) < 50) continue;
      if (this._distToRoad(x, z) < ROAD_WIDTH / 2 + 4) continue;
      const trunkH = 1.5 + rand();
      const r = 1.5 + rand() * 1.5;
      const h = 2.5 + rand() * 2;
      this._add(new THREE.CylinderGeometry(0.3, 0.4, trunkH, 6), bark, x, trunkH / 2, z);
      this._add(new THREE.ConeGeometry(r, h, 8), leaf, x, trunkH + h / 2 - 0.3, z);
    }

    for (const b of [
      { x: -30, z: 50, w: 8, d: 6, h: 10, c: 0x8eacc7 },
      { x: 30, z: 50, w: 7, d: 8, h: 12, c: 0xd4a88c },
      { x: -30, z: -55, w: 9, d: 7, h: 8, c: 0xa3c4a8 },
      { x: 30, z: -55, w: 6, d: 9, h: 14, c: 0xc9a3c9 },
    ]) {
      this._solid(this._add(new THREE.BoxGeometry(b.w, b.h, b.d), basic(b.c), b.x, b.h / 2, b.z));
    }
  }

  _boundary() {
    const e = 85, h = 5, t = 2;
    for (const [x, z, sx, sz] of [[0, e, e * 2, t], [0, -e, e * 2, t], [e, 0, t, e * 2], [-e, 0, t, e * 2]]) {
      this.colliders.push(new THREE.Box3(new THREE.Vector3(x - sx / 2, 0, z - sz / 2), new THREE.Vector3(x + sx / 2, h, z + sz / 2)));
    }
  }

  // Coarse distance to the nearest road centreline (every 5th sample point).
  _distToRoad(x, z) {
    let best = Infinity;
    for (const pts of [this.approachPts, this.leftPts, this.rightPts, this.exitPts]) {
      for (let i = 0; i < pts.length; i += 5) {
        best = Math.min(best, Math.hypot(x - pts[i].x, z - pts[i].z));
      }
    }
    return best;
  }
}
