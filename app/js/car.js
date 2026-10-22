import * as THREE from 'three';

// Arcade car. Physics constants and camera mounts match the setup the training
// data was recorded with; changing them shifts what the policy sees.

const MAX_SPEED = 18;
const MAX_REVERSE = 8;
const ACCEL = 14;
const BRAKE = 30;
const FRICTION = 8;
const TURN_RATE = 2.2;
const STEER_SMOOTHING = 10;

const BODY = { w: 2.2, h: 1.0, l: 4.0 }; // collision box + camera mount reference
const UP = new THREE.Vector3(0, 1, 0);

export class Car {
  constructor(scene) {
    this.position = new THREE.Vector3();
    this.heading = 0;
    this.speed = 0;
    this.steering = 0;
    this.collided = false;
    this.input = { forward: false, backward: false, left: false, right: false, brake: false };
    this.autoSteer = null; // continuous steer in [-1, 1] set by the autopilot

    this.group = this._buildMesh();
    scene.add(this.group);
    this._box = new THREE.Box3();
    this._prev = new THREE.Vector3();
  }

  // Open-wheel F1-style body (+z is forward). Purely cosmetic: collisions still
  // use the BODY box, and the car is hidden from the policy's camera.
  _buildMesh() {
    const g = new THREE.Group();
    const mat = (color, roughness = 0.45, metalness = 0.2) => new THREE.MeshStandardMaterial({ color, roughness, metalness });
    const red = mat(0xd0101a, 0.35, 0.3);
    const white = mat(0xf2f2f2, 0.4);
    const carbon = mat(0x18191c, 0.6, 0.4);
    const add = (geo, m, x, y, z, rot) => {
      const mesh = new THREE.Mesh(geo, m);
      mesh.position.set(x, y, z);
      if (rot) mesh.rotation.set(...rot);
      g.add(mesh);
      return mesh;
    };
    const box = (w, h, l, m, x, y, z, rot) => add(new THREE.BoxGeometry(w, h, l), m, x, y, z, rot);

    // Floor, diffuser and plank
    box(1.5, 0.06, 3.4, carbon, 0, 0.12, -0.2);
    box(1.3, 0.22, 0.5, carbon, 0, 0.2, -1.95, [0.25, 0, 0]);

    // Tub, cockpit and tapered nose (4-sided cylinder reads as a wedge)
    box(0.78, 0.42, 1.7, red, 0, 0.38, 0.35);
    box(0.5, 0.06, 0.75, carbon, 0, 0.6, 0.45);
    const nose = add(new THREE.CylinderGeometry(0.13, 0.4, 1.7, 4), red, 0, 0.3, 1.95, [Math.PI / 2, Math.PI / 4, 0]);
    nose.scale.set(1, 1, 0.5);
    box(0.08, 0.02, 1.4, white, 0, 0.37, 1.8, [-0.08, 0, 0]);

    // Front wing: two elements plus endplates
    box(2.1, 0.05, 0.48, carbon, 0, 0.13, 2.55);
    box(1.9, 0.04, 0.28, red, 0, 0.22, 2.48, [-0.35, 0, 0]);
    for (const s of [-1, 1]) box(0.04, 0.3, 0.62, carbon, s * 1.05, 0.22, 2.52);

    // Sidepods with dark intakes, engine cover and airbox
    for (const s of [-1, 1]) {
      box(0.5, 0.38, 1.6, red, s * 0.62, 0.34, -0.35);
      box(0.42, 0.3, 0.04, carbon, s * 0.62, 0.36, 0.46);
      box(0.06, 0.06, 0.26, carbon, s * 0.55, 0.78, 0.62); // mirror
      box(0.22, 0.1, 0.04, carbon, s * 0.66, 0.8, 0.62);
    }
    box(0.62, 0.5, 1.5, red, 0, 0.6, -0.75, [0.12, 0, 0]);
    box(0.34, 0.3, 0.55, carbon, 0, 0.95, -0.08);
    box(0.03, 0.38, 1.2, white, 0, 0.98, -0.9, [0.12, 0, 0]); // shark fin

    // Driver helmet and halo
    add(new THREE.SphereGeometry(0.17, 16, 12), mat(0xffd60a, 0.3), 0, 0.75, 0.22);
    add(new THREE.TorusGeometry(0.32, 0.035, 8, 20, Math.PI), carbon, 0, 0.88, 0.3, [Math.PI / 2, 0, 0]);
    box(0.05, 0.25, 0.05, carbon, 0, 0.76, 0.64);

    // Rear wing with endplates, DRS flap, pylon and rain light
    box(1.5, 0.05, 0.42, carbon, 0, 1.0, -2.05);
    box(1.5, 0.04, 0.26, red, 0, 1.13, -2.12, [0.4, 0, 0]);
    for (const s of [-1, 1]) box(0.04, 0.6, 0.62, red, s * 0.77, 0.85, -2.05);
    box(0.08, 0.6, 0.12, carbon, 0, 0.7, -1.95);
    box(0.18, 0.08, 0.04, new THREE.MeshBasicMaterial({ color: 0xff2020 }), 0, 0.35, -2.22);

    // Exposed wheels: wide rears, slimmer fronts; groups pivot for steering
    this.wheels = [];
    const tyreMat = mat(0x111111, 0.9, 0);
    const rimMat = mat(0x6b6e75, 0.3, 0.6);
    const stripeMat = new THREE.MeshBasicMaterial({ color: 0xffd60a });
    for (const [x, z, r, w] of [[-0.98, 1.4, 0.34, 0.36], [0.98, 1.4, 0.34, 0.36], [-0.95, -1.5, 0.37, 0.48], [0.95, -1.5, 0.37, 0.48]]) {
      const wheel = new THREE.Group();
      wheel.position.set(x, r, z);
      const tyre = new THREE.Mesh(new THREE.CylinderGeometry(r, r, w, 24), tyreMat);
      tyre.rotation.z = Math.PI / 2;
      const rim = new THREE.Mesh(new THREE.CylinderGeometry(r * 0.62, r * 0.62, w + 0.01, 16), rimMat);
      rim.rotation.z = Math.PI / 2;
      const stripe = new THREE.Mesh(new THREE.TorusGeometry(r * 0.8, 0.02, 6, 24), stripeMat);
      stripe.rotation.y = Math.PI / 2;
      stripe.position.x = Math.sign(x) * (w / 2 + 0.005);
      wheel.add(tyre, rim, stripe);
      g.add(wheel);
      this.wheels.push(wheel);
    }
    // Suspension wishbones from tub to each wheel
    for (const wheel of this.wheels) {
      const { x, y, z } = wheel.position;
      const len = Math.abs(x) - 0.35;
      box(len, 0.03, 0.05, carbon, Math.sign(x) * (0.35 + len / 2), y + 0.05, z);
    }
    return g;
  }

  setKey(code, down) {
    const map = { ArrowUp: 'forward', ArrowDown: 'backward', ArrowLeft: 'left', ArrowRight: 'right', Space: 'brake' };
    if (map[code]) this.input[map[code]] = down;
  }

  // Autopilot command: throttle on/off plus continuous steer (+1 = left).
  drive(forward, steer) {
    Object.assign(this.input, { forward, backward: false, left: false, right: false, brake: false });
    this.autoSteer = steer;
  }

  // Binary command (thresholded autopilot or manual takeover).
  press(forward, backward, left, right) {
    Object.assign(this.input, { forward, backward, left, right });
    this.autoSteer = null;
  }

  reset({ position, heading }) {
    this.position.copy(position);
    this.heading = heading;
    this.speed = 0;
    this.steering = 0;
    this.autoSteer = null;
    this.input = { forward: false, backward: false, left: false, right: false, brake: false };
    this._sync();
  }

  update(dt, colliders) {
    if (dt <= 0 || dt > 0.1) return;
    const k = this.input;
    this._prev.copy(this.position);

    if (k.brake) {
      this.speed = Math.abs(this.speed) < 0.5 ? 0 : this.speed - Math.sign(this.speed) * BRAKE * dt;
    } else if (k.forward) {
      this.speed += ACCEL * dt;
    } else if (k.backward) {
      this.speed -= ACCEL * dt;
    } else {
      this.speed = Math.abs(this.speed) < FRICTION * dt ? 0 : this.speed - Math.sign(this.speed) * FRICTION * dt;
    }
    this.speed = THREE.MathUtils.clamp(this.speed, -MAX_REVERSE, MAX_SPEED);

    const target = this.autoSteer ?? (k.left ? 1 : k.right ? -1 : 0);
    this.steering += (target - this.steering) * (1 - Math.exp(-STEER_SMOOTHING * dt));
    this.heading += this.steering * TURN_RATE * THREE.MathUtils.clamp(Math.abs(this.speed) / 5, 0, 1) * dt;

    this.position.x += Math.sin(this.heading) * this.speed * dt;
    this.position.z += Math.cos(this.heading) * this.speed * dt;

    this.collided = this._collide(colliders);
    this._sync();

    this.wheels[0].rotation.y = this.wheels[1].rotation.y = this.steering * 0.5;
  }

  _collide(colliders) {
    const hw = BODY.w / 2 + 0.3, hl = BODY.l / 2 + 0.1;
    this._box.min.set(this.position.x - hw, 0, this.position.z - hl);
    this._box.max.set(this.position.x + hw, BODY.h + 0.5, this.position.z + hl);
    for (const c of colliders) {
      if (this._box.intersectsBox(c)) {
        this.position.copy(this._prev);
        this.speed *= -0.3;
        if (Math.abs(this.speed) < 1) this.speed = 0;
        return true;
      }
    }
    return false;
  }

  _sync() {
    this.group.position.copy(this.position);
    this.group.rotation.y = this.heading;
  }

  _local(x, y, z) {
    return new THREE.Vector3(x, y, z).applyAxisAngle(UP, this.heading).add(this.position);
  }

  // Driver-eye camera: the view the policy sees (~13° down-tilt).
  placeDriverCamera(cam) {
    cam.position.copy(this._local(0, 1.8, BODY.l / 2 - 0.5));
    cam.lookAt(this._local(0, 0.3, BODY.l / 2 + 6));
  }

  placeChaseCamera(cam) {
    cam.position.copy(this._local(0, 8, -14));
    cam.lookAt(this._local(0, 1, 5));
  }

  get actions() {
    const k = this.input;
    return { forward: +k.forward, backward: +k.backward, left: +k.left, right: +k.right };
  }
}
