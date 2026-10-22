import * as THREE from 'three';

// Renders the driver camera into a small square target and exposes it as
// (a) a CHW float tensor for the policy, (b) a JPEG data URL for datasets,
// (c) a preview canvas. Pixels are read bottom-up from WebGL and flipped.

export class CameraFeed {
  constructor(renderer, scene, camera, size = 128) {
    this.renderer = renderer;
    this.scene = scene;
    this.camera = camera;
    this.canvas = document.createElement('canvas');
    this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });
    this.hidden = []; // objects left out of the policy's view (e.g. the car body)
    this.resize(size);
  }

  resize(size) {
    this.size = size;
    this.target?.dispose();
    this.target = new THREE.WebGLRenderTarget(size, size, {
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      format: THREE.RGBAFormat,
    });
    this.pixels = new Uint8Array(size * size * 4);
    this.tensor = new Float32Array(3 * size * size);
    this.canvas.width = this.canvas.height = size;
    this.image = this.ctx.createImageData(size, size);
  }

  // Render the driver view and copy it (flipped to top-down) into `this.image`.
  grab() {
    const { renderer, size: S } = this;
    const prev = renderer.getRenderTarget();
    for (const o of this.hidden) o.visible = false;
    renderer.setRenderTarget(this.target);
    renderer.render(this.scene, this.camera);
    renderer.setRenderTarget(prev);
    for (const o of this.hidden) o.visible = true;
    renderer.readRenderTargetPixels(this.target, 0, 0, S, S, this.pixels);
    const row = S * 4;
    for (let y = 0; y < S; y++) {
      this.image.data.set(this.pixels.subarray((S - 1 - y) * row, (S - y) * row), y * row);
    }
    this.ctx.putImageData(this.image, 0, 0);
    return this;
  }

  toTensor() {
    const S = this.size, plane = S * S, d = this.image.data, t = this.tensor;
    for (let i = 0; i < plane; i++) {
      t[i] = d[i * 4] / 255;
      t[plane + i] = d[i * 4 + 1] / 255;
      t[2 * plane + i] = d[i * 4 + 2] / 255;
    }
    return t;
  }

  toJPEG() {
    return this.canvas.toDataURL('image/jpeg', 0.8);
  }

  drawTo(ctx) {
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(this.canvas, 0, 0, ctx.canvas.width, ctx.canvas.height);
  }
}
