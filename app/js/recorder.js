// Records (driver-camera frame, language intent, key actions) samples in the
// JSON layout train.py reads: { metadata: {...}, samples: [{ image, language_id, actions, ... }] }.

export const INTENTS = [
  { label: 'always_take_left', text: 'Take Left' },
  { label: 'always_take_right', text: 'Take Right' },
];

// Straight driving dominates a lap, so keep only every Nth forward-only frame.
const FORWARD_KEEP_EVERY = 10;

export class Recorder {
  constructor(feed) {
    this.feed = feed;
    this.samples = [];
    this.recording = false;
    this._forwardRun = 0;
  }

  get count() { return this.samples.length; }

  start() { this.recording = true; }
  stop() { this.recording = false; }
  clear() { this.samples = []; }

  tick(car, languageId) {
    if (!this.recording) return;
    const a = car.actions;
    const turning = a.left || a.right;
    if (!turning) {
      if (!a.forward || ++this._forwardRun < FORWARD_KEEP_EVERY) return;
      this._forwardRun = 0;
    }
    this.samples.push({
      timestamp: performance.now(),
      image: this.feed.grab().toJPEG(),
      language_id: languageId,
      language_text: INTENTS[languageId].text,
      actions: a,
      speed: car.speed / 18,
      steering: car.steering,
    });
  }

  download() {
    if (!this.count) return;
    const S = this.feed.size;
    const dataset = {
      metadata: {
        image_width: S,
        image_height: S,
        frame_stack: 1,
        num_intents: INTENTS.length,
        intent_labels: INTENTS.map((i) => i.label),
        intent_texts: INTENTS.map((i) => i.text),
        num_samples: this.count,
        created: new Date().toISOString(),
      },
      samples: this.samples,
    };
    const url = URL.createObjectURL(new Blob([JSON.stringify(dataset)], { type: 'application/json' }));
    Object.assign(document.createElement('a'), { href: url, download: `fork_road_${S}px_${Date.now()}.json` }).click();
    URL.revokeObjectURL(url);
  }
}
