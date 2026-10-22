# Fork Road Pilot

Static web app that runs the language-conditioned ACT + CVAE policy from `train.py` live in the browser.

- **Drive & Record**: drive the fork road yourself and record `(frame, instruction, keys)` datasets in the JSON format `train.py` reads (`DATA_DIR=...`).
- **Autopilot**: loads `model.json` (bundled, or drop your own), runs the CNN + transformer in plain JS, plays each 5-step action chunk, and scores whether the car took the instructed road.
- **Model**: architecture diagram plus the hyperparameters read from the loaded model.

## Run locally

```bash
python3 -m http.server 5174 --directory app
```

Then open http://localhost:5174.

## Deploy

The folder has no build step. Deploy `app/` as a static site, for example with `vercel deploy app`.

## Notes

- The scene, colors, seeded scenery, driver camera and car physics must match the training setup. The CNN only works on frames that look like its training data, so don't restyle `js/world.js` or `js/car.js` without retraining.
- After retraining in Colab, replace `app/model.json` with the new export (or drag it onto the page).
- `js/policy.js` matches the reference JS implementation to within 5e-7 on the logits.
