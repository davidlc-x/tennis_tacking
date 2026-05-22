# CourtSpeed Tennis Tracker

CourtSpeed is a phone-friendly web app for recording tennis from the back of the court and estimating ball speed from camera tracking. It is inspired by the public feature set of SwingVision: single-camera capture, court calibration, ball tracking, shot speed, rally counts, and exportable session data.

## What it does

- Opens the rear phone camera with `getUserMedia`.
- Lets you tap four visible court corners to build a court homography.
- Can attempt automatic calibration from visible court sidelines and baselines.
- Detects the ball with a local YOLOv8 ONNX model, with color/blob tracking as a fallback.
- Projects detections onto a real tennis court plane.
- Estimates current, peak, serve, and return speed.
- Segments likely serve, return, and rally shots.
- Labels each shot with an approximate court location.
- Includes a virtual benchmark feed with known ground-truth ball position and speed.
- Shows an overhead court trail and exports shot data as JSON.
- Installs as a PWA on supported mobile browsers.

## Running locally

From this folder:

```powershell
python -m http.server 5173
```

Open:

```text
http://localhost:5173
```

Camera access works on `localhost`. To test from an actual phone over Wi-Fi, serve the app over HTTPS or use a secure tunnel such as Cloudflare Tunnel, ngrok, or a deployed static host. Mobile browsers usually block camera access on plain `http://192.168.x.x`.

## Court setup

Mount the phone behind one baseline with the full court visible. A higher fence mount improves the court projection and speed estimates. Tap the four court corners in this order:

1. Near-left baseline corner
2. Near-right baseline corner
3. Far-right baseline corner
4. Far-left baseline corner

Use doubles mode if you tap doubles corners; use singles mode if you tap singles corners.

## Auto calibration

Tap Camera first, then Auto. The app captures the current video frame, finds bright low-saturation court-line pixels, runs a Hough-style line search, and selects the most plausible pair of sidelines plus two cross-court lines. If it succeeds, blue guide lines and four calibration points appear on the video. If it fails, use Manual and tap the four corners.

Auto calibration works best when:

- The full court is visible.
- The phone is behind the baseline, ideally high on the fence.
- Court lines contrast clearly against the court surface.
- Players, bags, and fences do not cover the baseline corners.

## Virtual benchmark

Tap Sim to run a generated court feed through the same tracking pipeline. The simulator auto-calibrates the synthetic court, starts tracking, and reports mean position error and mean speed error in the stats panel. It uses a clean generated feed, so it is mainly for testing geometry, speed math, smoothing, and app regressions. It does not prove real-world camera accuracy, lighting robustness, or model performance against actual match footage.

## Accuracy notes

This prototype estimates speed from a single RGB camera and a 2D court-plane projection. It cannot fully recover ball height or true 3D trajectory, so speeds are approximate and depend heavily on phone frame rate, camera placement, lighting, calibration, and whether the ball is visible against the background. A production SwingVision-class system would add trained ball/court/player models, temporal tracking, bounce/hit classifiers, camera pose estimation, and server-side review.

## Model benchmarks to try next

The built-in auto calibration is the no-API court baseline. The app now includes a local `RJTPP/tennis-ball-detection` YOLOv8 ONNX model in `models/tennis-ball-yolov8n.onnx` and runs it in-browser through ONNX Runtime Web. The Detector setting defaults to Hybrid model, which runs YOLO periodically and uses the lightweight tracker between model frames.

For the next ML benchmark, the most useful addition is a court keypoint model that returns known tennis court points, then feeds those points into the same homography code.

Useful candidates:

- Roboflow tennis court keypoint models for court landmarks.
- Hugging Face YOLOv8/ONNX tennis-ball detectors for ball detection.
- TrackNet-style ball tracking for high-speed tiny-object sports tracking.

The practical production path is usually hybrid: keypoint model for the court, detector or heatmap tracker for the ball, then geometric filtering and temporal smoothing.

## Deploying to iPhone or iPad

Camera access requires a secure context. `http://localhost` works only on the same machine, but an iPhone or iPad opening your computer's LAN IP usually needs HTTPS.

Fastest options:

1. Deploy this folder as a static site on Netlify, Vercel, Cloudflare Pages, or GitHub Pages.
2. Or run the local server and expose it through an HTTPS tunnel such as Cloudflare Tunnel or ngrok.
3. Open the HTTPS URL in Safari on the iPhone or iPad.
4. Tap Share, then Add to Home Screen.
5. Launch CourtSpeed from the Home Screen and allow camera access.

For App Store/TestFlight later, wrap the same web app with Capacitor and build the iOS project in Xcode on a Mac.
