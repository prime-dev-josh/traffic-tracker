# macOS (later)

Windows is the primary ship target. The same Python app can run on Mac for development and a later `.app`.

## Runtime

[`traffic_tracker/core/device.py`](../traffic_tracker/core/device.py) already prefers **CUDA**, then **MPS** (Apple Silicon), then CPU.

| Chip | Recommended weights | Notes |
|------|---------------------|--------|
| Apple Silicon | `.pt` on **MPS**, or `yolo export format=coreml` then load the `.mlpackage` via Ultralytics | Measure both; keep ByteTrack in `model.track()` |
| Intel Mac | ONNX CPU or `.pt` CPU | OpenVINO is Intel-oriented; less relevant on Apple Silicon |

```bash
# CoreML export when you are ready to test on a Mac
yolo export model=yolo11n.pt format=coreml imgsz=640
```

Set `TRAFFIC_TRACKER_MODEL` to the `.mlpackage` path if you do not want AutoBackend’s default file search.

## Packaging checklist (when you ship)

1. Build on a Mac with Python 3.11 and the project venv.
2. PyInstaller one-folder / `.app` (same idea as [`TrafficTracker.spec`](../TrafficTracker.spec); collect PyQt6 or PySide6 plugins).
3. Bundle `yolo11n.onnx` or the CoreML package.
4. **Apple Developer ID** sign the `.app`.
5. **Notarize** and staple if you distribute outside the Mac App Store / your org’s MDM.
6. Test MPS vs CPU on an M-series machine and confirm TMC parity with Windows ONNX on the same clip (`TRAFFIC_TRACKER_GOLDEN_VIDEO`).

Hard parts are **signing and notarization**, not the counting code.

## What not to do

Do not start a second Electron or Swift UI just for Mac. PyQt/PySide is already cross-platform.
