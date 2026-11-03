# Packaging Traffic Tracker

Two Windows bundle flavors. **Do not expect a tiny installer while Ultralytics + PyTorch remain in the runtime.**

## Flavor 1 — Current (ship this)

**What:** PyInstaller **one-folder** with PyTorch, Ultralytics, PyQt6, OpenCV.

```powershell
.\build_exe.ps1
```

Output: `dist\Traffic Tracker\` (zip the whole folder). Typical size is **1–3 GB+**.

Bundle weights before building so field PCs do not download models:

1. `.\export_models.ps1` (writes `yolo11n.onnx`)
2. Keep `yolo11n.pt` in the project root if you still want a PyTorch fallback
3. Rebuild

[`TrafficTracker.spec`](../TrafficTracker.spec) collects `torch` on purpose. CUDA GPU machines still need NVIDIA drivers. Shipping CUDA/cuDNN inside the folder keeps the bundle large whether you use PyTorch or `onnxruntime-gpu`.

## Flavor 2 — Later slim CPU (not implemented)

A **hundreds-of-MB** CPU exe is only possible after **removing the Ultralytics Python package and PyTorch** (custom ONNX Runtime or a Rust sidecar). That is Phase 2 in the stack plan. Until then, do not advertise a slim installer.

A future slim spec would collect PyQt6, OpenCV, numpy, onnxruntime, and `yolo11n.onnx` — not `torch`.

## CUDA vs CPU field machines

| Audience | Bundle | Notes |
|----------|--------|--------|
| Mixed / unknown GPUs | Flavor 1 + `yolo11n.onnx` | Optional WinML extra on the **dev** machine (`requirements-winml.txt`), then rebuild. On this repo’s Windows test box, `onnxruntime-windowsml` 1.30 exposed `DmlExecutionProvider` and `CPUExecutionProvider`. |
| Known NVIDIA fleet | Flavor 1 with CUDA-enabled torch, or ORT GPU | Still a large folder |
| CPU-only laptops | Flavor 1 + ONNX or OpenVINO weights | Faster than `.pt` on CPU; folder size stays large |

## Rebuild after a bad Qt plugin path

```powershell
Remove-Item -Recurse -Force build, dist -ErrorAction SilentlyContinue
.\build_exe.ps1
```
