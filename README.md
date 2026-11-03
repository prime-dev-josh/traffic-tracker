# Traffic Tracker

Desktop app for turning-movement counts from **overhead drone** intersection videos. Upload one video or a folder of videos, set North, draw the entry and exit lines that exist, then export Overview, Gate Statistics, and a 15-minute TMC Summary.

## Stack

- Python 3.11
- Ultralytics YOLO11 **detect + ByteTrack** (`model.track`) — do not replace with a custom NMS path
- Path-intersection gate crossings (not Supervision LineZone)
- Default **nadir drone** detection (larger imgsz; boat/airplane labels kept as cars)
- PyQt6 desktop UI (see [docs/LICENSE_UI.md](docs/LICENSE_UI.md) before shipping a closed-source `.exe`)
- Auto device: NVIDIA CUDA, then Apple MPS, otherwise CPU
- Optional exported weights via Ultralytics AutoBackend: **ONNX**, **OpenVINO**, later CoreML

## Install

Use Python 3.11 (the version this stack is built for):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The first run downloads `yolo11n.pt` automatically.

## Run

```powershell
.\run.ps1
```

Or: `.\.venv\Scripts\Activate.ps1` then `python run.py`.

## Faster inference (ONNX / OpenVINO / WinML)

Export ONNX once, then the app prefers `yolo11n.onnx` automatically:

```powershell
.\export_models.ps1
```

| Extra | Install | When |
|-------|---------|------|
| ONNX Runtime CPU | `pip install -r requirements-onnx.txt` | Default exported path |
| WinML (mixed Windows GPUs) | Uninstall other ORT wheels, then `pip install -r requirements-winml.txt` | Windows 11 24H2+; NVIDIA/AMD/Intel |
| OpenVINO (Intel CPU/iGPU) | `pip install -r requirements-openvino.txt` then `.\export_models.ps1 --openvino` | Office Intel laptops |

Backend override:

```powershell
$env:TRAFFIC_TRACKER_BACKEND = "auto"   # onnx if present, else OpenVINO, else .pt
$env:TRAFFIC_TRACKER_BACKEND = "pt"
$env:TRAFFIC_TRACKER_BACKEND = "onnx"
$env:TRAFFIC_TRACKER_BACKEND = "openvino"
$env:TRAFFIC_TRACKER_MODEL = "C:\path\to\weights.onnx"
```

Compare `.pt` vs ONNX TMC on a known clip:

```powershell
$env:TRAFFIC_TRACKER_GOLDEN_VIDEO = "C:\path\to\clip.mp4"
$env:TRAFFIC_TRACKER_GOLDEN_ROUTES = "C:\path\to\routes.json"
python -m unittest tests.test_backend_parity -v
```

## Build a Windows `.exe`

See [docs/PACKAGING.md](docs/PACKAGING.md). Short version:

```powershell
.\export_models.ps1
.\build_exe.ps1
```

Ship the whole `dist\Traffic Tracker\` folder. It is large (often 1–3 GB+) because PyTorch is still required by Ultralytics. A slim torch-free installer is a later project.

## Workflow

1. Add a bundle of clips from the same camera (15–30 min / multi‑GB files stay on disk). Leave **Footage is overhead drone** checked for nadir (straight-down) video.
2. Optionally enter the video start clock time so 15-minute bins use real time instead of elapsed time.
3. On a random frame of the **first** clip, drag an arrow toward **North**, then draw the **start (entry)** and **end (exit)** lines that exist. One-way streets can omit a side. Lines are named from that heading: entries `N/S/E/W`, exits `NB/SB/EB/WB`. A completed movement is `{origin}-{bound}` (example: `N-SB`).
4. **Align** later clips: drag/scale/rotate the overlay (or match 2–4 landmarks). Confirm each clip. Unconfirmed clips are skipped.
5. Process confirmed clips sequentially. Each writes `{video_name}_tmc.xlsx` with **Overview**, **Gate Statistics**, and **TMC Summary** sheets.

Save **project** (`bundle.ttproj.json`) so alignments survive an overnight stop. Prefer a local disk — OneDrive sync of 2–5 GB files stalls seeking.

A trip is recorded when a track crosses any start line and later any end line. Start destination is the entry origin (`N`, `S`, `E`, `W`); end destination is the exit bound (`NB`, `SB`, `EB`, `WB`). MVP counts cars, motorcycles, buses, and trucks as vehicles. Pedestrians and bicycles are ignored.

Draw each line so vehicles cross it (roughly perpendicular to travel). One completed trip is recorded per vehicle.

## More docs

- [docs/PACKAGING.md](docs/PACKAGING.md) — full vs later slim PyInstaller flavors
- [docs/LICENSE_UI.md](docs/LICENSE_UI.md) — PyQt6 (GPL) vs PySide6 (LGPL)
- [docs/MACOS.md](docs/MACOS.md) — MPS/CoreML and notarization (later)

## Project layout

```
run.py / run.ps1
export_models.ps1 / scripts/export_models.py
traffic_tracker/
  main.py
  config.py
  core/          # video, geometry, YOLO+LineZone processor, Excel export
  gui/           # PyQt6 window and drawing canvas
```
