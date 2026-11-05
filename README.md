# Traffic Tracker

Desktop app for turning-movement counts from **overhead drone** intersection videos.

Home opens **Process Traffic Footage**, **Training**, or **Settings**. Process is the existing count wizard: set North, draw the entry and exit lines that exist, optionally mask typical vehicles, then export **Overview**, **Gate Statistics**, and a 15-minute **TMC Summary**.

There is no car classifier. Processing is an OpenCV motion mask. A moving blob that sits in the vehicle-size band and crosses two geographic sides of the intersection becomes one trip. Every vehicle counts the same.

## Recent changes

- **Hold stopped vehicles.** After a car is seen twice, its last box stays for up to 90 s at a red light. The background subtractor does not learn that box away. Preview labels held IDs as `hold`.
- **Home dashboard.** Process, Training, and Settings are separate features. The 1–5 step strip only appears inside Process.
- **Training.** Box every vehicle on sample frames. That tightens the size band immediately. **Train detector** fine-tunes YOLO11 and writes `training/weights/best.pt`.
- **Motion mask instead of YOLO.** Nadir drone cars were missed or labeled boat/airplane. Counts now come from MOG2 foreground blobs.
- **Size, not class.** Drawn route masks plus Training boxes decide the size band. Pedestrians and flicker fall below the min; lighting washes fall above the max.
- **Vehicle reference masks** on the drawing step. Drag one or more boxes around typical vehicles. Processing keeps blobs from 35% of the smallest sample to 350% of the largest.
- **Export is unclassified.** Overview and Gate Statistics show **Motor Vehicle Count** only. TMC Summary is still by movement (`N-SB`, …).
- **Gates, not required 8 lines.** Set North, then only the start/end lines that exist (one-way is fine). Names are `{origin}-{bound}` such as `N-SB`.
- **Path crossings.** A trip is a centroid path that hits two different sides. ByteTrack IDs are not used.
- **Las Olas-style workbook** (Overview / Gate Statistics / TMC Summary). No copied logos or branding.

## Stack

- Python 3.11
- OpenCV **MOG2** motion mask + centroid association
- Path-intersection gate crossings
- PyQt6 desktop UI (see [docs/LICENSE_UI.md](docs/LICENSE_UI.md) before shipping a closed-source `.exe`)

Developer internals: [docs/DEVELOPER.md](docs/DEVELOPER.md).

## Install

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-cuda.txt
```

`pip install torch` from PyPI is CPU-only on Windows, even if the machine has an NVIDIA GPU. `requirements-cuda.txt` installs the CUDA 12.8 wheels so **Train detector** can use the card.

No model download is required for motion counting.

## Run

```powershell
.\run.ps1
```

Or: `.\.venv\Scripts\Activate.ps1` then `python run.py`.

## Workflow

### Home

The app starts on a dashboard. Open one of the three cards.

### Process Traffic Footage

1. Add a bundle of clips from the same camera (15–30 min / multi‑GB files stay on disk).
2. Optionally enter the video start clock time so 15-minute bins use real time.
3. On a random frame of the **first** clip:
   - Drag an arrow toward **geographic North** (the top of the photo does not have to be North).
   - Draw **start (entry)** and **end (exit)** lines that exist. One-way streets can omit a side.
   - **Add vehicle mask**: drag a box around a typical vehicle. Repeat for other sizes. That size band is the processing filter.
4. **Align** later clips: drag/scale/rotate the overlay (or match 2–4 landmarks). Confirm each clip. Unconfirmed clips are skipped.
5. Process confirmed clips. Each writes `{video_name}_tmc.xlsx`.

Save a **project** (`bundle.ttproj.json`) so alignments survive an overnight stop. Prefer a local disk — OneDrive sync of 2–5 GB files stalls seeking.

Draw each gate so vehicles **cross** it (a tripwire across the lanes, not along the curb).

A trip is recorded when a moving blob crosses any start side and later any other side. Start destination is `N/S/E/W`; end destination is `NB/SB/EB/WB`.

If you skip route masks **and** Training boxes, a default size band from the frame size is used. Add a smaller mask if people or leaves count; add a larger mask if trucks are missed.

### Training

Boxing vehicles does two things:

1. Updates `training/size_prior.json` so Motion mask keeps blobs in that size band even if you drew no route masks.
2. Saves each frame JPEG plus JSON and YOLO `.txt` labels under `training/sessions/{video_stem}/`.

Open a clip, box **every** vehicle you can see (including stopped queue cars), then **Next frame**. Repeat. **Save session** writes the labels. Aim for **150+ frames / 400+ boxes** across clips and times of day. The hard minimum to train is 20 frames and 40 boxes.

**Train detector** flattens those sessions into `training/dataset/`, fine-tunes `yolo11n.pt` at image size 1280, and copies `best.pt` to `training/weights/best.pt`. Process then switches to **Trained detector**. CPU training is allowed but slow; a GPU is much faster.

Stock COCO weights miss nadir drone cars. Use a fine-tuned `best.pt`, not `yolo11n.pt` alone.

**Train on VisDrone** (or `python -m traffic_tracker.train_visdrone`) fine-tunes the same YOLO11n on VisDrone2019-DET (~2 GB download). Weights go to `training/weights/visdrone.pt` and are copied to `training/weights/best.pt`. Boxed sessions stay so we can compare. If VisDrone counts better, the site Training feature can be removed later.

Trained detector keeps **cars, vans, trucks, motorcycles, and buses** at confidence **0.30** (Settings slider 0.15–0.60) and IoU **0.45**. People, bicycles, boats, tricycles, and parked vehicles are ignored. Motion mask still counts every moving blob in the size band.

### Settings

- Dark / Light theme.
- Default export folder (also on the Process videos page).
- OneDrive warning on/off.
- Detection mode: Motion (default) or Trained detector (YOLO). After **Train detector**, the app selects the site weights. You can still choose another `.pt` or switch back to Motion on Align before Process.
- Detector confidence (trained mode only). Raise it if the count is still high.
- Training data: view sessions, export a zip, import a zip or folder, open the `training/` folder.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_crossings tests.test_motion tests.test_geometry_names tests.test_report tests.test_transform tests.test_models tests.test_training tests.test_device -q
```

## Build a Windows `.exe`

See [docs/PACKAGING.md](docs/PACKAGING.md). Counting no longer needs YOLO at runtime; the current PyInstaller spec may still bundle leftover torch/Ultralytics until that spec is slimmed.

## More docs

- [docs/DEVELOPER.md](docs/DEVELOPER.md) — architecture, modules, data model, processing, GUI, tests
- [docs/PACKAGING.md](docs/PACKAGING.md) — PyInstaller flavors
- [docs/LICENSE_UI.md](docs/LICENSE_UI.md) — PyQt6 (GPL) vs PySide6 (LGPL)
- [docs/MACOS.md](docs/MACOS.md) — later macOS notes

## Project layout

```
run.py / run.ps1
traffic_tracker/
  main.py            # Qt plugin path + app launch
  config.py          # intervals, motion knobs, training paths
  core/
    geometry.py      # North, gates, vehicle mask, size-band helpers
    motion.py        # MOG2 blobs + size filter
    crossings.py     # segment intersection
    processor.py     # frame loop, tracks, trips
    report.py        # Overview / Gate Statistics / TMC Summary
    project.py       # bundle.ttproj.json
    training.py      # labeled frames, YOLO txt, size_prior.json
    settings.py      # export folder, warnings, detector mode
    video.py         # probe, random frame, OneDrive hint
  gui/
    main_window.py   # home → process / training / settings
    worker.py        # background process_video
    widgets/frame_canvas.py
tests/
docs/
```
