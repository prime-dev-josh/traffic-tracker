# Traffic Tracker — developer guide

This is the internals document for people changing the code. Product workflow for operators is in the [README](../README.md).

## What it does

Traffic Tracker turns a hovering-drone video of an intersection into a turning-movement count.

The home dashboard opens Process, Training, or Settings. In Process the operator sets **North**, draws the **entry/exit tripwires** that exist, and optionally boxes **typical vehicles**. Training boxes the same way on sample frames and writes a size prior plus a YOLO-ready dataset. The processor looks for moving blobs in that size band, follows each blob’s center, and records a trip when the path visits two different geographic sides of the intersection. The workbook matches the Las Olas sheet set: Overview, Gate Statistics, TMC Summary.

Counting does **not** classify cars vs trucks vs buses. Every completed trip is **Motor Vehicle Count**.

## Design rules

- **Motion is the default count path.** `detector_mode=trained` runs Ultralytics YOLO detect (not ByteTrack) at `TRAIN_IMGSZ` 1280 and feeds the same trip logic. Prefer site `training/weights/best.pt`; otherwise a user-chosen `.pt` or `yolo11n.pt`.
- **No vehicle classes.** Size is the filter; the export row is Motor Vehicle Count.
- **North is whatever the operator drew**, not image-up.
- **Gates are optional per side.** One-way intersections are valid (`is_ready` = North + ≥1 start + ≥1 end).
- **A trip is first side → second different side**, from path intersection or from first/last point sides if a gate was missed.
- **Do not copy client logos** into the export.
- Python **3.11**. GUI is **PyQt6**.

## End-to-end data flow

```
home dashboard
    → Process: videos → routes → align → process → report
    → Training: sample frames → boxes → session JPEG/JSON/YOLO + size_prior.json
         optional Train detector → training/dataset + training/weights/best.pt
    → Settings: theme, export folder, OneDrive warning, detector mode, training import/export
videos on disk
    → setup page (clock, export folder)
    → reference frame (random, with margin)
    → Geometry: north, starts[], ends[], vehicle_refs[]
    → per-clip Transform (nudge / rotate / homography)
    → process_video(path, fitted Geometry + training size prior)
         MOG2 mask → blobs in size band → centroid tracks → crossings → Trip[]
    → export_tmc_workbook → {stem}_tmc.xlsx
```

`BundleProject` (`bundle.ttproj.json`) stores the reference geometry and each clip’s alignment so a long batch can stop overnight.

## Modules

| Path | Role |
|------|------|
| `traffic_tracker/main.py` | Fixes `QT_PLUGIN_PATH` (venv / PyInstaller), starts `QApplication`. |
| `traffic_tracker/config.py` | App name, 15-min interval, motion constants, vehicle-mask factors, training paths. |
| `traffic_tracker/core/geometry.py` | `Line`, `Box`, `NamedLine`, `Geometry`, `Transform`, North-relative naming, size-band helpers. |
| `traffic_tracker/core/motion.py` | MOG2 subtractor, downscale, morphology, contour → boxes. |
| `traffic_tracker/core/crossings.py` | Segment intersection and padded gate hits. |
| `traffic_tracker/core/processor.py` | Video loop, tracks, trip assembly. Unions training size prior with drawn masks. |
| `traffic_tracker/core/report.py` | Trip table + three Excel sheets. |
| `traffic_tracker/core/project.py` | Clip list, confirm/skip, `ready_jobs`. |
| `traffic_tracker/core/training.py` | Session folders, JSON + YOLO labels, `size_prior.json`, flattened `training/dataset`. |
| `traffic_tracker/core/trainer.py` | Fine-tune YOLO11n from boxed sessions; copy `training/weights/best.pt`. |
| `traffic_tracker/core/detector.py` | YOLO predict at imgsz 1280 for the trained process path. |
| `traffic_tracker/core/settings.py` | `app_settings.json`: export dir, OneDrive warning, detector mode, theme. |
| `traffic_tracker/core/video.py` | Probe, duration, random frame, OneDrive path hint. |
| `traffic_tracker/core/device.py` | CUDA / MPS / CPU probe. Distinguishes a missing GPU from a CPU-only PyTorch wheel. |
| `traffic_tracker/core/models.py` | Resolves bundled `yolo11n.pt` as the fine-tune start. |
| `traffic_tracker/gui/main_window.py` | Home + Process wizard + Training + Settings. |
| `traffic_tracker/gui/worker.py` | `ProcessWorker` and `TrainWorker` threads. |
| `traffic_tracker/gui/widgets/frame_canvas.py` | Draw line / draw box / nudge / match points. |

## Geometry

`Geometry` is in **image pixels** of the frame it was drawn on.

- **North** is an arrow. Tail = intersection center. Head = geographic North. If North is at the bottom of the photo, the operator drags downward.
- **Starts** are named `N S E W` from the line midpoint’s heading vs North.
- **Ends** are named `NB SB EB WB` the same way.
- Movement labels are the cartesian product of drawn origins and drawn bounds (`N-SB`, …). Unused sides are omitted, not filled in order.
- **`vehicle_refs`** is a list of axis-aligned `Box`es around typical vehicles.

`name_from_north` / `side_from_point` use heading `0 = toward North, clockwise`.

`Geometry.scaled_to(w, h)` multiplies all lines and the vehicle box when the process frame differs from the drawing frame.

`Transform.apply_geometry` maps the reference overlay onto a shifted clip (translate, scale, rotate, or 4-point homography). The vehicle mask goes with it so size stays correct after alignment.

### Vehicle size band

`Geometry.vehicle_area_limits(extra_areas=None)`:

```
min = max(50, smallest_area × VEHICLE_REF_MIN_FACTOR)   # 0.35
max = max(200, largest_area × VEHICLE_REF_MAX_FACTOR)   # 3.5
```

Areas are drawn route masks plus optional Training box areas from `training/size_prior.json`. `process_video` calls `prior_areas()` and unions them with masks. If both are empty it uses frame-fraction defaults (`MOTION_MIN_AREA_FRAC` / `MOTION_MAX_AREA_FRAC`). There are no size spinboxes on the setup page.

JSON includes `vehicle_refs`. Older files with a single `vehicle_ref` still load.

## Motion processing

`traffic_tracker/core/motion.py`

1. Downscale so the long side is at most `MOTION_PROCESS_MAX_SIDE` (1280) for 4K speed.
2. `createBackgroundSubtractorMOG2` (`history=400`, `varThreshold=16`, shadows on). Live track boxes are replaced with the current background image before `apply`, so a stopped car is not learned away.
3. Keep pixels ≥ 200 (drop shadow value 127). OR a short frame-diff (`MOTION_DIFF_THRESHOLD`) so slow crawls still light up.
4. Open then close with a 5×5 ellipse.
5. External contours → bounding boxes.
6. Drop area outside `[min_area, max_area]` (areas are compared in **original** pixels; the scaled frame uses `area × scale²`).
7. Drop extreme aspect ratios (`MOTION_MAX_ASPECT`, default 5).

The first `MOTION_WARMUP_S` seconds (1.5 s) only train the background. Crossings start after that.

### Why vehicles are missed (typical on nadir drone)

The mask does **not** recognize cars. It finds moving pixels, turns them into boxes, and keeps boxes that look vehicle-sized.

- Stopped or slow cars become background and disappear.
- Tiny / low-contrast cars produce weak FG after the 1280 long-side downscale.
- Two cars merge into one blob (or one car splits and fails size).
- Tree sway / lighting creates noise, so the size band is kept tight and real cars fall outside it.
- A car seen only in the middle of the intersection never hits two sides, so it is detected in preview but not counted.

Training boxes tighten the size band immediately. **Train detector** builds `training/dataset` and writes `training/weights/best.pt`. Process uses those weights when the mode is Trained detector.

Preview tints the mask green and labels each box `#{id} width×height`.

## Tracks and trips

`processor.py` does **not** use ByteTrack. Each frame’s boxes are associated by nearest unused centroid within `_adopt_radius` (about 8% of the short frame side, minimum 120 px) and a 4× area similarity check.

A track stores first/last point, last box size, and an optional first-gate origin.

Gate hit: the segment from previous center to current center intersects a padded gate (`pad = max(24, 2% of the long side)`), **or the current bounding box sits on the gate**. A car that is only detected while it is on the line still counts. All start and end lines are treated as gates. The **geographic side** of the line (from North) is what matters, not whether the operator drew it green or red.

- First distinct side → origin (`N/S/E/W`) and start time.
- Second distinct side → bound (`NB/SB/EB/WB`) and a `Trip`.

Once a track has two sightings and is moving slowly (or already held), it stays at its last box for `MOTION_HOLD_S` (90 s) even if MOG2 loses the car — red lights. The background image is pasted over those boxes before the subtractor learns, so a stopped car is not absorbed. When a **trip completes** (second side / exit), hold is released immediately: the track is dropped from `pending.live`, `last_xyxy` is cleared, and MOG2 is no longer frozen on that box. Fast vanish still expires after `MOTION_COAST_S` (5 s).

If the track dies with only one gate, the last point’s side can finish the trip. If it never hit a gate but first and last sides differ and travel exceeds `_min_travel`, that path still becomes a trip.

`Trip.vehicle_class` is unused in the workbook. Every trip is Motor Vehicle Count.

## Export

`export_tmc_workbook` writes three sheets, no images:

- **Overview** — **Motor Vehicle Count** (all trips).
- **Gate Statistics** — one **Motor Vehicle Count** row per entry/exit.
- **TMC Summary** — 15-minute bins (`INTERVAL_SECONDS` = 900) grouped by entry, columns `{origin}-{bound}`.

Clock time is optional. If the operator sets a start datetime, bin labels are wall-clock; otherwise they are elapsed `MM:SS`.

## GUI pages

`MainWindow` is a `QStackedWidget`. The header Home button is hidden on the dashboard. The `1. Videos → …` step strip is visible only on Process pages.

0. **Home** — Process / Training / Settings cards.
1. **Setup** — add videos/folder, OneDrive warning, clock, export folder, load/save project.
2. **Routes** — North, add start/end (Auto or explicit name), add one or more vehicle masks, save/load geometry JSON.
3. **Align** — per-clip nudge, wheel zoom, rotate, match points, optional redraw, confirm.
4. **Process** — log, progress, live preview.
5. **Report** — tabs matching the workbook (Overview / Gate Statistics / TMC Summary), plus Open export folder.
6. **Training** — pick a video, box vehicles on random frames, save/load/delete sessions, **Train detector**.
7. **Settings** — theme, export folder, OneDrive warning, detector mode, view/import/export training data.

`FrameCanvas` modes: `view`, `draw` (line or box), `nudge`, `match`. Vehicle mask and Training use `set_drawing_enabled(True, box=True)` and `on_box_drawn`.

Route masks live on the drawing step. Training boxes live in `training/sessions/`. Processing unions both into the size band.

## Training sessions

`traffic_tracker/core/training.py`

- Folder: `training/sessions/{video_stem}/session.json`, `images/*.jpg`, `labels/*.txt`.
- YOLO line: `0 xc yc w h` (normalized, class 0 = vehicle).
- `rebuild_size_prior` writes `training/size_prior.json` with all boxed areas and the same 35%–350% band.
- `build_yolo_dataset` copies frames into `training/dataset/images/{train,val}` and `labels/{train,val}` (80/20 by frame, seed 42) and writes `data.yaml` (`vehicle`).
- Minimum to train: `TRAIN_MIN_FRAMES` (20) and `TRAIN_MIN_BOXES` (40).
- North/gates are not required.

`traffic_tracker/core/trainer.py` fine-tunes `yolo11n.pt` for 60 epochs at imgsz 1280 on a `TrainWorker` thread. Site boxing writes `training/weights/best.pt`. **Train on VisDrone** (`python -m traffic_tracker.train_visdrone`) downloads VisDrone2019-DET, writes `training/weights/visdrone.pt`, and copies it to `best.pt`. Detector keeps VisDrone motor-vehicle classes (car, van, truck, bus, motor, tricycles) and drops people/bicycles. Do not run a live train in unit tests. The boxed Training UI stays until VisDrone is proven better.

## Project file

`bundle.ttproj.json`:

```json
{
  "geometry": { "frame_width": 1920, "north": {...}, "starts": [...], "ends": [...], "vehicle_refs": [{...}] },
  "clips": [
    { "path": "...", "status": "reference|confirmed|needs_align", "transform": {...} }
  ]
}
```

`ready_jobs` returns `(path, fitted_geometry)` only for the reference clip and confirmed clips. Unconfirmed clips are skipped.

## Tests

Run from the repo root with the venv:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_crossings tests.test_motion tests.test_geometry_names tests.test_report tests.test_transform tests.test_models tests.test_training tests.test_device -q
```

| File | Covers |
|------|--------|
| `tests/test_crossings.py` | Segment math; start-then-end; swapped colors; missed gates inferred from path sides. |
| `tests/test_motion.py` | Default area vs resolution; downscale; keep a moving square; reject huge blobs. |
| `tests/test_geometry_names.py` | North naming, one-way, vehicle mask band + scale + JSON. |
| `tests/test_report.py` | Bins, Las Olas sheets, geometry round-trip including vehicle masks. |
| `tests/test_transform.py` | Translate/scale/homography, project file. |
| `tests/test_models.py` | Legacy weight-path resolver (unused by the count path). |
| `tests/test_training.py` | Session JPEG/JSON/YOLO round-trip, size prior, dataset flatten/split, settings fallback. |
| `tests/test_backend_parity.py` | Opt-in golden-video compare; both calls now hit the same motion path. |

Avoid spinning a `QApplication` in unit tests (Qt plugin path issues on some Windows setups). Prefer testing geometry, motion, and report without the GUI.

## Config knobs

In `traffic_tracker/config.py`:

| Name | Meaning |
|------|---------|
| `INTERVAL_SECONDS` | TMC bin width (900). |
| `MOTION_MIN_AREA_FRAC` / `MOTION_MAX_AREA_FRAC` | Fallback size band as a fraction of the frame. |
| `MOTION_MAX_ASPECT` | Reject skinny glare streaks. |
| `MOTION_HISTORY` / `MOTION_VAR_THRESHOLD` | MOG2 background model. |
| `MOTION_PROCESS_MAX_SIDE` | Infer at this long-side cap. |
| `MOTION_WARMUP_S` / `MOTION_COAST_S` | Ignore early frames; how long a fast-moving track may vanish. |
| `MOTION_HOLD_S` / `MOTION_HOLD_SPEED_PX_S` | How long a slow/stopped track is kept at its last box. |
| `MOTION_DIFF_THRESHOLD` | Frame-diff OR’d with MOG2 for slow motion. |
| `VEHICLE_REF_MIN_FACTOR` / `VEHICLE_REF_MAX_FACTOR` | Band around drawn masks and Training boxes. |
| `TRAINING_DIR_NAME` / `SIZE_PRIOR_NAME` | Local dataset + size prior. |
| `TRAINED_WEIGHTS_REL` | `weights/best.pt` under the training folder. |
| `SETTINGS_FILE_NAME` | `app_settings.json`. |
| `RANDOM_FRAME_MARGIN` | Avoid black first/last frames when picking a draw frame. |

## Running and debugging

- Entry: `python run.py` or `.\run.ps1`.
- If Qt fails to start, check `QT_PLUGIN_PATH` is not pointing at a stale folder. `main.py` sets it from the venv PyQt6 tree. Do not leave `QT_QPA_PLATFORM=offscreen` set in a normal desktop session.
- Large files under OneDrive: `video.py` warns. Seeking 2–5 GB over sync stalls. Copy locally.
- Process log prints whether size came from route masks, Training boxes, or the default frame band, plus start hits / end hits / tracked frames per clip.

## Leftover YOLO / packaging

The count path does not load `yolo11n.pt`. These still exist from the earlier stack and should not be treated as required for a motion-only run:

- `traffic_tracker/core/models.py`, `device.py`, `bytetrack.yaml`
- `scripts/export_models.py`, `export_models.ps1`
- `requirements.txt` still lists `ultralytics`, `torch`, `lap`, `supervision`
- [PACKAGING.md](PACKAGING.md) still describes a large PyTorch folder

A later slim-down can drop those dependencies and shrink the exe. Until then, `pip install -r requirements.txt` still pulls them.

## Known limits

- A **moving** drone (not a hover) will paint almost the whole frame as foreground.
- Two cars that merge into one blob can become one trip.
- A parked car that the background has not absorbed can blob if the camera jitters.
- Class-specific TMC rows are unused while every trip is `vehicle`.
- Alignment is manual. There is no automatic frame-to-frame registration.
