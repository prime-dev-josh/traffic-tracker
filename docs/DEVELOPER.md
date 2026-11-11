# Traffic Tracker — developer guide

This is the internals document for people changing the code. Product workflow for operators is in the [README](../README.md).

## What it does

Traffic Tracker turns a hovering-drone video of an intersection into a turning-movement count.

The home dashboard opens Process, Training, or Settings. In Process the operator sets **North**, draws the **entry/exit tripwires** that exist, and optionally boxes **typical vehicles**. Training downloads VisDrone and writes `training/weights/visdrone.pt`. The processor looks for moving blobs (or VisDrone boxes) in the size band, follows each blob’s center, and records a trip when the path visits two different geographic sides of the intersection. The workbook matches the Las Olas sheet set: Overview, Gate Statistics, TMC Summary.

Trained detector classifies cars, vans, trucks, motorcycles, and buses. Motion mask still writes every trip as `vehicle` and only fills **Number of all vehicles**. Each track stores a centroid path that is drawn on a darkened still and embedded under Gate Statistics.

## Design rules

- **Motion is the default count path.** `detector_mode=trained` runs Ultralytics YOLO detect (not ByteTrack) at `DETECTOR_IMGSZ` (default 960; Settings 640 / 960 / 1280) on CUDA with FP16 when available, and feeds the same trip logic. Training stays at `TRAIN_IMGSZ` 1280. Prefer site `training/weights/best.pt`; otherwise a user-chosen `.pt` or `yolo11n.pt`.
- **Classes come from the trained detector only.** Motion trips stay `vehicle`. Export rows are Car / Van / Truck / Motorcycle / Bus count plus Number of all vehicles.
- **North is whatever the operator drew**, not image-up.
- **Gates are optional per side.** One-way intersections are valid (`is_ready` = North + ≥1 start + ≥1 end).
- **A trip is first side → second different side**, confirmed only between real observations. Predictions and held boxes do not count. Motion mode may still infer a trip from the first and last observed positions when a gate was missed.
- **Do not copy client logos** into the export.
- Python **3.11**. GUI is **PyQt6**.

## End-to-end data flow

```
home dashboard
    → Process: videos → routes → align → process → report
    → Training: Train VisDrone → training/weights/visdrone.pt (not bundled)
    → Settings: theme, export folder, OneDrive warning, Motion vs Trained
videos on disk
    → setup page (clock, export folder)
    → reference frame (random, with margin)
    → Geometry: north, starts[], ends[], vehicle_refs[]
    → per-clip Transform (nudge / rotate / homography)
    → process_video(path, fitted Geometry)
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
| `traffic_tracker/core/drone_datasets.py` | VisDrone + UAVDT → 5-class `training/datasets/drone_vehicles`. |
| `traffic_tracker/core/trainer.py` | Fine-tune YOLO11n (site / VisDrone / drone base). Site only copies `best.pt`. |
| `traffic_tracker/core/detector.py` | YOLO predict at `DETECTOR_IMGSZ` with CUDA device + FP16 when available. |
| `traffic_tracker/core/settings.py` | `app_settings.json`: export dir, OneDrive warning, detector mode, theme, precision, detect size, stride, live preview. |
| `traffic_tracker/core/video.py` | Probe, duration, random frame, OneDrive path hint. |
| `traffic_tracker/core/device.py` | CUDA / MPS / CPU probe. `yolo_runtime` maps CUDA to device index 0 + FP16. Distinguishes a missing GPU from a CPU-only PyTorch wheel. |
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

Areas are drawn route masks. If none are drawn it uses frame-fraction defaults (`MOTION_MIN_AREA_FRAC` / `MOTION_MAX_AREA_FRAC`). There are no size spinboxes on the setup page.

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

**Train VisDrone** writes `training/weights/visdrone.pt`. Process uses those weights when the mode is Trained detector. Weights are not shipped with a new install.

Preview tints the mask green and labels each box `#{id} width×height`.

## Tracks and trips

`processor.py` does **not** use ByteTrack. Each frame’s boxes are matched to live tracks with greedy IoU first, then nearest unused centroid within a speed-capped radius (trained) or `_adopt_radius` (motion), plus a 4× area similarity check. Path traces skip inactive (parked) tracks and split on teleport jumps. Active tracks are drawn whether or not they completed a trip. Stroke width is 3 px and color matches the UI primary orange (`path_color_bgr` from the active theme). User-drawn **drop zones** (`geometry.occlusions`) mark dark patches that hide cars. An active track keeps a lead point `prediction_reach_px` ahead (set on the drawing step). When the box is lost in a zone that tip freezes as a pickup; a nearby re-detection reconnects the same ID. Drop-zone recovery is capped at prediction reach plus the span of the zone the gap actually crossed, so a long coast cannot attach across the map. A detection sitting on a stopped car is reserved for that ID; lost tracks cannot attach to a stoplight box. Vehicles on a drawn roadway keep their ID at standstill after a single sighting (hold up to `MOTION_HOLD_S`) without counting or tracing until they move. The drawn trace stitches the last real point to the next real point in a straight line — predicted leads are not drawn, because that sprayed a fan on every flicker. Predictions do not count gate crossings.

A track stores first/last point, last box size, and an optional first-gate origin.

Gate hit: the segment from previous center to current center intersects a padded gate (`pad = max(24, 2% of the long side)`), **or the current bounding box sits on the gate**. A car that is only detected while it is on the line still counts. All start and end lines are treated as gates. The **geographic side** of the line (from North) is what matters, not whether the operator drew it green or red.

- First distinct side → origin (`N/S/E/W`) and start time.
- Second distinct side → bound (`NB/SB/EB/WB`) and a `Trip`.

Once a track has two sightings and is moving slowly (or already held), it stays at its last box for `MOTION_HOLD_S` (90 s) even if MOG2 loses the car — red lights. The background image is pasted over those boxes before the subtractor learns, so a stopped car is not absorbed. When a **trip completes** (second side / exit), hold is released immediately: the track is dropped from `pending.live`, `last_xyxy` is cleared, and MOG2 is no longer frozen on that box. Fast vanish still expires after `MOTION_COAST_S` (5 s).

If the track dies with only one gate, the last point’s side can finish the trip. If it never hit a gate but first and last sides differ and travel exceeds `_min_travel`, that path still becomes a trip.

`Trip.vehicle_class` is `car` / `van` / `truck` / `motorcycle` / `bus` from the trained detector, or `vehicle` from motion. `Trip.path` is the centroid polyline used for the export overlay. Tracks that never complete a trip do not keep a finished path and are not drawn.

## Export

`export_tmc_workbook` writes three sheets. If `path_image` is set, Gate Statistics embeds that PNG and writes `{stem}_paths.png` (all trip traces) beside the workbook. `route_images` writes `{stem}_paths_{route}.png` for each movement that has trips.

- **Overview** — Car / Van / Truck / Motorcycle / Bus count, then **Number of all vehicles**.
- **Gate Statistics** — the same class rows per entry/exit, plus the darkened path-trace still under the table.
- **TMC Summary** — 15-minute bins (`INTERVAL_SECONDS` = 900) grouped by entry, columns `{origin}-{bound}`.

Clock time is optional. If the operator sets a start datetime, bin labels are wall-clock; otherwise they are elapsed `MM:SS`.

## GUI pages

`MainWindow` is a `QStackedWidget`. The header Home button is hidden on the dashboard. The `1. Videos → …` step strip is visible only on Process pages.

0. **Home** — Process / Training / Settings cards.
1. **Setup** — add videos/folder, OneDrive warning, clock, export folder, load/save project.
2. **Routes** — North, add start/end (Auto or explicit name), add one or more vehicle masks, optional drop zones over dark patches, prediction reach (px on this frame), save/load geometry JSON.
3. **Align** — per-clip nudge, wheel zoom, rotate, match points, optional redraw, confirm.
4. **Process** — log, progress, live preview.
5. **Report** — tabs matching the workbook (Overview / Gate Statistics / TMC Summary), plus Open export folder.
6. **Training** — **Train VisDrone** downloads VisDrone2019-DET and writes `training/weights/visdrone.pt`. Cancel, progress, and log. Open weights folder.
7. **Settings** — theme (Prime AE navy / orange dark + white / navy light), export folder, OneDrive warning, Motion vs Trained. Logos live in `traffic_tracker/gui/brand/`.

`FrameCanvas` modes: `view`, `draw` (line or box), `nudge`, `match`. Vehicle masks on Routes use `set_drawing_enabled(True, box=True)` and `on_box_drawn`.

Route masks live on the drawing step. Processing uses those masks for the size band.

## Training sessions

`traffic_tracker/core/training.py`

- Folder: `training/sessions/{video_stem}/session.json`, `images/*.jpg`, `labels/*.txt`.
- YOLO line: `{class} xc yc w h` (normalized). Classes are car, van, truck, bus, motorcycle (0–4). Boxed Training sessions default to **car**.
- `rebuild_size_prior` writes `training/size_prior.json` with all boxed areas and the same 35%–350% band.
- `build_yolo_dataset` copies frames into `training/dataset/images/{train,val}` and `labels/{train,val}` (80/20 by frame, seed 42) and writes 5-class `data.yaml`.
- `traffic_tracker/core/drone_datasets.py` merges VisDrone vehicle labels with subsampled UAVDT into `training/datasets/drone_vehicles/`. UAVDT ingest accepts MOT (`UAV-benchmark-M` + `GT/*_gt_det.txt`), a YOLO split, or a Dataset Ninja Supervisely extract (`train|test/img` + `ann`).
- Minimum to train: `TRAIN_MIN_FRAMES` (20) and `TRAIN_MIN_BOXES` (40).
- North/gates are not required.

`traffic_tracker/core/trainer.py` fine-tunes `yolo11n.pt` at imgsz 1280 on a `TrainWorker` thread. **Train VisDrone** (`python -m traffic_tracker.train_visdrone`) downloads VisDrone2019-DET and writes `training/weights/visdrone.pt`. Site boxing, UAVDT, and custom `.pt` selection were removed from the app. Detector keeps cars, vans, trucks, buses, and motorcycles. Settings stores a precision preset (`strict` / `balanced` / `sensitive` / `custom`) and passes confidence, NMS IoU, parked radius, link speed, path-jump, detect size, detect stride, and live-preview interval into `process_video`. Speed knobs do not switch the preset to Custom. GPU inference uses `yolo_runtime` (CUDA index 0 + FP16). Detect stride skips YOLO on non-multiples of N and still calls `_record_crossings` so hold / coast / drop-zone pickup continue; motion mode still runs MOG2 every frame. Balanced defaults are confidence 0.35 and NMS IoU 0.40. Frame-to-frame association prefers `box_iou` ≥ `DETECTOR_MATCH_IOU_MIN`, then speed-capped centroid linking. It drops people, bicycles, boats/watercraft, and tricycles. Trained mode only activates a track after it walks out of a settle radius (`DETECTOR_STATIC_RADIUS_*`) with small steps (`DETECTOR_LINK_*`). The settle point creeps with hover drift so parked cars and mislabeled boats do not appear in preview, traces, or trips. A hop to another stall starts a new track instead of a lot-spanning line. It does not infer first→last trips without a gate start. Do not run a live train in unit tests.

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
& "$env:LOCALAPPDATA\traffic-tracker\venv\Scripts\python.exe" -m unittest tests.test_crossings tests.test_motion tests.test_geometry_names tests.test_report tests.test_transform tests.test_models tests.test_training tests.test_drone_datasets tests.test_device tests.test_tracking_rules -q
```

| File | Covers |
|------|--------|
| `tests/test_crossings.py` | Segment math; start-then-end; swapped colors; missed gates inferred from path sides. |
| `tests/test_motion.py` | Default area vs resolution; downscale; keep a moving square; reject huge blobs. |
| `tests/test_geometry_names.py` | North naming, one-way, vehicle mask band + scale + JSON. |
| `tests/test_report.py` | Bins, Las Olas sheets, geometry round-trip including vehicle masks. |
| `tests/test_transform.py` | Translate/scale/homography, project file. |
| `tests/test_models.py` | Legacy weight-path resolver (unused by the count path). |
| `tests/test_training.py` | Session JPEG/JSON/YOLO round-trip, size prior, dataset flatten/split, settings fallback, detect size / GPU predict kwargs. |
| `tests/test_drone_datasets.py` | VisDrone/UAVDT class remap, subsample, merged `data.yaml`. |
| `tests/test_device.py` | CUDA probe, `yolo_runtime` FP16 mapping. |
| `tests/test_tracking_rules.py` | Hold / drop-zone pickup, detect stride, live-preview gating. |
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
| `VEHICLE_REF_MIN_FACTOR` / `VEHICLE_REF_MAX_FACTOR` | Band around drawn vehicle masks. |
| `TRAINING_DIR_NAME` / `SIZE_PRIOR_NAME` | Local dataset + size prior. |
| `TRAINED_WEIGHTS_REL` | `weights/best.pt` under the training folder. |
| `TRAIN_IMGSZ` | Training letterbox (1280). Process does not use this. |
| `DETECTOR_IMGSZ` / `DETECTOR_IMGSZ_CHOICES` | Process-time YOLO letterbox (default 960). |
| `DETECTOR_STRIDE` / `DETECTOR_STRIDE_MAX` | Run YOLO every N frames (1–3). Motion still every frame. |
| `PREVIEW_EVERY` / `PREVIEW_EVERY_CHOICES` | Live preview interval; 0 is off. |
| `SETTINGS_FILE_NAME` | `app_settings.json`. |
| `RANDOM_FRAME_MARGIN` | Avoid black first/last frames when picking a draw frame. |

## Running and debugging

- Entry: `python run.py` or `.\run.ps1`.
- If Qt fails to start, check `QT_PLUGIN_PATH` is not pointing at a stale folder. `main.py` sets it from the venv PyQt6 tree. Do not leave `QT_QPA_PLATFORM=offscreen` set in a normal desktop session.
- Large files under OneDrive: `video.py` warns. Seeking 2–5 GB over sync stalls. Copy locally.
- Process log prints whether size came from route masks or the default frame band, plus start hits / end hits / tracked frames per clip.

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
