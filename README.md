# Traffic Tracker

Desktop app for turning-movement counts from intersection videos. Upload one video or a folder of videos, draw a start line and an end line for every route on a random frame, then export a 15-minute TMC spreadsheet.

## Stack

- Python 3.11
- Ultralytics YOLO11 for vehicle detection
- Supervision ByteTrack + line zones for tracking and crossings
- PyQt6 desktop UI
- Auto device: NVIDIA CUDA when available, otherwise CPU

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
python run.py
```

## Workflow

1. Add one video, or a folder of videos from the same camera.
2. Optionally enter the video start clock time so 15-minute bins use real time instead of elapsed time.
3. On a random frame, add each route and drag its **start** line then **end** line (for example `NB Left`).
4. Save the routes to reuse them on the next video of the same intersection.
5. Process. Each video writes `{video_name}_tmc.xlsx` to the export folder.

A vehicle is counted on a route when its track crosses that route's start line and later its end line. MVP counts cars, motorcycles, buses, and trucks as vehicles. Pedestrians and bicycles are ignored.

Draw each line so vehicles cross it (roughly perpendicular to travel). One completed route is recorded per vehicle.

## Project layout

```
run.py
traffic_tracker/
  main.py
  config.py
  core/          # video, geometry, YOLO processor, Excel export
  gui/           # PyQt6 window and drawing canvas
```
