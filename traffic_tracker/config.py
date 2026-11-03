"""App-wide constants. Keep v1 settings here so the GUI stays thin."""

from __future__ import annotations

APP_NAME = "Traffic Tracker"
APP_VERSION = "0.1.0"

# 15-minute TMC bins
INTERVAL_SECONDS = 15 * 60

# COCO IDs. Street-level YOLO often tags nadir (top-down drone) cars as
# airplane (4), train (6), or boat (8) — keep those when overhead mode is on.
STREET_VEHICLE_CLASS_IDS = (2, 3, 5, 7)
DRONE_VEHICLE_CLASS_IDS = (2, 3, 4, 5, 6, 7, 8)
VEHICLE_CLASS_IDS = DRONE_VEHICLE_CLASS_IDS
VEHICLE_CLASS_NAMES = {
    2: "car",
    3: "motorcycle",
    4: "car",
    5: "bus",
    6: "car",
    7: "truck",
    8: "car",
}

# Nano is the right default for a first useful desktop build.
# Ultralytics AutoBackend loads .pt, .onnx, or an OpenVINO folder.
YOLO_PT_NAME = "yolo11n.pt"
YOLO_ONNX_NAME = "yolo11n.onnx"
YOLO_MODEL_NAME = YOLO_PT_NAME
OPENVINO_DIR_NAME = "yolo11n_openvino_model"
EXPORT_IMGSZ = 640
TRACK_IMGSZ = 1920
TRACK_CONF = 0.08
STREET_TRACK_IMGSZ = 640
STREET_TRACK_CONF = 0.25

# auto | pt | onnx | openvino
BACKEND_ENV = "TRAFFIC_TRACKER_BACKEND"
MODEL_ENV = "TRAFFIC_TRACKER_MODEL"

# Skip the very start/end when picking a drawing frame (often black or empty).
RANDOM_FRAME_MARGIN = 0.10

SUPPORTED_VIDEO_SUFFIXES = (".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv")
