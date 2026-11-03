"""App-wide constants. Keep v1 settings here so the GUI stays thin."""

from __future__ import annotations

APP_NAME = "Traffic Tracker"
APP_VERSION = "0.1.0"

# 15-minute TMC bins
INTERVAL_SECONDS = 15 * 60

# COCO class IDs used by YOLO11
# 2=car, 3=motorcycle, 5=bus, 7=truck
# Bus is included so city-bus movements are not dropped from a TMC.
VEHICLE_CLASS_IDS = (2, 3, 5, 7)
VEHICLE_CLASS_NAMES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Nano is the right default for a first useful desktop build.
# Ultralytics AutoBackend loads .pt, .onnx, or an OpenVINO folder.
YOLO_PT_NAME = "yolo11n.pt"
YOLO_ONNX_NAME = "yolo11n.onnx"
YOLO_MODEL_NAME = YOLO_PT_NAME
OPENVINO_DIR_NAME = "yolo11n_openvino_model"
EXPORT_IMGSZ = 640

# auto | pt | onnx | openvino
BACKEND_ENV = "TRAFFIC_TRACKER_BACKEND"
MODEL_ENV = "TRAFFIC_TRACKER_MODEL"

# Skip the very start/end when picking a drawing frame (often black or empty).
RANDOM_FRAME_MARGIN = 0.10

SUPPORTED_VIDEO_SUFFIXES = (".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv")
