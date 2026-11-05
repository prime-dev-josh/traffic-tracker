"""App-wide constants. Keep v1 settings here so the GUI stays thin."""

from __future__ import annotations

APP_NAME = "Traffic Tracker"
APP_VERSION = "0.1.0"

# 15-minute TMC bins
INTERVAL_SECONDS = 15 * 60

# Motion-mask vehicle filter. All blobs in this size band count the same.
MOTION_MIN_AREA_FRAC = 0.00008
MOTION_MAX_AREA_FRAC = 0.03
MOTION_MAX_ASPECT = 5.5
MOTION_HISTORY = 400
MOTION_VAR_THRESHOLD = 16.0
MOTION_PROCESS_MAX_SIDE = 1280
MOTION_WARMUP_S = 1.5
MOTION_COAST_S = 5.0
# Keep a slow or stopped track at its last box (red lights). Fast vanish still uses coast.
MOTION_HOLD_S = 90.0
MOTION_HOLD_SPEED_PX_S = 40.0
MOTION_DIFF_THRESHOLD = 12.0
# Drawn vehicle mask → keep blobs from this fraction to this multiple of its area.
VEHICLE_REF_MIN_FACTOR = 0.35
VEHICLE_REF_MAX_FACTOR = 3.5

# Kept for leftover packaging / model-export scripts. Processing does not use YOLO.
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

TRAINING_DIR_NAME = "training"
SIZE_PRIOR_NAME = "size_prior.json"
TRAINED_WEIGHTS_REL = "weights/best.pt"
VISDRONE_WEIGHTS_REL = "weights/visdrone.pt"
TRAIN_DATASET_REL = "dataset"
TRAIN_RUNS_REL = "runs"
TRAIN_MIN_FRAMES = 20
TRAIN_MIN_BOXES = 40
TRAIN_EPOCHS = 60
TRAIN_IMGSZ = 1280
TRAIN_BATCH = 4
TRAIN_PATIENCE = 15
TRAIN_VAL_FRACTION = 0.2
TRAIN_SPLIT_SEED = 42
SETTINGS_FILE_NAME = "app_settings.json"
