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
SITE_WEIGHTS_REL = "weights/site.pt"
TRAIN_DATASET_REL = "dataset"
TRAIN_RUNS_REL = "runs"
TRAIN_MIN_FRAMES = 20
TRAIN_MIN_BOXES = 40
TRAIN_EPOCHS = 60
TRAIN_SITE_EPOCHS = 40
TRAIN_IMGSZ = 1280
# Process-time detect size. Training stays at TRAIN_IMGSZ.
DETECTOR_IMGSZ = 960
DETECTOR_IMGSZ_CHOICES = (640, 960, 1280)
DETECTOR_STRIDE = 1
DETECTOR_STRIDE_MAX = 3
PREVIEW_EVERY = 15
PREVIEW_EVERY_CHOICES = (0, 15, 30, 60)
TRAIN_BATCH = 4
TRAIN_PATIENCE = 15
TRAIN_VAL_FRACTION = 0.2
TRAIN_SPLIT_SEED = 42
SETTINGS_FILE_NAME = "app_settings.json"

# Trained detector inference. Balanced preset is the default.
DETECTOR_CONF = 0.35
DETECTOR_CONF_MIN = 0.15
DETECTOR_CONF_MAX = 0.60
DETECTOR_IOU = 0.40
DETECTOR_IOU_MIN = 0.20
DETECTOR_IOU_MAX = 0.70
DETECTOR_MATCH_IOU_MIN = 0.10
# Parked cars / boats stay inside this radius; the settle point creeps with hover drift.
DETECTOR_STATIC_RADIUS_MIN = 56.0
DETECTOR_STATIC_RADIUS_FRAC = 0.07
DETECTOR_PARKED_RADIUS = 56.0
DETECTOR_PARKED_RADIUS_MIN = 24.0
DETECTOR_PARKED_RADIUS_MAX = 120.0
DETECTOR_STATIC_SETTLE_S = 1.5
DETECTOR_STATIC_EXPIRE_S = 2.0
# Trained-mode association. A wide radius stitches parked stalls into one track.
DETECTOR_LINK_SPEED_PX_S = 180.0
DETECTOR_LINK_SPEED_MIN = 60.0
DETECTOR_LINK_SPEED_MAX = 300.0
DETECTOR_LINK_MIN_PX = 36.0
DETECTOR_LINK_CAP_MIN = 96.0
DETECTOR_LINK_CAP_FRAC = 0.10
DETECTOR_PATH_JUMP = 90.0
DETECTOR_PATH_JUMP_MIN = 40.0
DETECTOR_PATH_JUMP_MAX = 200.0
DETECTOR_PATH_JUMP_FRAC = 0.08
# Drop-zone lead: how far ahead (px on the reference frame) to freeze a pickup.
PREDICTION_REACH_PX = 64.0
PREDICTION_REACH_MIN = 24.0
PREDICTION_REACH_MAX = 160.0

# Slider bundles. Custom means the user moved a slider off a preset.
PRECISION_PRESETS = {
    "strict": {
        "detector_conf": 0.45,
        "detector_iou": 0.50,
        "parked_radius": 96.0,
        "link_speed": 100.0,
        "path_jump": 60.0,
    },
    "balanced": {
        "detector_conf": DETECTOR_CONF,
        "detector_iou": DETECTOR_IOU,
        "parked_radius": DETECTOR_PARKED_RADIUS,
        "link_speed": DETECTOR_LINK_SPEED_PX_S,
        "path_jump": DETECTOR_PATH_JUMP,
    },
    "sensitive": {
        "detector_conf": 0.25,
        "detector_iou": 0.30,
        "parked_radius": 32.0,
        "link_speed": 260.0,
        "path_jump": 140.0,
    },
}
