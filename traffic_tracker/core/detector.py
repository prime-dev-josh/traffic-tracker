"""YOLO box detector used when Settings selects Trained detector."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from traffic_tracker.config import DETECTOR_CONF, DETECTOR_IOU, TRAIN_IMGSZ, YOLO_PT_NAME
from traffic_tracker.core.models import resolve_model_source
from traffic_tracker.core.training import (
    has_trained_weights,
    site_weights_path,
    trained_weights_path,
    visdrone_weights_path,
)

# COCO: car, motorcycle, bus, truck. Used only when names are not a known vehicle set.
COCO_VEHICLE_IDS = {2, 3, 5, 7}
VEHICLE_CLASS_NAMES = {
    "car",
    "van",
    "truck",
    "bus",
    "motor",
    "motorcycle",
    "motorbike",
}
IGNORE_CLASS_NAMES = {
    "pedestrian",
    "people",
    "person",
    "bicycle",
    "bike",
    "boat",
    "ship",
    "vessel",
    "yacht",
    "canoe",
    "kayak",
    "jet-ski",
    "jetski",
    "watercraft",
    "ferry",
    "sailboat",
    "speedboat",
    "dinghy",
    "airplane",
    "plane",
    "tricycle",
    "awning-tricycle",
    "parking",
    "parked",
}


CLASS_CANONICAL = {
    "car": "car",
    "van": "van",
    "truck": "truck",
    "bus": "bus",
    "motor": "motorcycle",
    "motorcycle": "motorcycle",
    "motorbike": "motorcycle",
}


def _norm_class_name(raw: object) -> str:
    return str(raw).strip().lower().replace("_", "-").replace(" ", "-")


def normalize_vehicle_class(raw: object) -> str:
    """Map detector names to car / van / truck / motorcycle / bus."""
    return CLASS_CANONICAL.get(_norm_class_name(raw), "vehicle")


def keep_detector_class(names: dict, cls: int) -> bool:
    """Keep cars, vans, trucks, motorcycles, and buses. Drop people, bikes, boats, tricycles."""
    raw = names.get(cls, "") if names else ""
    name = _norm_class_name(raw)
    if name in IGNORE_CLASS_NAMES or "boat" in name or "ship" in name:
        return False
    if not names or len(names) <= 1:
        return True
    known = {_norm_class_name(value) for value in names.values()}
    if known & VEHICLE_CLASS_NAMES:
        return name in VEHICLE_CLASS_NAMES
    return cls in COCO_VEHICLE_IDS

_MODEL_CACHE: dict[str, object] = {}


def reset_detector_cache() -> None:
    _MODEL_CACHE.clear()


def weights_for_source(source: str, explicit: str | None = None, base: Path | None = None) -> Path:
    """Pick VisDrone, the site fine-tune, or a custom file. Missing files fall through."""
    if source == "visdrone" and visdrone_weights_path(base).is_file():
        return visdrone_weights_path(base)
    if source == "custom" and explicit:
        path = Path(explicit)
        if path.is_file():
            return path
    if source == "site":
        if site_weights_path(base).is_file():
            return site_weights_path(base)
        if has_trained_weights(base):
            return trained_weights_path(base)
    return resolve_detector_weights(explicit, base)


def resolve_detector_weights(explicit: str | None = None, base: Path | None = None) -> Path:
    if explicit:
        path = Path(explicit)
        if path.is_file():
            return path
    if has_trained_weights(base):
        return trained_weights_path(base)
    try:
        source = resolve_model_source("pt")
        if source.path.is_file():
            return source.path
    except Exception:
        pass
    return Path(YOLO_PT_NAME)


def load_detector(weights: Path):
    key = str(Path(weights))
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached
    from ultralytics import YOLO

    model = YOLO(str(weights))
    _MODEL_CACHE[key] = model
    return model


def boxes_from_detector(
    model,
    frame: np.ndarray,
    *,
    min_area: float,
    max_area: float,
    conf: float = DETECTOR_CONF,
    iou: float = DETECTOR_IOU,
) -> tuple[list[tuple[tuple[float, float, float, float], str, float]], np.ndarray]:
    """Return ((x1, y1, x2, y2), class, confidence) detections and a preview mask."""
    height, width = frame.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    results = model.predict(frame, verbose=False, imgsz=TRAIN_IMGSZ, conf=conf, iou=iou)
    names = getattr(model, "names", {}) or {}
    if not isinstance(names, dict):
        names = {index: name for index, name in enumerate(names)}

    detections: list[tuple[tuple[float, float, float, float], str, float]] = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls = int(box.cls[0]) if box.cls is not None else 0
            if not keep_detector_class(names, cls):
                continue
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            x1 = min(max(x1, 0.0), width)
            y1 = min(max(y1, 0.0), height)
            x2 = min(max(x2, 0.0), width)
            y2 = min(max(y2, 0.0), height)
            if x2 <= x1 or y2 <= y1:
                continue
            area = (x2 - x1) * (y2 - y1)
            if area < min_area or area > max_area:
                continue
            xyxy = (x1, y1, x2, y2)
            label = normalize_vehicle_class(names.get(cls, "vehicle"))
            score = float(box.conf[0]) if getattr(box, "conf", None) is not None else float(conf)
            detections.append((xyxy, label, score))
            cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, -1)
    return detections, mask
