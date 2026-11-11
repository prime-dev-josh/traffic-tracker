"""YOLO box detector used when Settings selects Trained detector."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from traffic_tracker.config import (
    DETECTOR_CONF,
    DETECTOR_CONTAIN_FRAC,
    DETECTOR_IMGSZ,
    DETECTOR_IOU,
    DETECTOR_STACK_IOU,
    YOLO_PT_NAME,
)
from traffic_tracker.core.crossings import box_iou
from traffic_tracker.core.device import yolo_runtime
from traffic_tracker.core.models import resolve_model_source
from traffic_tracker.core.training import visdrone_weights_path

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


CLASS_RANK = {
    "motorcycle": 3,
    "van": 2,
    "truck": 2,
    "bus": 2,
    "car": 1,
    "vehicle": 0,
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


def _box_area(xyxy: tuple[float, float, float, float]) -> float:
    return max(0.0, xyxy[2] - xyxy[0]) * max(0.0, xyxy[3] - xyxy[1])


def _intersection_area(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def _same_vehicle_stack(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
    *,
    iou_min: float = DETECTOR_STACK_IOU,
    contain_frac: float = DETECTOR_CONTAIN_FRAC,
) -> bool:
    if box_iou(a, b) >= iou_min:
        return True
    inter = _intersection_area(a, b)
    if inter <= 0.0:
        return False
    smaller = min(_box_area(a), _box_area(b))
    return smaller > 0.0 and inter / smaller >= contain_frac


def _detection_key(item: tuple[tuple[float, float, float, float], str, float]) -> tuple[float, int, float]:
    xyxy, label, score = item
    return (float(score), CLASS_RANK.get(label, 0), _box_area(xyxy))


def suppress_duplicate_boxes(
    detections: list[tuple[tuple[float, float, float, float], str, float]],
    *,
    iou_min: float = DETECTOR_STACK_IOU,
    contain_frac: float = DETECTOR_CONTAIN_FRAC,
) -> list[tuple[tuple[float, float, float, float], str, float]]:
    """Keep one box per vehicle. Prefer higher confidence, then a more specific class."""
    kept: list[tuple[tuple[float, float, float, float], str, float]] = []
    for item in sorted(detections, key=_detection_key, reverse=True):
        xyxy = item[0]
        if any(_same_vehicle_stack(xyxy, other[0], iou_min=iou_min, contain_frac=contain_frac) for other in kept):
            continue
        kept.append(item)
    return kept


def draw_detection_preview(
    frame: np.ndarray,
    detections: list[tuple[tuple[float, float, float, float], str, float]],
) -> np.ndarray:
    """BGR still with surviving detector boxes. Used by the Settings frame check."""
    preview = frame.copy()
    for xyxy, label, score in detections:
        x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 220, 255), 2)
        caption = f"{label} {score:.2f}"
        cv2.putText(
            preview,
            caption,
            (x1, max(16, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 220, 255),
            1,
            cv2.LINE_AA,
        )
    return preview


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


def weights_for_source(source: str | None = None, explicit: str | None = None, base: Path | None = None) -> Path:
    """Trained detector always uses VisDrone weights when they exist."""
    del source
    return resolve_detector_weights(explicit, base)


def resolve_detector_weights(explicit: str | None = None, base: Path | None = None) -> Path:
    if explicit:
        path = Path(explicit)
        if path.is_file():
            return path
    visdrone = visdrone_weights_path(base)
    if visdrone.is_file():
        return visdrone
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
    imgsz: int = DETECTOR_IMGSZ,
    device: str | None = None,
) -> tuple[list[tuple[tuple[float, float, float, float], str, float]], np.ndarray]:
    """Return ((x1, y1, x2, y2), class, confidence) detections and a preview mask."""
    height, width = frame.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    runtime, quantize = yolo_runtime(device)
    results = model.predict(
        frame,
        verbose=False,
        imgsz=int(imgsz) or DETECTOR_IMGSZ,
        conf=conf,
        iou=iou,
        agnostic_nms=True,
        device=runtime,
        quantize=quantize,
    )
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
    detections = suppress_duplicate_boxes(detections)
    for xyxy, _label, _score in detections:
        x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    return detections, mask
