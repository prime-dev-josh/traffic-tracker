"""Run YOLO11 + ByteTrack and count completed start→end route crossings."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import cv2
import numpy as np

from traffic_tracker.config import VEHICLE_CLASS_IDS, VEHICLE_CLASS_NAMES, YOLO_MODEL_NAME
from traffic_tracker.core.device import describe_device, detect_device
from traffic_tracker.core.geometry import Geometry
from traffic_tracker.core.report import Trip
from traffic_tracker.core.video import VideoInfo, probe_video


ProgressCb = Callable[[int, int, str], None]
PreviewCb = Callable[[np.ndarray], None]


@dataclass
class ProcessResult:
    video: VideoInfo
    trips: list[Trip]
    device_label: str
    geometry: Geometry
    error: str | None = None


@dataclass
class _Pending:
    """Most recent start-line crossing for each route, per track."""

    starts: dict[str, float] = field(default_factory=dict)
    classes: dict[int, str] = field(default_factory=dict)
    completed: set[int] = field(default_factory=set)


_model = None


def _class_name(class_id: int) -> str:
    return VEHICLE_CLASS_NAMES.get(int(class_id), "vehicle")


def _get_model():
    """Load YOLO11 once so a batch of videos does not reload weights each time."""
    global _model
    if _model is None:
        from ultralytics import YOLO

        _model = YOLO(YOLO_MODEL_NAME)
    return _model


def _reset_tracker(model) -> None:
    """Clear Ultralytics persist-state so IDs do not carry across videos."""
    model.predictor = None


def process_video(
    video_path: str,
    geometry: Geometry,
    *,
    device: str | None = None,
    cancel_flag: Callable[[], bool] | None = None,
    progress_cb: ProgressCb | None = None,
    preview_cb: PreviewCb | None = None,
    preview_every: int = 15,
) -> ProcessResult:
    """Process one video. Heavy imports stay inside so the GUI can open quickly."""
    import supervision as sv

    info = probe_video(video_path)
    device = device or detect_device(timeout_s=20)
    device_label = describe_device(device)
    fitted = geometry.scaled_to(info.width, info.height)

    if progress_cb:
        progress_cb(0, info.frame_count or 1, "Loading YOLO11 model…")

    model = _get_model()
    _reset_tracker(model)

    start_zones: list[tuple[str, sv.LineZone]] = []
    end_zones: list[tuple[str, sv.LineZone]] = []
    for route in fitted.routes:
        start_zones.append(
            (
                route.name,
                sv.LineZone(
                    start=sv.Point(int(route.start.x1), int(route.start.y1)),
                    end=sv.Point(int(route.start.x2), int(route.start.y2)),
                ),
            )
        )
        end_zones.append(
            (
                route.name,
                sv.LineZone(
                    start=sv.Point(int(route.end.x1), int(route.end.y1)),
                    end=sv.Point(int(route.end.x2), int(route.end.y2)),
                ),
            )
        )

    annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.4)

    pending = _Pending()
    trips: list[Trip] = []

    cap = cv2.VideoCapture(str(info.path))
    if not cap.isOpened():
        return ProcessResult(info, [], device_label, fitted, error=f"Could not open {info.path}")

    frame_idx = 0
    try:
        while True:
            if cancel_flag and cancel_flag():
                return ProcessResult(info, trips, device_label, fitted, error="Cancelled")

            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            elapsed_s = frame_idx / info.fps

            result = model.track(
                frame,
                persist=True,
                verbose=False,
                classes=list(VEHICLE_CLASS_IDS),
                device=device,
                imgsz=640,
            )[0]
            detections = sv.Detections.from_ultralytics(result)

            if detections.tracker_id is not None:
                _record_crossings(
                    detections,
                    start_zones,
                    end_zones,
                    pending,
                    trips,
                    info.name,
                    elapsed_s,
                )

            if preview_cb and (frame_idx == 1 or frame_idx % preview_every == 0):
                preview = frame.copy()
                if detections.tracker_id is not None and detections.class_id is not None and len(detections) > 0:
                    labels = [
                        f"#{tid} {_class_name(cls)}"
                        for tid, cls in zip(detections.tracker_id, detections.class_id)
                    ]
                    preview = annotator.annotate(preview, detections)
                    preview = label_annotator.annotate(preview, detections, labels)
                preview = _draw_route_overlay(preview, fitted)
                preview_cb(preview)

            if progress_cb and (frame_idx == 1 or frame_idx % 10 == 0):
                progress_cb(
                    frame_idx,
                    info.frame_count or frame_idx,
                    f"{info.name}  •  {frame_idx}/{info.frame_count or '?'}  •  {len(trips)} trips",
                )
    finally:
        cap.release()

    if progress_cb:
        progress_cb(info.frame_count or frame_idx, info.frame_count or frame_idx, f"Finished {info.name}")
    return ProcessResult(info, trips, device_label, fitted)


def _draw_route_overlay(frame: np.ndarray, geometry: Geometry) -> np.ndarray:
    for route in geometry.routes:
        start = (int(route.start.x1), int(route.start.y1)), (int(route.start.x2), int(route.start.y2))
        end = (int(route.end.x1), int(route.end.y1)), (int(route.end.x2), int(route.end.y2))
        cv2.line(frame, start[0], start[1], (140, 214, 61), 3)
        cv2.line(frame, end[0], end[1], (107, 107, 255), 3)
        cv2.putText(frame, f"{route.name} start", start[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 214, 61), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{route.name} end", end[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (107, 107, 255), 1, cv2.LINE_AA)
    return frame


def _crossed_ids(zone, detections) -> list[int]:
    crossed_in, crossed_out = zone.trigger(detections)
    mask = np.logical_or(crossed_in, crossed_out)
    if detections.tracker_id is None or not np.any(mask):
        return []
    return [int(tid) for tid in detections.tracker_id[mask] if tid is not None and int(tid) >= 0]


def _record_crossings(
    detections,
    start_zones,
    end_zones,
    pending: _Pending,
    trips: list[Trip],
    video_name: str,
    elapsed_s: float,
) -> None:
    class_lookup = {}
    if detections.tracker_id is not None and detections.class_id is not None:
        for tid, cls in zip(detections.tracker_id, detections.class_id):
            class_lookup[int(tid)] = _class_name(int(cls))

    for route_name, zone in start_zones:
        for tid in _crossed_ids(zone, detections):
            if tid in pending.completed:
                continue
            pending.starts[f"{tid}:{route_name}"] = elapsed_s
            if tid in class_lookup:
                pending.classes[tid] = class_lookup[tid]

    for route_name, zone in end_zones:
        for tid in _crossed_ids(zone, detections):
            if tid in pending.completed:
                continue
            key = f"{tid}:{route_name}"
            if key not in pending.starts:
                continue
            trips.append(
                Trip(
                    video_name=video_name,
                    track_id=tid,
                    route_name=route_name,
                    completed_at_s=elapsed_s,
                    vehicle_class=pending.classes.get(tid, class_lookup.get(tid, "vehicle")),
                )
            )
            pending.completed.add(tid)
