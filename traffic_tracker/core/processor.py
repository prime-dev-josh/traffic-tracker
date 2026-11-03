"""Run YOLO11 + ByteTrack and record start/end crossings."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from traffic_tracker.config import (
    DRONE_VEHICLE_CLASS_IDS,
    STREET_TRACK_CONF,
    STREET_TRACK_IMGSZ,
    STREET_VEHICLE_CLASS_IDS,
    TRACK_CONF,
    TRACK_IMGSZ,
    VEHICLE_CLASS_NAMES,
)
from traffic_tracker.core.crossings import bbox_center, path_cross_t
from traffic_tracker.core.device import describe_device, detect_device
from traffic_tracker.core.geometry import (
    ORIGIN_TO_BOUND,
    ORIGINS,
    Geometry,
    NamedLine,
    name_from_north,
    side_from_point,
)
from traffic_tracker.core.models import ModelSource, resolve_model_source
from traffic_tracker.core.report import Trip
from traffic_tracker.core.video import VideoInfo, probe_video

COAST_S = 4.0


ProgressCb = Callable[[int, int, str], None]
PreviewCb = Callable[[np.ndarray], None]
TRACKER_PATH = Path(__file__).with_name("bytetrack.yaml")


@dataclass
class ProcessResult:
    video: VideoInfo
    trips: list[Trip]
    device_label: str
    geometry: Geometry
    error: str | None = None
    model_label: str = ""
    start_crossings: int = 0
    end_crossings: int = 0
    frames_with_tracks: int = 0


@dataclass
class _TrackState:
    tid: int
    xy: tuple[float, float]
    first_xy: tuple[float, float]
    first_time_s: float
    last_time_s: float
    cls: str = "vehicle"
    start: tuple[str, float] | None = None
    bbox_wh: tuple[float, float] = (20.0, 20.0)
    had_gate_trip: bool = False


@dataclass
class _Pending:
    """Live tracks. IDs are remapped when ByteTrack drops a crowded nadir car."""

    live: dict[int, _TrackState] = field(default_factory=dict)
    next_synth: int = 10000


_model = None
_model_source: ModelSource | None = None


def _class_name(class_id: int) -> str:
    return VEHICLE_CLASS_NAMES.get(int(class_id), "vehicle")


def _get_model(source: ModelSource | None = None):
    """Load YOLO11 once via Ultralytics AutoBackend (.pt, .onnx, or OpenVINO).

    Do not replace this with a custom ORT/NMS path. ByteTrack stays in
    ``model.track(persist=True)``.
    """
    global _model, _model_source
    source = source or resolve_model_source()
    if _model is None or _model_source != source:
        from ultralytics import YOLO

        _model = YOLO(str(source.path))
        _model_source = source
    return _model, source


def reset_model_cache() -> None:
    """Drop the cached Ultralytics model so a new backend can load."""
    global _model, _model_source
    _model = None
    _model_source = None


def _reset_tracker(model) -> None:
    """Clear Ultralytics persist-state so IDs do not carry across videos."""
    model.predictor = None


def _track_imgsz(width: int, height: int, *, overhead_drone: bool) -> int:
    longest = max(width, height)
    if overhead_drone:
        return TRACK_IMGSZ if longest >= 1280 else 1280
    if longest >= 1600:
        return max(STREET_TRACK_IMGSZ, 1280)
    return STREET_TRACK_IMGSZ


def _detections_from_result(sv, result):
    detections = sv.Detections.from_ultralytics(result)
    boxes = getattr(result, "boxes", None)
    if detections.tracker_id is None and boxes is not None and getattr(boxes, "id", None) is not None:
        detections.tracker_id = boxes.id.cpu().numpy().astype(int)
    return detections


def process_video(
    video_path: str,
    geometry: Geometry,
    *,
    device: str | None = None,
    cancel_flag: Callable[[], bool] | None = None,
    progress_cb: ProgressCb | None = None,
    preview_cb: PreviewCb | None = None,
    preview_every: int = 15,
    overhead_drone: bool = True,
) -> ProcessResult:
    """Process one video. Heavy imports stay inside so the GUI can open quickly."""
    import supervision as sv

    info = probe_video(video_path)
    device = device or detect_device(timeout_s=20)
    device_label = describe_device(device)
    fitted = geometry.scaled_to(info.width, info.height)
    imgsz = _track_imgsz(info.width, info.height, overhead_drone=overhead_drone)
    class_ids = list(DRONE_VEHICLE_CLASS_IDS if overhead_drone else STREET_VEHICLE_CLASS_IDS)
    conf = TRACK_CONF if overhead_drone else STREET_TRACK_CONF

    if progress_cb:
        mode = "overhead drone" if overhead_drone else "street-level"
        progress_cb(0, info.frame_count or 1, f"Loading YOLO11 ({mode}, imgsz={imgsz})…")

    model, source = _get_model()
    _reset_tracker(model)
    device_label = f"{device_label}  •  {source.label}"

    annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.4)

    pending = _Pending()
    trips: list[Trip] = []
    start_crossings = 0
    end_crossings = 0
    frames_with_tracks = 0

    cap = cv2.VideoCapture(str(info.path))
    if not cap.isOpened():
        return ProcessResult(
            info, [], device_label, fitted, error=f"Could not open {info.path}", model_label=source.label
        )

    frame_idx = 0
    try:
        while True:
            if cancel_flag and cancel_flag():
                return ProcessResult(
                    info,
                    trips,
                    device_label,
                    fitted,
                    error="Cancelled",
                    model_label=source.label,
                    start_crossings=start_crossings,
                    end_crossings=end_crossings,
                    frames_with_tracks=frames_with_tracks,
                )

            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            elapsed_s = frame_idx / info.fps

            result = model.track(
                frame,
                persist=True,
                verbose=False,
                classes=class_ids,
                device=device,
                imgsz=imgsz,
                conf=conf,
                max_det=500,
                iou=0.7,
                tracker=str(TRACKER_PATH),
            )[0]
            detections = _detections_from_result(sv, result)

            if len(detections) > 0:
                frames_with_tracks += 1
                starts, ends = _record_crossings(
                    detections,
                    fitted,
                    pending,
                    trips,
                    info.name,
                    elapsed_s,
                )
                start_crossings += starts
                end_crossings += ends
            end_crossings += _expire_tracks(pending, fitted, trips, info.name, elapsed_s)

            if preview_cb and (frame_idx == 1 or frame_idx % preview_every == 0):
                preview = frame.copy()
                if len(detections) > 0:
                    if detections.tracker_id is not None and detections.class_id is not None:
                        labels = [
                            f"#{tid} {_class_name(cls)}"
                            for tid, cls in zip(detections.tracker_id, detections.class_id)
                        ]
                    else:
                        labels = ["vehicle"] * len(detections)
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

    end_crossings += _expire_tracks(
        pending, fitted, trips, info.name, (frame_idx / info.fps) + COAST_S + 1.0, coast_s=0.0
    )
    if progress_cb:
        progress_cb(info.frame_count or frame_idx, info.frame_count or frame_idx, f"Finished {info.name}")
    return ProcessResult(
        info,
        trips,
        device_label,
        fitted,
        model_label=source.label,
        start_crossings=start_crossings,
        end_crossings=end_crossings,
        frames_with_tracks=frames_with_tracks,
    )


def _draw_route_overlay(frame: np.ndarray, geometry: Geometry) -> np.ndarray:
    if geometry.north is not None:
        p1 = (int(geometry.north.x1), int(geometry.north.y1))
        p2 = (int(geometry.north.x2), int(geometry.north.y2))
        cv2.arrowedLine(frame, p1, p2, (253, 184, 61), 3, tipLength=0.18)
        cv2.putText(frame, "N", p2, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (253, 184, 61), 2, cv2.LINE_AA)
    for item in geometry.starts:
        a = (int(item.line.x1), int(item.line.y1))
        b = (int(item.line.x2), int(item.line.y2))
        cv2.line(frame, a, b, (140, 214, 61), 3)
        cv2.putText(frame, item.name, a, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 214, 61), 1, cv2.LINE_AA)
    for item in geometry.ends:
        a = (int(item.line.x1), int(item.line.y1))
        b = (int(item.line.x2), int(item.line.y2))
        cv2.line(frame, a, b, (107, 107, 255), 3)
        cv2.putText(frame, item.name, a, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (107, 107, 255), 1, cv2.LINE_AA)
    return frame


def _gate_origin(item: NamedLine, geometry: Geometry) -> str:
    """Side of the intersection this gate sits on, from North — not draw order or color."""
    if geometry.north is not None:
        return name_from_north(
            item.line, ORIGINS, geometry.north, geometry.frame_width, geometry.frame_height
        )
    if item.name in ORIGINS:
        return item.name
    for origin, bound in ORIGIN_TO_BOUND.items():
        if item.name == bound:
            return origin
    return item.name or "N"


def _adopt_radius(geometry: Geometry) -> float:
    return max(120.0, 0.08 * min(geometry.frame_width, geometry.frame_height))


def _min_travel(geometry: Geometry) -> float:
    return max(40.0, 0.08 * min(geometry.frame_width, geometry.frame_height))


def _point_side(xy: tuple[float, float], geometry: Geometry) -> str | None:
    if geometry.north is None:
        return None
    return side_from_point(xy[0], xy[1], geometry.north, geometry.frame_width, geometry.frame_height)


def _size_ok(a: tuple[float, float], b: tuple[float, float]) -> bool:
    area_a = max(a[0] * a[1], 1.0)
    area_b = max(b[0] * b[1], 1.0)
    return max(area_a, area_b) / min(area_a, area_b) <= 4.0


def _nearest_unseen(
    pending: _Pending,
    xy: tuple[float, float],
    wh: tuple[float, float],
    seen: set[int],
    max_dist: float,
) -> _TrackState | None:
    best: _TrackState | None = None
    best_d = max_dist
    for state in pending.live.values():
        if state.tid in seen:
            continue
        dist = ((state.xy[0] - xy[0]) ** 2 + (state.xy[1] - xy[1]) ** 2) ** 0.5
        if dist < best_d and _size_ok(state.bbox_wh, wh):
            best = state
            best_d = dist
    return best


def _resolve_state(
    pending: _Pending,
    raw_tid: int,
    xy: tuple[float, float],
    wh: tuple[float, float],
    cls: str,
    elapsed_s: float,
    seen: set[int],
    max_dist: float,
) -> tuple[_TrackState, bool]:
    """Return (state, is_new). Reuses a nearby unseen track when ByteTrack switches IDs."""
    if raw_tid >= 0 and raw_tid in pending.live and raw_tid not in seen:
        return pending.live[raw_tid], False

    adopted = _nearest_unseen(pending, xy, wh, seen, max_dist)
    if adopted is not None:
        pending.live.pop(adopted.tid, None)
        if raw_tid >= 0 and raw_tid not in pending.live:
            adopted.tid = raw_tid
        pending.live[adopted.tid] = adopted
        return adopted, False

    tid = raw_tid if raw_tid >= 0 and raw_tid not in seen else pending.next_synth
    if tid == pending.next_synth:
        pending.next_synth += 1
    state = _TrackState(
        tid=tid,
        xy=xy,
        first_xy=xy,
        first_time_s=elapsed_s,
        last_time_s=elapsed_s,
        cls=cls,
        bbox_wh=wh,
    )
    pending.live[tid] = state
    return state, True


def _apply_gate_hits(
    state: _TrackState,
    prev: tuple[float, float],
    xy: tuple[float, float],
    gates: list[tuple[str, object]],
    pad: float,
    trips: list[Trip],
    video_name: str,
    elapsed_s: float,
) -> tuple[int, int]:
    hits: list[tuple[float, str]] = []
    for origin, line in gates:
        t = path_cross_t(prev, xy, line, pad_px=pad)
        if t is not None:
            hits.append((t, origin))
    hits.sort()
    start_hits = 0
    end_hits = 0
    for _, origin in hits:
        if state.start is None:
            state.start = (origin, elapsed_s)
            start_hits += 1
        elif state.start[0] != origin:
            start_origin, start_time_s = state.start
            trips.append(
                Trip(
                    video_name=video_name,
                    track_id=state.tid,
                    start_destination=start_origin,
                    start_time_s=start_time_s,
                    end_destination=ORIGIN_TO_BOUND.get(origin, origin),
                    end_time_s=elapsed_s,
                    vehicle_class=state.cls,
                )
            )
            state.start = None
            state.had_gate_trip = True
            end_hits += 1
            break
    return start_hits, end_hits


def _finalize_track(state: _TrackState, geometry: Geometry, trips: list[Trip], video_name: str) -> int:
    """Finish a dead track from last side or first→last path when gates were missed."""
    if state.start is not None:
        last_side = _point_side(state.xy, geometry)
        if last_side and last_side != state.start[0]:
            start_origin, start_time_s = state.start
            trips.append(
                Trip(
                    video_name=video_name,
                    track_id=state.tid,
                    start_destination=start_origin,
                    start_time_s=start_time_s,
                    end_destination=ORIGIN_TO_BOUND.get(last_side, last_side),
                    end_time_s=state.last_time_s,
                    vehicle_class=state.cls,
                )
            )
            state.start = None
            state.had_gate_trip = True
            return 1
        return 0
    if state.had_gate_trip:
        return 0
    first_side = _point_side(state.first_xy, geometry)
    last_side = _point_side(state.xy, geometry)
    travel = ((state.xy[0] - state.first_xy[0]) ** 2 + (state.xy[1] - state.first_xy[1]) ** 2) ** 0.5
    if first_side and last_side and first_side != last_side and travel >= _min_travel(geometry):
        trips.append(
            Trip(
                video_name=video_name,
                track_id=state.tid,
                start_destination=first_side,
                start_time_s=state.first_time_s,
                end_destination=ORIGIN_TO_BOUND.get(last_side, last_side),
                end_time_s=state.last_time_s,
                vehicle_class=state.cls,
            )
        )
        state.had_gate_trip = True
        return 1
    return 0


def _expire_tracks(
    pending: _Pending,
    geometry: Geometry,
    trips: list[Trip],
    video_name: str,
    now_s: float,
    *,
    coast_s: float = COAST_S,
) -> int:
    keep: dict[int, _TrackState] = {}
    inferred = 0
    for tid, state in pending.live.items():
        if now_s - state.last_time_s > coast_s:
            inferred += _finalize_track(state, geometry, trips, video_name)
        else:
            keep[tid] = state
    pending.live = keep
    return inferred


def _record_crossings(
    detections,
    geometry: Geometry,
    pending: _Pending,
    trips: list[Trip],
    video_name: str,
    elapsed_s: float,
) -> tuple[int, int]:
    gates = [(_gate_origin(item, geometry), item.line) for item in (*geometry.starts, *geometry.ends)]
    pad = max(24.0, 0.02 * max(geometry.frame_width, geometry.frame_height))
    max_dist = _adopt_radius(geometry)
    start_hits = 0
    end_hits = 0
    seen: set[int] = set()
    n = len(detections.xyxy)
    raw_ids = [int(t) for t in detections.tracker_id] if detections.tracker_id is not None else [-1] * n
    classes = (
        [_class_name(int(c)) for c in detections.class_id]
        if detections.class_id is not None
        else ["vehicle"] * n
    )

    for i, xyxy in enumerate(detections.xyxy):
        xy = bbox_center(xyxy)
        x1, y1, x2, y2 = (float(v) for v in xyxy)
        wh = (max(x2 - x1, 1.0), max(y2 - y1, 1.0))
        state, is_new = _resolve_state(
            pending, raw_ids[i], xy, wh, classes[i], elapsed_s, seen, max_dist
        )
        seen.add(state.tid)
        state.cls = classes[i]
        state.bbox_wh = wh
        if not is_new:
            added_s, added_e = _apply_gate_hits(
                state, state.xy, xy, gates, pad, trips, video_name, elapsed_s
            )
            start_hits += added_s
            end_hits += added_e
        state.xy = xy
        state.last_time_s = elapsed_s
    return start_hits, end_hits
