"""Motion-mask or trained boxes, record start/end crossings, and keep path traces."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from traffic_tracker.config import (
    DETECTOR_CONF,
    DETECTOR_LINK_CAP_FRAC,
    DETECTOR_LINK_CAP_MIN,
    DETECTOR_LINK_MIN_PX,
    DETECTOR_IOU,
    DETECTOR_LINK_SPEED_PX_S,
    DETECTOR_MATCH_IOU_MIN,
    DETECTOR_PATH_JUMP,
    DETECTOR_PATH_JUMP_FRAC,
    DETECTOR_STATIC_EXPIRE_S,
    DETECTOR_STATIC_RADIUS_FRAC,
    DETECTOR_STATIC_RADIUS_MIN,
    DETECTOR_STATIC_SETTLE_S,
    MOTION_COAST_S,
    MOTION_HOLD_S,
    MOTION_HOLD_SPEED_PX_S,
    MOTION_WARMUP_S,
)
from traffic_tracker.core.crossings import bbox_center, box_iou, confirmed_cross_t, path_cross_t, signed_offset
from traffic_tracker.core.geometry import (
    ORIGIN_TO_BOUND,
    ORIGINS,
    Geometry,
    NamedLine,
    name_from_north,
    point_in_roadway,
    side_from_point,
)
from traffic_tracker.core.detector import (
    boxes_from_detector,
    load_detector,
    reset_detector_cache,
    resolve_detector_weights,
)
from traffic_tracker.core.motion import (
    blobs_from_frame,
    create_subtractor,
    default_area_limits,
    process_scale,
)
from traffic_tracker.core.report import Trip
from traffic_tracker.core.training import prior_areas
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
    model_label: str = "motion mask"
    start_crossings: int = 0
    end_crossings: int = 0
    frames_with_tracks: int = 0
    trace_image: np.ndarray | None = None


@dataclass
class _TrackState:
    tid: int
    xy: tuple[float, float]
    first_xy: tuple[float, float]
    first_time_s: float
    last_time_s: float
    last_motion_s: float = 0.0
    cls: str = "vehicle"
    start: tuple[str, float] | None = None
    bbox_wh: tuple[float, float] = (20.0, 20.0)
    last_xyxy: tuple[float, float, float, float] | None = None
    speed: float = 0.0
    sightings: int = 1
    holding: bool = False
    had_gate_trip: bool = False
    last_side: str | None = None
    path: list[tuple[float, float]] = field(default_factory=list)
    active: bool = False
    settle_xy: tuple[float, float] | None = None
    settle_s: float = 0.0
    vel: tuple[float, float] = (0.0, 0.0)
    # last_observed_* is a real detection. xy / last_time_s may be a prediction.
    last_observed_s: float = 0.0
    last_observed_xy: tuple[float, float] | None = None
    last_observed_xyxy: tuple[float, float, float, float] | None = None
    estimate: str = "observed"
    confidence: float = 0.0
    start_evidence: str = "observed"


@dataclass
class _Pending:
    """Centroid tracks. A nearby new blob continues the same vehicle."""

    live: dict[int, _TrackState] = field(default_factory=dict)
    next_synth: int = 1
    finished_paths: list[tuple[str, list[tuple[float, float]]]] = field(default_factory=list)
    parked_radius: float | None = None
    link_speed: float | None = None
    path_jump: float | None = None
    diagnostics_enabled: bool = False
    diagnostics: list[dict] = field(default_factory=list)
    frame_index: int = 0
    dbg_counts: dict[str, int] = field(default_factory=dict)


def _dbg(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    # #region agent log
    try:
        import json
        import time

        with open(
            r"c:\Users\Josh.Marrero\OneDrive - PRIME AE Group, Inc\Desktop\Project Process AI Work\traffic-tracker\.cursor\debug-0630c2.log",
            "a",
            encoding="utf-8",
        ) as handle:
            handle.write(
                json.dumps(
                    {
                        "sessionId": "0630c2",
                        "hypothesisId": hypothesis_id,
                        "location": location,
                        "message": message,
                        "data": data,
                        "timestamp": int(time.time() * 1000),
                    }
                )
                + "\n"
            )
    except Exception:
        pass
    # #endregion


def _dbg_bump(pending: _Pending, key: str, n: int = 1) -> None:
    pending.dbg_counts[key] = pending.dbg_counts.get(key, 0) + n


def reset_model_cache() -> None:
    reset_detector_cache()


def process_video(
    video_path: str,
    geometry: Geometry,
    *,
    device: str | None = None,
    cancel_flag: Callable[[], bool] | None = None,
    progress_cb: ProgressCb | None = None,
    preview_cb: PreviewCb | None = None,
    preview_every: int = 15,
    min_area: int | None = None,
    max_area: int | None = None,
    overhead_drone: bool = True,
    training_base: Path | None = None,
    detector_mode: str = "motion",
    weights_path: str | None = None,
    detector_conf: float = DETECTOR_CONF,
    detector_iou: float = DETECTOR_IOU,
    parked_radius: float | None = None,
    link_speed: float | None = None,
    path_jump: float | None = None,
    diagnostics: bool = False,
) -> ProcessResult:
    """Process one video. Moving blobs or YOLO boxes in the size band are vehicles."""
    del device, overhead_drone
    info = probe_video(video_path)
    fitted = geometry.scaled_to(info.width, info.height)
    suggested_min, suggested_max = default_area_limits(info.width, info.height)
    ref_limits = fitted.vehicle_area_limits(extra_areas=prior_areas(training_base))
    if ref_limits is not None:
        min_area, max_area = ref_limits
    else:
        min_area = suggested_min if min_area is None else int(min_area)
        max_area = suggested_max if max_area is None else int(max_area)
    scale = process_scale(info.width, info.height)
    use_detector = detector_mode == "trained"
    detector = None
    weights = None
    if use_detector:
        weights = resolve_detector_weights(weights_path, training_base)
        try:
            detector = load_detector(weights)
        except Exception as exc:
            return ProcessResult(
                info,
                [],
                f"trained detector ({weights.name})",
                fitted,
                error=f"Could not load detector weights {weights}: {exc}",
            )
        device_label = f"trained detector ({weights.name})"
        model_label = device_label
    else:
        device_label = "motion mask (CPU)"
        model_label = "motion mask"

    if progress_cb:
        progress_cb(
            0,
            info.frame_count or 1,
            f"{device_label}  •  box {min_area}–{max_area} px²",
        )

    pending = _Pending(
        parked_radius=parked_radius,
        link_speed=link_speed,
        path_jump=path_jump,
        diagnostics_enabled=diagnostics,
    )
    trips: list[Trip] = []
    start_crossings = 0
    end_crossings = 0
    frames_with_tracks = 0
    subtractor = None if use_detector else create_subtractor()
    prev_gray = None
    trace_frame: np.ndarray | None = None
    trace_at = max(1, int((info.frame_count or 1) * 0.4))

    cap = cv2.VideoCapture(str(info.path))
    if not cap.isOpened():
        return ProcessResult(
            info, [], device_label, fitted, error=f"Could not open {info.path}"
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
                    start_crossings=start_crossings,
                    end_crossings=end_crossings,
                    frames_with_tracks=frames_with_tracks,
                )

            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            elapsed_s = frame_idx / info.fps
            if use_detector and detector is not None:
                detections, mask = boxes_from_detector(
                    detector,
                    frame,
                    min_area=min_area,
                    max_area=max_area,
                    conf=detector_conf,
                    iou=detector_iou,
                )
                boxes = [item[0] for item in detections]
                classes = [item[1] for item in detections]
                confs = [float(item[2]) for item in detections]
            else:
                freeze = [
                    state.last_xyxy
                    for state in pending.live.values()
                    if state.last_xyxy and not state.had_gate_trip
                ]
                boxes, mask, prev_gray = blobs_from_frame(
                    subtractor,
                    frame,
                    min_area=min_area,
                    max_area=max_area,
                    scale=scale,
                    prev_gray=prev_gray,
                    freeze_boxes=freeze,
                )
                classes = ["vehicle"] * len(boxes)
                confs = [1.0] * len(boxes)

            if frame_idx == 1 or frame_idx == trace_at:
                trace_frame = frame.copy()

            if use_detector or elapsed_s >= MOTION_WARMUP_S:
                pending.frame_index = frame_idx
                starts, ends = _record_crossings(
                    boxes,
                    fitted,
                    pending,
                    trips,
                    info.name,
                    elapsed_s,
                    moving_only=use_detector,
                    classes=classes,
                    confidences=confs,
                )
                start_crossings += starts
                end_crossings += ends
                if boxes or pending.live:
                    frames_with_tracks += 1
                end_crossings += _expire_tracks(
                    pending, fitted, trips, info.name, elapsed_s, infer_path=not use_detector
                )

            if preview_cb and (frame_idx == 1 or frame_idx % preview_every == 0):
                preview = _draw_motion_preview(
                    frame, mask, boxes, pending, fitted, trips, active_only=use_detector
                )
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
        pending,
        fitted,
        trips,
        info.name,
        (frame_idx / max(info.fps, 1e-6)) + MOTION_COAST_S + 1.0,
        coast_s=0.0,
        infer_path=not use_detector,
    )
    if progress_cb:
        progress_cb(info.frame_count or frame_idx, info.frame_count or frame_idx, f"Finished {info.name}")
    _dbg(
        "H5",
        "processor.py:process_video",
        "process finished",
        {
            "trips": len(trips),
            "start_crossings": start_crossings,
            "end_crossings": end_crossings,
            "use_detector": use_detector,
            "roadways": len(fitted.roadways),
            "starts": [item.name for item in fitted.starts],
            "ends": [item.name for item in fitted.ends],
            "counts": dict(pending.dbg_counts),
        },
    )
    overlay = None
    if trace_frame is not None:
        overlay = _draw_path_overlay(trace_frame, trips, pending, fitted)
    return ProcessResult(
        info,
        trips,
        device_label,
        fitted,
        model_label=model_label,
        start_crossings=start_crossings,
        end_crossings=end_crossings,
        frames_with_tracks=frames_with_tracks,
        trace_image=overlay,
    )


PATH_COLOR = (220, 20, 255)
PATH_WIDTH = 1


def _path_color(_cls: str = "") -> tuple[int, int, int]:
    return PATH_COLOR


def _append_path(state: _TrackState, xy: tuple[float, float], geometry: Geometry | None = None) -> None:
    if state.path and geometry is not None and geometry.occlusions:
        last = state.path[-1]
        dx, dy = xy[0] - last[0], xy[1] - last[1]
        dist = (dx * dx + dy * dy) ** 0.5
        if dist >= 16.0 and (
            _segment_hits_user_occlusion(last, xy, geometry)
            or _points_share_drop_zone(last, xy, geometry)
        ):
            steps = max(2, int(dist / 16.0))
            for index in range(1, steps):
                t = index / steps
                point = (last[0] + dx * t, last[1] + dy * t)
                prev = state.path[-1]
                if ((point[0] - prev[0]) ** 2 + (point[1] - prev[1]) ** 2) ** 0.5 >= 1.0:
                    state.path.append(point)
    if not state.path:
        state.path.append(xy)
        return
    last = state.path[-1]
    if ((xy[0] - last[0]) ** 2 + (xy[1] - last[1]) ** 2) ** 0.5 >= 1.0:
        state.path.append(xy)


def _path_jump_limit(geometry: Geometry | None = None, path_jump: float | None = None) -> float:
    if path_jump is not None:
        return float(path_jump)
    if geometry is None:
        return DETECTOR_PATH_JUMP
    return max(DETECTOR_PATH_JUMP, DETECTOR_PATH_JUMP_FRAC * min(geometry.frame_width, geometry.frame_height))


def _path_chunks(
    points: list[tuple[float, float]],
    max_jump: float,
    geometry: Geometry | None = None,
) -> list[list[tuple[float, float]]]:
    """Split a polyline whenever consecutive points jump farther than max_jump."""
    if len(points) < 2:
        return []
    chunks: list[list[tuple[float, float]]] = []
    current = [points[0]]
    for point in points[1:]:
        last = current[-1]
        step = ((point[0] - last[0]) ** 2 + (point[1] - last[1]) ** 2) ** 0.5
        through_zone = bool(
            geometry
            and (
                _segment_hits_user_occlusion(last, point, geometry)
                or _points_share_drop_zone(last, point, geometry)
            )
        )
        if step > max_jump and not through_zone:
            if len(current) >= 2:
                chunks.append(current)
            current = [point]
        else:
            current.append(point)
    if len(current) >= 2:
        chunks.append(current)
    return chunks


def _keep_path(pending: _Pending, state: _TrackState, geometry: Geometry | None = None) -> None:
    if not state.active:
        return
    for chunk in _path_chunks(state.path, _path_jump_limit(geometry, pending.path_jump), geometry):
        pending.finished_paths.append((state.cls, chunk))


def _make_trip(
    state: _TrackState,
    video_name: str,
    start_origin: str,
    start_time_s: float,
    end_origin: str,
    end_time_s: float,
    evidence: str = "observed",
) -> Trip:
    return Trip(
        video_name=video_name,
        track_id=state.tid,
        start_destination=start_origin,
        start_time_s=start_time_s,
        end_destination=ORIGIN_TO_BOUND.get(end_origin, end_origin),
        end_time_s=end_time_s,
        vehicle_class=state.cls or "vehicle",
        path=list(state.path),
        evidence=evidence,
    )


def _halo_polyline(frame: np.ndarray, points: list[tuple[float, float]], color: tuple[int, int, int], width: int = PATH_WIDTH) -> None:
    pts = np.array([(int(round(x)), int(round(y))) for x, y in points], dtype=np.int32)
    if len(pts) < 2:
        return
    cv2.polylines(frame, [pts], False, (0, 0, 0), width + 1, cv2.LINE_AA)
    cv2.polylines(frame, [pts], False, color, width, cv2.LINE_AA)


def _draw_paths(
    frame: np.ndarray,
    pending: _Pending,
    trips: list[Trip] | None = None,
    geometry: Geometry | None = None,
) -> None:
    del trips
    traces: list[tuple[str, list[tuple[float, float]]]] = list(pending.finished_paths)
    traces.extend(
        (state.cls, state.path)
        for state in pending.live.values()
        if state.active and len(state.path) >= 2
    )
    jump = _path_jump_limit(geometry, pending.path_jump)
    for _cls, points in traces:
        for chunk in _path_chunks(points, jump, geometry):
            _halo_polyline(frame, chunk, PATH_COLOR, PATH_WIDTH)


def _draw_path_overlay(
    frame: np.ndarray,
    trips: list[Trip],
    pending: _Pending,
    geometry: Geometry,
) -> np.ndarray:
    """Darken the still so neon traces stay readable on pale drone pavement."""
    preview = (frame.astype(np.float32) * 0.72).astype(np.uint8)
    _draw_paths(preview, pending, trips, geometry)
    return _draw_route_overlay(preview, geometry)


def _draw_motion_preview(
    frame: np.ndarray,
    mask: np.ndarray,
    boxes: list[tuple[float, float, float, float]],
    pending: _Pending,
    geometry: Geometry,
    trips: list[Trip] | None = None,
    active_only: bool = False,
) -> np.ndarray:
    preview = frame.copy()
    draw_boxes = boxes
    if active_only:
        draw_boxes = [
            state.last_xyxy
            for state in pending.live.values()
            if state.active and state.last_xyxy is not None
        ]
        mask = np.zeros(preview.shape[:2], dtype=np.uint8)
        for x1, y1, x2, y2 in draw_boxes:
            cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, -1)
    if mask is not None and mask.size:
        tint = np.zeros_like(preview)
        tint[:, :, 1] = 200
        cover = (mask > 0).astype(np.uint8)
        for channel in range(3):
            preview[:, :, channel] = np.where(
                cover,
                (preview[:, :, channel] * 0.65 + tint[:, :, channel] * 0.35).astype(np.uint8),
                preview[:, :, channel],
            )
    states = [state for state in pending.live.values() if (not active_only or state.active)]
    used = set()
    for x1, y1, x2, y2 in draw_boxes:
        xyxy = (x1, y1, x2, y2)
        tid = None
        best_iou = DETECTOR_MATCH_IOU_MIN
        for state in states:
            if state.tid in used or state.last_xyxy is None:
                continue
            overlap = box_iou(xyxy, state.last_xyxy)
            if overlap > best_iou:
                best_iou = overlap
                tid = state.tid
        if tid is None:
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            best = 40.0
            for state in states:
                if state.tid in used:
                    continue
                dist = ((state.xy[0] - cx) ** 2 + (state.xy[1] - cy) ** 2) ** 0.5
                if dist < best:
                    tid = state.tid
                    best = dist
        if tid is not None:
            used.add(tid)
        w, h = int(max(x2 - x1, 1)), int(max(y2 - y1, 1))
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        _halo_rect(preview, p1, p2, (220, 20, 255))
        label = f"#{tid} {w}x{h}" if tid is not None else f"{w}x{h}"
        _halo_text(preview, label, (p1[0], max(16, p1[1] - 6)), (220, 20, 255), 0.45)
    for state in pending.live.values():
        if state.tid in used or not state.last_xyxy:
            continue
        if active_only and not state.active:
            continue
        x1, y1, x2, y2 = state.last_xyxy
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        if state.estimate == "predicted":
            color = (255, 200, 0)
            label = f"#{state.tid} pred"
        elif state.estimate == "held":
            color = (40, 200, 255)
            label = f"#{state.tid} hold"
        else:
            color = (220, 20, 255)
            label = f"#{state.tid}"
        _halo_rect(preview, p1, p2, color)
        _halo_text(preview, label, (p1[0], max(16, p1[1] - 6)), color, 0.45)
    _draw_paths(preview, pending, trips, geometry)
    return _draw_route_overlay(preview, geometry)


def _halo_line(frame: np.ndarray, a: tuple[int, int], b: tuple[int, int], color: tuple[int, int, int], width: int = 3) -> None:
    cv2.line(frame, a, b, (0, 0, 0), width + 3, cv2.LINE_AA)
    cv2.line(frame, a, b, color, width, cv2.LINE_AA)


def _halo_rect(frame: np.ndarray, p1: tuple[int, int], p2: tuple[int, int], color: tuple[int, int, int]) -> None:
    cv2.rectangle(frame, p1, p2, (0, 0, 0), 5)
    cv2.rectangle(frame, p1, p2, color, 2)


def _halo_text(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    scale: float,
) -> None:
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _draw_route_overlay(frame: np.ndarray, geometry: Geometry) -> np.ndarray:
    if geometry.north is not None:
        p1 = (int(geometry.north.x1), int(geometry.north.y1))
        p2 = (int(geometry.north.x2), int(geometry.north.y2))
        cv2.arrowedLine(frame, p1, p2, (0, 0, 0), 5, tipLength=0.18)
        cv2.arrowedLine(frame, p1, p2, (255, 200, 40), 3, tipLength=0.18)
        _halo_text(frame, "N", p2, (255, 200, 40), 0.6)
    for item in geometry.starts:
        a = (int(item.line.x1), int(item.line.y1))
        b = (int(item.line.x2), int(item.line.y2))
        _halo_line(frame, a, b, (60, 230, 0))
        _halo_text(frame, item.name, a, (60, 230, 0), 0.55)
    for item in geometry.ends:
        a = (int(item.line.x1), int(item.line.y1))
        b = (int(item.line.x2), int(item.line.y2))
        _halo_line(frame, a, b, (40, 40, 255))
        _halo_text(frame, item.name, a, (40, 40, 255), 0.55)
    for index, raw in enumerate(geometry.occlusions, start=1):
        box = raw.normalized()
        p1 = (int(box.x1), int(box.y1))
        p2 = (int(box.x2), int(box.y2))
        _halo_rect(frame, p1, p2, (0, 170, 255))
        _halo_text(frame, f"Drop zone {index}", (p1[0], max(16, p1[1] - 6)), (0, 170, 255), 0.45)
    if geometry.roadways:
        overlay = frame.copy()
        for poly in geometry.roadways:
            if len(poly.points) < 3:
                continue
            pts = np.array([(int(round(x)), int(round(y))) for x, y in poly.points], dtype=np.int32)
            cv2.fillPoly(overlay, [pts], (255, 170, 40))
            cv2.polylines(overlay, [pts], True, (255, 210, 80), 2, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.28, frame, 0.72, 0, frame)
        _halo_text(frame, "Roadway", (16, frame.shape[0] - 16), (255, 210, 80), 0.5)
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
    return max(180.0, 0.15 * min(geometry.frame_width, geometry.frame_height))


def _link_radius(
    geometry: Geometry,
    dt: float,
    moving_only: bool,
    link_speed: float | None = None,
) -> float:
    short = min(geometry.frame_width, geometry.frame_height)
    if not moving_only:
        return _adopt_radius(geometry)
    dt = max(float(dt), 1e-3)
    speed = DETECTOR_LINK_SPEED_PX_S if link_speed is None else float(link_speed)
    cap = max(DETECTOR_LINK_CAP_MIN, DETECTOR_LINK_CAP_FRAC * short)
    return min(max(DETECTOR_LINK_MIN_PX, speed * dt), cap)


def _min_travel(geometry: Geometry) -> float:
    return max(40.0, 0.08 * min(geometry.frame_width, geometry.frame_height))


def _static_radius(geometry: Geometry, parked_radius: float | None = None) -> float:
    if parked_radius is not None:
        return float(parked_radius)
    return max(DETECTOR_STATIC_RADIUS_MIN, DETECTOR_STATIC_RADIUS_FRAC * min(geometry.frame_width, geometry.frame_height))


def _looks_moving(
    state: _TrackState,
    xy: tuple[float, float],
    geometry: Geometry,
    elapsed_s: float,
    *,
    parked_radius: float | None = None,
    link_speed: float | None = None,
) -> bool:
    """True once observed travel leaves the parked radius.

    The anchor stays put. Jitter inside the radius never counts, and a slow
    creep still qualifies once the net move exceeds it. Identity is separate:
    callers keep the track even while this returns False.
    """
    if state.active:
        return True
    origin = state.last_observed_xy or state.xy
    step = ((xy[0] - origin[0]) ** 2 + (xy[1] - origin[1]) ** 2) ** 0.5
    dt = max(elapsed_s - (state.last_observed_s or state.last_time_s), 1e-3)
    if step > _link_radius(geometry, dt, True, link_speed):
        return False
    if state.settle_xy is None:
        state.settle_xy = state.first_xy
        state.settle_s = state.first_time_s
    travel = ((xy[0] - state.settle_xy[0]) ** 2 + (xy[1] - state.settle_xy[1]) ** 2) ** 0.5
    return travel >= _static_radius(geometry, parked_radius)


def _point_side(xy: tuple[float, float], geometry: Geometry) -> str | None:
    if geometry.north is None:
        return None
    return side_from_point(xy[0], xy[1], geometry.north, geometry.frame_width, geometry.frame_height)


def _size_ok(a: tuple[float, float], b: tuple[float, float]) -> bool:
    area_a = max(a[0] * a[1], 1.0)
    area_b = max(b[0] * b[1], 1.0)
    return max(area_a, area_b) / min(area_a, area_b) <= 4.0


def _inflate_box(xyxy: tuple[float, float, float, float], pad: float) -> tuple[float, float, float, float]:
    return (xyxy[0] - pad, xyxy[1] - pad, xyxy[2] + pad, xyxy[3] + pad)


def _near_user_occlusion(xyxy: tuple[float, float, float, float], geometry: Geometry, pad: float = 12.0) -> bool:
    if not geometry.occlusions:
        return False
    grown = _inflate_box(xyxy, pad)
    for box in geometry.occlusions:
        if box_iou(grown, (box.x1, box.y1, box.x2, box.y2)) > 0.0:
            return True
    return False


def _point_in_occlusion(xy: tuple[float, float], geometry: Geometry, pad: float = 8.0) -> bool:
    if not geometry.occlusions:
        return False
    x, y = xy
    for box in geometry.occlusions:
        item = box.normalized()
        if item.x1 - pad <= x <= item.x2 + pad and item.y1 - pad <= y <= item.y2 + pad:
            return True
    return False


def _buried_in_drop_zone(xy: tuple[float, float], geometry: Geometry) -> bool:
    """True when a detection sits well inside a drop zone, not just on the lip."""
    if not geometry.occlusions:
        return False
    x, y = xy
    for box in geometry.occlusions:
        item = box.normalized()
        inset_x = min(16.0, 0.25 * item.width())
        inset_y = min(16.0, 0.25 * item.height())
        if item.x1 + inset_x < x < item.x2 - inset_x and item.y1 + inset_y < y < item.y2 - inset_y:
            return True
    return False


def _occlusion_span(geometry: Geometry) -> float:
    if not geometry.occlusions:
        return 0.0
    return max(max(box.width(), box.height()) for box in geometry.occlusions)


def _near_drop_box(xy: tuple[float, float], box, pad: float) -> bool:
    item = box.normalized()
    return item.x1 - pad <= xy[0] <= item.x2 + pad and item.y1 - pad <= xy[1] <= item.y2 + pad


def _points_share_drop_zone(
    p0: tuple[float, float],
    p1: tuple[float, float],
    geometry: Geometry,
    pad: float = 48.0,
) -> bool:
    """True when both points sit on/near the same marked drop zone."""
    if not geometry.occlusions:
        return False
    return any(_near_drop_box(p0, box, pad) and _near_drop_box(p1, box, pad) for box in geometry.occlusions)


def _shared_drop_span(
    p0: tuple[float, float],
    p1: tuple[float, float],
    geometry: Geometry,
    pad: float = 48.0,
) -> float:
    span = 0.0
    for box in geometry.occlusions:
        if _near_drop_box(p0, box, pad) and _near_drop_box(p1, box, pad):
            span = max(span, box.width(), box.height())
    return span


def _segment_hits_user_occlusion(
    p0: tuple[float, float],
    p1: tuple[float, float],
    geometry: Geometry,
) -> bool:
    if not geometry.occlusions:
        return False
    steps = max(2, int(((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2) ** 0.5))
    for index in range(steps + 1):
        t = index / steps
        x = p0[0] + (p1[0] - p0[0]) * t
        y = p0[1] + (p1[1] - p0[1]) * t
        for box in geometry.occlusions:
            if box.x1 - 64 <= x <= box.x2 + 64 and box.y1 - 64 <= y <= box.y2 + 64:
                return True
    return False


def _observed_origin(state: _TrackState) -> tuple[float, float]:
    return state.last_observed_xy or state.xy


def _observed_box(state: _TrackState) -> tuple[float, float, float, float] | None:
    return state.last_observed_xyxy or state.last_xyxy


def _predicted_xy(state: _TrackState, elapsed_s: float) -> tuple[float, float]:
    """One step from the last real detection. Predictions are not integrated."""
    origin = _observed_origin(state)
    dt = max(elapsed_s - (state.last_observed_s or state.last_time_s), 1e-3)
    return (origin[0] + state.vel[0] * dt, origin[1] + state.vel[1] * dt)


def _hits_drop_zone(
    state: _TrackState,
    xyxy: tuple[float, float, float, float] | None,
    xy: tuple[float, float],
    geometry: Geometry,
) -> bool:
    last = state.last_xyxy
    if last is not None and _near_user_occlusion(last, geometry):
        return True
    if xyxy is not None and _near_user_occlusion(xyxy, geometry):
        return True
    if _segment_hits_user_occlusion(state.xy, xy, geometry):
        return True
    if _points_share_drop_zone(state.xy, xy, geometry):
        return True
    return _point_in_occlusion(state.xy, geometry) or _point_in_occlusion(xy, geometry)


def _gap_uses_drop_zone(
    origin: tuple[float, float],
    xy: tuple[float, float],
    geometry: Geometry,
) -> bool:
    """The gap is eligible for recovery when it actually meets a marked drop zone."""
    if not geometry.occlusions:
        return False
    return (
        _point_in_occlusion(origin, geometry, pad=48.0)
        or _point_in_occlusion(xy, geometry, pad=48.0)
        or _segment_hits_user_occlusion(origin, xy, geometry)
        or _points_share_drop_zone(origin, xy, geometry)
    )


def _match_limit(
    state: _TrackState,
    dt: float,
    geometry: Geometry,
    moving_only: bool,
    link_speed: float | None,
    origin: tuple[float, float],
    xy: tuple[float, float],
) -> float:
    """Reachable distance since the last real detection.

    Outside a drop zone the normal frame cap applies. Through a drop zone the
    cap lifts to configured speed × elapsed time, not to the zone's width.
    A settled stationary track only accepts jitter around its own box.
    """
    capped = _link_radius(geometry, dt, moving_only, link_speed)
    speed = DETECTOR_LINK_SPEED_PX_S if link_speed is None else float(link_speed)
    slack = 0.5 * max(state.bbox_wh)
    speed_disk = speed * max(dt, 1e-3) + slack
    if not state.active and state.sightings >= 2 and state.speed <= MOTION_HOLD_SPEED_PX_S:
        jitter = max(18.0, 0.45 * max(state.bbox_wh))
        return min(capped, jitter)
    if _gap_uses_drop_zone(origin, xy, geometry):
        return max(capped, speed_disk)
    return capped


def _along_cross(state: _TrackState, pred: tuple[float, float], xy: tuple[float, float]) -> tuple[float, float]:
    dx, dy = xy[0] - pred[0], xy[1] - pred[1]
    speed = (state.vel[0] ** 2 + state.vel[1] ** 2) ** 0.5
    if speed < 8.0:
        dist = (dx ** 2 + dy ** 2) ** 0.5
        return dist, dist
    ux, uy = state.vel[0] / speed, state.vel[1] / speed
    return abs(dx * ux + dy * uy), abs(-uy * dx + ux * dy)


def _drop_zone_cross_limit(
    state: _TrackState,
    dt: float,
    geometry: Geometry,
    moving_only: bool,
    link_speed: float | None,
) -> float:
    return max(
        _link_radius(geometry, dt, moving_only, link_speed),
        max(state.bbox_wh),
        0.5 * _occlusion_span(geometry),
        80.0,
    )


def _note(pending: _Pending, **payload) -> None:
    if not pending.diagnostics_enabled or len(pending.diagnostics) >= 4000:
        return
    payload.setdefault("frame", pending.frame_index)
    pending.diagnostics.append(payload)


def _association_cost(
    state: _TrackState,
    xyxy: tuple[float, float, float, float],
    elapsed_s: float,
    geometry: Geometry,
    moving_only: bool,
    link_speed: float | None,
) -> tuple[float, str] | None:
    """Cost of giving this detection to this track. None means impossible.

    Prediction is computed once from the last observation. A turn stays legal
    as long as the detection is inside the reachable disk.
    """
    xy = bbox_center(xyxy)
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    wh = (max(x2 - x1, 1.0), max(y2 - y1, 1.0))
    if not _size_ok(state.bbox_wh, wh):
        return None
    origin = _observed_origin(state)
    dt = max(elapsed_s - (state.last_observed_s or state.last_time_s), 1e-3)
    pred = (origin[0] + state.vel[0] * dt, origin[1] + state.vel[1] * dt)
    dist_last = ((xy[0] - origin[0]) ** 2 + (xy[1] - origin[1]) ** 2) ** 0.5
    dist_pred = ((xy[0] - pred[0]) ** 2 + (xy[1] - pred[1]) ** 2) ** 0.5
    limit = _match_limit(state, dt, geometry, moving_only, link_speed, origin, xy)
    if dist_last > limit:
        return None
    # A hard turn is inside the disk. A neighbor off the velocity is not,
    # unless the step itself is still small.
    if dist_pred > limit * 1.35 and dist_last > max(48.0, 0.55 * limit):
        return None
    observed = _observed_box(state)
    overlap = box_iou(observed, xyxy) if observed is not None else 0.0
    cost = dist_pred * (1.2 - 0.7 * min(overlap, 1.0))
    method = "motion"
    if overlap >= 0.15:
        method = "overlap"
        if not state.active:
            cost *= 0.2
    if _gap_uses_drop_zone(origin, xy, geometry) and dist_last > _link_radius(geometry, dt, moving_only, link_speed):
        method = "drop-zone"
    return cost, method


def _can_coast_drop_zone(state: _TrackState, elapsed_s: float, geometry: Geometry) -> bool:
    if not state.active or state.had_gate_trip:
        return False
    if elapsed_s - state.last_observed_s > MOTION_COAST_S:
        return False
    pred = _predicted_xy(state, elapsed_s)
    origin = _observed_origin(state)
    last = _observed_box(state)
    return (
        _point_in_occlusion(origin, geometry, pad=80.0)
        or _point_in_occlusion(pred, geometry, pad=80.0)
        or (last is not None and _near_user_occlusion(last, geometry, pad=64.0))
        or _segment_hits_user_occlusion(origin, pred, geometry)
        or _points_share_drop_zone(origin, pred, geometry)
    )


def _nearest_unseen(
    pending: _Pending,
    xyxy: tuple[float, float, float, float],
    wh: tuple[float, float],
    seen: set[int],
    elapsed_s: float,
    geometry: Geometry,
    moving_only: bool,
) -> _TrackState | None:
    """Lowest-cost track for one detection. Impossible pairs are left unmatched."""
    del wh
    best: _TrackState | None = None
    best_cost = float("inf")
    for state in pending.live.values():
        if state.tid in seen:
            continue
        scored = _association_cost(state, xyxy, elapsed_s, geometry, moving_only, pending.link_speed)
        if scored is None:
            continue
        cost, _method = scored
        if cost < best_cost:
            best = state
            best_cost = cost
    return best


def _new_track(
    pending: _Pending,
    xy: tuple[float, float],
    wh: tuple[float, float],
    elapsed_s: float,
) -> _TrackState:
    tid = pending.next_synth
    pending.next_synth += 1
    state = _TrackState(
        tid=tid,
        xy=xy,
        first_xy=xy,
        first_time_s=elapsed_s,
        last_time_s=elapsed_s,
        last_motion_s=elapsed_s,
        last_observed_s=elapsed_s,
        last_observed_xy=xy,
        bbox_wh=wh,
        estimate="observed",
    )
    pending.live[tid] = state
    return state


def _assign_boxes_to_states(
    pending: _Pending,
    boxes: list[tuple[float, float, float, float]],
    elapsed_s: float,
    geometry: Geometry,
    moving_only: bool,
) -> list[tuple[_TrackState, bool, str]]:
    """One-to-one assignment. Implausible pairs stay unmatched."""
    if not boxes:
        return []
    tracks = list(pending.live.values())
    methods: dict[tuple[int, int], str] = {}
    if tracks:
        from lap import lapjv

        cost = np.full((len(tracks), len(boxes)), 1.0e6, dtype=np.float64)
        for row, state in enumerate(tracks):
            for col, xyxy in enumerate(boxes):
                scored = _association_cost(
                    state, xyxy, elapsed_s, geometry, moving_only, pending.link_speed
                )
                if scored is None:
                    continue
                value, method = scored
                cost[row, col] = value
                methods[(state.tid, col)] = method
        _total, row_to_col, _col_to_row = lapjv(cost, extend_cost=True, cost_limit=1.0e5)
        chosen: dict[int, _TrackState] = {}
        for row, col in enumerate(row_to_col):
            if col < 0 or col >= len(boxes) or cost[row, col] >= 1.0e5:
                continue
            chosen[int(col)] = tracks[row]
    else:
        chosen = {}

    assigned: list[tuple[_TrackState, bool, str]] = []
    for index, xyxy in enumerate(boxes):
        xy = bbox_center(xyxy)
        x1, y1, x2, y2 = (float(v) for v in xyxy)
        wh = (max(x2 - x1, 1.0), max(y2 - y1, 1.0))
        state = chosen.get(index)
        if state is None:
            created = _new_track(pending, xy, wh, elapsed_s)
            assigned.append((created, True, "new"))
            continue
        assigned.append((state, False, methods.get((state.tid, index), "motion")))
    return assigned


def _resolve_state(
    pending: _Pending,
    xy: tuple[float, float],
    wh: tuple[float, float],
    elapsed_s: float,
    seen: set[int],
    geometry: Geometry,
    moving_only: bool,
) -> tuple[_TrackState, bool]:
    half_w, half_h = wh[0] / 2.0, wh[1] / 2.0
    xyxy = (xy[0] - half_w, xy[1] - half_h, xy[0] + half_w, xy[1] + half_h)
    adopted = _nearest_unseen(pending, xyxy, wh, seen, elapsed_s, geometry, moving_only)
    if adopted is not None:
        return adopted, False
    return _new_track(pending, xy, wh, elapsed_s), True


def _apply_gate_hits(
    state: _TrackState,
    prev: tuple[float, float],
    xy: tuple[float, float],
    xyxy: tuple[float, float, float, float],
    gates: list[tuple[str, object]],
    pad: float,
    trips: list[Trip],
    video_name: str,
    elapsed_s: float,
    *,
    evidence: str = "observed",
    pending: _Pending | None = None,
) -> tuple[int, int]:
    """Confirm a crossing only between two real observations that change sides."""
    del xyxy
    hits: list[tuple[float, str]] = []
    seen_origins: set[str] = set()
    for origin, line in gates:
        t = confirmed_cross_t(prev, xy, line, pad_px=pad, clearance_px=8.0)
        legacy = path_cross_t(prev, xy, line, pad_px=pad)
        if pending is not None and legacy is not None:
            _dbg_bump(pending, "legacy_path_hit")
        if pending is not None and t is not None:
            _dbg_bump(pending, "confirmed_hit")
        elif pending is not None and legacy is not None:
            before = signed_offset(prev, line)
            after = signed_offset(xy, line)
            reason = "clearance" if before * after <= 0.0 and abs(before) < 8.0 and abs(after) < 8.0 else "pad_or_finite"
            _dbg_bump(pending, f"confirmed_miss_{reason}")
            if pending.dbg_counts.get(f"confirmed_miss_{reason}", 0) <= 8:
                _dbg(
                    "H2",
                    "processor.py:_apply_gate_hits",
                    "legacy path hit rejected by confirmed_cross_t",
                    {
                        "tid": state.tid,
                        "origin": origin,
                        "reason": reason,
                        "before": round(before, 2),
                        "after": round(after, 2),
                        "prev": [round(prev[0], 1), round(prev[1], 1)],
                        "xy": [round(xy[0], 1), round(xy[1], 1)],
                        "had_start": state.start[0] if state.start else None,
                    },
                )
        if t is None or origin in seen_origins:
            continue
        seen_origins.add(origin)
        hits.append((t, origin))
    hits.sort()
    start_hits = 0
    end_hits = 0
    if not hits:
        return 0, 0
    for _, origin in hits:
        if state.start is None:
            if origin == state.last_side:
                continue
            state.start = (origin, elapsed_s)
            state.start_evidence = evidence
            state.last_side = origin
            start_hits += 1
        elif state.start[0] != origin:
            start_origin, start_time_s = state.start
            trip_evidence = "gap" if "gap" in (state.start_evidence, evidence) else "observed"
            trips.append(
                _make_trip(
                    state,
                    video_name,
                    start_origin,
                    start_time_s,
                    origin,
                    elapsed_s,
                    evidence=trip_evidence,
                )
            )
            state.start = None
            state.had_gate_trip = True
            state.last_side = origin
            end_hits += 1
            _release_track_hold(state)
            break
    return start_hits, end_hits


def _release_track_hold(state: _TrackState) -> None:
    """After a full trip, stop freezing MOG2 and drop the long red-light hold."""
    state.holding = False
    state.last_xyxy = None


def _finalize_track(
    state: _TrackState,
    geometry: Geometry,
    trips: list[Trip],
    video_name: str,
    *,
    infer_path: bool = True,
) -> int:
    """Finish from real observations only. A predicted position cannot complete a trip."""
    end_xy = state.last_observed_xy or state.first_xy
    end_t = state.last_observed_s or state.first_time_s
    if state.start is not None:
        last_side = _point_side(end_xy, geometry)
        if last_side and last_side != state.start[0]:
            start_origin, start_time_s = state.start
            evidence = "gap" if state.start_evidence == "gap" else "observed"
            trips.append(_make_trip(state, video_name, start_origin, start_time_s, last_side, end_t, evidence=evidence))
            state.start = None
            state.had_gate_trip = True
            return 1
        return 0
    if not infer_path or state.had_gate_trip or not state.active:
        return 0
    first_side = _point_side(state.first_xy, geometry)
    last_side = _point_side(end_xy, geometry)
    travel = ((end_xy[0] - state.first_xy[0]) ** 2 + (end_xy[1] - state.first_xy[1]) ** 2) ** 0.5
    if first_side and last_side and first_side != last_side and travel >= _min_travel(geometry):
        trips.append(
            _make_trip(state, video_name, first_side, state.first_time_s, last_side, end_t, evidence="gap")
        )
        state.had_gate_trip = True
        return 1
    return 0


def _hold_limit(state: _TrackState) -> float:
    if not state.active:
        return DETECTOR_STATIC_EXPIRE_S
    if state.had_gate_trip:
        return MOTION_COAST_S
    if state.sightings < 2:
        return MOTION_COAST_S
    if state.holding or state.speed <= MOTION_HOLD_SPEED_PX_S:
        return MOTION_HOLD_S
    return MOTION_COAST_S


def _can_hold(state: _TrackState, now_s: float) -> bool:
    if not state.active:
        return False
    if state.had_gate_trip:
        return False
    if _observed_box(state) is None:
        return False
    if now_s - state.last_observed_s > MOTION_HOLD_S:
        return False
    if state.sightings < 2 and (now_s - state.first_time_s) < 0.35:
        return False
    return state.holding or state.speed <= MOTION_HOLD_SPEED_PX_S


def _expire_tracks(
    pending: _Pending,
    geometry: Geometry,
    trips: list[Trip],
    video_name: str,
    now_s: float,
    *,
    coast_s: float | None = None,
    infer_path: bool = True,
) -> int:
    keep: dict[int, _TrackState] = {}
    inferred = 0
    for tid, state in pending.live.items():
        limit = 0.0 if coast_s is not None else _hold_limit(state)
        if coast_s is not None:
            limit = coast_s
        age = now_s - state.last_observed_s
        if age > limit:
            had_start = state.start[0] if state.start else None
            was_active = state.active
            added = _finalize_track(state, geometry, trips, video_name, infer_path=infer_path)
            inferred += added
            if added == 0:
                if had_start:
                    _dbg_bump(pending, "finalize_dropped_start")
                elif was_active:
                    _dbg_bump(pending, "finalize_dropped_active")
            else:
                _dbg_bump(pending, "finalize_trip")
            _keep_path(pending, state, geometry)
            _note(
                pending,
                time_s=now_s,
                tid=state.tid,
                cls=state.cls,
                conf=state.confidence,
                status=state.estimate,
                age_s=age,
                action="expired",
                match="",
                crossing="discarded" if state.start is None else "confirmed",
                reason="unobserved",
            )
        else:
            keep[tid] = state
    pending.live = keep
    return inferred


def _record_crossings(
    boxes: list[tuple[float, float, float, float]],
    geometry: Geometry,
    pending: _Pending,
    trips: list[Trip],
    video_name: str,
    elapsed_s: float,
    *,
    moving_only: bool = False,
    classes: list[str] | None = None,
    confidences: list[float] | None = None,
) -> tuple[int, int]:
    gates = [(_gate_origin(item, geometry), item.line) for item in (*geometry.starts, *geometry.ends)]
    pad = max(32.0, 0.03 * max(geometry.frame_width, geometry.frame_height))
    start_hits = 0
    end_hits = 0
    seen: set[int] = set()
    track_boxes: list[tuple[float, float, float, float]] = []
    track_classes: list[str | None] = []
    track_confs: list[float] = []
    for index, xyxy in enumerate(boxes):
        center = bbox_center(xyxy)
        if not point_in_roadway(center[0], center[1], geometry):
            _dbg_bump(pending, "roadway_reject")
            _note(
                pending,
                time_s=elapsed_s,
                tid=0,
                cls=(classes[index] if classes and index < len(classes) else "vehicle"),
                conf=(confidences[index] if confidences and index < len(confidences) else 0.0),
                status="observed",
                age_s=0.0,
                action="rejected",
                match="roadway",
                crossing="rejected",
                reason="outside roadway",
            )
            continue
        track_boxes.append(xyxy)
        track_classes.append(classes[index] if classes and index < len(classes) else None)
        track_confs.append(confidences[index] if confidences and index < len(confidences) else 0.0)
    box_states = _assign_boxes_to_states(pending, track_boxes, elapsed_s, geometry, moving_only)

    for index, xyxy in enumerate(track_boxes):
        xy = bbox_center(xyxy)
        x1, y1, x2, y2 = (float(v) for v in xyxy)
        wh = (max(x2 - x1, 1.0), max(y2 - y1, 1.0))
        state, is_new, method = box_states[index]
        cls = track_classes[index] or "vehicle"
        if is_new or state.cls == "vehicle":
            state.cls = cls
        prev_xy = state.last_observed_xy or state.xy
        prev_t = state.last_observed_s or state.first_time_s
        gap = (not is_new) and (elapsed_s - prev_t) > 0.25
        evidence = "gap" if gap or method == "drop-zone" else "observed"
        seen.add(state.tid)
        dt = max(elapsed_s - prev_t, 1e-3)
        step_x, step_y = xy[0] - prev_xy[0], xy[1] - prev_xy[1]
        inst = 0.0 if is_new else (step_x ** 2 + step_y ** 2) ** 0.5 / dt
        if not is_new:
            state.vel = (0.45 * (step_x / dt) + 0.55 * state.vel[0], 0.45 * (step_y / dt) + 0.55 * state.vel[1])
            state.speed = 0.45 * inst + 0.55 * state.speed
            state.sightings += 1
        else:
            state.speed = 0.0
        state.bbox_wh = wh
        state.holding = False
        state.estimate = "observed"
        state.confidence = track_confs[index]
        moving = (not moving_only) or _looks_moving(
            state,
            xy,
            geometry,
            elapsed_s,
            parked_radius=pending.parked_radius,
            link_speed=pending.link_speed,
        )
        state.last_observed_xy = xy
        state.last_observed_xyxy = (x1, y1, x2, y2)
        state.last_observed_s = elapsed_s
        state.last_xyxy = (x1, y1, x2, y2)
        state.xy = xy
        state.last_time_s = elapsed_s
        action = "created" if is_new else ("recovered" if gap else "matched")
        if not moving:
            for origin, line in gates:
                if path_cross_t(prev_xy, xy, line, pad_px=pad) is not None:
                    _dbg_bump(pending, "skipped_not_moving_cross")
                    if pending.dbg_counts.get("skipped_not_moving_cross", 0) <= 8:
                        _dbg(
                            "H1",
                            "processor.py:_record_crossings",
                            "gate step skipped because track not yet moving",
                            {
                                "tid": state.tid,
                                "origin": origin,
                                "travel": round(((xy[0] - (state.settle_xy or state.first_xy)[0]) ** 2 + (xy[1] - (state.settle_xy or state.first_xy)[1]) ** 2) ** 0.5, 1),
                                "sightings": state.sightings,
                            },
                        )
                    break
            _note(
                pending,
                time_s=elapsed_s,
                tid=state.tid,
                cls=state.cls,
                conf=state.confidence,
                status="observed",
                age_s=0.0,
                action=action,
                match=method,
                crossing="none",
                reason="stationary",
            )
            continue
        state.active = True
        state.last_motion_s = elapsed_s
        _append_path(state, xy, geometry)
        if is_new:
            added_s, added_e = 0, 0
            for origin, line in gates:
                if path_cross_t(prev_xy, xy, line, pad_px=pad) is not None:
                    _dbg_bump(pending, "skipped_new_cross")
                    break
        else:
            added_s, added_e = _apply_gate_hits(
                state,
                prev_xy,
                xy,
                (x1, y1, x2, y2),
                gates,
                pad,
                trips,
                video_name,
                elapsed_s,
                evidence=evidence,
                pending=pending,
            )
        start_hits += added_s
        end_hits += added_e
        crossing = "confirmed" if added_s or added_e else "none"
        _note(
            pending,
            time_s=elapsed_s,
            tid=state.tid,
            cls=state.cls,
            conf=state.confidence,
            status="observed",
            age_s=0.0,
            action="completed" if added_e else action,
            match=method,
            crossing=crossing,
            reason="",
        )
        if added_e > 0:
            _keep_path(pending, state, geometry)
            pending.live.pop(state.tid, None)
            continue

    for state in list(pending.live.values()):
        if state.tid in seen:
            continue
        age = elapsed_s - state.last_observed_s
        if _can_coast_drop_zone(state, elapsed_s, geometry):
            pred = _predicted_xy(state, elapsed_s)
            half_w, half_h = state.bbox_wh[0] / 2.0, state.bbox_wh[1] / 2.0
            state.xy = pred
            state.last_time_s = elapsed_s
            state.estimate = "predicted"
            state.last_xyxy = (pred[0] - half_w, pred[1] - half_h, pred[0] + half_w, pred[1] + half_h)
            _append_path(state, pred, geometry)
            _note(
                pending,
                time_s=elapsed_s,
                tid=state.tid,
                cls=state.cls,
                conf=state.confidence,
                status="predicted",
                age_s=age,
                action="predicted",
                match="drop-zone",
                crossing="rejected",
                reason="prediction is not a count",
            )
            continue
        if not _can_hold(state, elapsed_s) or _observed_box(state) is None:
            continue
        state.holding = True
        state.estimate = "held"
        state.speed = min(state.speed, MOTION_HOLD_SPEED_PX_S)
        state.xy = _observed_origin(state)
        state.last_xyxy = _observed_box(state)
        _note(
            pending,
            time_s=elapsed_s,
            tid=state.tid,
            cls=state.cls,
            conf=state.confidence,
            status="held",
            age_s=age,
            action="held",
            match="",
            crossing="rejected",
            reason="held box is not a count",
        )
    return start_hits, end_hits
