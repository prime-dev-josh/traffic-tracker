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
from traffic_tracker.core.crossings import bbox_center, box_hits_line, box_iou, path_cross_t
from traffic_tracker.core.geometry import (
    ORIGIN_TO_BOUND,
    ORIGINS,
    Geometry,
    NamedLine,
    name_from_north,
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


@dataclass
class _Pending:
    """Centroid tracks. A nearby new blob continues the same vehicle."""

    live: dict[int, _TrackState] = field(default_factory=dict)
    next_synth: int = 1
    finished_paths: list[tuple[str, list[tuple[float, float]]]] = field(default_factory=list)
    parked_radius: float | None = None
    link_speed: float | None = None
    path_jump: float | None = None


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

            if frame_idx == 1 or frame_idx == trace_at:
                trace_frame = frame.copy()

            if use_detector or elapsed_s >= MOTION_WARMUP_S:
                starts, ends = _record_crossings(
                    boxes,
                    fitted,
                    pending,
                    trips,
                    info.name,
                    elapsed_s,
                    moving_only=use_detector,
                    classes=classes,
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


PATH_COLORS = {
    "car": (220, 20, 255),
    "van": (255, 220, 0),
    "truck": (0, 140, 255),
    "motorcycle": (0, 255, 80),
    "bus": (0, 255, 255),
    "vehicle": (255, 255, 255),
}


def _path_color(cls: str) -> tuple[int, int, int]:
    return PATH_COLORS.get(cls, PATH_COLORS["vehicle"])


def _append_path(state: _TrackState, xy: tuple[float, float]) -> None:
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
) -> list[list[tuple[float, float]]]:
    """Split a polyline whenever consecutive points jump farther than max_jump."""
    if len(points) < 2:
        return []
    chunks: list[list[tuple[float, float]]] = []
    current = [points[0]]
    for point in points[1:]:
        last = current[-1]
        step = ((point[0] - last[0]) ** 2 + (point[1] - last[1]) ** 2) ** 0.5
        if step > max_jump:
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
    for chunk in _path_chunks(state.path, _path_jump_limit(geometry, pending.path_jump)):
        pending.finished_paths.append((state.cls, chunk))


def _make_trip(state: _TrackState, video_name: str, start_origin: str, start_time_s: float, end_origin: str, end_time_s: float) -> Trip:
    return Trip(
        video_name=video_name,
        track_id=state.tid,
        start_destination=start_origin,
        start_time_s=start_time_s,
        end_destination=ORIGIN_TO_BOUND.get(end_origin, end_origin),
        end_time_s=end_time_s,
        vehicle_class=state.cls or "vehicle",
        path=list(state.path),
    )


def _halo_polyline(frame: np.ndarray, points: list[tuple[float, float]], color: tuple[int, int, int], width: int = 3) -> None:
    pts = np.array([(int(round(x)), int(round(y))) for x, y in points], dtype=np.int32)
    if len(pts) < 2:
        return
    cv2.polylines(frame, [pts], False, (0, 0, 0), width + 3, cv2.LINE_AA)
    cv2.polylines(frame, [pts], False, color, width, cv2.LINE_AA)


def _draw_paths(frame: np.ndarray, pending: _Pending, trips: list[Trip] | None = None) -> None:
    del trips
    traces: list[tuple[str, list[tuple[float, float]]]] = list(pending.finished_paths)
    traces.extend(
        (state.cls, state.path)
        for state in pending.live.values()
        if state.active and len(state.path) >= 2
    )
    jump = _path_jump_limit(path_jump=pending.path_jump)
    for cls, points in traces:
        color = _path_color(cls)
        for chunk in _path_chunks(points, jump):
            _halo_polyline(frame, chunk, color, 3)


def _draw_path_overlay(
    frame: np.ndarray,
    trips: list[Trip],
    pending: _Pending,
    geometry: Geometry,
) -> np.ndarray:
    """Darken the still so neon traces stay readable on pale drone pavement."""
    preview = (frame.astype(np.float32) * 0.72).astype(np.uint8)
    _draw_paths(preview, pending, trips)
    y = 28
    for cls, color in PATH_COLORS.items():
        if cls == "vehicle":
            continue
        _halo_text(preview, cls.title(), (16, y), color, 0.55)
        y += 22
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
        _halo_rect(preview, p1, p2, (255, 200, 40))
        _halo_text(preview, f"#{state.tid} hold", (p1[0], max(16, p1[1] - 6)), (255, 200, 40), 0.45)
    _draw_paths(preview, pending, trips)
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
    """True once a trained-mode box leaves its parked/boat settle radius.

    The settle point creeps toward the current centroid so hover-drone drift
    does not eventually look like a trip.
    """
    if state.active:
        return True
    step = ((xy[0] - state.xy[0]) ** 2 + (xy[1] - state.xy[1]) ** 2) ** 0.5
    dt = max(elapsed_s - state.last_time_s, 1e-3)
    if step > _link_radius(geometry, dt, True, link_speed):
        return False
    if state.settle_xy is None:
        state.settle_xy = state.first_xy
        state.settle_s = state.first_time_s
    travel = ((xy[0] - state.settle_xy[0]) ** 2 + (xy[1] - state.settle_xy[1]) ** 2) ** 0.5
    if travel < _static_radius(geometry, parked_radius):
        if elapsed_s - state.settle_s >= DETECTOR_STATIC_SETTLE_S:
            state.settle_xy = xy
            state.settle_s = elapsed_s
        return False
    return True


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
            if box.x1 - 8 <= x <= box.x2 + 8 and box.y1 - 8 <= y <= box.y2 + 8:
                return True
    return False


def _nearest_unseen(
    pending: _Pending,
    xyxy: tuple[float, float, float, float],
    wh: tuple[float, float],
    seen: set[int],
    elapsed_s: float,
    geometry: Geometry,
    moving_only: bool,
) -> _TrackState | None:
    """Match a leftover box to the closest predicted centroid inside the link radius."""
    xy = bbox_center(xyxy)
    best: _TrackState | None = None
    best_d = float("inf")
    for state in pending.live.values():
        if state.tid in seen or not _size_ok(state.bbox_wh, wh):
            continue
        dt = max(elapsed_s - state.last_time_s, 1e-3)
        pred = (state.xy[0] + state.vel[0] * dt, state.xy[1] + state.vel[1] * dt)
        dist = ((pred[0] - xy[0]) ** 2 + (pred[1] - xy[1]) ** 2) ** 0.5
        max_dist = _link_radius(geometry, dt, moving_only, pending.link_speed)
        last = state.last_xyxy
        if (
            (last is not None and _near_user_occlusion(last, geometry))
            or _near_user_occlusion(xyxy, geometry)
            or _segment_hits_user_occlusion(state.xy, xy, geometry)
        ):
            max_dist += 48.0
        if dist <= max_dist and dist < best_d:
            best = state
            best_d = dist
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
        bbox_wh=wh,
    )
    pending.live[tid] = state
    return state


def _assign_boxes_to_states(
    pending: _Pending,
    boxes: list[tuple[float, float, float, float]],
    elapsed_s: float,
    geometry: Geometry,
    moving_only: bool,
) -> list[tuple[_TrackState, bool]]:
    """Match detections to tracks by box overlap first, then predicted centroid."""
    if not boxes:
        return []
    assigned: list[tuple[_TrackState, bool] | None] = [None] * len(boxes)
    used_tracks: set[int] = set()

    pairs: list[tuple[float, int, int]] = []
    for index, xyxy in enumerate(boxes):
        for tid, state in pending.live.items():
            if state.last_xyxy is None:
                continue
            overlap = box_iou(xyxy, state.last_xyxy)
            if overlap >= DETECTOR_MATCH_IOU_MIN:
                pairs.append((overlap, index, tid))
    pairs.sort(key=lambda item: item[0], reverse=True)
    for overlap, index, tid in pairs:
        if assigned[index] is not None or tid in used_tracks:
            continue
        assigned[index] = (pending.live[tid], False)
        used_tracks.add(tid)

    for index, xyxy in enumerate(boxes):
        if assigned[index] is not None:
            continue
        xy = bbox_center(xyxy)
        x1, y1, x2, y2 = (float(v) for v in xyxy)
        wh = (max(x2 - x1, 1.0), max(y2 - y1, 1.0))
        adopted = _nearest_unseen(pending, (x1, y1, x2, y2), wh, used_tracks, elapsed_s, geometry, moving_only)
        if adopted is not None:
            assigned[index] = (adopted, False)
            used_tracks.add(adopted.tid)
        else:
            assigned[index] = (_new_track(pending, xy, wh, elapsed_s), True)
            used_tracks.add(assigned[index][0].tid)

    return [item for item in assigned if item is not None]


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
) -> tuple[int, int]:
    hits: list[tuple[float, str]] = []
    seen_origins: set[str] = set()
    for origin, line in gates:
        t = path_cross_t(prev, xy, line, pad_px=pad)
        if t is None and box_hits_line(xyxy, line, pad_px=pad):
            t = 0.0 if prev == xy else 1.0
        if t is None or origin in seen_origins:
            continue
        seen_origins.add(origin)
        hits.append((t, origin))
    hits.sort()
    start_hits = 0
    end_hits = 0
    if not hits:
        state.last_side = None
        return 0, 0
    for _, origin in hits:
        if state.start is None:
            if origin == state.last_side:
                continue
            state.start = (origin, elapsed_s)
            state.last_side = origin
            start_hits += 1
        elif state.start[0] != origin:
            start_origin, start_time_s = state.start
            trips.append(_make_trip(state, video_name, start_origin, start_time_s, origin, elapsed_s))
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
    """Finish a dead track from last side or first→last path when gates were missed."""
    if state.start is not None:
        last_side = _point_side(state.xy, geometry)
        if last_side and last_side != state.start[0]:
            start_origin, start_time_s = state.start
            trips.append(_make_trip(state, video_name, start_origin, start_time_s, last_side, state.last_time_s))
            state.start = None
            state.had_gate_trip = True
            return 1
        return 0
    if not infer_path or state.had_gate_trip:
        return 0
    first_side = _point_side(state.first_xy, geometry)
    last_side = _point_side(state.xy, geometry)
    travel = ((state.xy[0] - state.first_xy[0]) ** 2 + (state.xy[1] - state.first_xy[1]) ** 2) ** 0.5
    if first_side and last_side and first_side != last_side and travel >= _min_travel(geometry):
        trips.append(_make_trip(state, video_name, first_side, state.first_time_s, last_side, state.last_time_s))
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
    if state.last_xyxy is None:
        return False
    if now_s - state.last_motion_s > MOTION_HOLD_S:
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
        age = now_s - state.last_motion_s
        if age > limit:
            inferred += _finalize_track(state, geometry, trips, video_name, infer_path=infer_path)
            _keep_path(pending, state, geometry)
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
) -> tuple[int, int]:
    gates = [(_gate_origin(item, geometry), item.line) for item in (*geometry.starts, *geometry.ends)]
    pad = max(32.0, 0.03 * max(geometry.frame_width, geometry.frame_height))
    start_hits = 0
    end_hits = 0
    seen: set[int] = set()
    box_states = _assign_boxes_to_states(pending, boxes, elapsed_s, geometry, moving_only)

    for index, xyxy in enumerate(boxes):
        xy = bbox_center(xyxy)
        x1, y1, x2, y2 = (float(v) for v in xyxy)
        wh = (max(x2 - x1, 1.0), max(y2 - y1, 1.0))
        state, is_new = box_states[index]
        cls = "vehicle"
        if classes and index < len(classes):
            cls = classes[index] or "vehicle"
        if is_new or state.cls == "vehicle":
            state.cls = cls
        seen.add(state.tid)
        dt = max(elapsed_s - state.last_time_s, 1e-3)
        step_x, step_y = xy[0] - state.xy[0], xy[1] - state.xy[1]
        inst = (step_x ** 2 + step_y ** 2) ** 0.5 / dt
        if not is_new:
            state.vel = (0.45 * (step_x / dt) + 0.55 * state.vel[0], 0.45 * (step_y / dt) + 0.55 * state.vel[1])
        state.speed = inst if is_new else (0.45 * inst + 0.55 * state.speed)
        state.bbox_wh = wh
        state.last_xyxy = (x1, y1, x2, y2)
        state.sightings += 0 if is_new else 1
        state.holding = False
        if not moving_only:
            state.active = True
        elif not _looks_moving(
            state,
            xy,
            geometry,
            elapsed_s,
            parked_radius=pending.parked_radius,
            link_speed=pending.link_speed,
        ):
            state.xy = xy
            state.last_time_s = elapsed_s
            continue
        else:
            state.active = True
        _append_path(state, xy)
        added_s, added_e = _apply_gate_hits(
            state, state.xy, xy, (x1, y1, x2, y2), gates, pad, trips, video_name, elapsed_s
        )
        start_hits += added_s
        end_hits += added_e
        if added_e > 0:
            _keep_path(pending, state, geometry)
            pending.live.pop(state.tid, None)
            continue
        state.xy = xy
        state.last_time_s = elapsed_s
        state.last_motion_s = elapsed_s

    for state in list(pending.live.values()):
        if state.tid in seen or not _can_hold(state, elapsed_s) or state.last_xyxy is None:
            continue
        state.holding = True
        state.speed = min(state.speed, MOTION_HOLD_SPEED_PX_S)
        x1, y1, x2, y2 = state.last_xyxy
        if moving_only and not state.active:
            continue
        added_s, added_e = _apply_gate_hits(
            state, state.xy, state.xy, (x1, y1, x2, y2), gates, pad, trips, video_name, elapsed_s
        )
        start_hits += added_s
        end_hits += added_e
        if added_e > 0:
            _keep_path(pending, state, geometry)
            pending.live.pop(state.tid, None)
            continue
        seen.add(state.tid)
    return start_hits, end_hits
