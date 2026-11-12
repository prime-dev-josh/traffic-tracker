"""Motion-mask or trained boxes, record start/end crossings, and keep path traces."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np

from traffic_tracker.config import (
    DETECTOR_CONF,
    DETECTOR_IMGSZ,
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
    DETECTOR_STRIDE,
    MOTION_COAST_S,
    MOTION_HOLD_S,
    MOTION_HOLD_SPEED_PX_S,
    MOTION_WARMUP_S,
    PATH_SMOOTH_ALPHA,
    PREDICTION_REACH_PX,
    PREVIEW_EVERY,
)
from traffic_tracker.core.crossings import (
    bbox_center,
    box_cross_t,
    box_iou,
    box_overlaps_line,
    trace_cross_t,
)
from traffic_tracker.core.geometry import (
    ORIGIN_TO_BOUND,
    ORIGINS,
    Geometry,
    NamedLine,
    clamp_prediction_reach,
    gate_origin_name,
    name_from_north,
    point_in_roadway,
    side_from_point,
)
from traffic_tracker.core.detector import (
    _same_vehicle_stack,
    boxes_from_detector,
    load_detector,
    reset_detector_cache,
    resolve_detector_weights,
    suppress_duplicate_boxes,
)
from traffic_tracker.core.device import detect_device, inference_device_label
from traffic_tracker.core.motion import (
    blobs_from_frame,
    create_subtractor,
    default_area_limits,
    process_scale,
)
from traffic_tracker.core.report import Trip
from traffic_tracker.core.video import VideoInfo, clamp_clip_window, probe_video
from traffic_tracker.gui.style import path_color_bgr


ProgressCb = Callable[[int, int, str], None]
PreviewCb = Callable[[np.ndarray], None]


ISSUE_LABELS = {
    "duplicate_stack": "Stacked boxes",
    "long_trace": "Trace jump",
    "drop_zone_attach": "Drop-zone attach",
}
ISSUE_COLORS = {
    "duplicate_stack": (220, 20, 255),
    "long_trace": (0, 140, 255),
    "drop_zone_attach": (0, 200, 255),
}


@dataclass
class TrackIssue:
    kind: str
    time_s: float
    tid: int
    message: str
    xy: tuple[float, float] | None = None
    from_xy: tuple[float, float] | None = None
    xyxy: tuple[float, float, float, float] | None = None


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
    route_images: dict[str, np.ndarray] = field(default_factory=dict)
    issues: list[TrackIssue] = field(default_factory=list)
    export_path: Path | None = None


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
    entry_origin: str | None = None
    exit_bound: str | None = None
    bbox_wh: tuple[float, float] = (20.0, 20.0)
    last_xyxy: tuple[float, float, float, float] | None = None
    speed: float = 0.0
    sightings: int = 1
    holding: bool = False
    had_gate_trip: bool = False
    last_side: str | None = None
    counted_gates: set[str] = field(default_factory=set)
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
    lead_xy: tuple[float, float] | None = None
    pickup_xy: tuple[float, float] | None = None
    smooth_xy: tuple[float, float] | None = None


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
    trace_color: tuple[int, int, int] = field(default_factory=lambda: path_color_bgr("dark"))
    issues: list[TrackIssue] = field(default_factory=list)


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


def should_run_detector(frame_idx: int, stride: int) -> bool:
    """True on frame 1 and every Nth frame after that (1-based index)."""
    step = max(1, int(stride))
    return (max(int(frame_idx), 1) - 1) % step == 0


def preview_interval(preview_every: int | Callable[[], int]) -> int:
    """Resolve a live preview interval. 0 means off. A callable is read each frame."""
    value = preview_every() if callable(preview_every) else preview_every
    try:
        return int(value)
    except (TypeError, ValueError):
        return PREVIEW_EVERY


def should_emit_preview(frame_idx: int, preview_every: int | Callable[[], int]) -> bool:
    """False when preview_every is 0 (live preview off)."""
    every = preview_interval(preview_every)
    if every <= 0:
        return False
    return frame_idx == 1 or frame_idx % every == 0


def process_video(
    video_path: str,
    geometry: Geometry,
    *,
    device: str | None = None,
    cancel_flag: Callable[[], bool] | None = None,
    progress_cb: ProgressCb | None = None,
    preview_cb: PreviewCb | None = None,
    preview_every: int | Callable[[], int] = PREVIEW_EVERY,
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
    detector_imgsz: int = DETECTOR_IMGSZ,
    detect_stride: int = DETECTOR_STRIDE,
    theme: str = "dark",
    infer_trips_from_traces: bool | None = None,
    start_s: float = 0.0,
    end_s: float | None = None,
) -> ProcessResult:
    """Process one video. Moving blobs or YOLO boxes in the size band are vehicles."""
    del overhead_drone
    info = probe_video(video_path)
    start_s, end_s = clamp_clip_window(start_s, end_s, info.duration_s)
    window_s = max(0.0, end_s - start_s)
    window_frames = max(1, int(round(window_s * info.fps)) if info.fps else 1)
    fitted = geometry.scaled_to(info.width, info.height)
    suggested_min, suggested_max = default_area_limits(info.width, info.height)
    ref_limits = fitted.vehicle_area_limits()
    if ref_limits is not None:
        min_area, max_area = ref_limits
    else:
        min_area = suggested_min if min_area is None else int(min_area)
        max_area = suggested_max if max_area is None else int(max_area)
    scale = process_scale(info.width, info.height)
    use_detector = detector_mode == "trained"
    use_trace = (not use_detector) if infer_trips_from_traces is None else bool(infer_trips_from_traces)
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
        infer_device = device or detect_device()
        device_label = f"trained detector ({weights.name}) on {inference_device_label(infer_device)}"
        model_label = device_label
    else:
        infer_device = None
        device_label = "motion mask (CPU)"
        model_label = "motion mask"
    stride = max(1, int(detect_stride or DETECTOR_STRIDE))
    imgsz = int(detector_imgsz or DETECTOR_IMGSZ)

    if progress_cb:
        progress_cb(
            0,
            window_frames,
            f"{device_label}  •  box {min_area}–{max_area} px²"
            + (f"  •  {start_s:.1f}–{end_s:.1f}s" if window_s < info.duration_s - 0.05 else ""),
        )

    pending = _Pending(
        parked_radius=parked_radius,
        link_speed=link_speed,
        path_jump=path_jump,
        diagnostics_enabled=diagnostics,
        trace_color=path_color_bgr(theme),
    )
    trips: list[Trip] = []
    start_crossings = 0
    end_crossings = 0
    frames_with_tracks = 0
    subtractor = None if use_detector else create_subtractor()
    prev_gray = None
    trace_frame: np.ndarray | None = None
    start_frame = int(round(start_s * info.fps)) if info.fps else 0
    trace_at = start_frame + max(1, int(window_frames * 0.4))

    cap = cv2.VideoCapture(str(info.path))
    if not cap.isOpened():
        return ProcessResult(
            info, [], device_label, fitted, error=f"Could not open {info.path}"
        )

    frame_idx = start_frame
    if start_s > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000.0)
        pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        if pos and pos > 0:
            frame_idx = int(pos)
    try:
        while True:
            if cancel_flag and cancel_flag():
                _shift_result_times(trips, pending.issues, start_s)
                return ProcessResult(
                    _windowed_info(info, start_s, end_s, frame_idx - start_frame),
                    trips,
                    device_label,
                    fitted,
                    error="Cancelled",
                    start_crossings=start_crossings,
                    end_crossings=end_crossings,
                    frames_with_tracks=frames_with_tracks,
                    issues=list(pending.issues),
                )

            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            elapsed_s = frame_idx / info.fps
            if elapsed_s < start_s - 1e-3:
                continue
            if elapsed_s > end_s + 1e-3:
                break
            if use_detector and detector is not None:
                if should_run_detector(frame_idx, stride):
                    detections, mask = boxes_from_detector(
                        detector,
                        frame,
                        min_area=min_area,
                        max_area=max_area,
                        conf=detector_conf,
                        iou=detector_iou,
                        imgsz=imgsz,
                        device=infer_device,
                    )
                    boxes = [item[0] for item in detections]
                    classes = [item[1] for item in detections]
                    confs = [float(item[2]) for item in detections]
                else:
                    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    boxes, classes, confs = [], [], []
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

            if frame_idx == start_frame + 1 or frame_idx == trace_at:
                trace_frame = frame.copy()

            if use_detector or (elapsed_s - start_s) >= MOTION_WARMUP_S:
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
                    infer_trips_from_traces=use_trace,
                )
                start_crossings += starts
                end_crossings += ends
                if boxes or pending.live:
                    frames_with_tracks += 1
                end_crossings += _expire_tracks(
                    pending, fitted, trips, info.name, elapsed_s, infer_path=use_trace
                )

            done = max(1, frame_idx - start_frame)
            if preview_cb and should_emit_preview(done, preview_every):
                preview = _draw_motion_preview(
                    frame, mask, boxes, pending, fitted, trips, active_only=use_detector
                )
                preview_cb(preview)

            if progress_cb and (done == 1 or frame_idx % 10 == 0):
                progress_cb(
                    min(done, window_frames),
                    window_frames,
                    f"{info.name}  •  {done}/{window_frames}  •  {len(trips)} trips",
                )
    finally:
        cap.release()

    last_s = min(end_s, frame_idx / max(info.fps, 1e-6))
    end_crossings += _expire_tracks(
        pending,
        fitted,
        trips,
        info.name,
        last_s + MOTION_COAST_S + 1.0,
        coast_s=0.0,
        infer_path=use_trace,
    )
    _shift_result_times(trips, pending.issues, start_s)
    result_info = _windowed_info(info, start_s, end_s, max(0, frame_idx - start_frame))
    if progress_cb:
        progress_cb(window_frames, window_frames, f"Finished {info.name}")
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
    route_images: dict[str, np.ndarray] = {}
    if trace_frame is not None:
        overlay = _draw_path_overlay(trace_frame, trips, pending, fitted)
        route_images = _route_path_overlays(trace_frame, trips, pending, fitted)
    return ProcessResult(
        result_info,
        trips,
        device_label,
        fitted,
        model_label=model_label,
        start_crossings=start_crossings,
        end_crossings=end_crossings,
        frames_with_tracks=frames_with_tracks,
        trace_image=overlay,
        route_images=route_images,
        issues=list(pending.issues),
    )


def _windowed_info(info: VideoInfo, start_s: float, end_s: float, frames: int) -> VideoInfo:
    return replace(info, duration_s=max(0.0, end_s - start_s), frame_count=max(0, frames))


def _shift_result_times(trips: list[Trip], issues: list[TrackIssue], origin_s: float) -> None:
    if origin_s <= 0:
        return
    for trip in trips:
        trip.start_time_s -= origin_s
        trip.end_time_s -= origin_s
    for issue in issues:
        issue.time_s -= origin_s


PATH_WIDTH = 1


def _path_push(state: _TrackState, xy: tuple[float, float]) -> None:
    if not state.path:
        state.path.append(xy)
        return
    last = state.path[-1]
    if ((xy[0] - last[0]) ** 2 + (xy[1] - last[1]) ** 2) ** 0.5 >= 1.0:
        state.path.append(xy)


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
                _path_push(state, point)
    _path_push(state, xy)


def _append_reconnect_path(state: _TrackState, xy: tuple[float, float], geometry: Geometry) -> None:
    """Stitch a drop-zone gap with a straight line between real observations.

    Pickup/lead is only for re-acquiring the same ID. Drawing through that
    point sprays a fan every time detection flickers in the zone.
    """
    _append_path(state, xy, geometry)


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


def _trace_polylines(
    pending: _Pending,
    trips: list[Trip] | None,
    geometry: Geometry | None,
    *,
    include_unlinked: bool,
) -> list[list[tuple[float, float]]]:
    """Polylines to stroke. Trip paths are skipped when the same points were already kept."""
    jump = _path_jump_limit(geometry, pending.path_jump)
    lines: list[list[tuple[float, float]]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()

    def _add(points: list[tuple[float, float]]) -> None:
        for chunk in _path_chunks(points, jump, geometry):
            key = tuple((int(round(x)), int(round(y))) for x, y in chunk)
            if len(key) < 2 or key in seen:
                continue
            seen.add(key)
            lines.append(chunk)

    if include_unlinked:
        for _cls, points in pending.finished_paths:
            _add(list(points))
        for state in pending.live.values():
            if state.active:
                _add(list(state.path))
    for trip in trips or []:
        _add(list(trip.path))
    return lines


def _draw_paths(
    frame: np.ndarray,
    pending: _Pending,
    trips: list[Trip] | None = None,
    geometry: Geometry | None = None,
    *,
    include_unlinked: bool = True,
) -> None:
    for chunk in _trace_polylines(pending, trips, geometry, include_unlinked=include_unlinked):
        _halo_polyline(frame, chunk, pending.trace_color, PATH_WIDTH)


def _draw_path_overlay(
    frame: np.ndarray,
    trips: list[Trip],
    pending: _Pending,
    geometry: Geometry,
    *,
    title: str | None = None,
    include_unlinked: bool = True,
) -> np.ndarray:
    """Darken the still so neon traces stay readable on pale drone pavement."""
    preview = (frame.astype(np.float32) * 0.72).astype(np.uint8)
    _draw_paths(preview, pending, trips, geometry, include_unlinked=include_unlinked)
    preview = _draw_route_overlay(preview, geometry)
    if title:
        _halo_text(preview, title, (16, 32), pending.trace_color, 0.8)
    return preview


def _route_path_overlays(
    frame: np.ndarray,
    trips: list[Trip],
    pending: _Pending,
    geometry: Geometry,
) -> dict[str, np.ndarray]:
    """One overlay per movement that has at least one completed trip."""
    names = list(geometry.movement_names())
    for trip in trips:
        if trip.route_name not in names:
            names.append(trip.route_name)
    images: dict[str, np.ndarray] = {}
    for name in names:
        subset = [trip for trip in trips if trip.route_name == name]
        if not subset:
            continue
        images[name] = _draw_path_overlay(
            frame, subset, pending, geometry, title=name, include_unlinked=False
        )
    return images


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
    labels: list[tuple[tuple[int, int], int | None, tuple[int, int, int], _TrackState | None, str]] = []
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
        color = (220, 20, 255)
        _halo_rect(preview, p1, p2, color)
        state = next((item for item in states if item.tid == tid), None) if tid is not None else None
        extra = "" if tid is not None else f"{w}x{h}"
        labels.append((p1, tid, color, state, extra))
    for state in pending.live.values():
        if state.tid in used or not state.last_xyxy:
            continue
        if active_only and not state.active:
            continue
        x1, y1, x2, y2 = state.last_xyxy
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        if state.estimate == "predicted":
            color = (255, 200, 0)
        elif state.estimate == "held":
            color = (40, 200, 255)
        else:
            color = (220, 20, 255)
        _halo_rect(preview, p1, p2, color)
        labels.append((p1, state.tid, color, state, ""))
    _draw_leads(preview, pending)
    _draw_paths(preview, pending, trips, geometry)
    preview = _draw_route_overlay(preview, geometry)
    for p1, tid, color, state, extra in labels:
        _draw_id_label(preview, p1, tid, color, state=state, extra=extra)
    return preview


def _draw_leads(frame: np.ndarray, pending: _Pending) -> None:
    for state in pending.live.values():
        if not state.active:
            continue
        tip = state.pickup_xy or state.lead_xy
        if tip is None:
            continue
        origin = _observed_origin(state)
        frozen = state.pickup_xy is not None
        color = (0, 200, 255) if frozen else (255, 200, 0)
        a = (int(round(origin[0])), int(round(origin[1])))
        b = (int(round(tip[0])), int(round(tip[1])))
        _halo_line(frame, a, b, color, 1)
        cv2.circle(frame, b, 5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.circle(frame, b, 4, color, -1, cv2.LINE_AA)


def _halo_line(frame: np.ndarray, a: tuple[int, int], b: tuple[int, int], color: tuple[int, int, int], width: int = 3) -> None:
    cv2.line(frame, a, b, (0, 0, 0), width + 3, cv2.LINE_AA)
    cv2.line(frame, a, b, color, width, cv2.LINE_AA)


def _halo_rect(frame: np.ndarray, p1: tuple[int, int], p2: tuple[int, int], color: tuple[int, int, int]) -> None:
    cv2.rectangle(frame, p1, p2, (0, 0, 0), 5)
    cv2.rectangle(frame, p1, p2, color, 2)


ID_LIVE_BGR = (40, 40, 255)
ID_ENTRY_BGR = (255, 90, 40)
ID_COUNTED_BGR = (40, 220, 40)


def _halo_text(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    scale: float,
    thickness: int = 1,
) -> None:
    halo = max(thickness + 2, 3)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), halo, cv2.LINE_AA)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _id_label_color(state: _TrackState | None) -> tuple[int, int, int]:
    if state is None or not state.entry_origin:
        return ID_LIVE_BGR
    if state.exit_bound:
        return ID_COUNTED_BGR
    return ID_ENTRY_BGR


def _route_label(state: _TrackState | None) -> str:
    if state is None or not state.entry_origin:
        return ""
    if state.exit_bound:
        return f"{state.entry_origin}→{state.exit_bound}"
    return f"{state.entry_origin}→"


def _outlined_id_text(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    *,
    scale: float = 1.2,
    thickness: int = 3,
    outline: int = 2,
) -> None:
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    black = (0, 0, 0)
    cv2.putText(frame, text, origin, font, scale, black, thickness + outline * 2, cv2.LINE_AA)
    for dx in (-outline, 0, outline):
        for dy in (-outline, 0, outline):
            if dx == 0 and dy == 0:
                continue
            cv2.putText(frame, text, (x + dx, y + dy), font, scale, black, thickness, cv2.LINE_AA)
    cv2.putText(frame, text, origin, font, scale, color, thickness, cv2.LINE_AA)


def _draw_id_label(
    frame: np.ndarray,
    p1: tuple[int, int],
    tid: int | None,
    color: tuple[int, int, int],
    *,
    state: _TrackState | None = None,
    extra: str = "",
) -> None:
    """Large bold ID: red until a trip counts, then green. Black outline for contrast."""
    if tid is None:
        if extra:
            _halo_text(frame, extra, (p1[0], max(20, p1[1] - 8)), color, 0.7, thickness=2)
        return
    tag = f"#{tid}"
    route = _route_label(state)
    if route:
        tag = f"{tag} {route}"
    origin = (p1[0], max(36, p1[1] - 10))
    id_color = _id_label_color(state)
    _outlined_id_text(frame, tag, origin, id_color)
    bits: list[str] = []
    if extra:
        bits.append(extra)
    if state is not None:
        if state.estimate == "predicted":
            bits.append("pred")
        elif state.estimate == "held":
            bits.append("hold")
    if bits:
        _halo_text(frame, " ".join(bits), (origin[0], origin[1] + 22), id_color, 0.55, thickness=2)


def _draw_route_overlay(frame: np.ndarray, geometry: Geometry) -> np.ndarray:
    if geometry.north is not None:
        p1 = (int(geometry.north.x1), int(geometry.north.y1))
        p2 = (int(geometry.north.x2), int(geometry.north.y2))
        cv2.arrowedLine(frame, p1, p2, (0, 0, 0), 5, tipLength=0.18)
        cv2.arrowedLine(frame, p1, p2, (255, 200, 40), 3, tipLength=0.18)
        _halo_text(frame, "N", p2, (255, 200, 40), 0.6)
    for item in geometry.all_gates():
        a = (int(item.line.x1), int(item.line.y1))
        b = (int(item.line.x2), int(item.line.y2))
        _halo_line(frame, a, b, (60, 230, 0))
        _halo_text(frame, gate_origin_name(item.name), a, (60, 230, 0), 0.55)
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


def _issue_point(issue: TrackIssue) -> tuple[float, float] | None:
    if issue.xy is not None:
        return issue.xy
    if issue.xyxy is not None:
        return bbox_center(issue.xyxy)
    return None


def draw_issue_overlay(
    frame: np.ndarray,
    issues: list[TrackIssue],
    geometry: Geometry | None = None,
    *,
    highlight: TrackIssue | None = None,
) -> np.ndarray:
    """Mark issue locations on a still. Highlight one event, keep the rest as dots."""
    preview = frame.copy()
    if geometry is not None:
        preview = _draw_route_overlay(preview, geometry)
    for issue in issues:
        point = _issue_point(issue)
        if point is None or issue is highlight:
            continue
        color = ISSUE_COLORS.get(issue.kind, (200, 200, 200))
        center = (int(round(point[0])), int(round(point[1])))
        cv2.circle(preview, center, 7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.circle(preview, center, 6, color, -1, cv2.LINE_AA)
    if highlight is not None:
        color = ISSUE_COLORS.get(highlight.kind, (200, 200, 200))
        title = ISSUE_LABELS.get(highlight.kind, highlight.kind)
        caption = f"{title}  #{highlight.tid}"
        if highlight.from_xy is not None and highlight.xy is not None:
            a = (int(round(highlight.from_xy[0])), int(round(highlight.from_xy[1])))
            b = (int(round(highlight.xy[0])), int(round(highlight.xy[1])))
            _halo_line(preview, a, b, color, 2)
            cv2.circle(preview, a, 5, color, -1, cv2.LINE_AA)
        if highlight.xyxy is not None:
            x1, y1, x2, y2 = (int(round(v)) for v in highlight.xyxy)
            _halo_rect(preview, (x1, y1), (x2, y2), color)
            _halo_text(preview, caption, (x1, max(18, y1 - 8)), color, 0.55)
        elif highlight.xy is not None:
            center = (int(round(highlight.xy[0])), int(round(highlight.xy[1])))
            cv2.circle(preview, center, 12, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.circle(preview, center, 10, color, 2, cv2.LINE_AA)
            _halo_text(preview, caption, (center[0] + 12, max(18, center[1] - 8)), color, 0.55)
        legend = "  ".join(ISSUE_LABELS[kind] for kind in ("drop_zone_attach", "long_trace", "duplicate_stack"))
        _halo_text(preview, legend, (12, 22), (240, 240, 240), 0.45)
    else:
        counts: dict[str, int] = {}
        for issue in issues:
            counts[issue.kind] = counts.get(issue.kind, 0) + 1
        parts = [
            f"{ISSUE_LABELS.get(kind, kind)} {counts[kind]}"
            for kind in ("drop_zone_attach", "long_trace", "duplicate_stack")
            if counts.get(kind)
        ]
        if parts:
            _halo_text(preview, "  ".join(parts), (12, 22), (240, 240, 240), 0.5)
    return preview


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


def _crossed_drop_span(
    p0: tuple[float, float],
    p1: tuple[float, float],
    geometry: Geometry,
    pad: float = 16.0,
) -> float:
    """Span of the drop zone this gap actually uses, not the largest zone on the map."""
    shared = _shared_drop_span(p0, p1, geometry)
    if shared > 0:
        return shared
    span = 0.0
    if not geometry.occlusions:
        return 0.0
    steps = max(2, int(((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2) ** 0.5))
    for box in geometry.occlusions:
        item = box.normalized()
        hit = False
        for index in range(steps + 1):
            t = index / steps
            x = p0[0] + (p1[0] - p0[0]) * t
            y = p0[1] + (p1[1] - p0[1]) * t
            if item.x1 - pad <= x <= item.x2 + pad and item.y1 - pad <= y <= item.y2 + pad:
                hit = True
                break
        if hit:
            span = max(span, item.width(), item.height())
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


def _prediction_reach(geometry: Geometry) -> float:
    return clamp_prediction_reach(getattr(geometry, "prediction_reach_px", PREDICTION_REACH_PX) or PREDICTION_REACH_PX)


def _heading_unit(state: _TrackState) -> tuple[float, float] | None:
    speed = (state.vel[0] ** 2 + state.vel[1] ** 2) ** 0.5
    if speed < MOTION_HOLD_SPEED_PX_S:
        return None
    return state.vel[0] / speed, state.vel[1] / speed


def _lead_from_state(state: _TrackState, geometry: Geometry) -> tuple[float, float] | None:
    if not state.active:
        return None
    unit = _heading_unit(state)
    if unit is None:
        return None
    origin = _observed_origin(state)
    reach = _prediction_reach(geometry)
    return (origin[0] + unit[0] * reach, origin[1] + unit[1] * reach)


def _pickup_radius(state: _TrackState, geometry: Geometry) -> float:
    return max(0.5 * _prediction_reach(geometry), 0.6 * max(state.bbox_wh))


def _point_to_segment_dist(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length2 = dx * dx + dy * dy
    if length2 < 1e-9:
        return ((point[0] - start[0]) ** 2 + (point[1] - start[1]) ** 2) ** 0.5
    t = max(0.0, min(1.0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length2))
    qx, qy = start[0] + t * dx, start[1] + t * dy
    return ((point[0] - qx) ** 2 + (point[1] - qy) ** 2) ** 0.5


def _near_pickup(state: _TrackState, xy: tuple[float, float], geometry: Geometry) -> bool:
    pickup = state.pickup_xy
    if pickup is None:
        return False
    radius = _pickup_radius(state, geometry)
    dist = ((xy[0] - pickup[0]) ** 2 + (xy[1] - pickup[1]) ** 2) ** 0.5
    if dist <= radius:
        return True
    origin = _observed_origin(state)
    lateral = max(0.5 * max(state.bbox_wh), 0.35 * radius)
    return _point_to_segment_dist(xy, origin, pickup) <= lateral


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
        zone_cap = _prediction_reach(geometry) + _crossed_drop_span(origin, xy, geometry) + slack
        return min(max(capped, speed_disk), max(capped, zone_cap))
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


def _add_issue(
    pending: _Pending,
    kind: str,
    time_s: float,
    tid: int,
    message: str,
    *,
    xy: tuple[float, float] | None = None,
    from_xy: tuple[float, float] | None = None,
    xyxy: tuple[float, float, float, float] | None = None,
) -> None:
    if len(pending.issues) >= 200:
        return
    pending.issues.append(
        TrackIssue(
            kind=kind,
            time_s=float(time_s),
            tid=int(tid),
            message=message,
            xy=xy,
            from_xy=from_xy,
            xyxy=xyxy,
        )
    )


def _collapse_stacked_boxes(
    boxes: list[tuple[float, float, float, float]],
    classes: list[str] | None,
    confidences: list[float] | None,
) -> tuple[list[tuple[float, float, float, float]], list[str] | None, list[float] | None, int]:
    """One measurement per vehicle before LAP, so stacked class boxes share an ID."""
    if len(boxes) < 2:
        return boxes, classes, confidences, 0
    detections: list[tuple[tuple[float, float, float, float], str, float]] = []
    for index, xyxy in enumerate(boxes):
        label = classes[index] if classes and index < len(classes) else "vehicle"
        score = confidences[index] if confidences and index < len(confidences) else 0.0
        detections.append((xyxy, label, score))
    kept = suppress_duplicate_boxes(detections)
    dropped = len(detections) - len(kept)
    if dropped <= 0:
        return boxes, classes, confidences, 0
    out_boxes = [item[0] for item in kept]
    out_classes = [item[1] for item in kept] if classes is not None else None
    out_confs = [item[2] for item in kept] if confidences is not None else None
    return out_boxes, out_classes, out_confs, dropped


def _stack_owner(pending: _Pending, xyxy: tuple[float, float, float, float]) -> _TrackState | None:
    """Live track whose box already covers this detection (NMS leftover)."""
    best: _TrackState | None = None
    best_iou = -1.0
    for state in pending.live.values():
        box = state.last_xyxy or state.last_observed_xyxy
        if box is None:
            continue
        if not _same_vehicle_stack(xyxy, box):
            continue
        overlap = box_iou(xyxy, box)
        if overlap > best_iou:
            best_iou = overlap
            best = state
    return best


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
    if state.pickup_xy is not None and _near_pickup(state, xy, geometry):
        pickup_cap = _prediction_reach(geometry) + _pickup_radius(state, geometry)
        if dist_last <= max(limit, pickup_cap):
            pickup = state.pickup_xy
            dist_pick = ((xy[0] - pickup[0]) ** 2 + (xy[1] - pickup[1]) ** 2) ** 0.5
            return dist_pick, "lead-pickup"
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
    if not state.active:
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


def _drawn_roadway(geometry: Geometry) -> bool:
    return bool(getattr(geometry, "roadways", None))


def _roadway_standstill(state: _TrackState, geometry: Geometry) -> bool:
    """True when this track is sitting on a drawn roadway (stoplight / queue)."""
    if not _drawn_roadway(geometry):
        return False
    origin = _observed_origin(state)
    return point_in_roadway(origin[0], origin[1], geometry)


def _is_stationary_track(state: _TrackState) -> bool:
    if state.holding or not state.active:
        return True
    return state.pickup_xy is None and state.speed <= MOTION_HOLD_SPEED_PX_S


def _stall_radius(state: _TrackState) -> float:
    return max(18.0, 0.5 * max(state.bbox_wh))


def _stall_owner(
    pending: _Pending,
    xyxy: tuple[float, float, float, float],
    geometry: Geometry,
) -> _TrackState | None:
    """The stopped car this box is sitting on, if any.

    A lost ID must not attach to a stoplight box that already belongs to
    a vehicle we have already seen.
    """
    del geometry
    xy = bbox_center(xyxy)
    best: _TrackState | None = None
    best_score = float("inf")
    for state in pending.live.values():
        if not _is_stationary_track(state):
            continue
        last = _observed_box(state)
        origin = _observed_origin(state)
        dist = ((xy[0] - origin[0]) ** 2 + (xy[1] - origin[1]) ** 2) ** 0.5
        overlap = box_iou(last, xyxy) if last is not None else 0.0
        if overlap < DETECTOR_MATCH_IOU_MIN and dist > _stall_radius(state):
            continue
        score = dist * (1.0 - 0.8 * min(overlap, 1.0))
        if score < best_score:
            best = state
            best_score = score
    return best


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
    owner = _stall_owner(pending, xyxy, geometry)
    best: _TrackState | None = None
    best_cost = float("inf")
    for state in pending.live.values():
        if state.tid in seen:
            continue
        if owner is not None and state.tid != owner.tid:
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
        smooth_xy=xy,
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
        owners = {col: _stall_owner(pending, xyxy, geometry) for col, xyxy in enumerate(boxes)}
        for row, state in enumerate(tracks):
            for col, xyxy in enumerate(boxes):
                owner = owners.get(col)
                if owner is not None and state.tid != owner.tid:
                    continue
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
            owner = _stack_owner(pending, xyxy)
            if owner is not None:
                assigned.append((owner, False, "stack"))
                continue
            created = _new_track(pending, xy, wh, elapsed_s)
            created.last_xyxy = (x1, y1, x2, y2)
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
    infer_trips_from_traces: bool = False,
    prev_xyxy: tuple[float, float, float, float] | None = None,
) -> tuple[int, int]:
    """First gate that cuts this ID is entry; the next different gate is exit."""
    overlapping: set[str] = set()
    hits: list[tuple[float, str]] = []
    seen_origins: set[str] = set()
    for origin, line in gates:
        if box_overlaps_line(xyxy, line, pad_px=pad):
            overlapping.add(origin)
        t = box_cross_t(prev_xyxy, xyxy, line, pad_px=pad)
        if t is None and infer_trips_from_traces:
            t = trace_cross_t(prev, xy, line, pad_px=pad, clearance_px=8.0)
        if t is None or origin in seen_origins or origin in state.counted_gates:
            continue
        seen_origins.add(origin)
        hits.append((t, origin))
    hits.sort()
    start_hits = 0
    end_hits = 0
    for _, origin in hits:
        if state.start is None:
            state.start = (origin, elapsed_s)
            state.start_evidence = evidence
            state.last_side = origin
            state.entry_origin = origin
            state.exit_bound = None
            state.had_gate_trip = False
            state.counted_gates.add(origin)
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
            state.entry_origin = start_origin
            state.exit_bound = ORIGIN_TO_BOUND.get(origin, origin)
            state.last_side = origin
            state.counted_gates.add(origin)
            end_hits += 1
            _release_track_hold(state)
            break
    if state.start is None:
        state.counted_gates &= overlapping
    if pending is not None and (start_hits or end_hits):
        _dbg_bump(pending, "confirmed_hit")
    return start_hits, end_hits


def _release_track_hold(state: _TrackState) -> None:
    """After a full trip, stop freezing MOG2. Keep the box so the same ID continues."""
    state.holding = False


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
            state.entry_origin = start_origin
            state.exit_bound = ORIGIN_TO_BOUND.get(last_side, last_side)
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
        state.entry_origin = first_side
        state.exit_bound = ORIGIN_TO_BOUND.get(last_side, last_side)
        return 1
    return 0


def _hold_limit(state: _TrackState, geometry: Geometry) -> float:
    if not state.active:
        if _roadway_standstill(state, geometry):
            return MOTION_HOLD_S
        return DETECTOR_STATIC_EXPIRE_S
    if state.sightings < 2:
        return MOTION_COAST_S
    if state.holding or state.speed <= MOTION_HOLD_SPEED_PX_S:
        return MOTION_HOLD_S
    return MOTION_COAST_S


def _can_hold(state: _TrackState, now_s: float, geometry: Geometry) -> bool:
    if _observed_box(state) is None:
        return False
    if now_s - state.last_observed_s > MOTION_HOLD_S:
        return False
    if not state.active:
        return _roadway_standstill(state, geometry)
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
        limit = 0.0 if coast_s is not None else _hold_limit(state, geometry)
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
    infer_trips_from_traces: bool = False,
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
    collapsed_classes = [item or "vehicle" for item in track_classes]
    track_boxes, collapsed_classes, track_confs, dropped = _collapse_stacked_boxes(
        track_boxes, collapsed_classes, track_confs
    )
    track_classes = list(collapsed_classes or [])
    box_states = _assign_boxes_to_states(pending, track_boxes, elapsed_s, geometry, moving_only)
    if dropped > 0 and box_states:
        owner_tid = box_states[0][0].tid
        kept = track_boxes[0]
        _add_issue(
            pending,
            "duplicate_stack",
            elapsed_s,
            owner_tid,
            f"#{owner_tid} extra box(es) stacked on the same car at {elapsed_s:.1f}s",
            xy=bbox_center(kept),
            xyxy=kept,
        )

    for index, xyxy in enumerate(track_boxes):
        xy = bbox_center(xyxy)
        x1, y1, x2, y2 = (float(v) for v in xyxy)
        wh = (max(x2 - x1, 1.0), max(y2 - y1, 1.0))
        state, is_new, method = box_states[index]
        if state.tid in seen:
            _add_issue(
                pending,
                "duplicate_stack",
                elapsed_s,
                state.tid,
                f"#{state.tid} extra box stacked on the same car at {elapsed_s:.1f}s",
                xy=xy,
                xyxy=xyxy,
            )
            continue
        cls = track_classes[index] or "vehicle"
        if is_new or state.cls == "vehicle":
            state.cls = cls
        prev_xy = state.last_observed_xy or state.xy
        prev_xyxy = None if is_new else (state.last_observed_xyxy or state.last_xyxy)
        prev_t = state.last_observed_s or state.first_time_s
        gap = (not is_new) and (elapsed_s - prev_t) > 0.25
        evidence = "gap" if gap or method in {"drop-zone", "lead-pickup"} else "observed"
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
        reconnecting = state.pickup_xy is not None
        if is_new or state.smooth_xy is None or reconnecting:
            state.smooth_xy = xy
        else:
            prev_s = state.smooth_xy
            step = ((xy[0] - prev_s[0]) ** 2 + (xy[1] - prev_s[1]) ** 2) ** 0.5
            jitter = max(6.0, 0.25 * max(state.bbox_wh))
            if step > jitter:
                state.smooth_xy = xy
            else:
                state.smooth_xy = (
                    PATH_SMOOTH_ALPHA * xy[0] + (1.0 - PATH_SMOOTH_ALPHA) * prev_s[0],
                    PATH_SMOOTH_ALPHA * xy[1] + (1.0 - PATH_SMOOTH_ALPHA) * prev_s[1],
                )
        action = "created" if is_new else ("recovered" if gap else "matched")
        if moving:
            state.active = True
            state.last_motion_s = elapsed_s
            path_xy = xy if reconnecting else (state.smooth_xy or xy)
            if state.path:
                last = state.path[-1]
                dist = ((path_xy[0] - last[0]) ** 2 + (path_xy[1] - last[1]) ** 2) ** 0.5
                jump = _path_jump_limit(geometry, pending.path_jump)
                if dist > jump:
                    _add_issue(
                        pending,
                        "long_trace",
                        elapsed_s,
                        state.tid,
                        f"#{state.tid} trace jumped farther than the split limit at {elapsed_s:.1f}s",
                        xy=path_xy,
                        from_xy=last,
                        xyxy=(x1, y1, x2, y2),
                    )
                if method in {"drop-zone", "lead-pickup"}:
                    _add_issue(
                        pending,
                        "drop_zone_attach",
                        elapsed_s,
                        state.tid,
                        f"#{state.tid} reattached across a drop zone at {elapsed_s:.1f}s",
                        xy=path_xy,
                        from_xy=last,
                        xyxy=(x1, y1, x2, y2),
                    )
            if reconnecting:
                _append_reconnect_path(state, xy, geometry)
                state.smooth_xy = xy
            else:
                _append_path(state, path_xy, geometry)
            state.lead_xy = _lead_from_state(state, geometry)
            state.pickup_xy = None
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
            infer_trips_from_traces=infer_trips_from_traces,
            prev_xyxy=prev_xyxy,
        )
        start_hits += added_s
        end_hits += added_e
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
            crossing="confirmed" if added_s or added_e else "none",
            reason="" if moving else "stationary",
        )

    for state in list(pending.live.values()):
        if state.tid in seen:
            continue
        age = elapsed_s - state.last_observed_s
        if _can_coast_drop_zone(state, elapsed_s, geometry):
            if state.pickup_xy is None:
                state.pickup_xy = state.lead_xy or _lead_from_state(state, geometry)
            pred = state.pickup_xy or _predicted_xy(state, elapsed_s)
            state.pickup_xy = pred
            state.lead_xy = pred
            half_w, half_h = state.bbox_wh[0] / 2.0, state.bbox_wh[1] / 2.0
            state.xy = pred
            state.last_time_s = elapsed_s
            state.estimate = "predicted"
            state.last_xyxy = (pred[0] - half_w, pred[1] - half_h, pred[0] + half_w, pred[1] + half_h)
            _note(
                pending,
                time_s=elapsed_s,
                tid=state.tid,
                cls=state.cls,
                conf=state.confidence,
                status="predicted",
                age_s=age,
                action="predicted",
                match="lead-pickup" if state.pickup_xy is not None else "drop-zone",
                crossing="rejected",
                reason="prediction is not a count",
            )
            continue
        if not _can_hold(state, elapsed_s, geometry) or _observed_box(state) is None:
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
