"""Motion-mask blobs and record start/end crossings. No vehicle classes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from traffic_tracker.config import (
    MOTION_COAST_S,
    MOTION_HOLD_S,
    MOTION_HOLD_SPEED_PX_S,
    MOTION_WARMUP_S,
)
from traffic_tracker.core.crossings import bbox_center, box_hits_line, path_cross_t
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


@dataclass
class _Pending:
    """Centroid tracks. A nearby new blob continues the same vehicle."""

    live: dict[int, _TrackState] = field(default_factory=dict)
    next_synth: int = 1


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

    pending = _Pending()
    trips: list[Trip] = []
    start_crossings = 0
    end_crossings = 0
    frames_with_tracks = 0
    subtractor = None if use_detector else create_subtractor()
    prev_gray = None

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
                boxes, mask = boxes_from_detector(
                    detector, frame, min_area=min_area, max_area=max_area
                )
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

            if use_detector or elapsed_s >= MOTION_WARMUP_S:
                starts, ends = _record_crossings(
                    boxes, fitted, pending, trips, info.name, elapsed_s
                )
                start_crossings += starts
                end_crossings += ends
                if boxes or pending.live:
                    frames_with_tracks += 1
                end_crossings += _expire_tracks(pending, fitted, trips, info.name, elapsed_s)

            if preview_cb and (frame_idx == 1 or frame_idx % preview_every == 0):
                preview = _draw_motion_preview(frame, mask, boxes, pending, fitted)
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
        pending, fitted, trips, info.name, (frame_idx / max(info.fps, 1e-6)) + MOTION_COAST_S + 1.0, coast_s=0.0
    )
    if progress_cb:
        progress_cb(info.frame_count or frame_idx, info.frame_count or frame_idx, f"Finished {info.name}")
    return ProcessResult(
        info,
        trips,
        device_label,
        fitted,
        model_label=model_label,
        start_crossings=start_crossings,
        end_crossings=end_crossings,
        frames_with_tracks=frames_with_tracks,
    )


def _draw_motion_preview(
    frame: np.ndarray,
    mask: np.ndarray,
    boxes: list[tuple[float, float, float, float]],
    pending: _Pending,
    geometry: Geometry,
) -> np.ndarray:
    preview = frame.copy()
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
    by_xy = {state.tid: state for state in pending.live.values()}
    used = set()
    for x1, y1, x2, y2 in boxes:
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        tid = None
        best = 40.0
        for state in by_xy.values():
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
        x1, y1, x2, y2 = state.last_xyxy
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        _halo_rect(preview, p1, p2, (255, 200, 40))
        _halo_text(preview, f"#{state.tid} hold", (p1[0], max(16, p1[1] - 6)), (255, 200, 40), 0.45)
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
    xy: tuple[float, float],
    wh: tuple[float, float],
    elapsed_s: float,
    seen: set[int],
    max_dist: float,
) -> tuple[_TrackState, bool]:
    adopted = _nearest_unseen(pending, xy, wh, seen, max_dist)
    if adopted is not None:
        return adopted, False
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
    return state, True


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
            trips.append(
                Trip(
                    video_name=video_name,
                    track_id=state.tid,
                    start_destination=start_origin,
                    start_time_s=start_time_s,
                    end_destination=ORIGIN_TO_BOUND.get(origin, origin),
                    end_time_s=elapsed_s,
                    vehicle_class="vehicle",
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
                    vehicle_class="vehicle",
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
                vehicle_class="vehicle",
            )
        )
        state.had_gate_trip = True
        return 1
    return 0


def _hold_limit(state: _TrackState) -> float:
    if state.had_gate_trip:
        return MOTION_COAST_S
    if state.sightings < 2:
        return MOTION_COAST_S
    if state.holding or state.speed <= MOTION_HOLD_SPEED_PX_S:
        return MOTION_HOLD_S
    return MOTION_COAST_S


def _can_hold(state: _TrackState, now_s: float) -> bool:
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
) -> int:
    keep: dict[int, _TrackState] = {}
    inferred = 0
    for tid, state in pending.live.items():
        limit = 0.0 if coast_s is not None else _hold_limit(state)
        if coast_s is not None:
            limit = coast_s
        age = now_s - state.last_motion_s
        if age > limit:
            inferred += _finalize_track(state, geometry, trips, video_name)
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
) -> tuple[int, int]:
    gates = [(_gate_origin(item, geometry), item.line) for item in (*geometry.starts, *geometry.ends)]
    pad = max(32.0, 0.03 * max(geometry.frame_width, geometry.frame_height))
    max_dist = _adopt_radius(geometry)
    start_hits = 0
    end_hits = 0
    seen: set[int] = set()

    for xyxy in boxes:
        xy = bbox_center(xyxy)
        x1, y1, x2, y2 = (float(v) for v in xyxy)
        wh = (max(x2 - x1, 1.0), max(y2 - y1, 1.0))
        state, is_new = _resolve_state(pending, xy, wh, elapsed_s, seen, max_dist)
        seen.add(state.tid)
        dt = max(elapsed_s - state.last_time_s, 1e-3)
        inst = ((xy[0] - state.xy[0]) ** 2 + (xy[1] - state.xy[1]) ** 2) ** 0.5 / dt
        state.speed = inst if is_new else (0.45 * inst + 0.55 * state.speed)
        state.bbox_wh = wh
        state.last_xyxy = (x1, y1, x2, y2)
        state.sightings += 0 if is_new else 1
        state.holding = False
        added_s, added_e = _apply_gate_hits(
            state, state.xy, xy, (x1, y1, x2, y2), gates, pad, trips, video_name, elapsed_s
        )
        start_hits += added_s
        end_hits += added_e
        if added_e > 0:
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
        added_s, added_e = _apply_gate_hits(
            state, state.xy, state.xy, (x1, y1, x2, y2), gates, pad, trips, video_name, elapsed_s
        )
        start_hits += added_s
        end_hits += added_e
        if added_e > 0:
            pending.live.pop(state.tid, None)
            continue
        state.last_time_s = elapsed_s
        seen.add(state.tid)
    return start_hits, end_hits
