"""Count a gate crossing when a track's path intersects the drawn segment."""

from __future__ import annotations

from traffic_tracker.core.geometry import Line


def bbox_center(xyxy) -> tuple[float, float]:
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def bbox_corners(xyxy) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Top-left, top-right, bottom-right, bottom-left of an axis-aligned box."""
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    return ((x1, y1), (x2, y1), (x2, y2), (x1, y2))


def box_iou(a, b) -> float:
    """Intersection-over-union for axis-aligned boxes (x1, y1, x2, y2)."""
    ax1, ay1, ax2, ay2 = (float(v) for v in a)
    bx1, by1, bx2, by2 = (float(v) for v in b)
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _orient(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> float:
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def _on_segment(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> bool:
    return min(ax, bx) - 1e-6 <= cx <= max(ax, bx) + 1e-6 and min(ay, by) - 1e-6 <= cy <= max(ay, by) + 1e-6


def segments_intersect(
    a1: tuple[float, float],
    a2: tuple[float, float],
    b1: tuple[float, float],
    b2: tuple[float, float],
) -> bool:
    ax, ay = a1
    bx, by = a2
    cx, cy = b1
    dx, dy = b2
    o1 = _orient(ax, ay, bx, by, cx, cy)
    o2 = _orient(ax, ay, bx, by, dx, dy)
    o3 = _orient(cx, cy, dx, dy, ax, ay)
    o4 = _orient(cx, cy, dx, dy, bx, by)
    if (o1 > 0 and o2 < 0 or o1 < 0 and o2 > 0) and (o3 > 0 and o4 < 0 or o3 < 0 and o4 > 0):
        return True
    if abs(o1) <= 1e-6 and _on_segment(ax, ay, bx, by, cx, cy):
        return True
    if abs(o2) <= 1e-6 and _on_segment(ax, ay, bx, by, dx, dy):
        return True
    if abs(o3) <= 1e-6 and _on_segment(cx, cy, dx, dy, ax, ay):
        return True
    if abs(o4) <= 1e-6 and _on_segment(cx, cy, dx, dy, bx, by):
        return True
    return False


def pad_line(line: Line, pad_px: float) -> Line:
    dx = line.x2 - line.x1
    dy = line.y2 - line.y1
    length = (dx * dx + dy * dy) ** 0.5
    if length < 1e-6:
        return line
    ux, uy = dx / length, dy / length
    return Line(line.x1 - ux * pad_px, line.y1 - uy * pad_px, line.x2 + ux * pad_px, line.y2 + uy * pad_px)


def path_cross_t(
    prev: tuple[float, float],
    curr: tuple[float, float],
    line: Line,
    *,
    pad_px: float = 24.0,
) -> float | None:
    """Distance 0–1 along prev→curr where the gate is hit, or None."""
    if prev == curr:
        return None
    gated = pad_line(line, pad_px)
    return _segment_t(prev, curr, (gated.x1, gated.y1), (gated.x2, gated.y2))


def box_hits_line(xyxy, line: Line, *, pad_px: float = 24.0) -> bool:
    """True when the bounding box sits on or straddles the (padded) gate."""
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    left, right = min(x1, x2), max(x1, x2)
    top, bottom = min(y1, y2), max(y1, y2)
    gated = pad_line(line, pad_px)
    g1, g2 = (gated.x1, gated.y1), (gated.x2, gated.y2)
    edges = (
        ((left, top), (right, top)),
        ((right, top), (right, bottom)),
        ((right, bottom), (left, bottom)),
        ((left, bottom), (left, top)),
    )
    if any(segments_intersect(g1, g2, a, b) for a, b in edges):
        return True
    if left <= gated.x1 <= right and top <= gated.y1 <= bottom:
        return True
    if left <= gated.x2 <= right and top <= gated.y2 <= bottom:
        return True
    cx, cy = (left + right) / 2.0, (top + bottom) / 2.0
    return _point_near_segment((cx, cy), gated, pad_px)


def _point_near_segment(point: tuple[float, float], line: Line, pad_px: float) -> bool:
    px, py = point
    ax, ay, bx, by = line.x1, line.y1, line.x2, line.y2
    dx, dy = bx - ax, by - ay
    length2 = dx * dx + dy * dy
    if length2 < 1e-9:
        return ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5 <= pad_px
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length2))
    qx, qy = ax + t * dx, ay + t * dy
    return ((px - qx) ** 2 + (py - qy) ** 2) ** 0.5 <= pad_px


def signed_offset(point: tuple[float, float], line: Line) -> float:
    """Signed perpendicular distance from a point to the infinite gate line."""
    dx = line.x2 - line.x1
    dy = line.y2 - line.y1
    length = (dx * dx + dy * dy) ** 0.5 or 1.0
    px, py = point
    return ((px - line.x1) * dy - (py - line.y1) * dx) / length


def confirmed_cross_t(
    prev: tuple[float, float],
    curr: tuple[float, float],
    line: Line,
    *,
    pad_px: float = 24.0,
    clearance_px: float = 8.0,
) -> float | None:
    """Crossing only when two observations change sides of the finite gate.

    A stationary box on the line is not a crossing. A slow step that actually
    changes sides is. Hover jitter of a couple of pixels is ignored.
    """
    if prev == curr:
        return None
    step = ((curr[0] - prev[0]) ** 2 + (curr[1] - prev[1]) ** 2) ** 0.5
    before = signed_offset(prev, line)
    after = signed_offset(curr, line)
    if before * after > 0.0:
        return None
    if step < 2.0 and abs(after - before) < clearance_px:
        return None
    return path_cross_t(prev, curr, line, pad_px=pad_px)


def path_crosses_line(
    prev: tuple[float, float],
    curr: tuple[float, float],
    line: Line,
    *,
    pad_px: float = 24.0,
) -> bool:
    """True when the track step from prev→curr crosses the (slightly padded) gate."""
    return path_cross_t(prev, curr, line, pad_px=pad_px) is not None


def _segment_t(
    a1: tuple[float, float],
    a2: tuple[float, float],
    b1: tuple[float, float],
    b2: tuple[float, float],
) -> float | None:
    ax, ay = a1
    bx, by = a2
    cx, cy = b1
    dx, dy = b2
    rx, ry = bx - ax, by - ay
    sx, sy = dx - cx, dy - cy
    den = rx * sy - ry * sx
    if abs(den) < 1e-9:
        return 0.0 if segments_intersect(a1, a2, b1, b2) else None
    t = ((cx - ax) * sy - (cy - ay) * sx) / den
    u = ((cx - ax) * ry - (cy - ay) * rx) / den
    if -1e-6 <= t <= 1 + 1e-6 and -1e-6 <= u <= 1 + 1e-6:
        return min(1.0, max(0.0, t))
    return None
