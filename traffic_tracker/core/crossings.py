"""Count a gate crossing when a track's path intersects the drawn segment."""

from __future__ import annotations

from traffic_tracker.core.geometry import Line


def bbox_center(xyxy) -> tuple[float, float]:
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


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
