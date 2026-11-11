"""Count gates named N, S, E, W from a user-set North heading."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json
import math

from traffic_tracker.config import (
    PREDICTION_REACH_MAX,
    PREDICTION_REACH_MIN,
    PREDICTION_REACH_PX,
    VEHICLE_REF_MAX_FACTOR,
    VEHICLE_REF_MIN_FACTOR,
)

ORIGINS = ("N", "S", "E", "W")
BOUNDS = ("NB", "SB", "EB", "WB")
ORIGIN_TO_BOUND = {"N": "NB", "S": "SB", "E": "EB", "W": "WB"}
BOUND_TO_ORIGIN = {bound: origin for origin, bound in ORIGIN_TO_BOUND.items()}
MAX_GATES = 4
GATE_FLOWS = ("both", "in", "out")
GATE_FLOW_LABELS = {"both": "Two-way", "in": "One-way in", "out": "One-way out"}
MAX_VEHICLE_REFS = 12
MAX_OCCLUSIONS = 12
MAX_ROADWAYS = 8
# Centers this close to a roadway edge still count as inside, so box jitter
# on the boundary does not create and destroy tracks.
ROADWAY_MARGIN_PX = 12.0


def clamp_prediction_reach(value: float) -> float:
    return max(PREDICTION_REACH_MIN, min(PREDICTION_REACH_MAX, float(value)))


def area_limits_from_areas(areas: list[float]) -> tuple[int, int] | None:
    valid = [area for area in areas if area >= 1]
    if not valid:
        return None
    return (
        max(50, int(min(valid) * VEHICLE_REF_MIN_FACTOR)),
        max(200, int(max(valid) * VEHICLE_REF_MAX_FACTOR)),
    )


def merge_area_limits(*pairs: tuple[int, int] | None) -> tuple[int, int] | None:
    mins = [pair[0] for pair in pairs if pair]
    maxs = [pair[1] for pair in pairs if pair]
    if not mins:
        return None
    return min(mins), max(maxs)


_NAME_HEADING = {"N": 0.0, "NB": 0.0, "E": 90.0, "EB": 90.0, "S": 180.0, "SB": 180.0, "W": 270.0, "WB": 270.0}


def movement_name(origin: str, bound: str) -> str:
    """Cardinal origin plus exit bound, e.g. N-SB."""
    return f"{origin}-{bound}"


def all_movement_names() -> list[str]:
    return [movement_name(origin, bound) for origin in ORIGINS for bound in BOUNDS]


def ordered_origins(names: list[str]) -> list[str]:
    return [name for name in ORIGINS if name in names]


def ordered_bounds(names: list[str]) -> list[str]:
    return [name for name in BOUNDS if name in names]


def gate_origin_name(name: str) -> str:
    """N/S/E/W for a count gate, including legacy Exit names like SB."""
    if name in ORIGINS:
        return name
    if name in BOUND_TO_ORIGIN:
        return BOUND_TO_ORIGIN[name]
    return name or "N"


def normalize_gate_flow(value: str | None) -> str:
    if value in GATE_FLOWS:
        return value
    return "both"


def image_heading_deg(dx: float, dy: float) -> float:
    """Heading in image space: 0 = up, clockwise, 0–360."""
    return math.degrees(math.atan2(dx, -dy)) % 360.0


@dataclass
class Line:
    x1: float
    y1: float
    x2: float
    y2: float

    def scaled(self, sx: float, sy: float) -> "Line":
        return Line(self.x1 * sx, self.y1 * sy, self.x2 * sx, self.y2 * sy)

    def midpoint(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    def length(self) -> float:
        return math.hypot(self.x2 - self.x1, self.y2 - self.y1)


def line_with_length(line: Line, length: float) -> Line:
    dx, dy = line.x2 - line.x1, line.y2 - line.y1
    mag = math.hypot(dx, dy) or 1.0
    ux, uy = dx / mag, dy / mag
    return Line(line.x1, line.y1, line.x1 + ux * length, line.y1 + uy * length)


@dataclass
class Box:
    """Axis-aligned vehicle mask used as the processing size reference."""

    x1: float
    y1: float
    x2: float
    y2: float
    label: str = "car"

    def normalized(self) -> "Box":
        return Box(
            min(self.x1, self.x2),
            min(self.y1, self.y2),
            max(self.x1, self.x2),
            max(self.y1, self.y2),
            label=self.label,
        )

    def width(self) -> float:
        return abs(self.x2 - self.x1)

    def height(self) -> float:
        return abs(self.y2 - self.y1)

    def area(self) -> float:
        return max(self.width() * self.height(), 0.0)

    def scaled(self, sx: float, sy: float) -> "Box":
        return Box(self.x1 * sx, self.y1 * sy, self.x2 * sx, self.y2 * sy, label=self.label)

    @classmethod
    def from_line(cls, line: Line) -> "Box":
        return cls(line.x1, line.y1, line.x2, line.y2).normalized()


@dataclass
class Polygon:
    """Allowed roadway region. An empty list on Geometry means the full frame."""

    points: list[tuple[float, float]] = field(default_factory=list)

    def scaled(self, sx: float, sy: float) -> "Polygon":
        return Polygon([(x * sx, y * sy) for x, y in self.points])


def _distinct_vertices(points: list[tuple[float, float]], tol: float = 1.5) -> list[tuple[float, float]]:
    cleaned: list[tuple[float, float]] = []
    for x, y in points:
        if cleaned and math.hypot(x - cleaned[-1][0], y - cleaned[-1][1]) <= tol:
            continue
        cleaned.append((float(x), float(y)))
    if len(cleaned) >= 2 and math.hypot(cleaned[0][0] - cleaned[-1][0], cleaned[0][1] - cleaned[-1][1]) <= tol:
        cleaned.pop()
    return cleaned


def polygon_area(points: list[tuple[float, float]]) -> float:
    verts = _distinct_vertices(points)
    if len(verts) < 3:
        return 0.0
    area = 0.0
    for index, (x, y) in enumerate(verts):
        nx, ny = verts[(index + 1) % len(verts)]
        area += x * ny - nx * y
    return abs(area) * 0.5


def polygon_is_valid(points: list[tuple[float, float]]) -> bool:
    """At least three distinct vertices and a non-degenerate area."""
    return len(_distinct_vertices(points)) >= 3 and polygon_area(points) >= 1.0


def point_in_polygon(x: float, y: float, points: list[tuple[float, float]]) -> bool:
    """Ray cast. Points on an edge count as inside."""
    verts = _distinct_vertices(points, tol=0.0) if len(points) >= 3 else list(points)
    if len(verts) < 3:
        return False
    inside = False
    j = len(verts) - 1
    for i, (xi, yi) in enumerate(verts):
        xj, yj = verts[j]
        if _point_on_segment(x, y, xi, yi, xj, yj):
            return True
        if (yi > y) != (yj > y):
            denom = (yj - yi) or 1e-12
            x_cross = (xj - xi) * (y - yi) / denom + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def _point_on_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float, tol: float = 1.5) -> bool:
    dx, dy = bx - ax, by - ay
    length2 = dx * dx + dy * dy
    if length2 < 1e-9:
        return math.hypot(px - ax, py - ay) <= tol
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length2))
    qx, qy = ax + t * dx, ay + t * dy
    return math.hypot(px - qx, py - qy) <= tol


def distance_to_polygon(x: float, y: float, points: list[tuple[float, float]]) -> float:
    """Zero when the point is inside; otherwise the distance to the nearest edge."""
    if point_in_polygon(x, y, points):
        return 0.0
    verts = points
    if len(verts) < 2:
        return float("inf")
    best = float("inf")
    for index, (ax, ay) in enumerate(verts):
        bx, by = verts[(index + 1) % len(verts)]
        dx, dy = bx - ax, by - ay
        length2 = dx * dx + dy * dy
        if length2 < 1e-9:
            best = min(best, math.hypot(x - ax, y - ay))
            continue
        t = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / length2))
        best = min(best, math.hypot(x - (ax + t * dx), y - (ay + t * dy)))
    return best


def point_in_roadway(x: float, y: float, geometry: "Geometry", margin: float = ROADWAY_MARGIN_PX) -> bool:
    """True when no roadway is drawn, or the point lies in the union (plus margin)."""
    if not geometry.roadways:
        return True
    for poly in geometry.roadways:
        if point_in_polygon(x, y, poly.points):
            return True
        if margin > 0.0 and distance_to_polygon(x, y, poly.points) <= margin:
            return True
    return False


def roadway_approach_warnings(geometry: "Geometry", probe_px: float = 48.0) -> list[str]:
    """Gates need observed roadway on both sides so a crossing can be confirmed."""
    if not geometry.roadways:
        return []
    warnings: list[str] = []
    labeled = [("Gate", item) for item in geometry.all_gates()]
    for kind, item in labeled:
        dx = item.line.x2 - item.line.x1
        dy = item.line.y2 - item.line.y1
        length = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / length, dx / length
        mx, my = item.line.midpoint()
        ahead = point_in_roadway(mx + nx * probe_px, my + ny * probe_px, geometry, margin=0.0)
        behind = point_in_roadway(mx - nx * probe_px, my - ny * probe_px, geometry, margin=0.0)
        if ahead and behind:
            continue
        warnings.append(
            f"{kind} {item.name} is clipped by the roadway. Leave roadway on both sides of the gate."
        )
    return warnings


DROP_ZONE_MAX_CAR_LENGTHS = 3.0


def typical_vehicle_length_px(geometry: "Geometry") -> float | None:
    if not geometry.vehicle_refs:
        return None
    return max(max(box.width(), box.height()) for box in geometry.vehicle_refs)


def drop_zone_too_wide(
    box: Box,
    geometry: "Geometry",
    *,
    car_lengths: float = DROP_ZONE_MAX_CAR_LENGTHS,
) -> bool:
    """True when a drop zone is many typical-vehicle lengths across."""
    typical = typical_vehicle_length_px(geometry)
    if typical is None or typical <= 1.0:
        return False
    return max(box.width(), box.height()) > car_lengths * typical


@dataclass
class NamedLine:
    name: str
    line: Line
    flow: str = "both"

    def __post_init__(self) -> None:
        self.flow = normalize_gate_flow(self.flow)

    def allows_entry(self) -> bool:
        return self.flow in ("both", "in")

    def allows_exit(self) -> bool:
        return self.flow in ("both", "out")


@dataclass
class Route:
    """Derived movement label, not a drawn pair."""

    name: str
    start: Line
    end: Line


@dataclass
class Transform:
    """Map reference-scaled lines onto a shifted camera frame."""

    tx: float = 0.0
    ty: float = 0.0
    scale: float = 1.0
    rotation_deg: float = 0.0
    homography: list[float] | None = None

    def is_identity(self) -> bool:
        if self.homography is not None:
            return False
        return (
            abs(self.tx) < 1e-6
            and abs(self.ty) < 1e-6
            and abs(self.scale - 1.0) < 1e-6
            and abs(self.rotation_deg) < 1e-6
        )

    def apply_point(self, x: float, y: float, cx: float, cy: float) -> tuple[float, float]:
        if self.homography is not None and len(self.homography) == 9:
            h = self.homography
            num_x = h[0] * x + h[1] * y + h[2]
            num_y = h[3] * x + h[4] * y + h[5]
            den = h[6] * x + h[7] * y + h[8]
            if abs(den) < 1e-9:
                return x, y
            return num_x / den, num_y / den
        dx = (x - cx) * self.scale
        dy = (y - cy) * self.scale
        rad = math.radians(self.rotation_deg)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        rx = dx * cos_a - dy * sin_a
        ry = dx * sin_a + dy * cos_a
        return rx + cx + self.tx, ry + cy + self.ty

    def apply_line(self, line: Line, cx: float, cy: float) -> Line:
        x1, y1 = self.apply_point(line.x1, line.y1, cx, cy)
        x2, y2 = self.apply_point(line.x2, line.y2, cx, cy)
        return Line(x1, y1, x2, y2)

    def apply_box(self, box: Box, cx: float, cy: float) -> Box:
        x1, y1 = self.apply_point(box.x1, box.y1, cx, cy)
        x2, y2 = self.apply_point(box.x2, box.y2, cx, cy)
        return Box(x1, y1, x2, y2).normalized()

    def apply_polygon(self, poly: Polygon, cx: float, cy: float) -> Polygon:
        return Polygon([self.apply_point(x, y, cx, cy) for x, y in poly.points])

    def apply_geometry(self, geom: "Geometry") -> "Geometry":
        cx = geom.frame_width / 2.0
        cy = geom.frame_height / 2.0
        north = None
        if geom.north is not None:
            north = self.apply_line(geom.north, cx, cy)
        reach = None
        reach_px = geom.prediction_reach_px
        if geom.prediction_reach is not None:
            reach = self.apply_line(geom.prediction_reach, cx, cy)
            reach_px = clamp_prediction_reach(reach.length())
        else:
            x1, y1 = self.apply_point(0.0, 0.0, cx, cy)
            x2, y2 = self.apply_point(geom.prediction_reach_px, 0.0, cx, cy)
            reach_px = clamp_prediction_reach(math.hypot(x2 - x1, y2 - y1))
        return Geometry(
            frame_width=geom.frame_width,
            frame_height=geom.frame_height,
            north=north,
            starts=[NamedLine(item.name, self.apply_line(item.line, cx, cy), item.flow) for item in geom.starts],
            ends=[NamedLine(item.name, self.apply_line(item.line, cx, cy), item.flow) for item in geom.ends],
            vehicle_refs=[self.apply_box(box, cx, cy) for box in geom.vehicle_refs],
            occlusions=[self.apply_box(box, cx, cy) for box in geom.occlusions],
            roadways=[self.apply_polygon(poly, cx, cy) for poly in geom.roadways],
            prediction_reach_px=reach_px,
            prediction_reach=reach,
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "tx": self.tx,
            "ty": self.ty,
            "scale": self.scale,
            "rotation_deg": self.rotation_deg,
        }
        if self.homography is not None:
            data["homography"] = list(self.homography)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "Transform":
        if not data:
            return cls()
        homo = data.get("homography")
        return cls(
            tx=float(data.get("tx", 0.0)),
            ty=float(data.get("ty", 0.0)),
            scale=float(data.get("scale", 1.0)),
            rotation_deg=float(data.get("rotation_deg", 0.0)),
            homography=[float(v) for v in homo] if homo else None,
        )


def transform_from_matches(
    src_points: list[tuple[float, float]],
    dst_points: list[tuple[float, float]],
) -> Transform:
    if len(src_points) < 2 or len(src_points) != len(dst_points):
        raise ValueError("Need at least two matched landmark pairs.")
    import cv2
    import numpy as np

    src = np.float32(src_points)
    dst = np.float32(dst_points)
    if len(src_points) >= 4:
        matrix, _ = cv2.findHomography(src, dst, method=0)
        if matrix is None:
            raise ValueError("Could not estimate a homography from those points.")
        return Transform(homography=[float(v) for v in matrix.flatten()])

    matrix, _ = cv2.estimateAffinePartial2D(src, dst)
    if matrix is None:
        raise ValueError("Could not estimate a similarity from those points.")
    a, b, tx = float(matrix[0, 0]), float(matrix[0, 1]), float(matrix[0, 2])
    c, d, ty = float(matrix[1, 0]), float(matrix[1, 1]), float(matrix[1, 2])
    scale = math.hypot(a, c)
    rotation_deg = math.degrees(math.atan2(c, a)) if scale > 1e-6 else 0.0
    homo = [a, b, tx, c, d, ty, 0.0, 0.0, 1.0]
    return Transform(tx=tx, ty=ty, scale=scale or 1.0, rotation_deg=rotation_deg, homography=homo)


def _angle_error(heading: float, target: float) -> float:
    return abs((heading - target + 180.0) % 360.0 - 180.0)


def intersection_center(north: Line | None, frame_w: int, frame_h: int) -> tuple[float, float]:
    """Intersection is the tail of the North arrow; otherwise the frame center."""
    if north is not None:
        return north.x1, north.y1
    return frame_w / 2.0, frame_h / 2.0


def heading_from_north(x: float, y: float, north: Line, frame_w: int, frame_h: int) -> float:
    """0 = toward North, clockwise, from the intersection center."""
    cx, cy = intersection_center(north, frame_w, frame_h)
    north_heading = image_heading_deg(north.x2 - north.x1, north.y2 - north.y1)
    return (image_heading_deg(x - cx, y - cy) - north_heading) % 360.0


def side_from_point(
    x: float,
    y: float,
    north: Line,
    frame_w: int,
    frame_h: int,
) -> str:
    """Geographic side of a point: N, S, E, or W from the North arrow."""
    heading = heading_from_north(x, y, north, frame_w, frame_h)
    return min(ORIGINS, key=lambda name: _angle_error(heading, _NAME_HEADING[name]))


def name_from_north(
    line: Line,
    names: tuple[str, ...],
    north: Line,
    frame_w: int,
    frame_h: int,
    taken: set[str] | None = None,
) -> str:
    """Nearest cardinal to this line. Unused sides are skipped, not filled in order."""
    mx, my = line.midpoint()
    heading = heading_from_north(mx, my, north, frame_w, frame_h)
    taken = taken or set()
    available = [name for name in names if name in _NAME_HEADING and name not in taken] or [
        name for name in names if name in _NAME_HEADING
    ]
    return min(available, key=lambda name: _angle_error(heading, _NAME_HEADING[name]))


def _assign_by_north(lines: list[Line], names: tuple[str, ...], north: Line, frame_w: int, frame_h: int) -> list[NamedLine]:
    """Map 1–4 lines to the nearest unused cardinal from North, not draw order."""
    if not lines:
        return []
    scored: list[tuple[float, Line]] = []
    for line in lines:
        mx, my = line.midpoint()
        scored.append((heading_from_north(mx, my, north, frame_w, frame_h), line))

    unused = [name for name in names if name in _NAME_HEADING]
    remaining = set(range(len(scored)))
    assigned: list[NamedLine | None] = [None] * len(scored)
    while remaining and unused:
        best_i = None
        best_name = None
        best_err = 1e9
        for index in remaining:
            heading = scored[index][0]
            for name in unused:
                err = _angle_error(heading, _NAME_HEADING[name])
                if err < best_err:
                    best_i = index
                    best_name = name
                    best_err = err
        if best_i is None or best_name is None:
            break
        assigned[best_i] = NamedLine(best_name, scored[best_i][1])
        remaining.remove(best_i)
        unused.remove(best_name)
    return [item for item in assigned if item is not None]


@dataclass
class Geometry:
    """North heading plus up to four count gates (N, S, E, W)."""

    frame_width: int
    frame_height: int
    north: Line | None = None
    starts: list[NamedLine] = field(default_factory=list)
    ends: list[NamedLine] = field(default_factory=list)
    vehicle_refs: list[Box] = field(default_factory=list)
    occlusions: list[Box] = field(default_factory=list)
    roadways: list[Polygon] = field(default_factory=list)
    prediction_reach_px: float = PREDICTION_REACH_PX
    prediction_reach: Line | None = None

    def __post_init__(self) -> None:
        self.prediction_reach_px = clamp_prediction_reach(self.prediction_reach_px)

    def set_prediction_reach(self, px: float, line: Line | None = None) -> None:
        self.prediction_reach_px = clamp_prediction_reach(px)
        source = line if line is not None else self.prediction_reach
        if source is not None:
            self.prediction_reach = line_with_length(source, self.prediction_reach_px)

    def sample_prediction_reach_line(self) -> Line | None:
        """Drawn reach line, or a sample arrow of length D next to a drop zone / vehicle."""
        if self.prediction_reach is not None:
            return self.prediction_reach
        d = self.prediction_reach_px
        if self.occlusions:
            box = self.occlusions[0].normalized()
            y = (box.y1 + box.y2) / 2.0
            return Line(box.x1, y, box.x1 + d, y)
        if self.vehicle_refs:
            box = self.vehicle_refs[0].normalized()
            y = (box.y1 + box.y2) / 2.0
            return Line(box.x2 + 8.0, y, box.x2 + 8.0 + d, y)
        return None

    def all_gates(self) -> list[NamedLine]:
        return list(self.starts) + list(self.ends)

    def gate_count(self) -> int:
        return len(self.starts) + len(self.ends)

    @property
    def routes(self) -> list[Route]:
        """Synthetic pairs for overlays that still expect Route objects."""
        pairs: list[Route] = []
        if self.starts and self.ends:
            for start in self.starts:
                for end in self.ends:
                    pairs.append(Route(movement_name(start.name, end.name), start.line, end.line))
            return pairs
        gates = self.all_gates()
        for start in gates:
            oa = gate_origin_name(start.name)
            if oa not in ORIGIN_TO_BOUND:
                continue
            for end in gates:
                if start is end:
                    continue
                ob = gate_origin_name(end.name)
                if ob not in ORIGIN_TO_BOUND or oa == ob:
                    continue
                pairs.append(Route(movement_name(oa, ORIGIN_TO_BOUND[ob]), start.line, end.line))
        return pairs

    def is_ready(self) -> bool:
        return self.north is not None and self.gate_count() >= 2

    def origin_names(self) -> list[str]:
        if self.starts and self.ends:
            return ordered_origins([item.name for item in self.starts if item.name and item.allows_entry()])
        return ordered_origins(
            [gate_origin_name(item.name) for item in self.all_gates() if item.name and item.allows_entry()]
        )

    def bound_names(self) -> list[str]:
        if self.ends:
            return ordered_bounds([item.name for item in self.ends if item.name and item.allows_exit()])
        return ordered_bounds(
            [
                ORIGIN_TO_BOUND[gate_origin_name(item.name)]
                for item in self.all_gates()
                if item.name and item.allows_exit() and gate_origin_name(item.name) in ORIGIN_TO_BOUND
            ]
        )

    def movement_names(self) -> list[str]:
        origins = self.origin_names()
        bounds = self.bound_names()
        if self.starts and self.ends:
            return [movement_name(origin, bound) for origin in origins for bound in bounds]
        return [
            movement_name(origin, bound)
            for origin in origins
            for bound in bounds
            if ORIGIN_TO_BOUND.get(origin) != bound
        ]

    def vehicle_area_limits(self, extra_areas: list[float] | None = None) -> tuple[int, int] | None:
        """Min/max blob area from drawn masks plus optional training-box areas."""
        areas = [box.area() for box in self.vehicle_refs if box.area() >= 1]
        if extra_areas:
            areas.extend(a for a in extra_areas if a >= 1)
        return area_limits_from_areas(areas)

    def assign_names(self) -> None:
        if self.north is None:
            return
        start_flows = [item.flow for item in self.starts]
        end_flows = [item.flow for item in self.ends]
        start_lines = [item.line for item in self.starts]
        end_lines = [item.line for item in self.ends]
        if start_lines:
            self.starts = _assign_by_north(start_lines, ORIGINS, self.north, self.frame_width, self.frame_height)
            for item, flow in zip(self.starts, start_flows):
                item.flow = flow
        if end_lines:
            self.ends = _assign_by_north(end_lines, BOUNDS, self.north, self.frame_width, self.frame_height)
            for item, flow in zip(self.ends, end_flows):
                item.flow = flow

    def add_gate(self, kind: str, line: Line, name: str | None = None) -> NamedLine | None:
        """Add one count gate. New drawings use N/S/E/W; ``end`` keeps legacy NB/SB names."""
        if kind not in ("start", "end") or self.north is None:
            return None
        pool = self.starts if kind == "start" else self.ends
        names = ORIGINS if kind == "start" else BOUNDS
        if len(pool) >= MAX_GATES:
            return None
        taken = {item.name for item in pool if item.name}
        if kind == "start":
            taken |= {gate_origin_name(item.name) for item in self.ends if item.name}
        chosen = name if name in names else name_from_north(
            line, names, self.north, self.frame_width, self.frame_height, taken
        )
        if chosen in taken:
            return None
        item = NamedLine(chosen, line)
        pool.append(item)
        return item

    def scaled_to(self, width: int, height: int) -> "Geometry":
        if width == self.frame_width and height == self.frame_height:
            return self
        if self.frame_width <= 0 or self.frame_height <= 0:
            return self
        sx = width / self.frame_width
        sy = height / self.frame_height
        north = self.north.scaled(sx, sy) if self.north else None
        reach = self.prediction_reach.scaled(sx, sy) if self.prediction_reach else None
        if reach is not None:
            reach_px = clamp_prediction_reach(reach.length())
        else:
            reach_px = clamp_prediction_reach(self.prediction_reach_px * ((abs(sx) + abs(sy)) / 2.0))
        return Geometry(
            frame_width=width,
            frame_height=height,
            north=north,
            starts=[NamedLine(item.name, item.line.scaled(sx, sy), item.flow) for item in self.starts],
            ends=[NamedLine(item.name, item.line.scaled(sx, sy), item.flow) for item in self.ends],
            vehicle_refs=[box.scaled(sx, sy) for box in self.vehicle_refs],
            occlusions=[box.scaled(sx, sy) for box in self.occlusions],
            roadways=[poly.scaled(sx, sy) for poly in self.roadways],
            prediction_reach_px=reach_px,
            prediction_reach=reach,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "north": asdict(self.north) if self.north else None,
            "starts": [{"name": item.name, "line": asdict(item.line), "flow": item.flow} for item in self.starts],
            "ends": [{"name": item.name, "line": asdict(item.line), "flow": item.flow} for item in self.ends],
            "vehicle_refs": [asdict(box) for box in self.vehicle_refs],
            "occlusions": [asdict(box) for box in self.occlusions],
            "roadways": [{"points": [[x, y] for x, y in poly.points]} for poly in self.roadways],
            "prediction_reach_px": self.prediction_reach_px,
            "prediction_reach": asdict(self.prediction_reach) if self.prediction_reach else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Geometry":
        north = Line(**data["north"]) if data.get("north") else None
        starts = [
            NamedLine(str(item["name"]), Line(**item["line"]), normalize_gate_flow(item.get("flow")))
            for item in data.get("starts", [])
        ]
        ends = [
            NamedLine(str(item["name"]), Line(**item["line"]), normalize_gate_flow(item.get("flow")))
            for item in data.get("ends", [])
        ]
        refs = [Box(**item).normalized() for item in data.get("vehicle_refs") or []]
        if not refs and data.get("vehicle_ref"):
            refs = [Box(**data["vehicle_ref"]).normalized()]
        occlusions = [Box(**item).normalized() for item in data.get("occlusions") or []]
        roadways: list[Polygon] = []
        for item in data.get("roadways") or []:
            raw_points = item.get("points") if isinstance(item, dict) else item
            points = [(float(pt[0]), float(pt[1])) for pt in (raw_points or []) if len(pt) >= 2]
            if polygon_is_valid(points):
                roadways.append(Polygon(_distinct_vertices(points)))
        reach_raw = data.get("prediction_reach")
        reach = Line(**reach_raw) if isinstance(reach_raw, dict) else None
        reach_px = data.get("prediction_reach_px")
        if reach_px is None and reach is not None:
            reach_px = reach.length()
        geom = cls(
            frame_width=int(data.get("frame_width", 1)),
            frame_height=int(data.get("frame_height", 1)),
            north=north,
            starts=starts,
            ends=ends,
            vehicle_refs=refs,
            occlusions=occlusions,
            roadways=roadways,
            prediction_reach_px=float(reach_px) if reach_px is not None else PREDICTION_REACH_PX,
            prediction_reach=reach,
        )
        # Legacy pair files: treat each pair as unnamed start/end segments.
        if not starts and data.get("routes"):
            for item in data["routes"]:
                geom.starts.append(NamedLine(str(item.get("name", "start")), Line(**item["start"])))
                geom.ends.append(NamedLine(str(item.get("name", "end")), Line(**item["end"])))
        return geom

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Geometry":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
