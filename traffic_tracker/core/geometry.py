"""Entry/exit lines named from a user-set North heading."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json
import math

ORIGINS = ("N", "S", "E", "W")
BOUNDS = ("NB", "SB", "EB", "WB")
ORIGIN_TO_BOUND = {"N": "NB", "S": "SB", "E": "EB", "W": "WB"}
MAX_GATES = 4
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


@dataclass
class NamedLine:
    name: str
    line: Line


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

    def apply_geometry(self, geom: "Geometry") -> "Geometry":
        cx = geom.frame_width / 2.0
        cy = geom.frame_height / 2.0
        north = None
        if geom.north is not None:
            north = self.apply_line(geom.north, cx, cy)
        return Geometry(
            frame_width=geom.frame_width,
            frame_height=geom.frame_height,
            north=north,
            starts=[NamedLine(item.name, self.apply_line(item.line, cx, cy)) for item in geom.starts],
            ends=[NamedLine(item.name, self.apply_line(item.line, cx, cy)) for item in geom.ends],
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
    """North heading plus four entry lines and four exit lines."""

    frame_width: int
    frame_height: int
    north: Line | None = None
    starts: list[NamedLine] = field(default_factory=list)
    ends: list[NamedLine] = field(default_factory=list)

    @property
    def routes(self) -> list[Route]:
        """Synthetic pairs for overlays that still expect Route objects."""
        pairs: list[Route] = []
        for start in self.starts:
            for end in self.ends:
                pairs.append(Route(movement_name(start.name, end.name), start.line, end.line))
        return pairs

    def is_ready(self) -> bool:
        return self.north is not None and len(self.starts) >= 1 and len(self.ends) >= 1

    def origin_names(self) -> list[str]:
        return ordered_origins([item.name for item in self.starts if item.name])

    def bound_names(self) -> list[str]:
        return ordered_bounds([item.name for item in self.ends if item.name])

    def movement_names(self) -> list[str]:
        return [movement_name(origin, bound) for origin in self.origin_names() for bound in self.bound_names()]

    def assign_names(self) -> None:
        if self.north is None:
            return
        start_lines = [item.line for item in self.starts]
        end_lines = [item.line for item in self.ends]
        if start_lines:
            self.starts = _assign_by_north(start_lines, ORIGINS, self.north, self.frame_width, self.frame_height)
        if end_lines:
            self.ends = _assign_by_north(end_lines, BOUNDS, self.north, self.frame_width, self.frame_height)

    def add_gate(self, kind: str, line: Line, name: str | None = None) -> NamedLine | None:
        """Add one entry or exit. ``name`` is used as-is; otherwise derive from North."""
        if kind not in ("start", "end") or self.north is None:
            return None
        pool = self.starts if kind == "start" else self.ends
        names = ORIGINS if kind == "start" else BOUNDS
        if len(pool) >= MAX_GATES:
            return None
        taken = {item.name for item in pool if item.name}
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
        return Geometry(
            frame_width=width,
            frame_height=height,
            north=north,
            starts=[NamedLine(item.name, item.line.scaled(sx, sy)) for item in self.starts],
            ends=[NamedLine(item.name, item.line.scaled(sx, sy)) for item in self.ends],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "north": asdict(self.north) if self.north else None,
            "starts": [{"name": item.name, "line": asdict(item.line)} for item in self.starts],
            "ends": [{"name": item.name, "line": asdict(item.line)} for item in self.ends],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Geometry":
        north = Line(**data["north"]) if data.get("north") else None
        starts = [NamedLine(str(item["name"]), Line(**item["line"])) for item in data.get("starts", [])]
        ends = [NamedLine(str(item["name"]), Line(**item["line"])) for item in data.get("ends", [])]
        geom = cls(
            frame_width=int(data.get("frame_width", 1)),
            frame_height=int(data.get("frame_height", 1)),
            north=north,
            starts=starts,
            ends=ends,
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
