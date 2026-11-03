"""Named start/end line pairs that define intersection routes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json
import math


@dataclass
class Line:
    x1: float
    y1: float
    x2: float
    y2: float

    def scaled(self, sx: float, sy: float) -> "Line":
        return Line(self.x1 * sx, self.y1 * sy, self.x2 * sx, self.y2 * sy)


@dataclass
class Route:
    name: str
    start: Line
    end: Line


@dataclass
class Transform:
    """Map reference-scaled lines onto a shifted camera frame.

    Similarity (translate / scale / rotate about the frame center) is the
    default. A 3x3 homography, when set, replaces that similarity.
    """

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
        return Geometry(
            frame_width=geom.frame_width,
            frame_height=geom.frame_height,
            routes=[
                Route(
                    name=route.name,
                    start=self.apply_line(route.start, cx, cy),
                    end=self.apply_line(route.end, cx, cy),
                )
                for route in geom.routes
            ],
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
    """Build a transform from landmark pairs (source → destination)."""
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
    # estimateAffinePartial2D is about the origin, not the frame center.
    # Store as homography so apply_point stays exact.
    homo = [a, b, tx, c, d, ty, 0.0, 0.0, 1.0]
    return Transform(tx=tx, ty=ty, scale=scale or 1.0, rotation_deg=rotation_deg, homography=homo)


@dataclass
class Geometry:
    """Routes drawn on a reference frame of known pixel size."""

    frame_width: int
    frame_height: int
    routes: list[Route] = field(default_factory=list)

    def scaled_to(self, width: int, height: int) -> "Geometry":
        if width == self.frame_width and height == self.frame_height:
            return self
        if self.frame_width <= 0 or self.frame_height <= 0:
            return self
        sx = width / self.frame_width
        sy = height / self.frame_height
        return Geometry(
            frame_width=width,
            frame_height=height,
            routes=[
                Route(name=route.name, start=route.start.scaled(sx, sy), end=route.end.scaled(sx, sy))
                for route in self.routes
            ],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "routes": [
                {
                    "name": route.name,
                    "start": asdict(route.start),
                    "end": asdict(route.end),
                }
                for route in self.routes
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Geometry":
        routes = []
        for item in data.get("routes", []):
            routes.append(
                Route(
                    name=str(item["name"]),
                    start=Line(**item["start"]),
                    end=Line(**item["end"]),
                )
            )
        return cls(
            frame_width=int(data["frame_width"]),
            frame_height=int(data["frame_height"]),
            routes=routes,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Geometry":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)
