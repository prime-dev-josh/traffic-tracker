"""Named start/end line pairs that define intersection routes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json


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
