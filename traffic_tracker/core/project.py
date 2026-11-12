"""Bundle project: reference routes plus per-clip alignment state."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
import json

from traffic_tracker.core.geometry import Geometry, Transform
from traffic_tracker.core.video import clamp_clip_window

ClipStatus = Literal["reference", "needs_align", "aligned", "redrawn"]


@dataclass
class ProcessJob:
    path: Path
    geometry: Geometry
    start_s: float = 0.0
    end_s: float | None = None
    duration_s: float = 0.0


@dataclass
class ClipRecord:
    path: Path
    status: ClipStatus = "needs_align"
    transform: Transform = field(default_factory=Transform)
    geometry_override: Geometry | None = None
    process_start_s: float = 0.0
    process_end_s: float | None = None

    def window(self, duration_s: float) -> tuple[float, float]:
        return clamp_clip_window(self.process_start_s, self.process_end_s, duration_s)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "path": str(self.path),
            "status": self.status,
            "transform": self.transform.to_dict(),
            "process_start_s": self.process_start_s,
        }
        if self.process_end_s is not None:
            data["process_end_s"] = self.process_end_s
        if self.geometry_override is not None:
            data["geometry_override"] = self.geometry_override.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClipRecord":
        override = data.get("geometry_override")
        status = data.get("status", "needs_align")
        if status not in ("reference", "needs_align", "aligned", "redrawn"):
            status = "needs_align"
        end_raw = data.get("process_end_s")
        try:
            start_s = float(data.get("process_start_s") or 0.0)
        except (TypeError, ValueError):
            start_s = 0.0
        try:
            end_s = float(end_raw) if end_raw is not None else None
        except (TypeError, ValueError):
            end_s = None
        return cls(
            path=Path(data["path"]),
            status=status,
            transform=Transform.from_dict(data.get("transform")),
            geometry_override=Geometry.from_dict(override) if override else None,
            process_start_s=max(0.0, start_s),
            process_end_s=None if end_s is None else max(0.0, end_s),
        )


@dataclass
class BundleProject:
    reference: Path | None = None
    geometry: Geometry = field(default_factory=lambda: Geometry(1, 1))
    clips: list[ClipRecord] = field(default_factory=list)

    def clip_for(self, path: Path) -> ClipRecord | None:
        resolved = path.resolve()
        for clip in self.clips:
            if clip.path.resolve() == resolved:
                return clip
        return None

    def ensure_clips(self, videos: list[Path]) -> None:
        existing = {clip.path.resolve(): clip for clip in self.clips}
        ordered: list[ClipRecord] = []
        for index, video in enumerate(videos):
            key = video.resolve()
            record = existing.get(key)
            if record is None:
                status: ClipStatus = "reference" if index == 0 else "needs_align"
                record = ClipRecord(path=video, status=status)
            elif index == 0:
                record.status = "reference"
            ordered.append(record)
        self.clips = ordered
        if videos:
            self.reference = videos[0]

    def ready_jobs(self, reference_geometry: Geometry) -> list[ProcessJob]:
        """Confirmed clips with geometry ready for processing."""
        jobs: list[ProcessJob] = []
        for clip in self.clips:
            if clip.status == "needs_align":
                continue
            duration_s = 0.0
            probe_w, probe_h = reference_geometry.frame_width, reference_geometry.frame_height
            try:
                from traffic_tracker.core.video import probe_video

                info = probe_video(clip.path)
                probe_w, probe_h = info.width, info.height
                duration_s = info.duration_s
            except Exception:
                pass
            start_s, end_s = clip.window(duration_s)
            if clip.status == "redrawn" and clip.geometry_override is not None:
                jobs.append(ProcessJob(clip.path, clip.geometry_override, start_s, end_s, duration_s))
                continue
            fitted = reference_geometry.scaled_to(probe_w, probe_h)
            jobs.append(
                ProcessJob(clip.path, clip.transform.apply_geometry(fitted), start_s, end_s, duration_s)
            )
        return jobs

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference": str(self.reference) if self.reference else "",
            "geometry": self.geometry.to_dict(),
            "clips": [clip.to_dict() for clip in self.clips],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BundleProject":
        ref = data.get("reference") or None
        geom = data.get("geometry") or {"frame_width": 1, "frame_height": 1, "routes": []}
        return cls(
            reference=Path(ref) if ref else None,
            geometry=Geometry.from_dict(geom),
            clips=[ClipRecord.from_dict(item) for item in data.get("clips", [])],
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BundleProject":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)
