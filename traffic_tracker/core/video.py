"""Small OpenCV helpers for probing videos and grabbing a drawing frame."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import cv2
import numpy as np

from traffic_tracker.config import RANDOM_FRAME_MARGIN, SUPPORTED_VIDEO_SUFFIXES


@dataclass
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: float

    @property
    def name(self) -> str:
        return self.path.name


def is_video_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_VIDEO_SUFFIXES


def list_videos_in_folder(folder: str | Path) -> list[Path]:
    root = Path(folder)
    videos = [p for p in root.iterdir() if p.is_file() and is_video_file(p)]
    return sorted(videos, key=lambda p: p.name.lower())


def probe_video(path: str | Path) -> VideoInfo:
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 1e-3:
            fps = 30.0
        duration_s = frame_count / fps if frame_count > 0 else 0.0
        return VideoInfo(
            path=path,
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration_s=duration_s,
        )
    finally:
        cap.release()


def grab_random_frame(path: str | Path, rng: random.Random | None = None) -> tuple[np.ndarray, int]:
    """Return (BGR frame, frame index) from the middle portion of the video."""
    info = probe_video(path)
    if info.frame_count <= 1:
        cap = cv2.VideoCapture(str(path))
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise RuntimeError(f"Could not read a frame from: {path}")
        return frame, 0

    margin = int(info.frame_count * RANDOM_FRAME_MARGIN)
    low = min(margin, info.frame_count - 1)
    high = max(low, info.frame_count - 1 - margin)
    picker = rng or random.Random()
    index = picker.randint(low, high)

    cap = cv2.VideoCapture(str(path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
            index = 0
        if not ok or frame is None:
            raise RuntimeError(f"Could not read a frame from: {path}")
        return frame, index
    finally:
        cap.release()
