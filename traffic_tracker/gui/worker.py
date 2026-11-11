"""Background video processing so the GUI stays responsive."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal

from traffic_tracker.config import (
    DETECTOR_CONF,
    DETECTOR_IMGSZ,
    DETECTOR_IOU,
    DETECTOR_LINK_SPEED_PX_S,
    DETECTOR_PARKED_RADIUS,
    DETECTOR_PATH_JUMP,
    DETECTOR_STRIDE,
    EXPORT_SUBFOLDER_PATTERN,
    PREVIEW_EVERY,
)
from traffic_tracker.core.geometry import Geometry
from traffic_tracker.core.processor import ProcessResult, process_video
from traffic_tracker.core.report import Trip, export_tmc_workbook
from traffic_tracker.core.settings import export_workbook_path
from traffic_tracker.core.trainer import train_visdrone_detector


class ProcessWorker(QThread):
    progress = pyqtSignal(int, int, str)
    preview = pyqtSignal(object)
    video_done = pyqtSignal(object)
    failed = pyqtSignal(str)
    all_done = pyqtSignal(list)

    def __init__(
        self,
        jobs: list[tuple[Path, Geometry]],
        export_dir: Path,
        clock_start: datetime | None,
        parent: QObject | None = None,
        detector_mode: str = "motion",
        weights_path: str | None = None,
        detector_conf: float = DETECTOR_CONF,
        detector_iou: float = DETECTOR_IOU,
        parked_radius: float = DETECTOR_PARKED_RADIUS,
        link_speed: float = DETECTOR_LINK_SPEED_PX_S,
        path_jump: float = DETECTOR_PATH_JUMP,
        detector_imgsz: int = DETECTOR_IMGSZ,
        detect_stride: int = DETECTOR_STRIDE,
        preview_every: int = PREVIEW_EVERY,
        export_pattern: str = EXPORT_SUBFOLDER_PATTERN,
        theme: str = "dark",
        infer_trips_from_traces: bool | None = None,
    ) -> None:
        super().__init__(parent)
        self._jobs = jobs
        self._export_dir = export_dir
        self._export_pattern = export_pattern
        self._clock_start = clock_start
        self._detector_mode = detector_mode
        self._weights_path = weights_path
        self._detector_conf = detector_conf
        self._detector_iou = detector_iou
        self._parked_radius = parked_radius
        self._link_speed = link_speed
        self._path_jump = path_jump
        self._detector_imgsz = detector_imgsz
        self._detect_stride = detect_stride
        self._preview_every = preview_every
        self._theme = theme if theme in ("dark", "light") else "dark"
        self._infer_trips_from_traces = infer_trips_from_traces
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        results: list[ProcessResult] = []
        started_at = datetime.now()
        try:
            for video, geometry in self._jobs:
                if self._cancel:
                    break

                def progress_cb(current: int, total: int, message: str) -> None:
                    self.progress.emit(current, total, message)

                def preview_cb(frame: np.ndarray) -> None:
                    self.preview.emit(frame)

                result = process_video(
                    str(video),
                    geometry,
                    cancel_flag=lambda: self._cancel,
                    progress_cb=progress_cb,
                    preview_cb=preview_cb if self._preview_every > 0 else None,
                    preview_every=self._preview_every,
                    detector_mode=self._detector_mode,
                    weights_path=self._weights_path,
                    detector_conf=self._detector_conf,
                    detector_iou=self._detector_iou,
                    parked_radius=self._parked_radius,
                    link_speed=self._link_speed,
                    path_jump=self._path_jump,
                    detector_imgsz=self._detector_imgsz,
                    detect_stride=self._detect_stride,
                    theme=self._theme,
                    infer_trips_from_traces=self._infer_trips_from_traces,
                )
                if result.error and result.error != "Cancelled":
                    self.failed.emit(f"{video.name}: {result.error}")
                elif not result.error:
                    export_path = export_workbook_path(
                        self._export_dir,
                        self._export_pattern,
                        video.stem,
                        when=started_at,
                    )
                    export_tmc_workbook(
                        export_path,
                        result.trips,
                        geometry,
                        video_name=result.video.name,
                        duration_s=result.video.duration_s,
                        clock_start=self._clock_start,
                        device_label=result.device_label,
                        path_image=result.trace_image,
                        route_images=result.route_images,
                    )
                    result.export_path = export_path
                results.append(result)
                self.video_done.emit(result)
            self.all_done.emit(results)
        except Exception as exc:
            self.failed.emit(str(exc))
            self.all_done.emit(results)


class TrainWorker(QThread):
    progress = pyqtSignal(int, int, str)
    log_line = pyqtSignal(str)
    failed = pyqtSignal(str)
    train_done = pyqtSignal(str)

    def __init__(self, parent: QObject | None = None, source: str = "visdrone") -> None:
        super().__init__(parent)
        self._cancel = False
        _ = source

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            path = train_visdrone_detector(
                progress_cb=self.progress.emit,
                log_cb=self.log_line.emit,
                cancel_flag=lambda: self._cancel,
            )
            if self._cancel:
                self.failed.emit("Training cancelled.")
                return
            self.train_done.emit(str(path))
        except Exception as exc:
            self.failed.emit(str(exc))


def trips_to_rows(trips: list[Trip]) -> list[tuple]:
    return [
        (t.start_destination, f"{t.start_time_s:.1f}s", t.end_destination, f"{t.end_time_s:.1f}s")
        for t in trips
    ]
