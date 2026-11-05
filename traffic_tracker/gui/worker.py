"""Background video processing so the GUI stays responsive."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal

from traffic_tracker.core.geometry import Geometry
from traffic_tracker.core.processor import ProcessResult, process_video
from traffic_tracker.core.report import Trip, export_tmc_workbook
from traffic_tracker.core.trainer import train_site_detector, train_visdrone_detector


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
    ) -> None:
        super().__init__(parent)
        self._jobs = jobs
        self._export_dir = export_dir
        self._clock_start = clock_start
        self._detector_mode = detector_mode
        self._weights_path = weights_path
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        results: list[ProcessResult] = []
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
                    preview_cb=preview_cb,
                    detector_mode=self._detector_mode,
                    weights_path=self._weights_path,
                )
                if result.error and result.error != "Cancelled":
                    self.failed.emit(f"{video.name}: {result.error}")
                elif not result.error:
                    export_path = self._export_dir / f"{video.stem}_tmc.xlsx"
                    export_tmc_workbook(
                        export_path,
                        result.trips,
                        geometry,
                        video_name=result.video.name,
                        duration_s=result.video.duration_s,
                        clock_start=self._clock_start,
                        device_label=result.device_label,
                    )
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

    def __init__(self, parent: QObject | None = None, source: str = "site") -> None:
        super().__init__(parent)
        self._cancel = False
        self._source = source if source in ("site", "visdrone") else "site"

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            train_fn = train_visdrone_detector if self._source == "visdrone" else train_site_detector
            path = train_fn(
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
