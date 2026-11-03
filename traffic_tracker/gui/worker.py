"""Background video processing so the GUI stays responsive."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal

from traffic_tracker.core.geometry import Geometry
from traffic_tracker.core.processor import ProcessResult, process_video
from traffic_tracker.core.report import Trip, build_tmc_table, export_tmc_workbook


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
    ) -> None:
        super().__init__(parent)
        self._jobs = jobs
        self._export_dir = export_dir
        self._clock_start = clock_start
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
                )
                if result.error and result.error != "Cancelled":
                    self.failed.emit(f"{video.name}: {result.error}")
                elif not result.error:
                    table = build_tmc_table(
                        result.trips,
                        [route.name for route in geometry.routes],
                        result.video.duration_s,
                        self._clock_start,
                    )
                    export_path = self._export_dir / f"{video.stem}_tmc.xlsx"
                    export_tmc_workbook(
                        export_path,
                        table,
                        video_name=result.video.name,
                        duration_s=result.video.duration_s,
                        clock_start=self._clock_start,
                        device_label=result.device_label,
                        trip_count=len(result.trips),
                    )
                results.append(result)
                self.video_done.emit(result)
            self.all_done.emit(results)
        except Exception as exc:
            self.failed.emit(str(exc))
            self.all_done.emit(results)


def trips_to_rows(trips: list[Trip]) -> list[tuple]:
    return [(t.video_name, t.route_name, t.track_id, t.vehicle_class, f"{t.completed_at_s:.1f}s") for t in trips]
