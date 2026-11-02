"""Four-step desktop flow: videos → routes → process → report."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import (
    QCheckBox,
    QDateTimeEdit,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from traffic_tracker.config import APP_NAME, APP_VERSION
from traffic_tracker.core.device import describe_device, detect_device
from traffic_tracker.core.geometry import Geometry, Line, Route
from traffic_tracker.core.processor import ProcessResult
from traffic_tracker.core.report import build_tmc_table
from traffic_tracker.core.video import grab_random_frame, is_video_file, list_videos_in_folder, probe_video
from traffic_tracker.gui.widgets.frame_canvas import FrameCanvas
from traffic_tracker.gui.worker import ProcessWorker


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} {APP_VERSION}")
        self.resize(1180, 760)

        self.videos: list[Path] = []
        self.geometry = Geometry(frame_width=1, frame_height=1)
        self.export_dir = Path.cwd() / "outputs"
        self.worker: ProcessWorker | None = None
        self.results: list[ProcessResult] = []
        self._route_name = ""
        self._pending_start: Line | None = None
        self._draw_step = "idle"

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        header = QVBoxLayout()
        title = QLabel(APP_NAME)
        title.setObjectName("title")
        subtitle = QLabel("Count vehicles across user-drawn start and end lines, then export a 15-minute TMC.")
        subtitle.setObjectName("subtitle")
        header.addWidget(title)
        header.addWidget(subtitle)
        layout.addLayout(header)

        self.steps = QLabel("1. Videos   →   2. Routes   →   3. Process   →   4. Report")
        self.steps.setObjectName("hint")
        layout.addWidget(self.steps)

        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_setup_page())
        self.stack.addWidget(self._build_routes_page())
        self.stack.addWidget(self._build_process_page())
        self.stack.addWidget(self._build_report_page())
        layout.addWidget(self.stack, 1)

        self.statusBar().showMessage("Ready")
        QTimer.singleShot(0, self._refresh_device_status)

    def _card(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("card")
        return frame

    def _build_setup_page(self) -> QWidget:
        page = QWidget()
        row = QHBoxLayout(page)
        row.setSpacing(16)

        left = self._card()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(self._step_title("1. Add intersection videos"))
        hint = QLabel("Add one video, or a folder of videos that share the same camera view.")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        left_layout.addWidget(hint)

        self.video_list = QListWidget()
        left_layout.addWidget(self.video_list, 1)

        buttons = QHBoxLayout()
        add_video = QPushButton("Add video")
        add_folder = QPushButton("Add folder")
        remove = QPushButton("Remove")
        add_video.clicked.connect(self._add_videos)
        add_folder.clicked.connect(self._add_folder)
        remove.clicked.connect(self._remove_selected_videos)
        buttons.addWidget(add_video)
        buttons.addWidget(add_folder)
        buttons.addWidget(remove)
        left_layout.addLayout(buttons)

        self.use_clock = QCheckBox("Use real clock time for 15-min bins")
        self.clock_edit = QDateTimeEdit()
        self.clock_edit.setDisplayFormat("yyyy-MM-dd HH:mm")
        self.clock_edit.setCalendarPopup(True)
        self.clock_edit.setDateTime(datetime.now())
        self.clock_edit.setEnabled(False)
        self.use_clock.toggled.connect(self.clock_edit.setEnabled)
        left_layout.addWidget(self.use_clock)
        left_layout.addWidget(self.clock_edit)

        export_row = QHBoxLayout()
        self.export_label = QLabel(f"Export folder: {self.export_dir}")
        self.export_label.setObjectName("hint")
        self.export_label.setWordWrap(True)
        choose_export = QPushButton("Export folder")
        choose_export.clicked.connect(self._choose_export_dir)
        export_row.addWidget(self.export_label, 1)
        export_row.addWidget(choose_export)
        left_layout.addLayout(export_row)

        next_btn = QPushButton("Next: draw routes")
        next_btn.setObjectName("primary")
        next_btn.clicked.connect(self._go_routes)
        left_layout.addWidget(next_btn)

        row.addWidget(left, 1)

        right = self._card()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(self._step_title("How a count works"))
        help_text = QLabel(
            "After the next step, a random frame is shown so you can draw a start line "
            "and an end line for every movement (for example NB Left, SB Through).\n\n"
            "A vehicle is counted on a route when its track crosses that route's start "
            "line and later its end line.\n\n"
            "Save the drawn routes and reuse them on the next video of the same intersection. "
            "A folder of videos can be processed in one batch with the same geometry."
        )
        help_text.setWordWrap(True)
        right_layout.addWidget(help_text)
        right_layout.addStretch(1)
        row.addWidget(right, 1)
        return page

    def _build_routes_page(self) -> QWidget:
        page = QWidget()
        row = QHBoxLayout(page)
        row.setSpacing(16)

        left = self._card()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(self._step_title("2. Draw start and end lines"))
        self.route_hint = QLabel("Add a route, then click and drag each line on the frame.")
        self.route_hint.setObjectName("hint")
        self.route_hint.setWordWrap(True)
        left_layout.addWidget(self.route_hint)

        self.route_list = QListWidget()
        left_layout.addWidget(self.route_list, 1)

        route_btns = QHBoxLayout()
        add_route = QPushButton("Add route")
        add_route.setObjectName("primary")
        delete_route = QPushButton("Delete")
        delete_route.setObjectName("danger")
        add_route.clicked.connect(self._start_add_route)
        delete_route.clicked.connect(self._delete_route)
        route_btns.addWidget(add_route)
        route_btns.addWidget(delete_route)
        left_layout.addLayout(route_btns)

        io_btns = QHBoxLayout()
        save_geom = QPushButton("Save routes")
        load_geom = QPushButton("Load routes")
        save_geom.clicked.connect(self._save_geometry)
        load_geom.clicked.connect(self._load_geometry)
        io_btns.addWidget(save_geom)
        io_btns.addWidget(load_geom)
        left_layout.addLayout(io_btns)

        nav = QHBoxLayout()
        back = QPushButton("Back")
        process = QPushButton("Process videos")
        process.setObjectName("primary")
        back.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        process.clicked.connect(self._go_process)
        nav.addWidget(back)
        nav.addWidget(process)
        left_layout.addLayout(nav)

        right = self._card()
        right_layout = QVBoxLayout(right)
        canvas_header = QHBoxLayout()
        canvas_header.addWidget(self._step_title("Random frame"))
        new_frame = QPushButton("New random frame")
        new_frame.clicked.connect(self._refresh_frame)
        canvas_header.addStretch(1)
        canvas_header.addWidget(new_frame)
        right_layout.addLayout(canvas_header)

        self.canvas = FrameCanvas()
        self.canvas.on_line_drawn = self._on_line_drawn
        right_layout.addWidget(self.canvas, 1)
        row.addWidget(left, 0)
        row.addWidget(right, 1)
        left.setMaximumWidth(360)
        return page

    def _build_process_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(12)
        layout.addWidget(self._step_title("3. Processing"))

        self.process_status = QLabel("Waiting to start…")
        self.process_status.setObjectName("hint")
        layout.addWidget(self.process_status)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        layout.addWidget(self.progress)

        body = QHBoxLayout()
        self.preview_canvas = FrameCanvas()
        self.preview_canvas.set_drawing_enabled(False)
        body.addWidget(self.preview_canvas, 3)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        body.addWidget(self.log, 2)
        layout.addLayout(body, 1)

        nav = QHBoxLayout()
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setObjectName("danger")
        self.cancel_btn.clicked.connect(self._cancel_process)
        back = QPushButton("Back to routes")
        back.clicked.connect(lambda: self.stack.setCurrentIndex(1))
        nav.addWidget(back)
        nav.addStretch(1)
        nav.addWidget(self.cancel_btn)
        layout.addLayout(nav)
        return page

    def _build_report_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(self._step_title("4. 15-minute TMC"))
        self.report_summary = QLabel("No results yet.")
        self.report_summary.setObjectName("hint")
        layout.addWidget(self.report_summary)

        self.table = QTableWidget()
        layout.addWidget(self.table, 1)

        nav = QHBoxLayout()
        again = QPushButton("New batch")
        again.clicked.connect(self._reset_flow)
        open_folder = QPushButton("Open export folder")
        open_folder.setObjectName("primary")
        open_folder.clicked.connect(self._open_export_dir)
        nav.addWidget(again)
        nav.addStretch(1)
        nav.addWidget(open_folder)
        layout.addLayout(nav)
        return page

    def _refresh_device_status(self) -> None:
        self.statusBar().showMessage(f"Ready  •  {describe_device(detect_device())}")

    def _step_title(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("stepTitle")
        return label

    def _clock_start(self) -> datetime | None:
        if not self.use_clock.isChecked():
            return None
        return self.clock_edit.dateTime().toPyDateTime()

    def _add_videos(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select videos",
            "",
            "Video files (*.mp4 *.mov *.avi *.mkv *.m4v *.wmv);;All files (*.*)",
        )
        self._append_videos([Path(p) for p in files])

    def _add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select folder of videos")
        if folder:
            self._append_videos(list_videos_in_folder(folder))

    def _append_videos(self, paths: list[Path]) -> None:
        for path in paths:
            if not is_video_file(path):
                continue
            if path not in self.videos:
                self.videos.append(path)
                try:
                    info = probe_video(path)
                    self.video_list.addItem(
                        f"{path.name}  ({info.width}x{info.height}, {info.duration_s/60:.1f} min)"
                    )
                except Exception as exc:
                    self.video_list.addItem(f"{path.name}  (unreadable: {exc})")
        self.statusBar().showMessage(f"{len(self.videos)} video(s) queued")

    def _remove_selected_videos(self) -> None:
        rows = sorted({idx.row() for idx in self.video_list.selectedIndexes()}, reverse=True)
        for row in rows:
            self.video_list.takeItem(row)
            if 0 <= row < len(self.videos):
                del self.videos[row]

    def _choose_export_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Choose export folder")
        if folder:
            self.export_dir = Path(folder)
            self.export_label.setText(f"Export folder: {self.export_dir}")

    def _go_routes(self) -> None:
        if not self.videos:
            QMessageBox.information(self, APP_NAME, "Add at least one video first.")
            return
        self.stack.setCurrentIndex(1)
        if not self.canvas.has_frame():
            self._refresh_frame()

    def _refresh_frame(self) -> None:
        if not self.videos:
            return
        try:
            frame, index = grab_random_frame(self.videos[0])
            self.canvas.set_frame(frame)
            width, height = self.canvas.image_size()
            self.geometry.frame_width = width
            self.geometry.frame_height = height
            self.canvas.set_routes(self.geometry.routes)
            self.route_hint.setText(
                f"Frame {index} from {self.videos[0].name}. Add a route, then drag its start line (green) and end line (red)."
            )
        except Exception as exc:
            QMessageBox.warning(self, APP_NAME, str(exc))

    def _refresh_route_list(self) -> None:
        self.route_list.clear()
        for route in self.geometry.routes:
            self.route_list.addItem(route.name)
        self.canvas.set_routes(self.geometry.routes)

    def _start_add_route(self) -> None:
        name, ok = QInputDialog.getText(self, "New route", "Route name (example: NB Left)")
        name = name.strip()
        if not ok or not name:
            return
        if any(route.name == name for route in self.geometry.routes):
            QMessageBox.information(self, APP_NAME, "That route name is already used.")
            return
        self._route_name = name
        self._pending_start = None
        self._draw_step = "start"
        self.canvas.set_drawing_enabled(True)
        self.route_hint.setText(f"Draw the START line for “{name}”. Click and drag on the frame.")

    def _on_line_drawn(self, line: Line) -> None:
        if self._draw_step == "start":
            self._pending_start = line
            self._draw_step = "end"
            self.route_hint.setText(f"Now draw the END line for “{self._route_name}”.")
            return
        if self._draw_step == "end" and self._pending_start is not None:
            self.geometry.routes.append(Route(name=self._route_name, start=self._pending_start, end=line))
            self._draw_step = "idle"
            self._pending_start = None
            self.canvas.set_drawing_enabled(False)
            self.canvas.clear_draft()
            self._refresh_route_list()
            self.route_hint.setText(f"Saved “{self._route_name}”. Add another route or process the videos.")

    def _delete_route(self) -> None:
        row = self.route_list.currentRow()
        if row < 0:
            return
        del self.geometry.routes[row]
        self._refresh_route_list()

    def _save_geometry(self) -> None:
        if not self.geometry.routes:
            QMessageBox.information(self, APP_NAME, "Draw at least one route first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save routes", "routes.json", "JSON (*.json)")
        if path:
            self.geometry.save(path)
            self.statusBar().showMessage(f"Saved routes to {path}")

    def _load_geometry(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load routes", "", "JSON (*.json)")
        if not path:
            return
        try:
            loaded = Geometry.load(path)
            width, height = self.canvas.image_size()
            self.geometry = loaded.scaled_to(width, height) if self.canvas.has_frame() else loaded
            self._refresh_route_list()
            self.route_hint.setText(f"Loaded {len(self.geometry.routes)} route(s) from {Path(path).name}.")
        except Exception as exc:
            QMessageBox.warning(self, APP_NAME, f"Could not load routes: {exc}")

    def _go_process(self) -> None:
        if not self.geometry.routes:
            QMessageBox.information(self, APP_NAME, "Draw or load at least one route.")
            return
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.stack.setCurrentIndex(2)
        self.log.clear()
        self.results = []
        self.progress.setValue(0)
        self.cancel_btn.setEnabled(True)
        self._append_log(f"Device: {describe_device(detect_device())}")
        self._append_log(f"Videos: {len(self.videos)}")
        self._append_log(f"Routes: {', '.join(r.name for r in self.geometry.routes)}")

        self.worker = ProcessWorker(
            videos=list(self.videos),
            geometry=self.geometry,
            export_dir=self.export_dir,
            clock_start=self._clock_start(),
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.preview.connect(self._on_preview)
        self.worker.video_done.connect(self._on_video_done)
        self.worker.failed.connect(self._on_failed)
        self.worker.all_done.connect(self._on_all_done)
        self.worker.start()

    def _cancel_process(self) -> None:
        if self.worker:
            self.worker.cancel()
            self._append_log("Cancel requested…")
            self.cancel_btn.setEnabled(False)

    def _on_progress(self, current: int, total: int, message: str) -> None:
        if total > 0:
            self.progress.setValue(int(current / total * 100))
        self.process_status.setText(message)

    def _on_preview(self, frame: np.ndarray) -> None:
        self.preview_canvas.set_frame(frame)

    def _on_video_done(self, result: ProcessResult) -> None:
        status = result.error or f"{len(result.trips)} trips"
        self._append_log(f"{result.video.name}: {status}")

    def _on_failed(self, message: str) -> None:
        self._append_log(f"Error: {message}")
        QMessageBox.warning(self, APP_NAME, message)

    def _on_all_done(self, results: list[ProcessResult]) -> None:
        self.results = results
        self.cancel_btn.setEnabled(False)
        if results and all(result.error == "Cancelled" for result in results):
            self.process_status.setText("Cancelled.")
            return
        self._populate_report()
        self.stack.setCurrentIndex(3)

    def _populate_report(self) -> None:
        ok_results = [result for result in self.results if not result.error]
        total_trips = sum(len(result.trips) for result in ok_results)
        self.report_summary.setText(
            f"{len(ok_results)} video(s) processed  •  {total_trips} vehicles assigned to a route  •  "
            f"Spreadsheets saved in {self.export_dir}"
        )
        if not ok_results:
            self.table.clear()
            return

        first = ok_results[0]
        table = build_tmc_table(
            first.trips,
            [route.name for route in self.geometry.routes],
            first.video.duration_s,
            self._clock_start(),
        )
        self.table.setRowCount(len(table))
        self.table.setColumnCount(len(table.columns))
        self.table.setHorizontalHeaderLabels(list(table.columns))
        for r_idx, row in table.iterrows():
            for c_idx, value in enumerate(row):
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(r_idx, c_idx, item)
        self.table.resizeColumnsToContents()

    def _open_export_dir(self) -> None:
        from PyQt6.QtGui import QDesktopServices
        from PyQt6.QtCore import QUrl

        self.export_dir.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.export_dir.resolve())))

    def _reset_flow(self) -> None:
        self.stack.setCurrentIndex(0)

    def _append_log(self, text: str) -> None:
        self.log.appendPlainText(text)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(2000)
        event.accept()
