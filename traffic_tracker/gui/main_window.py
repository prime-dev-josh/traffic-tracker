"""Videos → routes → align → process → report."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import (
    QCheckBox,
    QDateTimeEdit,
    QDoubleSpinBox,
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
    QSlider,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from traffic_tracker.config import APP_NAME, APP_VERSION
from traffic_tracker.core.device import describe_device, detect_device
from traffic_tracker.core.geometry import Geometry, Line, Route, Transform, transform_from_matches
from traffic_tracker.core.models import resolve_model_source
from traffic_tracker.core.processor import ProcessResult
from traffic_tracker.core.project import BundleProject, ClipRecord
from traffic_tracker.core.report import build_tmc_table
from traffic_tracker.core.video import (
    file_size_gb,
    format_duration,
    grab_frame_at,
    grab_random_frame,
    is_onedrive_path,
    is_video_file,
    list_videos_in_folder,
    probe_video,
)
from traffic_tracker.gui.widgets.frame_canvas import FrameCanvas
from traffic_tracker.gui.worker import ProcessWorker

PAGE_SETUP, PAGE_ROUTES, PAGE_ALIGN, PAGE_PROCESS, PAGE_REPORT = range(5)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} {APP_VERSION}")
        self.resize(1220, 800)

        self.videos: list[Path] = []
        self.project = BundleProject()
        self.geometry = Geometry(frame_width=1, frame_height=1)
        self.export_dir = Path.cwd() / "outputs"
        self.worker: ProcessWorker | None = None
        self.results: list[ProcessResult] = []
        self._route_name = ""
        self._pending_start: Line | None = None
        self._draw_step = "idle"
        self._align_index = 0
        self._ref_frame = None
        self._ref_size = (1, 1)
        self._match_ref: list[tuple[float, float]] = []
        self._match_clip: list[tuple[float, float]] = []
        self._match_side = "ref"
        self._redraw_queue: list[str] = []
        self._redrawing = False

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

        self.steps = QLabel("1. Videos   →   2. Routes   →   3. Align   →   4. Process   →   5. Report")
        self.steps.setObjectName("hint")
        layout.addWidget(self.steps)

        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_setup_page())
        self.stack.addWidget(self._build_routes_page())
        self.stack.addWidget(self._build_align_page())
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
        hint = QLabel(
            "Add a bundle of clips from the same camera. Large files (2–5 GB) stay on disk — "
            "the app only seeks frames. Draw routes once, then align each later clip."
        )
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        left_layout.addWidget(hint)

        self.video_list = QListWidget()
        left_layout.addWidget(self.video_list, 1)
        self.onedrive_warn = QLabel("")
        self.onedrive_warn.setObjectName("hint")
        self.onedrive_warn.setWordWrap(True)
        left_layout.addWidget(self.onedrive_warn)

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

        proj = QHBoxLayout()
        load_proj = QPushButton("Load project")
        save_proj = QPushButton("Save project")
        load_proj.clicked.connect(self._load_project)
        save_proj.clicked.connect(self._save_project)
        proj.addWidget(load_proj)
        proj.addWidget(save_proj)
        left_layout.addLayout(proj)

        next_btn = QPushButton("Next: draw routes")
        next_btn.setObjectName("primary")
        next_btn.clicked.connect(self._go_routes)
        left_layout.addWidget(next_btn)
        row.addWidget(left, 1)

        right = self._card()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(self._step_title("How a count works"))
        help_text = QLabel(
            "Draw start and end lines once on the first clip (the reference).\n\n"
            "Later clips from the same camera often shift (pan, zoom, remount). "
            "On the Align step, drag the overlay to match poles and curbs, then confirm. "
            "Unconfirmed clips are not processed.\n\n"
            "Keep large bundles on a local disk. OneDrive sync can stall seeking 2–5 GB files."
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
        self.route_hint = QLabel("Add a route, then click and drag each line on the reference frame.")
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
        nxt = QPushButton("Next: align clips")
        nxt.setObjectName("primary")
        back.clicked.connect(lambda: self.stack.setCurrentIndex(PAGE_SETUP))
        nxt.clicked.connect(self._go_align)
        nav.addWidget(back)
        nav.addWidget(nxt)
        left_layout.addLayout(nav)

        right = self._card()
        right_layout = QVBoxLayout(right)
        canvas_header = QHBoxLayout()
        canvas_header.addWidget(self._step_title("Reference frame"))
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

    def _build_align_page(self) -> QWidget:
        page = QWidget()
        row = QHBoxLayout(page)
        row.setSpacing(16)

        left = self._card()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(self._step_title("3. Align each clip"))
        self.align_hint = QLabel("Drag the overlay to match this clip. Wheel zooms. Confirm before processing.")
        self.align_hint.setObjectName("hint")
        self.align_hint.setWordWrap(True)
        left_layout.addWidget(self.align_hint)

        self.align_list = QListWidget()
        self.align_list.currentRowChanged.connect(self._on_align_row_changed)
        left_layout.addWidget(self.align_list, 1)

        self.rotate_spin = QDoubleSpinBox()
        self.rotate_spin.setRange(-45.0, 45.0)
        self.rotate_spin.setDecimals(1)
        self.rotate_spin.setSuffix("°")
        self.rotate_spin.setPrefix("Rotate  ")
        self.rotate_spin.valueChanged.connect(self._on_rotate_changed)
        left_layout.addWidget(self.rotate_spin)

        actions = QHBoxLayout()
        confirm = QPushButton("Confirm alignment")
        confirm.setObjectName("primary")
        confirm.clicked.connect(self._confirm_align)
        reset = QPushButton("Reset")
        reset.clicked.connect(self._reset_align)
        actions.addWidget(confirm)
        actions.addWidget(reset)
        left_layout.addLayout(actions)

        extra = QHBoxLayout()
        match = QPushButton("Match points")
        redraw = QPushButton("Redraw this clip")
        match.clicked.connect(self._start_match_points)
        redraw.clicked.connect(self._start_redraw_clip)
        extra.addWidget(match)
        extra.addWidget(redraw)
        left_layout.addLayout(extra)

        apply_match = QPushButton("Apply matched points")
        apply_match.clicked.connect(self._apply_match_points)
        left_layout.addWidget(apply_match)

        nav = QHBoxLayout()
        back = QPushButton("Back")
        nxt = QPushButton("Process confirmed")
        nxt.setObjectName("primary")
        back.clicked.connect(lambda: self.stack.setCurrentIndex(PAGE_ROUTES))
        nxt.clicked.connect(self._go_process)
        nav.addWidget(back)
        nav.addWidget(nxt)
        left_layout.addLayout(nav)
        left.setMaximumWidth(380)

        right = self._card()
        right_layout = QVBoxLayout(right)
        head = QHBoxLayout()
        head.addWidget(self._step_title("Clip frame"))
        rand = QPushButton("New random frame")
        rand.clicked.connect(self._align_random_frame)
        head.addStretch(1)
        head.addWidget(rand)
        right_layout.addLayout(head)

        self.align_time = QSlider(Qt.Orientation.Horizontal)
        self.align_time.setRange(0, 1000)
        self.align_time.sliderReleased.connect(self._align_seek)
        right_layout.addWidget(self.align_time)
        self.align_time_label = QLabel("0:00")
        self.align_time_label.setObjectName("hint")
        right_layout.addWidget(self.align_time_label)

        canvases = QHBoxLayout()
        ref_box = QVBoxLayout()
        ref_box.addWidget(QLabel("Reference (for match points)"))
        self.ref_canvas = FrameCanvas()
        self.ref_canvas.setMinimumSize(280, 180)
        self.ref_canvas.on_point = self._on_ref_point
        ref_box.addWidget(self.ref_canvas, 1)
        canvases.addLayout(ref_box, 1)

        clip_box = QVBoxLayout()
        clip_box.addWidget(QLabel("This clip — drag overlay"))
        self.align_canvas = FrameCanvas()
        self.align_canvas.on_nudge = self._on_nudge
        self.align_canvas.on_scale = self._on_scale
        self.align_canvas.on_point = self._on_clip_point
        self.align_canvas.on_line_drawn = self._on_redraw_line
        clip_box.addWidget(self.align_canvas, 1)
        canvases.addLayout(clip_box, 2)
        right_layout.addLayout(canvases, 1)

        row.addWidget(left, 0)
        row.addWidget(right, 1)
        return page

    def _build_process_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(12)
        layout.addWidget(self._step_title("4. Processing"))
        self.process_status = QLabel("Waiting to start…")
        self.process_status.setObjectName("hint")
        layout.addWidget(self.process_status)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        layout.addWidget(self.progress)
        body = QHBoxLayout()
        self.preview_canvas = FrameCanvas()
        self.preview_canvas.set_mode("view")
        body.addWidget(self.preview_canvas, 3)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        body.addWidget(self.log, 2)
        layout.addLayout(body, 1)
        nav = QHBoxLayout()
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setObjectName("danger")
        self.cancel_btn.clicked.connect(self._cancel_process)
        back = QPushButton("Back to align")
        back.clicked.connect(lambda: self.stack.setCurrentIndex(PAGE_ALIGN))
        nav.addWidget(back)
        nav.addStretch(1)
        nav.addWidget(self.cancel_btn)
        layout.addLayout(nav)
        return page

    def _build_report_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(self._step_title("5. 15-minute TMC"))
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

    def _video_label(self, path: Path) -> str:
        try:
            info = probe_video(path)
            size = file_size_gb(path)
            size_txt = f", {size:.1f} GB" if size >= 0.1 else ""
            return f"{path.name}  ({info.width}x{info.height}, {format_duration(info.duration_s)}{size_txt})"
        except Exception as exc:
            return f"{path.name}  (unreadable: {exc})"

    def _refresh_setup_list(self) -> None:
        self.video_list.clear()
        for path in self.videos:
            self.video_list.addItem(self._video_label(path))
        if self.videos and any(is_onedrive_path(p) for p in self.videos):
            self.onedrive_warn.setText(
                "These files are under OneDrive. Seeking 2–5 GB clips over sync is slow and can stall. "
                "Copy the bundle to a local disk when you can."
            )
        else:
            self.onedrive_warn.setText("")

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
        self.project.ensure_clips(self.videos)
        self._refresh_setup_list()
        self.statusBar().showMessage(f"{len(self.videos)} video(s) queued")

    def _remove_selected_videos(self) -> None:
        rows = sorted({idx.row() for idx in self.video_list.selectedIndexes()}, reverse=True)
        for row in rows:
            if 0 <= row < len(self.videos):
                del self.videos[row]
        self.project.ensure_clips(self.videos)
        self._refresh_setup_list()

    def _choose_export_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Choose export folder")
        if folder:
            self.export_dir = Path(folder)
            self.export_label.setText(f"Export folder: {self.export_dir}")

    def _go_routes(self) -> None:
        if not self.videos:
            QMessageBox.information(self, APP_NAME, "Add at least one video first.")
            return
        self.project.ensure_clips(self.videos)
        self.stack.setCurrentIndex(PAGE_ROUTES)
        if not self.canvas.has_frame():
            self._refresh_frame()

    def _refresh_frame(self) -> None:
        if not self.videos:
            return
        try:
            frame, index = grab_random_frame(self.videos[0])
            self._ref_frame = frame
            self.canvas.set_frame(frame)
            self.ref_canvas.set_frame(frame)
            width, height = self.canvas.image_size()
            self._ref_size = (width, height)
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
        self.project.geometry = self.geometry

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
            self.route_hint.setText(f"Saved “{self._route_name}”. Add another route or align the other clips.")

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

    def _current_clip(self) -> ClipRecord | None:
        if not self.project.clips:
            return None
        self._align_index = min(max(0, self._align_index), len(self.project.clips) - 1)
        return self.project.clips[self._align_index]

    def _fitted_overlay(self, clip: ClipRecord) -> list[Route]:
        if clip.status == "redrawn" and clip.geometry_override is not None:
            return clip.geometry_override.routes
        width, height = self.align_canvas.image_size()
        fitted = self.geometry.scaled_to(width, height)
        return clip.transform.apply_geometry(fitted).routes

    def _refresh_align_list(self) -> None:
        self.align_list.blockSignals(True)
        self.align_list.clear()
        labels = {
            "reference": "Reference",
            "needs_align": "Needs align",
            "aligned": "Aligned",
            "redrawn": "Custom redraw",
        }
        for clip in self.project.clips:
            self.align_list.addItem(f"[{labels.get(clip.status, clip.status)}]  {self._video_label(clip.path)}")
        if self.project.clips:
            self.align_list.setCurrentRow(self._align_index)
        self.align_list.blockSignals(False)

    def _go_align(self) -> None:
        if not self.geometry.routes:
            QMessageBox.information(self, APP_NAME, "Draw or load at least one route.")
            return
        self.project.geometry = self.geometry
        self.project.ensure_clips(self.videos)
        if self.project.clips:
            self.project.clips[0].status = "reference"
        self.stack.setCurrentIndex(PAGE_ALIGN)
        if self._ref_frame is None and self.videos:
            self._refresh_frame()
        elif self._ref_frame is not None:
            self.ref_canvas.set_frame(self._ref_frame)
        self._align_index = 0 if len(self.project.clips) <= 1 else 1
        self._refresh_align_list()
        self._load_align_clip()

    def _on_align_row_changed(self, row: int) -> None:
        if row < 0:
            return
        self._align_index = row
        self._load_align_clip()

    def _load_align_clip(self) -> None:
        clip = self._current_clip()
        if clip is None:
            return
        self._redrawing = False
        self._match_ref = []
        self._match_clip = []
        self._match_side = "ref"
        try:
            info = probe_video(clip.path)
            t = min(info.duration_s * 0.2, max(0.0, info.duration_s - 0.1))
            frame, _ = grab_frame_at(clip.path, t)
            self.align_canvas.set_frame(frame)
            self.align_time.setValue(int(t / info.duration_s * 1000) if info.duration_s else 0)
            self.align_time_label.setText(format_duration(t))
        except Exception as exc:
            QMessageBox.warning(self, APP_NAME, str(exc))
            return
        self.rotate_spin.blockSignals(True)
        self.rotate_spin.setValue(clip.transform.rotation_deg)
        self.rotate_spin.blockSignals(False)
        self.align_canvas.set_routes(self._fitted_overlay(clip))
        self.align_canvas.set_match_points([])
        self.ref_canvas.set_match_points([])
        if clip.status == "reference":
            self.align_canvas.set_mode("view")
            self.align_hint.setText("This is the reference clip. Routes were drawn here. Other clips need alignment.")
        else:
            self.align_canvas.set_mode("nudge")
            self.align_hint.setText(
                "Drag to pan the overlay. Mouse wheel scales. Rotate with the control. "
                "Confirm when lines sit on the same pavement / poles as the reference."
            )

    def _align_random_frame(self) -> None:
        clip = self._current_clip()
        if clip is None:
            return
        try:
            frame, index = grab_random_frame(clip.path)
            self.align_canvas.set_frame(frame)
            info = probe_video(clip.path)
            t = index / info.fps if info.fps else 0
            self.align_time.setValue(int(t / info.duration_s * 1000) if info.duration_s else 0)
            self.align_time_label.setText(format_duration(t))
            self.align_canvas.set_routes(self._fitted_overlay(clip))
        except Exception as exc:
            QMessageBox.warning(self, APP_NAME, str(exc))

    def _align_seek(self) -> None:
        clip = self._current_clip()
        if clip is None:
            return
        try:
            info = probe_video(clip.path)
            t = (self.align_time.value() / 1000.0) * info.duration_s
            frame, _ = grab_frame_at(clip.path, t)
            self.align_canvas.set_frame(frame)
            self.align_time_label.setText(format_duration(t))
            self.align_canvas.set_routes(self._fitted_overlay(clip))
        except Exception as exc:
            QMessageBox.warning(self, APP_NAME, str(exc))

    def _on_nudge(self, dx: float, dy: float) -> None:
        clip = self._current_clip()
        if clip is None or clip.status == "reference":
            return
        if clip.transform.homography is not None:
            clip.transform = Transform(tx=dx, ty=dy)
        else:
            clip.transform.tx += dx
            clip.transform.ty += dy
        if clip.status == "aligned":
            clip.status = "needs_align"
        self.align_canvas.set_routes(self._fitted_overlay(clip))

    def _on_scale(self, factor: float) -> None:
        clip = self._current_clip()
        if clip is None or clip.status == "reference":
            return
        if clip.transform.homography is not None:
            clip.transform = Transform(scale=max(0.3, min(3.0, factor)))
        else:
            clip.transform.scale = max(0.3, min(3.0, clip.transform.scale * factor))
        if clip.status == "aligned":
            clip.status = "needs_align"
        self.align_canvas.set_routes(self._fitted_overlay(clip))

    def _on_rotate_changed(self, value: float) -> None:
        clip = self._current_clip()
        if clip is None or clip.status == "reference":
            return
        if clip.transform.homography is not None:
            clip.transform = Transform(rotation_deg=value)
        else:
            clip.transform.rotation_deg = value
        if clip.status == "aligned":
            clip.status = "needs_align"
        self.align_canvas.set_routes(self._fitted_overlay(clip))

    def _confirm_align(self) -> None:
        clip = self._current_clip()
        if clip is None:
            return
        if clip.status == "reference":
            self._next_unaligned()
            return
        if self._redrawing:
            QMessageBox.information(self, APP_NAME, "Finish redrawing the routes on this clip first.")
            return
        clip.status = "aligned"
        self._refresh_align_list()
        self._next_unaligned()

    def _next_unaligned(self) -> None:
        for i, clip in enumerate(self.project.clips):
            if clip.status == "needs_align":
                self._align_index = i
                self._refresh_align_list()
                self._load_align_clip()
                return
        self.align_hint.setText("All clips are confirmed or are the reference. You can process.")

    def _reset_align(self) -> None:
        clip = self._current_clip()
        if clip is None or clip.status == "reference":
            return
        clip.transform = Transform()
        clip.geometry_override = None
        clip.status = "needs_align"
        self._redrawing = False
        self._match_ref = []
        self._match_clip = []
        self.rotate_spin.blockSignals(True)
        self.rotate_spin.setValue(0)
        self.rotate_spin.blockSignals(False)
        self.align_canvas.set_mode("nudge")
        self.align_canvas.set_routes(self._fitted_overlay(clip))
        self.align_canvas.set_match_points([])
        self.ref_canvas.set_match_points([])
        self._refresh_align_list()

    def _start_match_points(self) -> None:
        clip = self._current_clip()
        if clip is None or clip.status == "reference":
            QMessageBox.information(self, APP_NAME, "Pick a non-reference clip to match landmarks.")
            return
        self._match_ref = []
        self._match_clip = []
        self._match_side = "ref"
        self.align_canvas.set_mode("match")
        self.ref_canvas.set_mode("match")
        self.align_hint.setText(
            "Click a fixed landmark on the REFERENCE (pole, curb, signal), then the same point on THIS clip. "
            "Repeat 2–4 times, then Apply matched points."
        )
        self.align_canvas.set_match_points([])
        self.ref_canvas.set_match_points([])

    def _on_ref_point(self, x: float, y: float) -> None:
        if self._match_side != "ref":
            return
        self._match_ref.append((x, y))
        self._match_side = "clip"
        self.ref_canvas.set_match_points(self._match_ref)
        self.align_hint.setText(f"Now click the same landmark on THIS clip (pair {len(self._match_ref)}).")

    def _on_clip_point(self, x: float, y: float) -> None:
        if self.align_canvas._mode != "match":
            return
        if self._match_side != "clip":
            self.align_hint.setText("Click the landmark on the REFERENCE first.")
            return
        self._match_clip.append((x, y))
        self._match_side = "ref"
        self.align_canvas.set_match_points(self._match_clip)
        self.align_hint.setText(
            f"{len(self._match_clip)} pair(s). Click another reference landmark, or Apply matched points."
        )

    def _apply_match_points(self) -> None:
        clip = self._current_clip()
        if clip is None:
            return
        if len(self._match_ref) < 2 or len(self._match_ref) != len(self._match_clip):
            QMessageBox.information(self, APP_NAME, "Need at least two complete landmark pairs.")
            return
        clip_w, clip_h = self.align_canvas.image_size()
        ref_w, ref_h = self._ref_size
        sx = clip_w / ref_w if ref_w else 1.0
        sy = clip_h / ref_h if ref_h else 1.0
        src = [(x * sx, y * sy) for x, y in self._match_ref]
        try:
            clip.transform = transform_from_matches(src, self._match_clip)
        except Exception as exc:
            QMessageBox.warning(self, APP_NAME, str(exc))
            return
        clip.status = "needs_align"
        self.align_canvas.set_mode("nudge")
        self.ref_canvas.set_mode("view")
        self.align_canvas.set_routes(self._fitted_overlay(clip))
        self.align_hint.setText("Overlay updated from landmarks. Nudge if needed, then Confirm alignment.")
        self._refresh_align_list()

    def _start_redraw_clip(self) -> None:
        clip = self._current_clip()
        if clip is None or clip.status == "reference":
            QMessageBox.information(self, APP_NAME, "Redraw is for non-reference clips. Edit routes on step 2.")
            return
        if not self.geometry.routes:
            return
        width, height = self.align_canvas.image_size()
        clip.geometry_override = Geometry(width, height, [])
        clip.status = "needs_align"
        self._redraw_queue = [route.name for route in self.geometry.routes]
        self._redrawing = True
        self._pending_start = None
        self._draw_step = "start"
        self._route_name = self._redraw_queue[0]
        self.align_canvas.set_routes([])
        self.align_canvas.set_mode("draw")
        self.align_hint.setText(f"Redraw START line for “{self._route_name}” on this clip.")

    def _on_redraw_line(self, line: Line) -> None:
        if not self._redrawing:
            return
        clip = self._current_clip()
        if clip is None or clip.geometry_override is None:
            return
        if self._draw_step == "start":
            self._pending_start = line
            self._draw_step = "end"
            self.align_hint.setText(f"Now draw the END line for “{self._route_name}”.")
            return
        if self._draw_step == "end" and self._pending_start is not None:
            clip.geometry_override.routes.append(
                Route(name=self._route_name, start=self._pending_start, end=line)
            )
            self._redraw_queue.pop(0)
            self._pending_start = None
            self.align_canvas.set_routes(clip.geometry_override.routes)
            if not self._redraw_queue:
                clip.status = "redrawn"
                self._redrawing = False
                self.align_canvas.set_mode("view")
                self.align_hint.setText("Custom routes saved for this clip.")
                self._refresh_align_list()
                return
            self._route_name = self._redraw_queue[0]
            self._draw_step = "start"
            self.align_hint.setText(f"Redraw START line for “{self._route_name}”.")

    def _save_project(self) -> None:
        self.project.geometry = self.geometry
        self.project.ensure_clips(self.videos)
        path, _ = QFileDialog.getSaveFileName(self, "Save project", "bundle.ttproj.json", "Project (*.json)")
        if path:
            self.project.save(path)
            self.statusBar().showMessage(f"Saved project to {path}")

    def _load_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load project", "", "Project (*.json)")
        if not path:
            return
        try:
            loaded = BundleProject.load(path)
        except Exception as exc:
            QMessageBox.warning(self, APP_NAME, f"Could not load project: {exc}")
            return
        self.project = loaded
        self.geometry = loaded.geometry
        self.videos = [clip.path for clip in loaded.clips if clip.path.exists()]
        missing = [clip.path.name for clip in loaded.clips if not clip.path.exists()]
        self.project.ensure_clips(self.videos)
        self._refresh_setup_list()
        self._refresh_route_list()
        if missing:
            QMessageBox.information(self, APP_NAME, "Missing video files:\n" + "\n".join(missing))
        self.statusBar().showMessage(f"Loaded project ({len(self.videos)} clip(s) found)")

    def _go_process(self) -> None:
        if not self.geometry.routes:
            QMessageBox.information(self, APP_NAME, "Draw or load at least one route.")
            return
        self.project.geometry = self.geometry
        self.project.ensure_clips(self.videos)
        if self.project.clips:
            self.project.clips[0].status = "reference"
        jobs = self.project.ready_jobs(self.geometry)
        skipped = [c.path.name for c in self.project.clips if c.status == "needs_align"]
        if not jobs:
            QMessageBox.information(
                self,
                APP_NAME,
                "No confirmed clips. Align and confirm each shifted clip, or process the reference only after confirming.",
            )
            return
        if skipped:
            go = QMessageBox.question(
                self,
                APP_NAME,
                f"{len(jobs)} clip(s) will process. Skipped (not aligned):\n" + "\n".join(skipped),
            )
            if go != QMessageBox.StandardButton.Yes:
                return
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.stack.setCurrentIndex(PAGE_PROCESS)
        self.log.clear()
        self.results = []
        self.progress.setValue(0)
        self.cancel_btn.setEnabled(True)
        self._append_log(f"Device: {describe_device(detect_device())}")
        try:
            self._append_log(f"Model: {resolve_model_source().label}")
        except Exception as exc:
            self._append_log(f"Model: {exc}")
        self._append_log(f"Processing {len(jobs)} clip(s); skipped {len(skipped)}")
        self._append_log(f"Routes: {', '.join(r.name for r in self.geometry.routes)}")

        self.worker = ProcessWorker(
            jobs=jobs,
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
        extra = f"  •  {result.model_label}" if result.model_label else ""
        self._append_log(f"{result.video.name}: {status}{extra}")

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
        self.stack.setCurrentIndex(PAGE_REPORT)

    def _populate_report(self) -> None:
        ok_results = [result for result in self.results if not result.error]
        total_trips = sum(len(result.trips) for result in ok_results)
        skipped = len(self.project.clips) - len(ok_results)
        self.report_summary.setText(
            f"{len(ok_results)} video(s) processed  •  {total_trips} vehicles assigned to a route  •  "
            f"{max(0, skipped)} skipped  •  Spreadsheets saved in {self.export_dir}"
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
        from PyQt6.QtCore import QUrl
        from PyQt6.QtGui import QDesktopServices

        self.export_dir.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.export_dir.resolve())))

    def _reset_flow(self) -> None:
        self.stack.setCurrentIndex(PAGE_SETUP)

    def _append_log(self, text: str) -> None:
        self.log.appendPlainText(text)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(2000)
        event.accept()
