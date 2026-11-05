"""Videos → routes → align → process → report."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDateTimeEdit,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QStackedWidget,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from traffic_tracker.config import (
    APP_NAME,
    APP_VERSION,
    DETECTOR_CONF_MAX,
    DETECTOR_CONF_MIN,
    TRAIN_MIN_BOXES,
    TRAIN_MIN_FRAMES,
)
from traffic_tracker.core.geometry import (
    BOUNDS,
    MAX_GATES,
    ORIGINS,
    MAX_VEHICLE_REFS,
    Box,
    Geometry,
    Line,
    NamedLine,
    Transform,
    transform_from_matches,
)


def bind_row_action(callback):
    """Run a row callback without Qt's clicked(checked) overwriting default args."""

    def _run(*_args):
        callback()

    return _run
from traffic_tracker.core.processor import ProcessResult
from traffic_tracker.core.project import BundleProject, ClipRecord
from traffic_tracker.core.report import (
    build_gate_statistics_table,
    build_overview_table,
    build_tmc_summary_table,
)
from traffic_tracker.core.settings import AppSettings, clamp_detector_conf, load_settings, save_settings
from traffic_tracker.core.detector import reset_detector_cache, resolve_detector_weights
from traffic_tracker.core.device import device_report
from traffic_tracker.core.training import (
    LabeledFrame,
    TrainingSession,
    delete_session,
    export_training_archive,
    has_trained_weights,
    import_training_archive,
    list_sessions,
    load_session,
    prior_areas,
    save_session,
    trained_weights_path,
    training_inventory,
    training_root,
)
from traffic_tracker.gui.style import stylesheet_for
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
from traffic_tracker.gui.widgets.home_menu import HomeStage
from traffic_tracker.gui.worker import ProcessWorker, TrainWorker

(
    PAGE_HOME,
    PAGE_SETUP,
    PAGE_ROUTES,
    PAGE_ALIGN,
    PAGE_PROCESS,
    PAGE_REPORT,
    PAGE_TRAINING,
    PAGE_SETTINGS,
) = range(8)
PROCESS_PAGES = {PAGE_SETUP, PAGE_ROUTES, PAGE_ALIGN, PAGE_PROCESS, PAGE_REPORT}


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} {APP_VERSION}")
        self.resize(1220, 800)

        self.videos: list[Path] = []
        self.project = BundleProject()
        self.geometry = Geometry(frame_width=1, frame_height=1)
        self.settings = load_settings()
        self.export_dir = Path(self.settings.export_dir) if self.settings.export_dir else Path.cwd() / "outputs"
        self.worker: ProcessWorker | None = None
        self.train_worker: TrainWorker | None = None
        self.results: list[ProcessResult] = []
        self._draw_step = "idle"
        self._draw_gate_name: str | None = None
        self._redraw_index: int | None = None
        self._redraw_target = None
        self._align_index = 0
        self._ref_frame = None
        self._ref_size = (1, 1)
        self._match_ref: list[tuple[float, float]] = []
        self._match_clip: list[tuple[float, float]] = []
        self._match_side = "ref"
        self._redrawing = False
        self._train_video: Path | None = None
        self._train_session: TrainingSession | None = None
        self._train_frame = None
        self._train_frame_index = 0
        self._train_time_s = 0.0
        self._train_boxes: list[Box] = []
        self._train_frames_bgr: dict[int, np.ndarray] = {}

        root = QWidget()
        self.setCentralWidget(root)
        self._root_layout = QVBoxLayout(root)
        self._root_layout.setContentsMargins(0, 0, 0, 0)
        self._root_layout.setSpacing(0)

        self.header_bar = QWidget()
        title_row = QHBoxLayout(self.header_bar)
        title_row.setContentsMargins(16, 16, 16, 0)
        header = QVBoxLayout()
        title = QLabel(APP_NAME)
        title.setObjectName("title")
        subtitle = QLabel("Count vehicles across user-drawn start and end lines, then export a 15-minute TMC.")
        subtitle.setObjectName("subtitle")
        header.addWidget(title)
        header.addWidget(subtitle)
        title_row.addLayout(header, 1)
        self.home_btn = QPushButton("Home")
        self.home_btn.clicked.connect(self._go_home)
        self.home_btn.setVisible(False)
        title_row.addWidget(self.home_btn, 0, Qt.AlignmentFlag.AlignTop)
        self.header_bar.setVisible(False)
        self._root_layout.addWidget(self.header_bar)

        self.steps = QLabel("1. Videos   →   2. Routes   →   3. Align   →   4. Process   →   5. Report")
        self.steps.setObjectName("hint")
        self.steps.setContentsMargins(16, 8, 16, 0)
        self.steps.setVisible(False)
        self._root_layout.addWidget(self.steps)

        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_home_page())
        self.stack.addWidget(self._build_setup_page())
        self.stack.addWidget(self._build_routes_page())
        self.stack.addWidget(self._build_align_page())
        self.stack.addWidget(self._build_process_page())
        self.stack.addWidget(self._build_report_page())
        self.stack.addWidget(self._build_training_page())
        self.stack.addWidget(self._build_settings_page())
        self.stack.currentChanged.connect(self._on_page_changed)
        self.content_shell = QWidget()
        self.content_layout = QVBoxLayout(self.content_shell)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(0)
        self.content_layout.addWidget(self.stack)
        self._root_layout.addWidget(self.content_shell, 1)

        self.statusBar().hide()
        self._apply_theme(self.settings.theme)
        self._sync_detector_ui()
        QTimer.singleShot(0, self._refresh_device_status)

    def _card(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("card")
        return frame

    def _build_home_page(self) -> QWidget:
        self.home_stage = HomeStage(
            [
                (
                    "01",
                    "Process",
                    "Add clips, set North and gates, align later videos, then export the TMC.",
                    self._go_process_feature,
                ),
                (
                    "02",
                    "Training",
                    "Box every vehicle on sample frames, then Train detector for site weights.",
                    self._go_training,
                ),
                (
                    "03",
                    "Settings",
                    "Export folder, OneDrive warning, and detector mode. Motion stays on until site weights exist.",
                    self._go_settings,
                ),
            ]
        )
        return self.home_stage

    def _build_training_page(self) -> QWidget:
        page = QWidget()
        row = QHBoxLayout(page)
        row.setSpacing(16)

        left = self._card()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(self._step_title("Training"))
        hint = QLabel(
            "Box every vehicle you can see, including cars stopped at a light. "
            "Aim for 150+ frames and 400+ boxes across clips. "
            f"{TRAIN_MIN_FRAMES} frames and {TRAIN_MIN_BOXES} boxes is the minimum to Train detector. "
            "North and gates are not required here."
        )
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        left_layout.addWidget(hint)

        pick = QPushButton("Open video")
        pick.clicked.connect(self._training_pick_video)
        left_layout.addWidget(pick)

        self.train_video_label = QLabel("No video selected.")
        self.train_video_label.setObjectName("hint")
        self.train_video_label.setWordWrap(True)
        left_layout.addWidget(self.train_video_label)

        self.train_status = QLabel("Open a clip, then box every vehicle you can see.")
        self.train_status.setObjectName("hint")
        self.train_status.setWordWrap(True)
        left_layout.addWidget(self.train_status)

        left_layout.addWidget(self._section("Saved sessions"))
        self.train_sessions = QListWidget()
        self.train_sessions.itemSelectionChanged.connect(self._training_select_session)
        left_layout.addWidget(self.train_sessions, 1)

        session_btns = QHBoxLayout()
        load_sess = QPushButton("Load")
        load_sess.clicked.connect(self._training_load_selected)
        delete_sess = QPushButton("Delete session")
        delete_sess.setObjectName("danger")
        delete_sess.clicked.connect(self._training_delete_selected)
        session_btns.addWidget(load_sess)
        session_btns.addWidget(delete_sess)
        left_layout.addLayout(session_btns)

        actions = QHBoxLayout()
        next_frame = QPushButton("Next frame")
        next_frame.clicked.connect(self._training_next_frame)
        undo_box = QPushButton("Undo box")
        undo_box.clicked.connect(self._training_undo_box)
        save_sess = QPushButton("Save session")
        save_sess.setObjectName("primary")
        save_sess.clicked.connect(self._training_save)
        actions.addWidget(next_frame)
        actions.addWidget(undo_box)
        actions.addWidget(save_sess)
        left_layout.addLayout(actions)

        left_layout.addWidget(self._section("Site detector"))
        self.train_ready_label = QLabel("No labeled frames yet.")
        self.train_ready_label.setObjectName("hint")
        self.train_ready_label.setWordWrap(True)
        left_layout.addWidget(self.train_ready_label)
        train_row = QHBoxLayout()
        self.train_detector_btn = QPushButton("Train detector")
        self.train_detector_btn.setObjectName("primary")
        self.train_detector_btn.clicked.connect(self._start_detector_train)
        self.train_visdrone_btn = QPushButton("Train on VisDrone")
        self.train_visdrone_btn.clicked.connect(self._start_visdrone_train)
        self.cancel_train_btn = QPushButton("Cancel train")
        self.cancel_train_btn.setEnabled(False)
        self.cancel_train_btn.clicked.connect(self._cancel_detector_train)
        train_row.addWidget(self.train_detector_btn)
        train_row.addWidget(self.train_visdrone_btn)
        train_row.addWidget(self.cancel_train_btn)
        left_layout.addLayout(train_row)
        self.train_progress = QProgressBar()
        self.train_progress.setValue(0)
        left_layout.addWidget(self.train_progress)
        self.train_log = QPlainTextEdit()
        self.train_log.setReadOnly(True)
        self.train_log.setMaximumHeight(120)
        self.train_log.setPlaceholderText("Training log")
        left_layout.addWidget(self.train_log)

        right = self._card()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(self._step_title("Box every vehicle on this frame"))
        self.train_canvas = FrameCanvas()
        self.train_canvas.on_box_drawn = self._training_box_drawn
        right_layout.addWidget(self.train_canvas, 1)

        row.addWidget(left, 0)
        row.addWidget(right, 1)
        return page

    def _build_settings_page(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setSpacing(16)
        card = self._card()
        form = QVBoxLayout(card)
        form.addWidget(self._step_title("Settings"))

        form.addWidget(self._section("Appearance"))
        theme_row = QHBoxLayout()
        theme_label = QLabel("Theme")
        self.theme_combo = QComboBox()
        self.theme_combo.addItem("Dark", "dark")
        self.theme_combo.addItem("Light", "light")
        self.theme_combo.setCurrentIndex(max(0, self.theme_combo.findData(self.settings.theme)))
        self.theme_combo.currentIndexChanged.connect(self._theme_changed)
        theme_row.addWidget(theme_label)
        theme_row.addWidget(self.theme_combo, 1)
        form.addLayout(theme_row)

        export_row = QHBoxLayout()
        self.settings_export_label = QLabel(f"Export folder: {self.export_dir}")
        self.settings_export_label.setObjectName("hint")
        self.settings_export_label.setWordWrap(True)
        choose = QPushButton("Change folder")
        choose.clicked.connect(self._choose_export_dir)
        export_row.addWidget(self.settings_export_label, 1)
        export_row.addWidget(choose)
        form.addLayout(export_row)

        self.warn_onedrive_check = QCheckBox("Warn when videos are on OneDrive")
        self.warn_onedrive_check.setChecked(self.settings.warn_onedrive)
        self.warn_onedrive_check.toggled.connect(self._settings_changed)
        form.addWidget(self.warn_onedrive_check)

        form.addWidget(self._section("Detection mode"))
        mode_hint = QLabel(
            "Motion mask is the default. Trained detector uses a YOLO .pt file — "
            "site weights at training/weights/best.pt if present, otherwise a weights file you choose "
            "(or yolo11n.pt). You can also change this on the Align step before Process."
        )
        mode_hint.setObjectName("hint")
        mode_hint.setWordWrap(True)
        form.addWidget(mode_hint)
        self.detector_combo = QComboBox()
        self.detector_combo.addItem("Motion mask", "motion")
        self.detector_combo.addItem("Trained detector", "trained")
        self.detector_combo.currentIndexChanged.connect(self._detector_mode_changed)
        form.addWidget(self.detector_combo)
        weights_row = QHBoxLayout()
        self.detector_status = QLabel("")
        self.detector_status.setObjectName("hint")
        self.detector_status.setWordWrap(True)
        choose_weights = QPushButton("Choose weights…")
        choose_weights.clicked.connect(self._choose_detector_weights)
        weights_row.addWidget(self.detector_status, 1)
        weights_row.addWidget(choose_weights)
        form.addLayout(weights_row)

        conf_hint = QLabel(
            "Trained detector keeps cars, vans, trucks, motorcycles, and buses. "
            "People, bicycles, boats, and parked vehicles are ignored. "
            "Higher confidence drops extra boxes; lower finds smaller cars."
        )
        conf_hint.setObjectName("hint")
        conf_hint.setWordWrap(True)
        form.addWidget(conf_hint)
        conf_row = QHBoxLayout()
        self.detector_conf_label = QLabel(self._detector_conf_text(self.settings.detector_conf))
        self.detector_conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.detector_conf_slider.setRange(int(DETECTOR_CONF_MIN * 100), int(DETECTOR_CONF_MAX * 100))
        self.detector_conf_slider.setValue(int(round(self.settings.detector_conf * 100)))
        self.detector_conf_slider.valueChanged.connect(self._detector_conf_changed)
        conf_row.addWidget(self.detector_conf_label)
        conf_row.addWidget(self.detector_conf_slider, 1)
        form.addLayout(conf_row)

        form.addWidget(self._section("Training data"))
        train_hint = QLabel(
            "Boxed frames live under training/sessions. Use Training → Train detector to "
            "fine-tune site weights (training/weights/best.pt). Export a zip to share or back up. "
            "Import a zip or folder to merge sessions into this machine."
        )
        train_hint.setObjectName("hint")
        train_hint.setWordWrap(True)
        form.addWidget(train_hint)
        self.training_summary = QLabel("No training data yet.")
        self.training_summary.setObjectName("hint")
        self.training_summary.setWordWrap(True)
        form.addWidget(self.training_summary)
        self.settings_train_list = QListWidget()
        self.settings_train_list.setMinimumHeight(140)
        form.addWidget(self.settings_train_list)
        train_btns = QHBoxLayout()
        export_data = QPushButton("Export…")
        export_data.clicked.connect(self._export_training_data)
        import_data = QPushButton("Import…")
        import_data.clicked.connect(self._import_training_data)
        open_train = QPushButton("Open folder")
        open_train.clicked.connect(self._open_training_folder)
        train_btns.addWidget(export_data)
        train_btns.addWidget(import_data)
        train_btns.addWidget(open_train)
        train_btns.addStretch(1)
        form.addLayout(train_btns)

        form.addStretch(1)
        layout.addWidget(card, 1)
        scroll.setWidget(body)
        outer.addWidget(scroll)
        return page

    def _section(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("sectionLabel")
        return label

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
            "Keep large bundles on a local disk. OneDrive sync can stall seeking 2–5 GB files.\n\n"
            "Processing is a motion mask, not a car classifier. On the drawing step, "
            "mask one or more typical vehicles (car, truck, …) so processing knows "
            "the size band. The preview shows each box as width×height."
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
        left_layout.addWidget(self._step_title("2. Draw on the reference frame"))
        self.route_hint = QLabel(
            "1) Set North  2) Draw entry and exit lines across the lanes  "
            "3) Box a typical vehicle. Use the buttons on each row to rename, redraw, or delete."
        )
        self.route_hint.setObjectName("hint")
        self.route_hint.setWordWrap(True)
        left_layout.addWidget(self.route_hint)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        host = QWidget()
        self.route_items = QVBoxLayout(host)
        self.route_items.setContentsMargins(0, 0, 0, 0)
        self.route_items.setSpacing(6)
        self.route_items.addStretch(1)
        scroll.setWidget(host)
        left_layout.addWidget(scroll, 1)

        add_row = QHBoxLayout()
        set_north = QPushButton("Set North")
        set_north.setObjectName("primary")
        set_north.clicked.connect(self._start_set_north)
        add_entry = QPushButton("Add entry")
        add_entry.clicked.connect(lambda: self._start_draw_lines("start"))
        add_exit = QPushButton("Add exit")
        add_exit.clicked.connect(lambda: self._start_draw_lines("end"))
        add_mask = QPushButton("Add vehicle")
        add_mask.setToolTip("Drag a box around one typical vehicle. Add several sizes if needed.")
        add_mask.clicked.connect(self._start_mask_vehicle)
        add_row.addWidget(set_north)
        add_row.addWidget(add_entry)
        add_row.addWidget(add_exit)
        add_row.addWidget(add_mask)
        left_layout.addLayout(add_row)

        extra = QHBoxLayout()
        save_geom = QPushButton("Save")
        load_geom = QPushButton("Load")
        clear_geom = QPushButton("Clear all")
        clear_geom.setObjectName("danger")
        save_geom.clicked.connect(self._save_geometry)
        load_geom.clicked.connect(self._load_geometry)
        clear_geom.clicked.connect(self._clear_geometry_lines)
        extra.addWidget(save_geom)
        extra.addWidget(load_geom)
        extra.addWidget(clear_geom)
        left_layout.addLayout(extra)

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
        self.canvas.on_box_drawn = self._on_vehicle_mask_drawn
        right_layout.addWidget(self.canvas, 1)
        row.addWidget(left, 0)
        row.addWidget(right, 1)
        left.setMinimumWidth(400)
        left.setMaximumWidth(440)
        self._refresh_route_list()
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

        self.finish_gates_btn = QPushButton("Finish start lines")
        self.finish_gates_btn.clicked.connect(self._finish_redraw_phase)
        self.finish_gates_btn.setVisible(False)
        left_layout.addWidget(self.finish_gates_btn)

        apply_match = QPushButton("Apply matched points")
        apply_match.clicked.connect(self._apply_match_points)
        left_layout.addWidget(apply_match)

        left_layout.addWidget(self._section("Detection before process"))
        self.process_detector_combo = QComboBox()
        self.process_detector_combo.addItem("Motion mask", "motion")
        self.process_detector_combo.addItem("Trained detector", "trained")
        self.process_detector_combo.currentIndexChanged.connect(self._detector_mode_changed)
        left_layout.addWidget(self.process_detector_combo)
        self.process_detector_status = QLabel("")
        self.process_detector_status.setObjectName("hint")
        self.process_detector_status.setWordWrap(True)
        left_layout.addWidget(self.process_detector_status)

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
        layout.addWidget(self._step_title("5. Report"))
        self.report_summary = QLabel("No results yet.")
        self.report_summary.setObjectName("hint")
        self.report_summary.setWordWrap(True)
        layout.addWidget(self.report_summary)
        clip_row = QHBoxLayout()
        self.report_clip_label = QLabel("Workbook")
        clip_row.addWidget(self.report_clip_label)
        self.report_clip = QComboBox()
        self.report_clip.currentIndexChanged.connect(self._show_report_clip)
        clip_row.addWidget(self.report_clip, 1)
        layout.addLayout(clip_row)
        self.report_tabs = QTabWidget()
        self.overview_table = QTableWidget()
        self.table = QTableWidget()
        self.tmc_table = QTableWidget()
        self.report_tabs.addTab(self.overview_table, "Overview")
        self.report_tabs.addTab(self.table, "Gate Statistics")
        self.report_tabs.addTab(self.tmc_table, "TMC Summary")
        layout.addWidget(self.report_tabs, 1)
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
        self.statusBar().showMessage("Ready  •  motion mask")

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
        if self.settings.warn_onedrive and self.videos and any(is_onedrive_path(p) for p in self.videos):
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
            if hasattr(self, "settings_export_label"):
                self.settings_export_label.setText(f"Export folder: {self.export_dir}")
            self._persist_settings()

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
            self.canvas.set_geometry(self.geometry)
            self._refresh_route_list()
            self.route_hint.setText(
                f"Frame {index} from {self.videos[0].name}. Set North, then add entry and exit lines across the lanes."
            )
        except Exception as exc:
            QMessageBox.warning(self, APP_NAME, str(exc))

    def _clear_layout(self, layout: QVBoxLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.hide()
                widget.blockSignals(True)
                for child in widget.findChildren(QWidget):
                    child.blockSignals(True)
                widget.setParent(None)
                widget.deleteLater()

    def _section_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("sectionLabel")
        return label

    def _item_row(
        self,
        title: str,
        *,
        name_options: tuple[str, ...] | None = None,
        current_name: str | None = None,
        on_name=None,
        on_redraw=None,
        on_delete=None,
        redraw_label: str = "Redraw",
        delete_label: str = "Delete",
    ) -> QWidget:
        row = QWidget()
        row.setObjectName("itemRow")
        layout = QHBoxLayout(row)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)
        label = QLabel(title)
        layout.addWidget(label, 1)
        if name_options and on_name:
            combo = QComboBox()
            for name in name_options:
                combo.addItem(name, name)
            if current_name:
                combo.setCurrentText(current_name)
            combo.activated.connect(lambda _idx, box=combo: on_name(box.currentData()))
            layout.addWidget(combo)
        if on_redraw:
            redraw = QPushButton(redraw_label)
            redraw.setObjectName("rowAction")
            redraw.clicked.connect(bind_row_action(on_redraw))
            layout.addWidget(redraw)
        if on_delete:
            delete = QPushButton(delete_label)
            delete.setObjectName("rowDanger")
            delete.clicked.connect(bind_row_action(on_delete))
            layout.addWidget(delete)
        return row

    def _refresh_route_list(self) -> None:
        self._clear_layout(self.route_items)
        self.route_items.addWidget(self._section_label("NORTH"))
        if self.geometry.north is not None:
            self.route_items.addWidget(
                self._item_row(
                    "North heading",
                    on_redraw=self._start_set_north,
                    on_delete=self._clear_north,
                    delete_label="Reset",
                )
            )
        else:
            self.route_items.addWidget(
                self._item_row("North not set yet", on_redraw=self._start_set_north, redraw_label="Set")
            )

        self.route_items.addWidget(self._section_label("ENTRIES  (vehicles enter here)"))
        if not self.geometry.starts:
            empty = QLabel("No entry lines. Click Add entry, then drag across the inbound lanes.")
            empty.setObjectName("hint")
            empty.setWordWrap(True)
            self.route_items.addWidget(empty)
        for item in self.geometry.starts:
            self.route_items.addWidget(
                self._item_row(
                    "Entry",
                    name_options=ORIGINS,
                    current_name=item.name,
                    on_name=lambda name, gate=item: self._rename_gate_item(gate, name),
                    on_redraw=lambda gate=item: self._redraw_gate_item(gate),
                    on_delete=lambda gate=item: self._delete_gate_item(gate),
                )
            )

        self.route_items.addWidget(self._section_label("EXITS  (vehicles leave here)"))
        if not self.geometry.ends:
            empty = QLabel("No exit lines. Click Add exit, then drag across the outbound lanes.")
            empty.setObjectName("hint")
            empty.setWordWrap(True)
            self.route_items.addWidget(empty)
        for item in self.geometry.ends:
            self.route_items.addWidget(
                self._item_row(
                    "Exit",
                    name_options=BOUNDS,
                    current_name=item.name,
                    on_name=lambda name, gate=item: self._rename_gate_item(gate, name),
                    on_redraw=lambda gate=item: self._redraw_gate_item(gate),
                    on_delete=lambda gate=item: self._delete_gate_item(gate),
                )
            )

        self.route_items.addWidget(self._section_label("VEHICLE SIZE"))
        if not self.geometry.vehicle_refs:
            empty = QLabel("No size masks. Click Add vehicle and box a typical car or truck.")
            empty.setObjectName("hint")
            empty.setWordWrap(True)
            self.route_items.addWidget(empty)
        for box in self.geometry.vehicle_refs:
            self.route_items.addWidget(
                self._item_row(
                    f"Vehicle  {int(box.width())}×{int(box.height())}  ({int(box.area())} px²)",
                    on_redraw=lambda mask=box: self._redraw_vehicle_item(mask),
                    on_delete=lambda mask=box: self._delete_vehicle_item(mask),
                )
            )
        limits = self.geometry.vehicle_area_limits(extra_areas=prior_areas())
        if limits:
            extra = "masks plus Training boxes" if prior_areas() else "these masks"
            band = QLabel(f"Keeping blobs {limits[0]}–{limits[1]} px² from {extra}.")
            band.setObjectName("hint")
            band.setWordWrap(True)
            self.route_items.addWidget(band)
        if self.geometry.is_ready():
            ready = QLabel("Movements: " + ", ".join(self.geometry.movement_names()))
            ready.setObjectName("hint")
            ready.setWordWrap(True)
            self.route_items.addWidget(ready)
        self.route_items.addStretch(1)
        self.canvas.set_geometry(self.geometry)
        self.project.geometry = self.geometry

    def _start_set_north(self, *_args) -> None:
        self._draw_step = "north"
        self._redraw_index = None
        self._redraw_target = None
        self.canvas.set_drawing_enabled(True)
        self.route_hint.setText(
            "On the photo: click the middle of the intersection, drag toward geographic North, release. "
            "If North is at the bottom of the photo, drag downward."
        )

    def _start_draw_lines(self, kind: str, *, replace=None) -> None:
        if self.geometry.north is None:
            QMessageBox.information(self, APP_NAME, "Set North first so entry and exit lines can be named.")
            return
        existing = self.geometry.starts if kind == "start" else self.geometry.ends
        if replace is None and len(existing) >= MAX_GATES:
            QMessageBox.information(self, APP_NAME, f"At most {MAX_GATES} {kind} lines. Delete one to add another.")
            return
        live = self._live_gate(replace) if replace is not None else None
        if replace is not None and (live is None or live[0] != kind):
            return
        self._draw_step = kind
        self._redraw_target = live[1] if live is not None else None
        self._redraw_index = None
        self._draw_gate_name = live[1].name if live is not None else None
        label = "entry" if kind == "start" else "exit"
        self.route_hint.setText(f"On the photo: drag a line across the {label} lanes, then release.")
        self.canvas.set_drawing_enabled(True)

    def _start_mask_vehicle(self, *_args, replace=None) -> None:
        if replace is None and len(self.geometry.vehicle_refs) >= MAX_VEHICLE_REFS:
            QMessageBox.information(self, APP_NAME, f"At most {MAX_VEHICLE_REFS} vehicle masks. Delete one to add another.")
            return
        live = self._live_vehicle(replace) if replace is not None else None
        if replace is not None and live is None:
            return
        self._draw_step = "vehicle"
        self._redraw_target = live
        self._redraw_index = None
        self.canvas.set_drawing_enabled(True, box=True)
        self.route_hint.setText("On the photo: drag a box around one typical vehicle, then release.")

    def _live_gate(self, item) -> tuple[str, NamedLine] | None:
        if not isinstance(item, NamedLine) or not item.name:
            return None
        for gate in self.geometry.starts:
            if gate is item or gate.name == item.name:
                return "start", gate
        for gate in self.geometry.ends:
            if gate is item or gate.name == item.name:
                return "end", gate
        return None

    def _live_vehicle(self, box) -> Box | None:
        if not isinstance(box, Box):
            return None
        for item in self.geometry.vehicle_refs:
            if item is box or item == box:
                return item
        return None

    def _redraw_gate_item(self, item) -> None:
        live = self._live_gate(item)
        if live is None:
            return
        self._start_draw_lines(live[0], replace=live[1])

    def _redraw_vehicle_item(self, box: Box) -> None:
        live = self._live_vehicle(box)
        if live is None:
            return
        self._start_mask_vehicle(replace=live)

    def _delete_gate_item(self, item) -> None:
        live = self._live_gate(item)
        if live is None:
            return
        kind, gate = live
        if kind == "start":
            self.geometry.starts = [existing for existing in self.geometry.starts if existing is not gate]
        else:
            self.geometry.ends = [existing for existing in self.geometry.ends if existing is not gate]
        if self._redraw_target is gate:
            self._redraw_target = None
            self._draw_step = "idle"
            self.canvas.set_drawing_enabled(False)
        self._refresh_route_list()

    def _delete_vehicle_item(self, box: Box) -> None:
        live = self._live_vehicle(box)
        if live is None:
            return
        self.geometry.vehicle_refs = [item for item in self.geometry.vehicle_refs if item is not live]
        if self._redraw_target is live:
            self._redraw_target = None
            self._draw_step = "idle"
            self.canvas.set_drawing_enabled(False)
        self._refresh_route_list()

    def _rename_gate_item(self, item, name: str | None) -> None:
        if not name:
            return
        live = self._live_gate(item)
        if live is None:
            return
        _kind, item = live
        pool = self.geometry.starts if _kind == "start" else self.geometry.ends
        taken = {other.name for other in pool if other is not item}
        if name in taken:
            QMessageBox.information(self, APP_NAME, f"{name} is already used on another line.")
            self._refresh_route_list()
            return
        if item.name == name:
            return
        item.name = name
        self._refresh_route_list()

    def _clear_north(self, *_args) -> None:
        self.geometry.north = None
        self._refresh_route_list()
        self.route_hint.setText("North cleared. Set it again before adding entry or exit lines.")

    def _on_vehicle_mask_drawn(self, box: Box) -> None:
        if self._draw_step != "vehicle":
            return
        if box.area() < 20:
            QMessageBox.information(self, APP_NAME, "That box is too small. Drag around a whole vehicle.")
            return
        if self._redraw_target is not None:
            index = next(
                (i for i, item in enumerate(self.geometry.vehicle_refs) if item is self._redraw_target),
                None,
            )
            if index is None:
                return
            self.geometry.vehicle_refs[index] = box
        elif len(self.geometry.vehicle_refs) >= MAX_VEHICLE_REFS:
            QMessageBox.information(self, APP_NAME, f"At most {MAX_VEHICLE_REFS} vehicle masks.")
            return
        else:
            self.geometry.vehicle_refs.append(box)
        self._draw_step = "idle"
        self._redraw_index = None
        self._redraw_target = None
        self.canvas.set_drawing_enabled(False)
        self.canvas.clear_draft()
        self._refresh_route_list()
        limits = self.geometry.vehicle_area_limits(extra_areas=prior_areas())
        band = f" Keeping blobs {limits[0]}–{limits[1]} px²." if limits else ""
        self.route_hint.setText(f"Saved vehicle {int(box.width())}×{int(box.height())}.{band}")

    def _clear_geometry_lines(self) -> None:
        go = QMessageBox.question(
            self,
            APP_NAME,
            "Clear North, all entry/exit lines, and vehicle masks?",
        )
        if go != QMessageBox.StandardButton.Yes:
            return
        self.geometry.north = None
        self.geometry.starts = []
        self.geometry.ends = []
        self.geometry.vehicle_refs = []
        self._draw_step = "idle"
        self._redraw_index = None
        self._redraw_target = None
        self.canvas.set_drawing_enabled(False)
        self._refresh_route_list()
        self.route_hint.setText("Cleared. Set North, then add the entry and exit lines that exist.")

    def _on_line_drawn(self, line: Line) -> None:
        if self._draw_step == "north":
            self.geometry.north = line
            self.geometry.assign_names()
            self._draw_step = "idle"
            self._redraw_index = None
            self._redraw_target = None
            self.canvas.set_drawing_enabled(False)
            self.canvas.clear_draft()
            self._refresh_route_list()
            self.route_hint.setText("North set. Add entry and exit lines across the lanes.")
            return
        if self._draw_step in ("start", "end"):
            pool = self.geometry.starts if self._draw_step == "start" else self.geometry.ends
            if self._redraw_target is not None and any(item is self._redraw_target for item in pool):
                self._redraw_target.line = line
                added_name = self._redraw_target.name
            else:
                added = self.geometry.add_gate(self._draw_step, line, self._draw_gate_name)
                if added is None:
                    QMessageBox.information(self, APP_NAME, "Could not add that gate. It may already exist or the set is full.")
                    self.canvas.clear_draft()
                    self.canvas.set_drawing_enabled(False)
                    self._draw_step = "idle"
                    self._redraw_index = None
                    self._redraw_target = None
                    return
                added_name = added.name
            self.canvas.clear_draft()
            self.canvas.set_drawing_enabled(False)
            self._draw_step = "idle"
            self._redraw_index = None
            self._redraw_target = None
            self._refresh_route_list()
            kind = "entry" if added_name in ORIGINS else "exit"
            self.route_hint.setText(f"Saved {added_name} {kind} line. Change the name on that row if needed.")

    def _save_geometry(self) -> None:
        if not self.geometry.is_ready():
            QMessageBox.information(self, APP_NAME, "Set North and draw at least one start line and one end line.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save geometry", "geometry.json", "JSON (*.json)")
        if path:
            self.geometry.save(path)
            self.statusBar().showMessage(f"Saved geometry to {path}")

    def _load_geometry(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load geometry", "", "JSON (*.json)")
        if not path:
            return
        try:
            loaded = Geometry.load(path)
            width, height = self.canvas.image_size()
            self.geometry = loaded.scaled_to(width, height) if self.canvas.has_frame() else loaded
            self._refresh_route_list()
            self.route_hint.setText(f"Loaded geometry from {Path(path).name}.")
        except Exception as exc:
            QMessageBox.warning(self, APP_NAME, f"Could not load geometry: {exc}")

    def _current_clip(self) -> ClipRecord | None:
        if not self.project.clips:
            return None
        self._align_index = min(max(0, self._align_index), len(self.project.clips) - 1)
        return self.project.clips[self._align_index]

    def _fitted_overlay(self, clip: ClipRecord) -> Geometry:
        if clip.status == "redrawn" and clip.geometry_override is not None:
            return clip.geometry_override
        width, height = self.align_canvas.image_size()
        fitted = self.geometry.scaled_to(width, height)
        return clip.transform.apply_geometry(fitted)

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
        if not self.geometry.is_ready():
            QMessageBox.information(self, APP_NAME, "Set North and draw at least one start line and one end line.")
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
        self.finish_gates_btn.setVisible(False)
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
        self.align_canvas.set_geometry(self._fitted_overlay(clip))
        self.align_canvas.set_match_points([])
        self.ref_canvas.set_match_points([])
        if clip.status == "reference":
            self.align_canvas.set_mode("view")
            self.align_hint.setText("This is the reference clip. Entry and exit lines were drawn here. Other clips need alignment.")
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
            self.align_canvas.set_geometry(self._fitted_overlay(clip))
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
            self.align_canvas.set_geometry(self._fitted_overlay(clip))
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
        self.align_canvas.set_geometry(self._fitted_overlay(clip))

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
        self.align_canvas.set_geometry(self._fitted_overlay(clip))

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
        self.align_canvas.set_geometry(self._fitted_overlay(clip))

    def _confirm_align(self) -> None:
        clip = self._current_clip()
        if clip is None:
            return
        if clip.status == "reference":
            self._next_unaligned()
            return
        if self._redrawing:
            QMessageBox.information(self, APP_NAME, "Finish redrawing the entry and exit lines on this clip first.")
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
        self.finish_gates_btn.setVisible(False)
        self._match_ref = []
        self._match_clip = []
        self.rotate_spin.blockSignals(True)
        self.rotate_spin.setValue(0)
        self.rotate_spin.blockSignals(False)
        self.align_canvas.set_mode("nudge")
        self.align_canvas.set_geometry(self._fitted_overlay(clip))
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
        self.align_canvas.set_geometry(self._fitted_overlay(clip))
        self.align_hint.setText("Overlay updated from landmarks. Nudge if needed, then Confirm alignment.")
        self._refresh_align_list()

    def _start_redraw_clip(self) -> None:
        clip = self._current_clip()
        if clip is None or clip.status == "reference":
            QMessageBox.information(self, APP_NAME, "Redraw is for non-reference clips. Edit lines on step 2.")
            return
        if not self.geometry.is_ready():
            return
        width, height = self.align_canvas.image_size()
        fitted = clip.transform.apply_geometry(self.geometry.scaled_to(width, height))
        clip.geometry_override = Geometry(width, height, north=fitted.north)
        clip.status = "needs_align"
        self._redrawing = True
        self._draw_step = "start"
        self.finish_gates_btn.setText("Finish start lines")
        self.finish_gates_btn.setVisible(True)
        self.align_canvas.set_geometry(clip.geometry_override)
        self.align_canvas.set_mode("draw")
        self.align_hint.setText("Redraw entry lines on this clip. Add only the sides that exist, then Finish start lines.")

    def _finish_redraw_phase(self) -> None:
        clip = self._current_clip()
        if not self._redrawing or clip is None or clip.geometry_override is None:
            return
        if self._draw_step == "start":
            if not clip.geometry_override.starts:
                QMessageBox.information(self, APP_NAME, "Draw at least one start line.")
                return
            clip.geometry_override.assign_names()
            self._draw_step = "end"
            self.finish_gates_btn.setText("Finish end lines")
            self.align_canvas.set_geometry(clip.geometry_override)
            self.align_hint.setText("Now redraw exit lines. Add only the sides that exist, then Finish end lines.")
            return
        if self._draw_step == "end":
            if not clip.geometry_override.ends:
                QMessageBox.information(self, APP_NAME, "Draw at least one end line.")
                return
            clip.geometry_override.assign_names()
            clip.status = "redrawn"
            self._redrawing = False
            self._draw_step = "idle"
            self.finish_gates_btn.setVisible(False)
            self.align_canvas.set_geometry(clip.geometry_override)
            self.align_canvas.set_mode("view")
            self.align_hint.setText("Custom entry and exit lines saved for this clip.")
            self._refresh_align_list()

    def _on_redraw_line(self, line: Line) -> None:
        if not self._redrawing:
            return
        clip = self._current_clip()
        if clip is None or clip.geometry_override is None:
            return
        if self._draw_step == "start":
            added = clip.geometry_override.add_gate("start", line)
            if added is None:
                self.align_hint.setText("Could not add that start line. Finish start lines or delete one.")
                return
            self.align_canvas.set_geometry(clip.geometry_override)
            self.align_hint.setText(f"Saved {added.name} start. Draw another, or Finish start lines.")
            return
        if self._draw_step == "end":
            added = clip.geometry_override.add_gate("end", line)
            if added is None:
                self.align_hint.setText("Could not add that end line. Finish end lines or delete one.")
                return
            self.align_canvas.set_geometry(clip.geometry_override)
            self.align_hint.setText(f"Saved {added.name} end. Draw another, or Finish end lines.")

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
        if not self.geometry.is_ready():
            QMessageBox.information(self, APP_NAME, "Set North and draw at least one start line and one end line.")
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
        mode = self.settings.detector_mode
        weights = resolve_detector_weights(self.settings.detector_weights)
        if mode == "trained":
            self._append_log(
                f"Detection: trained detector  •  {weights}  •  "
                f"conf {self.settings.detector_conf:.2f}  •  cars/vans/trucks/motorcycles/buses"
            )
        else:
            self._append_log("Detection: motion mask (all vehicles the same)")
        limits = self.geometry.vehicle_area_limits(extra_areas=prior_areas())
        if limits:
            extra = f" + {len(prior_areas())} training box(es)" if prior_areas() else ""
            self._append_log(
                f"Box size from {len(self.geometry.vehicle_refs)} vehicle mask(s){extra}: "
                f"{limits[0]}–{limits[1]} px²"
            )
        else:
            self._append_log("Box size: default frame band (no vehicle masks or Training boxes)")
        self._append_log(f"Processing {len(jobs)} clip(s); skipped {len(skipped)}")
        self._append_log(f"Movements: {', '.join(self.geometry.movement_names())}")

        self.worker = ProcessWorker(
            jobs=jobs,
            export_dir=self.export_dir,
            clock_start=self._clock_start(),
            detector_mode=mode,
            weights_path=self.settings.detector_weights or str(weights),
            detector_conf=self.settings.detector_conf,
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
        counts = f"  •  {result.start_crossings} start hits  •  {result.end_crossings} end hits  •  {result.frames_with_tracks} tracked frames"
        self._append_log(f"{result.video.name}: {status}{extra}{counts}")

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
            f"{len(ok_results)} video(s) processed  •  {total_trips} trips  •  "
            f"{max(0, skipped)} skipped  •  Spreadsheets saved in {self.export_dir}"
        )
        self.report_clip.blockSignals(True)
        self.report_clip.clear()
        for result in ok_results:
            self.report_clip.addItem(result.video.name)
        multi = len(ok_results) > 1
        self.report_clip.setVisible(multi)
        self.report_clip_label.setVisible(multi)
        self.report_clip.blockSignals(False)
        if not ok_results:
            self._fill_table(self.overview_table, None)
            self._fill_table(self.table, None)
            self._fill_table(self.tmc_table, None)
            return
        self.report_clip.setCurrentIndex(0)
        self._show_report_clip()

    def _show_report_clip(self) -> None:
        ok_results = [result for result in self.results if not result.error]
        if not ok_results:
            return
        index = max(0, min(self.report_clip.currentIndex(), len(ok_results) - 1))
        result = ok_results[index]
        clock = self._clock_start()
        overview = build_overview_table(
            result.trips,
            video_name=result.video.name,
            duration_s=result.video.duration_s,
            clock_start=clock,
            device_label=result.device_label,
        )
        gates = build_gate_statistics_table(result.trips, result.geometry)
        tmc = build_tmc_summary_table(
            result.trips, result.geometry, result.video.duration_s, clock
        )
        self._fill_table(self.overview_table, overview, left_align_first=True)
        self._fill_table(self.table, gates)
        self._fill_table(self.tmc_table, tmc, left_align_first=True)

    def _fill_table(self, widget: QTableWidget, table, *, left_align_first: bool = False) -> None:
        widget.clear()
        widget.setRowCount(0)
        widget.setColumnCount(0)
        if table is None or table.empty:
            return
        widget.setRowCount(len(table))
        widget.setColumnCount(len(table.columns))
        widget.setHorizontalHeaderLabels([str(col) for col in table.columns])
        for r_idx, (_, row) in enumerate(table.iterrows()):
            for c_idx, value in enumerate(row):
                text = "" if value is None or value == "" else str(value)
                item = QTableWidgetItem(text)
                if c_idx == 0 and left_align_first:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
                else:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                widget.setItem(r_idx, c_idx, item)
        widget.resizeColumnsToContents()
        header = widget.horizontalHeader()
        if header is not None:
            header.setStretchLastSection(True)

    def _open_export_dir(self) -> None:
        from PyQt6.QtCore import QUrl
        from PyQt6.QtGui import QDesktopServices

        self.export_dir.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.export_dir.resolve())))

    def _reset_flow(self) -> None:
        self.stack.setCurrentIndex(PAGE_SETUP)

    def _on_page_changed(self, index: int) -> None:
        home = index == PAGE_HOME
        self.header_bar.setVisible(not home)
        self.steps.setVisible(index in PROCESS_PAGES)
        self.home_btn.setVisible(not home)
        if home:
            self.content_layout.setContentsMargins(0, 0, 0, 0)
            self.statusBar().hide()
            self.home_stage.focus_menu()
        else:
            self.content_layout.setContentsMargins(16, 12, 16, 16)
            self.statusBar().show()
        if index == PAGE_TRAINING:
            self._refresh_training_sessions()
        if index == PAGE_SETTINGS:
            self._refresh_settings_page()
        if index == PAGE_ALIGN:
            self._sync_detector_ui()

    def _go_home(self) -> None:
        self.stack.setCurrentIndex(PAGE_HOME)

    def _go_process_feature(self) -> None:
        self.stack.setCurrentIndex(PAGE_SETUP)

    def _go_training(self) -> None:
        self.stack.setCurrentIndex(PAGE_TRAINING)
        self._refresh_train_ready_status()

    def _go_settings(self) -> None:
        self.stack.setCurrentIndex(PAGE_SETTINGS)

    def _persist_settings(self) -> None:
        mode = "motion"
        if hasattr(self, "detector_combo"):
            mode = self.detector_combo.currentData() or "motion"
        warn = self.settings.warn_onedrive
        if hasattr(self, "warn_onedrive_check"):
            warn = self.warn_onedrive_check.isChecked()
        theme = self.settings.theme
        if hasattr(self, "theme_combo"):
            theme = self.theme_combo.currentData() or "dark"
        conf = self.settings.detector_conf
        if hasattr(self, "detector_conf_slider"):
            conf = clamp_detector_conf(self.detector_conf_slider.value() / 100.0)
        self.settings = AppSettings(
            export_dir=str(self.export_dir),
            warn_onedrive=warn,
            detector_mode=mode,
            detector_weights=self.settings.detector_weights,
            theme=theme if theme in ("dark", "light") else "dark",
            detector_conf=conf,
        )
        save_settings(self.settings)
        self._refresh_setup_list()

    def _settings_changed(self, *_args) -> None:
        self._persist_settings()

    def _detector_conf_text(self, conf: float) -> str:
        return f"Detector confidence: {conf:.2f}"

    def _detector_conf_changed(self, value: int) -> None:
        conf = clamp_detector_conf(value / 100.0)
        if hasattr(self, "detector_conf_label"):
            self.detector_conf_label.setText(self._detector_conf_text(conf))
        self._persist_settings()

    def _detector_mode_changed(self, *_args) -> None:
        sender = self.sender()
        if sender is not None and hasattr(sender, "currentData"):
            mode = sender.currentData() or "motion"
            for combo in (getattr(self, "detector_combo", None), getattr(self, "process_detector_combo", None)):
                if combo is None or combo is sender:
                    continue
                combo.blockSignals(True)
                combo.setCurrentIndex(max(0, combo.findData(mode)))
                combo.blockSignals(False)
        self._persist_settings()
        self._refresh_detector_status()

    def _choose_detector_weights(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose detector weights",
            str(trained_weights_path().parent),
            "YOLO weights (*.pt *.onnx);;All files (*.*)",
        )
        if not path:
            return
        self.settings.detector_weights = path
        if hasattr(self, "detector_combo"):
            self.detector_combo.blockSignals(True)
            self.detector_combo.setCurrentIndex(max(0, self.detector_combo.findData("trained")))
            self.detector_combo.blockSignals(False)
        if hasattr(self, "process_detector_combo"):
            self.process_detector_combo.blockSignals(True)
            self.process_detector_combo.setCurrentIndex(max(0, self.process_detector_combo.findData("trained")))
            self.process_detector_combo.blockSignals(False)
        self._persist_settings()
        self._refresh_detector_status()

    def _sync_detector_ui(self) -> None:
        mode = self.settings.detector_mode if self.settings.detector_mode in ("motion", "trained") else "motion"
        for combo in (getattr(self, "detector_combo", None), getattr(self, "process_detector_combo", None)):
            if combo is None:
                continue
            combo.blockSignals(True)
            combo.setCurrentIndex(max(0, combo.findData(mode)))
            combo.blockSignals(False)
        self._refresh_detector_status()

    def _refresh_detector_status(self) -> None:
        weights = resolve_detector_weights(self.settings.detector_weights)
        if has_trained_weights() and (
            not self.settings.detector_weights or Path(self.settings.detector_weights) == trained_weights_path()
        ):
            text = f"Using site weights: {trained_weights_path()}"
        elif self.settings.detector_weights and Path(self.settings.detector_weights).is_file():
            text = f"Using {self.settings.detector_weights}"
        elif weights.is_file():
            text = f"Using {weights} (no site best.pt yet)"
        else:
            text = (
                f"Will load {weights.name} on process (download if needed). "
                "Choose a .pt file to use your own weights."
            )
        if hasattr(self, "detector_status"):
            self.detector_status.setText(text)
        if hasattr(self, "process_detector_status"):
            self.process_detector_status.setText(text)

    def _theme_changed(self, *_args) -> None:
        self._persist_settings()
        self._apply_theme(self.settings.theme)

    def _apply_theme(self, theme: str) -> None:
        from PyQt6.QtWidgets import QApplication

        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(stylesheet_for(theme))
        if hasattr(self, "home_stage"):
            self.home_stage.set_theme(theme)

    def _refresh_training_inventory(self) -> None:
        info = training_inventory()
        prior = "size prior on" if info["has_prior"] else "no size prior"
        weights = "weights present" if info["has_weights"] else "no weights"
        self.training_summary.setText(
            f"{info['root']}  •  {info['session_count']} session(s), "
            f"{info['frame_count']} frame(s), {info['box_count']} box(es)  •  {prior}, {weights}"
        )
        self.settings_train_list.clear()
        for session in info["sessions"]:
            self.settings_train_list.addItem(self._training_session_label(session))
        self._refresh_train_ready_status()

    def _export_training_data(self) -> None:
        info = training_inventory()
        if info["session_count"] == 0:
            QMessageBox.information(self, APP_NAME, "No training sessions to export yet.")
            return
        dest, _ = QFileDialog.getSaveFileName(
            self,
            "Export training data",
            str(Path.cwd() / "training-data.zip"),
            "Zip archive (*.zip)",
        )
        if not dest:
            return
        try:
            path = export_training_archive(Path(dest))
        except (OSError, FileNotFoundError) as exc:
            QMessageBox.warning(self, APP_NAME, str(exc))
            return
        self.statusBar().showMessage(f"Exported training data to {path}")
        QMessageBox.information(self, APP_NAME, f"Saved {path}")

    def _import_training_data(self) -> None:
        src, _ = QFileDialog.getOpenFileName(
            self,
            "Import training data",
            "",
            "Zip archive (*.zip);;All files (*.*)",
        )
        folder = ""
        if not src:
            folder = QFileDialog.getExistingDirectory(self, "Or choose a training folder")
            if not folder:
                return
            src = folder
        incoming = training_inventory()
        replace = False
        if incoming["session_count"]:
            go = QMessageBox.question(
                self,
                APP_NAME,
                "Replace existing sessions with the same name?\n\n"
                "Yes replaces matches. No keeps yours and only adds new names.",
            )
            replace = go == QMessageBox.StandardButton.Yes
        try:
            result = import_training_archive(Path(src), replace=replace)
        except (OSError, ValueError, FileNotFoundError) as exc:
            QMessageBox.warning(self, APP_NAME, str(exc))
            return
        self._refresh_training_inventory()
        self._refresh_training_sessions()
        QMessageBox.information(
            self,
            APP_NAME,
            f"Imported {result['imported']}, replaced {result['replaced']}, skipped {result['skipped']}.",
        )

    def _open_training_folder(self) -> None:
        from PyQt6.QtCore import QUrl
        from PyQt6.QtGui import QDesktopServices

        folder = training_root()
        folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder.resolve())))

    def _refresh_settings_page(self) -> None:
        self.settings_export_label.setText(f"Export folder: {self.export_dir}")
        self.warn_onedrive_check.blockSignals(True)
        self.warn_onedrive_check.setChecked(self.settings.warn_onedrive)
        self.warn_onedrive_check.blockSignals(False)
        self.theme_combo.blockSignals(True)
        self.theme_combo.setCurrentIndex(max(0, self.theme_combo.findData(self.settings.theme)))
        self.theme_combo.blockSignals(False)
        self._refresh_training_inventory()
        self._sync_detector_ui()

    def _training_session_label(self, session: TrainingSession) -> str:
        saved = session.saved_at[:19].replace("T", " ") if session.saved_at else "unsaved"
        return (
            f"{session.video_stem}  •  {len(session.frames)} frame(s), "
            f"{session.box_count()} box(es)  •  {saved}"
        )

    def _refresh_training_sessions(self) -> None:
        self.train_sessions.clear()
        for session in list_sessions():
            self.train_sessions.addItem(self._training_session_label(session))
            self.train_sessions.item(self.train_sessions.count() - 1).setData(
                Qt.ItemDataRole.UserRole, session.video_stem
            )
        areas = prior_areas()
        if areas:
            limits = self.geometry.vehicle_area_limits(extra_areas=areas)
            if limits:
                self.train_status.setText(
                    f"{len(areas)} training box(es) in the size prior. "
                    f"Process will keep blobs {limits[0]}–{limits[1]} px²."
                )
        self._refresh_train_ready_status()

    def _refresh_train_ready_status(self) -> None:
        if not hasattr(self, "train_ready_label"):
            return
        info = training_inventory()
        report = device_report()
        extra = " — this will take a long time" if report.device == "cpu" else ""
        self.train_ready_label.setText(
            f"{info['frame_count']} labeled frame(s)  •  {info['box_count']} box(es)  •  "
            f"{report.label}{extra}"
        )

    def _set_train_busy(self, busy: bool) -> None:
        if hasattr(self, "train_detector_btn"):
            self.train_detector_btn.setEnabled(not busy)
        if hasattr(self, "train_visdrone_btn"):
            self.train_visdrone_btn.setEnabled(not busy)
        if hasattr(self, "cancel_train_btn"):
            self.cancel_train_btn.setEnabled(busy)

    def _start_detector_train(self) -> None:
        if self.train_worker and self.train_worker.isRunning():
            return
        info = training_inventory()
        if info["frame_count"] < TRAIN_MIN_FRAMES or info["box_count"] < TRAIN_MIN_BOXES:
            QMessageBox.information(
                self,
                APP_NAME,
                f"Need at least {TRAIN_MIN_FRAMES} labeled frames and {TRAIN_MIN_BOXES} boxes. "
                f"Found {info['frame_count']} frame(s) and {info['box_count']} box(es). "
                "Box every vehicle on more frames, then try again.",
            )
            return
        report = device_report()
        if report.device == "cpu":
            go = QMessageBox.question(
                self,
                APP_NAME,
                report.warning or "Fine-tuning YOLO on CPU can take hours. Continue?",
            )
            if go != QMessageBox.StandardButton.Yes:
                return
        self._launch_train_worker("site", "Starting site detector training…")

    def _start_visdrone_train(self) -> None:
        if self.train_worker and self.train_worker.isRunning():
            return
        go = QMessageBox.question(
            self,
            APP_NAME,
            "This downloads VisDrone2019-DET (~2 GB) and fine-tunes YOLO11n. "
            "It can take hours. Your boxed Training sessions stay in place so we can compare. Continue?",
        )
        if go != QMessageBox.StandardButton.Yes:
            return
        report = device_report()
        if report.device == "cpu":
            go = QMessageBox.question(
                self,
                APP_NAME,
                report.warning or "Fine-tuning YOLO on CPU can take hours. Continue?",
            )
            if go != QMessageBox.StandardButton.Yes:
                return
        self._launch_train_worker("visdrone", "Starting VisDrone training…")

    def _launch_train_worker(self, source: str, start_message: str) -> None:
        self.train_log.clear()
        self.train_progress.setValue(0)
        self.train_log.appendPlainText(start_message)
        self.train_worker = TrainWorker(self, source=source)
        self.train_worker.progress.connect(self._on_train_progress)
        self.train_worker.log_line.connect(self._on_train_log)
        self.train_worker.failed.connect(self._on_train_failed)
        self.train_worker.train_done.connect(self._on_train_done)
        self._set_train_busy(True)
        self.train_worker.start()

    def _cancel_detector_train(self) -> None:
        if self.train_worker and self.train_worker.isRunning():
            self.train_worker.cancel()
            self.train_log.appendPlainText("Cancel requested. The current epoch may finish first.")

    def _on_train_progress(self, current: int, total: int, message: str) -> None:
        self.train_progress.setMaximum(max(int(total), 1))
        self.train_progress.setValue(int(current))
        if message:
            self.train_log.appendPlainText(message)

    def _on_train_log(self, message: str) -> None:
        if message:
            self.train_log.appendPlainText(message)

    def _on_train_failed(self, message: str) -> None:
        self._set_train_busy(False)
        self.train_log.appendPlainText(message)
        QMessageBox.warning(self, APP_NAME, message)

    def _on_train_done(self, path: str) -> None:
        self._set_train_busy(False)
        self.train_progress.setValue(self.train_progress.maximum())
        self.train_log.appendPlainText(f"Saved {path}")
        reset_detector_cache()
        self.settings.detector_weights = str(trained_weights_path())
        for combo in (getattr(self, "detector_combo", None), getattr(self, "process_detector_combo", None)):
            if combo is None:
                continue
            combo.blockSignals(True)
            combo.setCurrentIndex(max(0, combo.findData("trained")))
            combo.blockSignals(False)
        self._persist_settings()
        self._refresh_detector_status()
        self._refresh_training_inventory()
        self._refresh_train_ready_status()
        QMessageBox.information(
            self,
            APP_NAME,
            "Site weights saved. Process will use Trained detector. "
            "Change it back to Motion mask in Settings if you need the old path.",
        )

    def _training_pick_video(self) -> None:
        start = str(self.videos[0].parent) if self.videos else ""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select a training video",
            start,
            "Video files (*.mp4 *.mov *.avi *.mkv *.m4v *.wmv);;All files (*.*)",
        )
        if not path:
            return
        video = Path(path)
        self._train_video = video
        self.train_video_label.setText(self._video_label(video))
        try:
            self._train_session = load_session(video.stem)
        except (OSError, ValueError, KeyError, TypeError):
            self._train_session = TrainingSession(video_path=str(video), video_stem=video.stem)
        else:
            self._train_session.video_path = str(video)
        self._train_frames_bgr = {}
        self._training_next_frame()

    def _training_select_session(self) -> None:
        return None

    def _training_load_selected(self) -> None:
        items = self.train_sessions.selectedItems()
        if not items:
            return
        stem = items[0].data(Qt.ItemDataRole.UserRole)
        if not stem:
            return
        try:
            session = load_session(str(stem))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            QMessageBox.warning(self, APP_NAME, f"Could not load session: {exc}")
            return
        self._train_session = session
        video = Path(session.video_path)
        if video.is_file():
            self._train_video = video
            self.train_video_label.setText(self._video_label(video))
        else:
            self._train_video = None
            self.train_video_label.setText(f"{session.video_stem} (video missing — pick the file to continue)")
        self._train_frames_bgr = {}
        if session.frames:
            last = session.frames[-1]
            self._train_boxes = list(last.boxes)
            self._train_frame_index = last.frame_index
            self._train_time_s = last.time_s
            if video.is_file():
                try:
                    frame, index = grab_frame_at(video, last.time_s)
                    self._train_frame = frame
                    self._train_frame_index = index
                    self._train_frames_bgr[index] = frame
                    self._show_training_frame()
                    return
                except Exception:
                    pass
        if video.is_file():
            self._training_next_frame()
        else:
            self._show_training_frame()

    def _training_delete_selected(self) -> None:
        items = self.train_sessions.selectedItems()
        if not items:
            return
        stem = items[0].data(Qt.ItemDataRole.UserRole)
        if not stem:
            return
        go = QMessageBox.question(self, APP_NAME, f"Delete training session {stem}?")
        if go != QMessageBox.StandardButton.Yes:
            return
        delete_session(str(stem))
        if self._train_session and self._train_session.video_stem == stem:
            self._train_session = None
            self._train_boxes = []
            self._train_frame = None
        self._refresh_training_sessions()

    def _stash_training_frame(self) -> None:
        if self._train_session is None or self._train_frame is None:
            return
        height, width = self._train_frame.shape[:2]
        image_name = f"frame_{self._train_frame_index:06d}.jpg"
        labeled = LabeledFrame(
            frame_index=self._train_frame_index,
            image_name=image_name,
            width=width,
            height=height,
            boxes=list(self._train_boxes),
            time_s=self._train_time_s,
        )
        existing = [f for f in self._train_session.frames if f.frame_index != self._train_frame_index]
        if self._train_boxes:
            existing.append(labeled)
            self._train_frames_bgr[self._train_frame_index] = self._train_frame
        self._train_session.frames = sorted(existing, key=lambda item: item.frame_index)

    def _training_next_frame(self) -> None:
        if self._train_video is None:
            if self.videos:
                self._train_video = self.videos[0]
                self.train_video_label.setText(self._video_label(self._train_video))
                if self._train_session is None:
                    try:
                        self._train_session = load_session(self._train_video.stem)
                    except (OSError, ValueError, KeyError, TypeError):
                        self._train_session = TrainingSession(
                            video_path=str(self._train_video),
                            video_stem=self._train_video.stem,
                        )
            else:
                QMessageBox.information(self, APP_NAME, "Open a video first.")
                return
        self._stash_training_frame()
        try:
            frame, index = grab_random_frame(self._train_video)
        except Exception as exc:
            QMessageBox.warning(self, APP_NAME, f"Could not read a frame: {exc}")
            return
        self._train_frame = frame
        self._train_frame_index = index
        info = probe_video(self._train_video)
        self._train_time_s = index / info.fps if info.fps else 0.0
        reused = None
        if self._train_session:
            reused = next((f for f in self._train_session.frames if f.frame_index == index), None)
        self._train_boxes = list(reused.boxes) if reused else []
        self._train_frames_bgr[index] = frame
        self._show_training_frame()

    def _show_training_frame(self) -> None:
        if self._train_frame is None:
            self.train_canvas.set_geometry(None)
            return
        self.train_canvas.set_frame(self._train_frame)
        height, width = self._train_frame.shape[:2]
        geom = Geometry(width, height, vehicle_refs=list(self._train_boxes))
        self.train_canvas.set_geometry(geom)
        self.train_canvas.set_drawing_enabled(True, box=True)
        boxed = len(self._train_boxes)
        labeled = len(self._train_session.frames) if self._train_session else 0
        self.train_status.setText(
            f"Frame {self._train_frame_index}  •  {boxed} box(es) on this frame  •  "
            f"{labeled} labeled frame(s) in the session. Drag a box around each vehicle."
        )

    def _training_box_drawn(self, box: Box) -> None:
        if box.area() < 4:
            return
        self._train_boxes.append(box.normalized())
        self._show_training_frame()

    def _training_undo_box(self) -> None:
        if self._train_boxes:
            self._train_boxes.pop()
            self._show_training_frame()

    def _training_save(self) -> None:
        if self._train_session is None:
            QMessageBox.information(self, APP_NAME, "Open a video and box vehicles first.")
            return
        self._stash_training_frame()
        if not self._train_session.frames:
            QMessageBox.information(self, APP_NAME, "Box at least one vehicle before saving.")
            return
        folder = save_session(self._train_session, self._train_frames_bgr)
        self._refresh_training_sessions()
        if hasattr(self, "settings_train_list"):
            self._refresh_training_inventory()
        self.statusBar().showMessage(f"Saved training session to {folder}")
        QMessageBox.information(
            self,
            APP_NAME,
            "Session saved. Process will use these box sizes even if you drew no route masks.",
        )

    def _append_log(self, text: str) -> None:
        self.log.appendPlainText(text)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(2000)
        if self.train_worker and self.train_worker.isRunning():
            self.train_worker.cancel()
            self.train_worker.wait(2000)
        event.accept()
