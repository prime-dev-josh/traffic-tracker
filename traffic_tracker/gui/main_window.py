"""Videos → routes → align → process → report."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QCloseEvent, QImage, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDateTimeEdit,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
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
    DETECTOR_IMGSZ_CHOICES,
    DETECTOR_IOU_MAX,
    DETECTOR_IOU_MIN,
    DETECTOR_LINK_SPEED_MAX,
    DETECTOR_LINK_SPEED_MIN,
    DETECTOR_PARKED_RADIUS_MAX,
    DETECTOR_PARKED_RADIUS_MIN,
    DETECTOR_PATH_JUMP_MAX,
    DETECTOR_PATH_JUMP_MIN,
    DETECTOR_STRIDE_MAX,
    EXPORT_PATTERN_PRESETS,
    PREVIEW_EVERY_CHOICES,
    PREDICTION_REACH_MAX,
    PREDICTION_REACH_MIN,
    PRECISION_MODE_LABELS,
    PREDICTION_REACH_PX,
)
from traffic_tracker.core.geometry import (
    GATE_FLOW_LABELS,
    GATE_FLOWS,
    MAX_GATES,
    ORIGIN_TO_BOUND,
    ORIGINS,
    MAX_OCCLUSIONS,
    MAX_ROADWAYS,
    MAX_VEHICLE_REFS,
    Box,
    Geometry,
    Line,
    NamedLine,
    Polygon,
    gate_origin_name,
    Transform,
    polygon_is_valid,
    drop_zone_too_wide,
    roadway_approach_warnings,
    transform_from_matches,
)


def bind_row_action(callback):
    """Run a row callback without Qt's clicked(checked) overwriting default args."""

    def _run(*_args):
        callback()

    return _run
from traffic_tracker.core.processor import ProcessResult, draw_issue_overlay
from traffic_tracker.core.project import BundleProject, ClipRecord
from traffic_tracker.core.report import (
    build_gate_statistics_table,
    build_overview_table,
    build_tmc_summary_table,
)
from traffic_tracker.core.settings import (
    AppSettings,
    apply_precision_preset,
    clamp_detect_stride,
    clamp_detector_conf,
    clamp_detector_imgsz,
    clamp_detector_iou,
    clamp_export_pattern,
    clamp_link_speed,
    clamp_parked_radius,
    clamp_path_jump,
    clamp_preview_every,
    load_settings,
    resolve_export_folder,
    save_settings,
)
from traffic_tracker.core.detector import (
    boxes_from_detector,
    draw_detection_preview,
    load_detector,
    reset_detector_cache,
)
from traffic_tracker.core.device import device_report
from traffic_tracker.core.training import visdrone_weights_path
from traffic_tracker.gui.style import load_brand_pixmap, stylesheet_for
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
        self._filling_preset = False
        self.export_dir = Path(self.settings.export_dir) if self.settings.export_dir else Path.cwd() / "outputs"
        self.worker: ProcessWorker | None = None
        self.train_worker: TrainWorker | None = None
        self.results: list[ProcessResult] = []
        self._draw_step = "idle"
        self._roadway_edit: int | None = None
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

        root = QWidget()
        self.setCentralWidget(root)
        self._root_layout = QVBoxLayout(root)
        self._root_layout.setContentsMargins(0, 0, 0, 0)
        self._root_layout.setSpacing(0)

        self.header_bar = QWidget()
        title_row = QHBoxLayout(self.header_bar)
        title_row.setContentsMargins(16, 16, 16, 0)
        self.header_logo = QLabel()
        self.header_logo.setObjectName("brandMark")
        self.header_logo.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        title_row.addWidget(self.header_logo, 0, Qt.AlignmentFlag.AlignTop)
        header = QVBoxLayout()
        title = QLabel(APP_NAME)
        title.setObjectName("title")
        subtitle = QLabel("Count vehicles across N, S, E, and W gates, then export a 15-minute TMC.")
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

    def _wide_button(
        self,
        text: str,
        *,
        primary: bool = False,
        danger: bool = False,
        quiet: bool = False,
        tooltip: str = "",
    ) -> QPushButton:
        button = QPushButton(text)
        if primary:
            button.setObjectName("primary")
        elif danger:
            button.setObjectName("danger")
        elif quiet:
            button.setObjectName("quiet")
        if tooltip:
            button.setToolTip(tooltip)
        button.setMinimumHeight(28 if quiet else 36)
        button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        return button

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
                    "Train VisDrone once. New installs need this; weights are not bundled.",
                    self._go_training,
                ),
                (
                    "03",
                    "Settings",
                    "Export folder, OneDrive warning, and Motion vs Trained detector.",
                    self._go_settings,
                ),
            ]
        )
        return self.home_stage

    def _build_training_page(self) -> QWidget:
        page = QWidget()
        row = QHBoxLayout(page)
        row.setSpacing(16)

        card = self._card()
        layout = QVBoxLayout(card)
        layout.addWidget(self._step_title("Train VisDrone"))
        hint = QLabel(
            "Weights are not bundled with a new install. Train once to download VisDrone2019-DET "
            "(~2 GB) and fine-tune YOLO11n. The result is training/weights/visdrone.pt. "
            "Then set Detection mode to Trained detector. This can take hours; a GPU is much faster."
        )
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.train_status = QLabel("")
        self.train_status.setObjectName("hint")
        self.train_status.setWordWrap(True)
        layout.addWidget(self.train_status)

        train_row = QHBoxLayout()
        self.train_visdrone_btn = QPushButton("Train VisDrone")
        self.train_visdrone_btn.setObjectName("primary")
        self.train_visdrone_btn.clicked.connect(self._start_visdrone_train)
        self.cancel_train_btn = QPushButton("Cancel")
        self.cancel_train_btn.setEnabled(False)
        self.cancel_train_btn.clicked.connect(self._cancel_detector_train)
        open_folder = QPushButton("Open weights folder")
        open_folder.clicked.connect(self._open_training_folder)
        train_row.addWidget(self.train_visdrone_btn)
        train_row.addWidget(self.cancel_train_btn)
        train_row.addWidget(open_folder)
        layout.addLayout(train_row)

        self.train_progress = QProgressBar()
        self.train_progress.setValue(0)
        layout.addWidget(self.train_progress)
        self.train_log = QPlainTextEdit()
        self.train_log.setReadOnly(True)
        self.train_log.setPlaceholderText("Training log")
        layout.addWidget(self.train_log, 1)

        row.addWidget(card, 1)
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

        form.addWidget(self._section("Export"))
        export_row = QHBoxLayout()
        self.settings_export_label = QLabel(f"Export folder: {self.export_dir}")
        self.settings_export_label.setObjectName("hint")
        self.settings_export_label.setWordWrap(True)
        choose = QPushButton("Change folder")
        choose.clicked.connect(self._choose_export_dir)
        export_row.addWidget(self.settings_export_label, 1)
        export_row.addWidget(choose)
        form.addLayout(export_row)

        pattern_row = QHBoxLayout()
        pattern_row.addWidget(QLabel("Subfolder"))
        self.export_pattern_combo = QComboBox()
        for pattern, label in EXPORT_PATTERN_PRESETS:
            self.export_pattern_combo.addItem(label, pattern)
        self.export_pattern_combo.addItem("Custom pattern", "custom")
        pattern_row.addWidget(self.export_pattern_combo, 1)
        form.addLayout(pattern_row)

        self.export_pattern_edit = QLineEdit()
        self.export_pattern_edit.setPlaceholderText("{video}/{date}_{time}")
        self.export_pattern_edit.setText(self.settings.export_pattern)
        self.export_pattern_edit.setToolTip(
            "Creates a new folder for each run. Tokens: {video} {date} {time} {datetime}. "
            "Leave blank to save directly in the export folder."
        )
        form.addWidget(self.export_pattern_edit)
        pattern_hint = QLabel(
            "Each process run writes the spreadsheet and path images into a new folder. "
            "Use {video} for the clip name, {date} and {time} for the run stamp. "
            "Blank saves into the export folder with no extra subfolder."
        )
        pattern_hint.setObjectName("hint")
        pattern_hint.setWordWrap(True)
        form.addWidget(pattern_hint)
        self.export_pattern_preview = QLabel("")
        self.export_pattern_preview.setObjectName("hint")
        self.export_pattern_preview.setWordWrap(True)
        form.addWidget(self.export_pattern_preview)
        self._sync_export_pattern_combo()
        self._refresh_export_preview()
        self.export_pattern_combo.currentIndexChanged.connect(self._export_pattern_preset_changed)
        self.export_pattern_edit.textChanged.connect(self._export_pattern_edited)

        self.warn_onedrive_check = QCheckBox("Warn when videos are on OneDrive")
        self.warn_onedrive_check.setChecked(self.settings.warn_onedrive)
        self.warn_onedrive_check.toggled.connect(self._settings_changed)
        form.addWidget(self.warn_onedrive_check)

        form.addWidget(self._section("Detection mode"))
        mode_hint = QLabel(
            "Motion mask is the default. Trained detector uses VisDrone weights at "
            "training/weights/visdrone.pt. Train those on the Training page — they are "
            "not shipped with a new install. You can also switch Motion vs Trained on Align."
        )
        mode_hint.setObjectName("hint")
        mode_hint.setWordWrap(True)
        form.addWidget(mode_hint)
        self.detector_combo = QComboBox()
        self.detector_combo.addItem("Motion mask", "motion")
        self.detector_combo.addItem("Trained detector", "trained")
        self.detector_combo.currentIndexChanged.connect(self._detector_mode_changed)
        form.addWidget(self.detector_combo)
        self.detector_status = QLabel("")
        self.detector_status.setObjectName("hint")
        self.detector_status.setWordWrap(True)
        form.addWidget(self.detector_status)
        self.infer_trips_check = QCheckBox("Infer trips from traces")
        self.infer_trips_check.setChecked(self.settings.infer_trips_from_traces)
        self.infer_trips_check.setToolTip(
            "When on, a path that crosses a gate still counts if two detector boxes "
            "did not straddle the drawn line cleanly. Off only counts a confirmed box-to-box crossing. "
            "Trained detector starts off so parked or predicted IDs do not invent trips."
        )
        self.infer_trips_check.toggled.connect(self._infer_trips_changed)
        form.addWidget(self.infer_trips_check)
        infer_hint = QLabel(
            "An ID can survive a gate without showing N→ when the boxes never cleanly straddle "
            "the tripwire (detect-every-N, a short gate, or a hold/predict across the line). "
            "Turn this on to take the entry/exit from the trace instead."
        )
        infer_hint.setObjectName("hint")
        infer_hint.setWordWrap(True)
        form.addWidget(infer_hint)

        form.addWidget(self._section("Tracking mode"))
        preset_hint = QLabel(
            "One box per car hides weak extra boxes. Through shadows keeps IDs across drop zones. "
            "Keep queued cars is looser when cars sit in a line. "
            "Numeric sliders are under Advanced. These apply to Trained detector."
        )
        preset_hint.setObjectName("hint")
        preset_hint.setWordWrap(True)
        form.addWidget(preset_hint)
        self.precision_combo = QComboBox()
        for key in ("strict", "balanced", "sensitive", "custom"):
            self.precision_combo.addItem(PRECISION_MODE_LABELS[key], key)
        self.precision_combo.setCurrentIndex(max(0, self.precision_combo.findData(self.settings.precision_preset)))
        self.precision_combo.currentIndexChanged.connect(self._precision_preset_changed)
        form.addWidget(self.precision_combo)

        self.advanced_btn = QPushButton("Show advanced sliders")
        self.advanced_btn.setCheckable(True)
        self.advanced_btn.setChecked(False)
        self.advanced_btn.toggled.connect(self._toggle_advanced_sliders)
        form.addWidget(self.advanced_btn)
        self.advanced_panel = QWidget()
        self.advanced_panel.setVisible(False)
        adv = QVBoxLayout(self.advanced_panel)
        adv.setContentsMargins(0, 8, 0, 0)
        adv.setSpacing(8)

        self.detector_conf_label, self.detector_conf_slider = self._add_setting_slider(
            adv,
            "Detector confidence: {value:.2f}",
            int(DETECTOR_CONF_MIN * 100),
            int(DETECTOR_CONF_MAX * 100),
            int(round(self.settings.detector_conf * 100)),
            100.0,
            "How sure the model must be before it draws a box. "
            "Raise to hide ghosts, poles, and boats labeled as cars. "
            "Lower if small moving cars disappear.",
        )
        self.detector_iou_label, self.detector_iou_slider = self._add_setting_slider(
            adv,
            "Overlap (NMS IoU): {value:.2f}",
            int(DETECTOR_IOU_MIN * 100),
            int(DETECTOR_IOU_MAX * 100),
            int(round(self.settings.detector_iou * 100)),
            100.0,
            "How much two boxes can overlap before one is removed. "
            "Lower means fewer double boxes on the same car. "
            "Raise to keep queued cars as separate boxes.",
        )
        self.parked_radius_label, self.parked_radius_slider = self._add_setting_slider(
            adv,
            "Parked ignore radius: {value:.0f} px",
            int(DETECTOR_PARKED_RADIUS_MIN),
            int(DETECTOR_PARKED_RADIUS_MAX),
            int(round(self.settings.parked_radius)),
            1.0,
            "How far a still car can drift (hover) and still stay parked. "
            "Raise if parked cars start traces. "
            "Lower if a car that just pulled out is ignored.",
        )
        self.link_speed_label, self.link_speed_slider = self._add_setting_slider(
            adv,
            "Track link speed: {value:.0f} px/s",
            int(DETECTOR_LINK_SPEED_MIN),
            int(DETECTOR_LINK_SPEED_MAX),
            int(round(self.settings.link_speed)),
            1.0,
            "How fast a car can move and still keep the same ID. "
            "Raise to stitch through missed frames. "
            "Lower if two nearby cars share an ID.",
        )
        self.path_jump_label, self.path_jump_slider = self._add_setting_slider(
            adv,
            "Trace jump split: {value:.0f} px",
            int(DETECTOR_PATH_JUMP_MIN),
            int(DETECTOR_PATH_JUMP_MAX),
            int(round(self.settings.path_jump)),
            1.0,
            "If the path hops farther than this, the line is cut. "
            "Raise to keep one stroke through a gap. "
            "Lower to stop lot-spanning teleport lines.",
        )
        form.addWidget(self.advanced_panel)

        form.addWidget(self._section("Speed"))
        speed_hint = QLabel(
            "These only change how fast a clip processes. GPU FP16 is used automatically when CUDA is available. "
            "A smaller detect size and detecting every 2nd or 3rd frame are faster but can miss small cars. "
            "Live preview off skips drawing while the clip runs."
        )
        speed_hint.setObjectName("hint")
        speed_hint.setWordWrap(True)
        form.addWidget(speed_hint)

        imgsz_row = QHBoxLayout()
        imgsz_row.addWidget(QLabel("Detect size"))
        self.detector_imgsz_combo = QComboBox()
        for size in DETECTOR_IMGSZ_CHOICES:
            self.detector_imgsz_combo.addItem(f"{size} px", size)
        self.detector_imgsz_combo.setCurrentIndex(
            max(0, self.detector_imgsz_combo.findData(self.settings.detector_imgsz))
        )
        self.detector_imgsz_combo.setToolTip(
            "YOLO letterbox at process time. Training stays 1280. "
            "640 is fastest; 1280 is closest to how the model was trained."
        )
        self.detector_imgsz_combo.currentIndexChanged.connect(self._settings_changed)
        imgsz_row.addWidget(self.detector_imgsz_combo, 1)
        form.addLayout(imgsz_row)

        stride_row = QHBoxLayout()
        stride_row.addWidget(QLabel("Detect every"))
        self.detect_stride_combo = QComboBox()
        for step in range(1, DETECTOR_STRIDE_MAX + 1):
            label = "frame" if step == 1 else f"{step} frames"
            self.detect_stride_combo.addItem(label, step)
        self.detect_stride_combo.setCurrentIndex(
            max(0, self.detect_stride_combo.findData(self.settings.detect_stride))
        )
        self.detect_stride_combo.setToolTip(
            "Trained detector only. Skipped frames keep the last ID via hold and drop-zone pickup."
        )
        self.detect_stride_combo.currentIndexChanged.connect(self._settings_changed)
        stride_row.addWidget(self.detect_stride_combo, 1)
        form.addLayout(stride_row)

        preview_row = QHBoxLayout()
        preview_row.addWidget(QLabel("Live preview"))
        self.preview_every_combo = QComboBox()
        preview_labels = {0: "Off", 15: "Every 15 frames", 30: "Every 30 frames", 60: "Every 60 frames"}
        for every in PREVIEW_EVERY_CHOICES:
            self.preview_every_combo.addItem(preview_labels.get(every, str(every)), every)
        self.preview_every_combo.setCurrentIndex(
            max(0, self.preview_every_combo.findData(self.settings.preview_every))
        )
        self.preview_every_combo.setToolTip("Off is fastest. Counts do not depend on the live picture.")
        self.preview_every_combo.currentIndexChanged.connect(self._settings_changed)
        preview_row.addWidget(self.preview_every_combo, 1)
        form.addLayout(preview_row)

        form.addStretch(1)
        layout.addWidget(card, 1)
        scroll.setWidget(body)

        split = QHBoxLayout()
        split.setContentsMargins(0, 0, 0, 0)
        split.setSpacing(16)
        split.addWidget(scroll, 1)

        preview_card = self._card()
        preview = QVBoxLayout(preview_card)
        preview.addWidget(self._step_title("Frame check"))
        self.frame_check_status = QLabel(
            "Freeze this still and rerun the detector when confidence, overlap, or weights change. Not a full clip."
        )
        self.frame_check_status.setObjectName("hint")
        self.frame_check_status.setWordWrap(True)
        preview.addWidget(self.frame_check_status)
        self.check_canvas = FrameCanvas()
        self.check_canvas.set_mode("view")
        self.check_canvas.setMinimumSize(360, 240)
        preview.addWidget(self.check_canvas, 1)
        split.addWidget(preview_card, 1)
        outer.addLayout(split)

        self._frame_check_timer = QTimer(self)
        self._frame_check_timer.setSingleShot(True)
        self._frame_check_timer.setInterval(350)
        self._frame_check_timer.timeout.connect(self._run_frame_check)
        return page

    def _add_setting_slider(
        self,
        form: QVBoxLayout,
        template: str,
        low: int,
        high: int,
        value: int,
        scale: float,
        hint: str = "",
    ) -> tuple[QLabel, QSlider]:
        row = QHBoxLayout()
        label = QLabel(template.format(value=value / scale))
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(low, high)
        slider.setValue(value)
        slider.valueChanged.connect(lambda raw, lab=label, text=template, div=scale: self._detection_slider_changed(lab, text, raw, div))
        row.addWidget(label)
        row.addWidget(slider, 1)
        form.addLayout(row)
        if hint:
            slider.setToolTip(hint)
            label.setToolTip(hint)
            note = QLabel(hint)
            note.setObjectName("hint")
            note.setWordWrap(True)
            note.setContentsMargins(0, 0, 0, 10)
            form.addWidget(note)
        return label, slider

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
        self.setup_export_preview = QLabel("")
        self.setup_export_preview.setObjectName("hint")
        self.setup_export_preview.setWordWrap(True)
        left_layout.addWidget(self.setup_export_preview)
        self._refresh_export_preview()

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
            "Draw North, then up to four gates (N, S, E, W) on the first clip (the reference).\n\n"
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
            "Set North, then click Gate and drag one line per approach (N, S, E, W)."
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

        left_layout.addWidget(self._section_label("DRAW"))
        draw = QGridLayout()
        draw.setContentsMargins(0, 0, 0, 0)
        draw.setHorizontalSpacing(8)
        draw.setVerticalSpacing(8)
        set_north = self._wide_button("North", tooltip="Drag from the intersection toward geographic North.")
        set_north.clicked.connect(self._start_set_north)
        add_gate = self._wide_button(
            "Gate",
            tooltip="Drag one line per approach. Named N, S, E, or W from North. First hit is entry, next is exit.",
        )
        add_gate.clicked.connect(self._start_draw_gate)
        add_road = self._wide_button(
            "Roadway",
            tooltip="Outline pavement only. Boats, parking, and sidewalks stay outside.",
        )
        add_road.clicked.connect(self._start_roadway)
        draw.addWidget(set_north, 0, 0)
        draw.addWidget(add_gate, 0, 1)
        draw.addWidget(add_road, 1, 0, 1, 2)
        left_layout.addLayout(draw)

        self.finish_road_btn = self._wide_button("Finish roadway", primary=True)
        self.finish_road_btn.clicked.connect(self._finish_roadway)
        self.finish_road_btn.setVisible(False)
        self.cancel_road_btn = self._wide_button("Cancel roadway")
        self.cancel_road_btn.clicked.connect(self._cancel_roadway)
        self.cancel_road_btn.setVisible(False)
        road_actions = QHBoxLayout()
        road_actions.setContentsMargins(0, 8, 0, 0)
        road_actions.addWidget(self.cancel_road_btn)
        road_actions.addWidget(self.finish_road_btn)
        left_layout.addLayout(road_actions)

        self.more_tools_btn = QPushButton("Vehicle size and drop zones")
        self.more_tools_btn.setCheckable(True)
        self.more_tools_btn.setChecked(False)
        self.more_tools_btn.toggled.connect(self._toggle_more_route_tools)
        left_layout.addWidget(self.more_tools_btn)
        self.more_tools = QWidget()
        self.more_tools.setVisible(False)
        more = QGridLayout(self.more_tools)
        more.setContentsMargins(0, 8, 0, 0)
        more.setHorizontalSpacing(8)
        more.setVerticalSpacing(8)
        add_mask = self._wide_button(
            "Vehicle size",
            tooltip="Box one typical car so processing knows the size band.",
        )
        add_mask.clicked.connect(self._start_mask_vehicle)
        add_shadow = self._wide_button(
            "Drop zone",
            tooltip="Box only the shadow or joint that hides cars, not the whole approach.",
        )
        add_shadow.clicked.connect(self._start_mask_occlusion)
        add_reach = self._wide_button(
            "Reach on photo",
            tooltip="Drag about one car-length through a drop zone. Or use the slider above.",
        )
        add_reach.clicked.connect(self._start_set_reach)
        more.addWidget(add_mask, 0, 0)
        more.addWidget(add_shadow, 0, 1)
        more.addWidget(add_reach, 1, 0, 1, 2)
        left_layout.addWidget(self.more_tools)

        files = QHBoxLayout()
        files.setContentsMargins(0, 4, 0, 0)
        save_geom = self._wide_button("Save", quiet=True, tooltip="Save this drawing to a file.")
        save_geom.clicked.connect(self._save_geometry)
        load_geom = self._wide_button("Load", quiet=True, tooltip="Load a saved drawing.")
        load_geom.clicked.connect(self._load_geometry)
        clear_geom = self._wide_button("Clear all", quiet=True, tooltip="Remove North, lines, roadway, and drop zones.")
        clear_geom.clicked.connect(self._clear_geometry_lines)
        files.addWidget(save_geom)
        files.addWidget(load_geom)
        files.addStretch(1)
        files.addWidget(clear_geom)
        left_layout.addLayout(files)

        nav = QHBoxLayout()
        nav.setContentsMargins(0, 8, 0, 0)
        back = self._wide_button("Back")
        back.clicked.connect(lambda: self.stack.setCurrentIndex(PAGE_SETUP))
        nxt = self._wide_button("Next: align clips", primary=True)
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
        self.canvas.on_box_drawn = self._on_setup_box_drawn
        self.canvas.on_polygon_finished = self._on_roadway_finished
        right_layout.addWidget(self.canvas, 1)
        row.addWidget(left, 0)
        row.addWidget(right, 1)
        left.setMinimumWidth(420)
        left.setMaximumWidth(480)
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

        self.finish_gates_btn = QPushButton("Finish gates")
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
        self.process_infer_trips_check = QCheckBox("Infer trips from traces")
        self.process_infer_trips_check.setChecked(self.settings.infer_trips_from_traces)
        self.process_infer_trips_check.setToolTip(
            "When on, a path that crosses a gate still counts if two detector boxes "
            "did not straddle the drawn line. Off only counts a confirmed box-to-box crossing."
        )
        self.process_infer_trips_check.toggled.connect(self._infer_trips_changed)
        left_layout.addWidget(self.process_infer_trips_check)

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
        gates_page = QWidget()
        gates_layout = QVBoxLayout(gates_page)
        gates_layout.setContentsMargins(0, 0, 0, 0)
        gates_layout.addWidget(self.table)
        self.gate_trace = QLabel("Path traces appear here after processing.")
        self.gate_trace.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.gate_trace.setMinimumHeight(220)
        trace_scroll = QScrollArea()
        trace_scroll.setWidgetResizable(True)
        trace_scroll.setWidget(self.gate_trace)
        gates_layout.addWidget(trace_scroll, 1)
        self.report_tabs.addTab(self.overview_table, "Overview")
        self.report_tabs.addTab(gates_page, "Gate Statistics")
        self.report_tabs.addTab(self.tmc_table, "TMC Summary")
        review_page = QWidget()
        review_layout = QHBoxLayout(review_page)
        review_layout.setContentsMargins(0, 0, 0, 0)
        review_col = QVBoxLayout()
        review_hint = QLabel(
            "Dots mark every drop, jump, and stacked box. Click a row to seek that still and highlight the event."
        )
        review_hint.setObjectName("hint")
        review_hint.setWordWrap(True)
        review_col.addWidget(review_hint)
        self.issue_list = QListWidget()
        self.issue_list.itemClicked.connect(self._seek_review_issue)
        review_col.addWidget(self.issue_list, 1)
        review_layout.addLayout(review_col, 1)
        review_still = QVBoxLayout()
        self.review_status = QLabel("All issues on this clip.")
        self.review_status.setObjectName("hint")
        self.review_status.setWordWrap(True)
        review_still.addWidget(self.review_status)
        self.review_canvas = FrameCanvas()
        self.review_canvas.set_mode("view")
        self.review_canvas.setMinimumSize(360, 220)
        review_still.addWidget(self.review_canvas, 1)
        review_layout.addLayout(review_still, 2)
        self.report_tabs.addTab(review_page, "Review")
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
        if hasattr(self, "setup_export_preview"):
            self._refresh_export_preview()

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
            self._refresh_export_preview()
            self._persist_settings()

    def _export_example_stem(self) -> str:
        if self.videos:
            return self.videos[0].stem
        return "clip"

    def _current_export_pattern(self) -> str:
        if hasattr(self, "export_pattern_edit"):
            return clamp_export_pattern(self.export_pattern_edit.text())
        return clamp_export_pattern(self.settings.export_pattern)

    def _sync_export_pattern_combo(self) -> None:
        if not hasattr(self, "export_pattern_combo"):
            return
        pattern = self._current_export_pattern()
        index = self.export_pattern_combo.findData(pattern)
        if index < 0:
            index = self.export_pattern_combo.findData("custom")
        self.export_pattern_combo.blockSignals(True)
        self.export_pattern_combo.setCurrentIndex(max(0, index))
        self.export_pattern_combo.blockSignals(False)

    def _refresh_export_preview(self) -> None:
        pattern = self._current_export_pattern()
        example = resolve_export_folder(
            self.export_dir,
            pattern,
            video_stem=self._export_example_stem(),
            when=datetime.now(),
        )
        text = f"Export folder: {self.export_dir}"
        if hasattr(self, "export_label"):
            self.export_label.setText(text)
        if hasattr(self, "settings_export_label"):
            self.settings_export_label.setText(text)
        preview = f"Next run saves into: {example}"
        if hasattr(self, "export_pattern_preview"):
            self.export_pattern_preview.setText(preview)
        if hasattr(self, "setup_export_preview"):
            self.setup_export_preview.setText(preview)

    def _export_pattern_preset_changed(self, *_args) -> None:
        if not hasattr(self, "export_pattern_combo"):
            return
        data = self.export_pattern_combo.currentData()
        if data == "custom":
            return
        if hasattr(self, "export_pattern_edit"):
            self.export_pattern_edit.blockSignals(True)
            self.export_pattern_edit.setText(str(data or ""))
            self.export_pattern_edit.blockSignals(False)
        self._refresh_export_preview()
        self._persist_settings()

    def _export_pattern_edited(self, *_args) -> None:
        self._sync_export_pattern_combo()
        self._refresh_export_preview()
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
                f"Frame {index} from {self.videos[0].name}. Set North, then drag a gate on each approach (N, S, E, W)."
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
        flow_options: tuple[str, ...] | None = None,
        current_flow: str | None = None,
        on_flow=None,
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
        if flow_options and on_flow:
            flow = QComboBox()
            for value in flow_options:
                flow.addItem(GATE_FLOW_LABELS.get(value, value), value)
            if current_flow:
                index = flow.findData(current_flow)
                if index >= 0:
                    flow.setCurrentIndex(index)
            flow.activated.connect(lambda _idx, box=flow: on_flow(box.currentData()))
            layout.addWidget(flow)
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

        missing = [name for name in ORIGINS if name not in {gate_origin_name(item.name) for item in self.geometry.all_gates()}]
        gate_heading = "GATES  (N, S, E, W — first hit is entry, next is exit)"
        self.route_items.addWidget(self._section_label(gate_heading))
        if not self.geometry.all_gates():
            empty = QLabel("None yet. Click Gate, then drag a line across each approach. Up to four: N, S, E, W.")
            empty.setObjectName("hint")
            empty.setWordWrap(True)
            self.route_items.addWidget(empty)
        else:
            hint = QLabel("Two-way, one-way in, or one-way out. The export hides unused entry/exit columns.")
            hint.setObjectName("hint")
            hint.setWordWrap(True)
            self.route_items.addWidget(hint)
        for item in self.geometry.all_gates():
            self.route_items.addWidget(
                self._item_row(
                    "Gate",
                    name_options=ORIGINS,
                    current_name=gate_origin_name(item.name),
                    on_name=lambda name, gate=item: self._rename_gate_item(gate, name),
                    flow_options=GATE_FLOWS,
                    current_flow=item.flow,
                    on_flow=lambda flow, gate=item: self._set_gate_flow(gate, flow),
                    on_redraw=lambda gate=item: self._redraw_gate_item(gate),
                    on_delete=lambda gate=item: self._delete_gate_item(gate),
                )
            )
        if missing and self.geometry.all_gates():
            left = QLabel(f"Still open: {', '.join(missing)}. Draw those sides if they exist.")
            left.setObjectName("hint")
            left.setWordWrap(True)
            self.route_items.addWidget(left)

        self.route_items.addWidget(self._section_label("VEHICLE SIZE"))
        if not self.geometry.vehicle_refs:
            empty = QLabel("Optional. Open Vehicle size and drop zones, then box one car.")
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
        limits = self.geometry.vehicle_area_limits()
        if limits:
            band = QLabel(f"Keeping blobs {limits[0]}–{limits[1]} px² from these masks.")
            band.setObjectName("hint")
            band.setWordWrap(True)
            self.route_items.addWidget(band)

        self.route_items.addWidget(self._section_label("ROADWAY"))
        if not self.geometry.roadways:
            empty = QLabel("Optional. Use Roadway to leave boats and parking outside. Otherwise the full frame is used.")
            empty.setObjectName("hint")
            empty.setWordWrap(True)
            self.route_items.addWidget(empty)
        for index, poly in enumerate(self.geometry.roadways):
            self.route_items.addWidget(
                self._item_row(
                    f"Roadway {index + 1}  {len(poly.points)} pts",
                    on_redraw=lambda i=index: self._start_roadway(replace=i),
                    on_delete=lambda i=index: self._delete_roadway(i),
                )
            )
        for warning in roadway_approach_warnings(self.geometry):
            note = QLabel(warning)
            note.setObjectName("hint")
            note.setWordWrap(True)
            self.route_items.addWidget(note)

        self.route_items.addWidget(self._section_label("DROP ZONES"))
        if not self.geometry.occlusions:
            empty = QLabel("Optional. Open Vehicle size and drop zones to box a shadow that hides cars.")
            empty.setObjectName("hint")
            empty.setWordWrap(True)
            self.route_items.addWidget(empty)
        for index, box in enumerate(self.geometry.occlusions, start=1):
            label = f"Drop zone {index}  {int(box.width())}×{int(box.height())}"
            if drop_zone_too_wide(box, self.geometry):
                label += "  — too wide (many car-lengths); auto-attach will stretch"
            self.route_items.addWidget(
                self._item_row(
                    label,
                    on_redraw=lambda mask=box: self._redraw_occlusion_item(mask),
                    on_delete=lambda mask=box: self._delete_occlusion_item(mask),
                )
            )
        if self.geometry.occlusions:
            reach_px = int(round(self.geometry.prediction_reach_px))
            reach_label = QLabel(f"Prediction reach: {reach_px} px")
            reach_label.setObjectName("hint")
            self.route_items.addWidget(reach_label)
            reach_slider = QSlider(Qt.Orientation.Horizontal)
            reach_slider.setRange(int(PREDICTION_REACH_MIN), int(PREDICTION_REACH_MAX))
            reach_slider.setValue(reach_px)
            reach_slider.setToolTip("About one car-length. Lost IDs look this far ahead inside a drop zone.")

            def _on_reach_slider(value: int, label=reach_label) -> None:
                self.geometry.set_prediction_reach(float(value))
                label.setText(f"Prediction reach: {int(round(self.geometry.prediction_reach_px))} px")
                self.canvas.set_geometry(self.geometry)
                self.project.geometry = self.geometry

            reach_slider.valueChanged.connect(_on_reach_slider)
            self.route_items.addWidget(reach_slider)
        if self.geometry.is_ready():
            count = len(self.geometry.movement_names())
            ready = QLabel(f"{count} movement{'s' if count != 1 else ''} ready.")
            ready.setObjectName("hint")
            ready.setWordWrap(True)
            self.route_items.addWidget(ready)
        self.route_items.addStretch(1)
        self.canvas.set_geometry(self.geometry)
        self.project.geometry = self.geometry

    def _set_active_tool(self, name: str, hint: str) -> None:
        self.route_hint.setText(f"Active tool: {name}. {hint}")

    def _start_set_north(self, *_args) -> None:
        self._draw_step = "north"
        self._redraw_index = None
        self._redraw_target = None
        self.canvas.set_drawing_enabled(True)
        self._set_active_tool(
            "North",
            "On the photo: click the middle of the intersection, drag toward geographic North, release. "
            "If North is at the bottom of the photo, drag downward.",
        )

    def _start_set_reach(self, *_args) -> None:
        if self._draw_step == "roadway":
            self._cancel_roadway()
        self._draw_step = "reach"
        self._redraw_index = None
        self._redraw_target = None
        self.canvas.set_drawing_enabled(True)
        self._set_active_tool(
            "Reach",
            "On the photo: drag a short line the distance a car should travel through a drop zone, then release. "
            "Use a typical gap on this footage, not a global pixel guess.",
        )

    def _start_draw_gate(self, *_args, replace=None) -> None:
        if self.geometry.north is None:
            QMessageBox.information(self, APP_NAME, "Set North first so gates can be named N, S, E, or W.")
            return
        if replace is None and self.geometry.gate_count() >= MAX_GATES:
            QMessageBox.information(self, APP_NAME, f"At most {MAX_GATES} gates (N, S, E, W). Delete one to add another.")
            return
        live = self._live_gate(replace) if replace is not None else None
        if replace is not None and live is None:
            return
        self._draw_step = "gate"
        self._redraw_target = live[1] if live is not None else None
        self._redraw_index = None
        self._draw_gate_name = live[1].name if live is not None else None
        remaining = [name for name in ORIGINS if name not in {gate_origin_name(item.name) for item in self.geometry.all_gates()}]
        if live is not None:
            hint = f"On the photo: drag a new line for {gate_origin_name(live[1].name)}, then release."
        elif remaining:
            hint = f"On the photo: drag a line across an approach, then release. Still open: {', '.join(remaining)}."
        else:
            hint = "On the photo: drag a line across an approach, then release."
        self._set_active_tool("Gate", hint)
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
        self._set_active_tool("Vehicle", "On the photo: drag a box around one typical vehicle, then release.")

    def _set_roadway_actions(self, drawing: bool) -> None:
        self.finish_road_btn.setVisible(drawing)
        self.cancel_road_btn.setVisible(drawing)

    def _toggle_more_route_tools(self, checked: bool) -> None:
        self.more_tools.setVisible(bool(checked))
        self.more_tools_btn.setText(
            "Hide vehicle size and drop zones" if checked else "Vehicle size and drop zones"
        )

    def _start_roadway(self, *_args, replace=None) -> None:
        if replace is None and len(self.geometry.roadways) >= MAX_ROADWAYS:
            QMessageBox.information(self, APP_NAME, f"At most {MAX_ROADWAYS} roadway areas. Delete one to add another.")
            return
        if replace is not None and not (0 <= int(replace) < len(self.geometry.roadways)):
            return
        self._draw_step = "roadway"
        self._roadway_edit = None if replace is None else int(replace)
        self._redraw_target = None
        self.canvas.begin_polygon()
        self._set_roadway_actions(True)
        self._set_active_tool(
            "Roadway",
            "Click around the intersection and its approaches. Include pavement on both sides of each gate. "
            "Exclude the marina, parking lots, buildings, and sidewalks. "
            "Finish roadway, double-click, or click the first point to close. Cancel roadway discards this shape. "
            "If you draw none, the full frame is used.",
        )

    def _finish_roadway(self, *_args) -> None:
        if self._draw_step != "roadway":
            return
        points = [(float(point.x()), float(point.y())) for point in self.canvas._poly_points]
        self.canvas.cancel_polygon()
        self._on_roadway_finished(points)

    def _cancel_roadway(self, *_args) -> None:
        self.canvas.cancel_polygon()
        self._draw_step = "idle"
        self._roadway_edit = None
        self._set_roadway_actions(False)
        self.route_hint.setText("Roadway cancelled. Add roadway when you want to exclude boats and parking.")

    def _on_roadway_finished(self, points: list[tuple[float, float]]) -> None:
        self._set_roadway_actions(False)
        if self._draw_step != "roadway" and self._roadway_edit is None and not points:
            return
        if not polygon_is_valid(points):
            self._draw_step = "idle"
            self._roadway_edit = None
            QMessageBox.information(
                self,
                APP_NAME,
                "A roadway needs at least three separated points that enclose an area.",
            )
            self.route_hint.setText("That roadway was not saved. Click Add roadway to try again.")
            return
        poly = Polygon([(float(x), float(y)) for x, y in points])
        if self._roadway_edit is not None and 0 <= self._roadway_edit < len(self.geometry.roadways):
            self.geometry.roadways[self._roadway_edit] = poly
        elif len(self.geometry.roadways) >= MAX_ROADWAYS:
            QMessageBox.information(self, APP_NAME, f"At most {MAX_ROADWAYS} roadway areas.")
            self._draw_step = "idle"
            return
        else:
            self.geometry.roadways.append(poly)
        self._draw_step = "idle"
        self._roadway_edit = None
        self._refresh_route_list()
        warnings = roadway_approach_warnings(self.geometry)
        if warnings:
            self.route_hint.setText("Saved roadway. " + " ".join(warnings))
        else:
            self.route_hint.setText(
                "Saved roadway. Vehicles outside it are ignored. Drop zones only recover brief gaps inside the road."
            )

    def _delete_roadway(self, index: int) -> None:
        if not (0 <= index < len(self.geometry.roadways)):
            return
        del self.geometry.roadways[index]
        if self._draw_step == "roadway":
            self._cancel_roadway()
        self._refresh_route_list()

    def _start_mask_occlusion(self, *_args, replace=None) -> None:
        if replace is None and len(self.geometry.occlusions) >= MAX_OCCLUSIONS:
            QMessageBox.information(self, APP_NAME, f"At most {MAX_OCCLUSIONS} drop zones. Delete one to add another.")
            return
        live = self._live_occlusion(replace) if replace is not None else None
        if replace is not None and live is None:
            return
        self._draw_step = "occlusion"
        self._redraw_target = live
        self._redraw_index = None
        self.canvas.set_drawing_enabled(True, box=True)
        self._set_active_tool(
            "Drop zone",
            "On the photo: drag a box over the dark patch, joint, or pole shadow. "
            "Cover the full strip that hides cars, not the whole street.",
        )

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

    def _live_occlusion(self, box) -> Box | None:
        if not isinstance(box, Box):
            return None
        for item in self.geometry.occlusions:
            if item is box or item == box:
                return item
        return None

    def _redraw_gate_item(self, item) -> None:
        live = self._live_gate(item)
        if live is None:
            return
        self._start_draw_gate(replace=live[1])

    def _redraw_vehicle_item(self, box: Box) -> None:
        live = self._live_vehicle(box)
        if live is None:
            return
        self._start_mask_vehicle(replace=live)

    def _redraw_occlusion_item(self, box: Box) -> None:
        live = self._live_occlusion(box)
        if live is None:
            return
        self._start_mask_occlusion(replace=live)

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

    def _delete_occlusion_item(self, box: Box) -> None:
        live = self._live_occlusion(box)
        if live is None:
            return
        self.geometry.occlusions = [item for item in self.geometry.occlusions if item is not live]
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
        kind, item = live
        taken = {
            gate_origin_name(other.name)
            for other in self.geometry.all_gates()
            if other is not item
        }
        if name not in ORIGINS:
            return
        if name in taken:
            QMessageBox.information(self, APP_NAME, f"{name} is already used on another gate.")
            self._refresh_route_list()
            return
        stored = name if kind == "start" else ORIGIN_TO_BOUND.get(name, name)
        if item.name == stored:
            return
        item.name = stored
        self._refresh_route_list()

    def _set_gate_flow(self, item, flow: str | None) -> None:
        live = self._live_gate(item)
        if live is None or not flow:
            return
        _kind, gate = live
        if flow not in GATE_FLOWS or gate.flow == flow:
            return
        gate.flow = flow
        self._refresh_route_list()

    def _clear_north(self, *_args) -> None:
        self.geometry.north = None
        self._refresh_route_list()
        self.route_hint.setText("North cleared. Set it again before adding gates.")

    def _on_setup_box_drawn(self, box: Box) -> None:
        if self._draw_step == "occlusion":
            self._on_occlusion_drawn(box)
            return
        self._on_vehicle_mask_drawn(box)

    def _on_occlusion_drawn(self, box: Box) -> None:
        if self._draw_step != "occlusion":
            return
        if box.area() < 20:
            QMessageBox.information(self, APP_NAME, "That box is too small. Drag over the whole dark patch or joint.")
            return
        if self._redraw_target is not None:
            index = next(
                (i for i, item in enumerate(self.geometry.occlusions) if item is self._redraw_target),
                None,
            )
            if index is None:
                return
            self.geometry.occlusions[index] = box
        elif len(self.geometry.occlusions) >= MAX_OCCLUSIONS:
            QMessageBox.information(self, APP_NAME, f"At most {MAX_OCCLUSIONS} drop zones.")
            return
        else:
            self.geometry.occlusions.append(box)
        self._draw_step = "idle"
        self._redraw_index = None
        self._redraw_target = None
        self.canvas.set_drawing_enabled(False)
        self.canvas.clear_draft()
        self._refresh_route_list()
        saved = "Saved drop zone. A car that is lost there keeps its ID and trace through the gap."
        if drop_zone_too_wide(box, self.geometry):
            wide = (
                "This drop zone is many car-lengths wide; that stretches auto-attach. "
                "Box only the dark strip that hides cars."
            )
            QMessageBox.warning(self, APP_NAME, wide)
            self.route_hint.setText(f"{saved} {wide}")
        else:
            self.route_hint.setText(saved)

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
        limits = self.geometry.vehicle_area_limits()
        band = f" Keeping blobs {limits[0]}–{limits[1]} px²." if limits else ""
        self.route_hint.setText(f"Saved vehicle {int(box.width())}×{int(box.height())}.{band}")

    def _clear_geometry_lines(self) -> None:
        go = QMessageBox.question(
            self,
            APP_NAME,
            "Clear North, all gates, the roadway, vehicle masks, drop zones, and prediction reach?",
        )
        if go != QMessageBox.StandardButton.Yes:
            return
        self.geometry.north = None
        self.geometry.starts = []
        self.geometry.ends = []
        self.geometry.vehicle_refs = []
        self.geometry.occlusions = []
        self.geometry.roadways = []
        self.geometry.prediction_reach = None
        self.geometry.set_prediction_reach(PREDICTION_REACH_PX)
        self._set_roadway_actions(False)
        self._draw_step = "idle"
        self._redraw_index = None
        self._redraw_target = None
        self.canvas.set_drawing_enabled(False)
        self._refresh_route_list()
        self.route_hint.setText("Cleared. Set North, then draw a gate on each approach (N, S, E, W).")

    def _on_line_drawn(self, line: Line) -> None:
        if self._draw_step == "reach":
            if line.length() < 8:
                QMessageBox.information(
                    self,
                    APP_NAME,
                    "That line is too short. Drag about one car-length through a typical drop-zone gap.",
                )
                self.canvas.clear_draft()
                return
            self.geometry.set_prediction_reach(line.length(), line=line)
            self._draw_step = "idle"
            self._redraw_index = None
            self._redraw_target = None
            self.canvas.set_drawing_enabled(False)
            self.canvas.clear_draft()
            self._refresh_route_list()
            self.route_hint.setText(
                f"Prediction reach set to {int(round(self.geometry.prediction_reach_px))} px. "
                "A lost car in a drop zone keeps a pickup that far ahead."
            )
            return
        if self._draw_step == "north":
            self.geometry.north = line
            self.geometry.assign_names()
            self._draw_step = "idle"
            self._redraw_index = None
            self._redraw_target = None
            self.canvas.set_drawing_enabled(False)
            self.canvas.clear_draft()
            self._refresh_route_list()
            self.route_hint.setText("North set. Click Gate and drag a line across each approach (N, S, E, W).")
            return
        if self._draw_step == "gate":
            pool = self.geometry.all_gates()
            if self._redraw_target is not None and any(item is self._redraw_target for item in pool):
                self._redraw_target.line = line
                added_name = gate_origin_name(self._redraw_target.name)
                self.canvas.clear_draft()
                self.canvas.set_drawing_enabled(False)
                self._draw_step = "idle"
                self._redraw_index = None
                self._redraw_target = None
                self._refresh_route_list()
                self.route_hint.setText(f"Updated {added_name} gate.")
                return
            added = self.geometry.add_gate("start", line)
            if added is None:
                QMessageBox.information(
                    self,
                    APP_NAME,
                    "Could not add that gate. That side may already be drawn, or all four (N, S, E, W) are set.",
                )
                self.canvas.clear_draft()
                return
            added_name = gate_origin_name(added.name)
            self.canvas.clear_draft()
            self._redraw_index = None
            self._redraw_target = None
            self._draw_gate_name = None
            self._refresh_route_list()
            remaining = [name for name in ORIGINS if name not in {gate_origin_name(item.name) for item in self.geometry.all_gates()}]
            if not remaining or self.geometry.gate_count() >= MAX_GATES:
                self.canvas.set_drawing_enabled(False)
                self._draw_step = "idle"
                self.route_hint.setText(f"Saved {added_name}. All four gates are set.")
            else:
                self.canvas.set_drawing_enabled(True)
                self.route_hint.setText(
                    f"Saved {added_name}. {self.geometry.gate_count()} of {MAX_GATES}. "
                    f"Drag the next approach ({', '.join(remaining)})."
                )

    def _save_geometry(self) -> None:
        if not self.geometry.is_ready():
            QMessageBox.information(self, APP_NAME, "Set North and draw at least two gates (N, S, E, or W).")
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
            QMessageBox.information(self, APP_NAME, "Set North and draw at least two gates (N, S, E, or W).")
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
            self.align_hint.setText("This is the reference clip. Gates were drawn here. Other clips need alignment.")
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
            QMessageBox.information(self, APP_NAME, "Finish redrawing the gates on this clip first.")
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
        self._draw_step = "gate"
        self.finish_gates_btn.setText("Finish gates")
        self.finish_gates_btn.setVisible(True)
        self.align_canvas.set_geometry(clip.geometry_override)
        self.align_canvas.set_mode("draw")
        self.align_hint.setText("Redraw gates on this clip (N, S, E, W). Draw the sides that exist, then Finish gates.")

    def _finish_redraw_phase(self) -> None:
        clip = self._current_clip()
        if not self._redrawing or clip is None or clip.geometry_override is None:
            return
        if clip.geometry_override.gate_count() < 2:
            QMessageBox.information(self, APP_NAME, "Draw at least two gates.")
            return
        clip.geometry_override.assign_names()
        clip.status = "redrawn"
        self._redrawing = False
        self._draw_step = "idle"
        self.finish_gates_btn.setVisible(False)
        self.align_canvas.set_geometry(clip.geometry_override)
        self.align_canvas.set_mode("view")
        self.align_hint.setText("Custom gates saved for this clip.")
        self._refresh_align_list()

    def _on_redraw_line(self, line: Line) -> None:
        if not self._redrawing:
            return
        clip = self._current_clip()
        if clip is None or clip.geometry_override is None:
            return
        added = clip.geometry_override.add_gate("start", line)
        if added is None:
            self.align_hint.setText("Could not add that gate. Finish gates, or that side is already drawn.")
            return
        self.align_canvas.set_geometry(clip.geometry_override)
        remaining = [
            name
            for name in ORIGINS
            if name not in {gate_origin_name(item.name) for item in clip.geometry_override.all_gates()}
        ]
        if remaining:
            self.align_hint.setText(f"Saved {added.name}. Draw {', '.join(remaining)}, or Finish gates.")
        else:
            self.align_hint.setText(f"Saved {added.name}. All four gates are set. Finish gates.")

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
            QMessageBox.information(self, APP_NAME, "Set North and draw at least two gates (N, S, E, or W).")
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
        mode = self.settings.detector_mode
        weights = visdrone_weights_path()
        if mode == "trained" and not weights.is_file():
            QMessageBox.information(
                self,
                APP_NAME,
                "Trained detector needs VisDrone weights. Open Training and run Train VisDrone first. "
                "Weights are not included with a new install.",
            )
            return
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.stack.setCurrentIndex(PAGE_PROCESS)
        self.log.clear()
        self.results = []
        self.progress.setValue(0)
        self.cancel_btn.setEnabled(True)
        if mode == "trained":
            self._append_log(
                f"Detection: trained detector  •  {weights}  •  "
                f"conf {self.settings.detector_conf:.2f}  •  NMS {self.settings.detector_iou:.2f}  •  "
                f"parked {self.settings.parked_radius:.0f}px  •  link {self.settings.link_speed:.0f}px/s  •  "
                f"imgsz {self.settings.detector_imgsz}  •  every {self.settings.detect_stride} frame(s)"
            )
        else:
            self._append_log("Detection: motion mask (all vehicles the same)")
        if self.settings.infer_trips_from_traces:
            self._append_log("Trips: traces can infer gate hits when boxes miss the tripwire")
        else:
            self._append_log("Trips: confirmed box-to-box gate crossings only")
        if self.settings.preview_every <= 0:
            self._append_log("Live preview: off")
        limits = self.geometry.vehicle_area_limits()
        if limits:
            self._append_log(
                f"Box size from {len(self.geometry.vehicle_refs)} vehicle mask(s): "
                f"{limits[0]}–{limits[1]} px²"
            )
        else:
            self._append_log("Box size: default frame band (no vehicle masks)")
        if self.geometry.roadways:
            self._append_log(f"Roadway areas: {len(self.geometry.roadways)} (boats and parking outside are ignored)")
        if self.geometry.occlusions:
            self._append_log(f"Drop zones: {len(self.geometry.occlusions)} (recover a brief gap; detections inside still count)")
        self._append_log(f"Processing {len(jobs)} clip(s); skipped {len(skipped)}")
        self._append_log(f"Movements: {', '.join(self.geometry.movement_names())}")
        example_stem = Path(jobs[0][0]).stem if jobs else self._export_example_stem()
        self._append_log(
            f"Export: {resolve_export_folder(self.export_dir, self._current_export_pattern(), video_stem=example_stem)}"
        )

        self.worker = ProcessWorker(
            jobs=jobs,
            export_dir=self.export_dir,
            export_pattern=self._current_export_pattern(),
            clock_start=self._clock_start(),
            detector_mode=mode,
            weights_path=str(weights),
            detector_conf=self.settings.detector_conf,
            detector_iou=self.settings.detector_iou,
            parked_radius=self.settings.parked_radius,
            link_speed=self.settings.link_speed,
            path_jump=self.settings.path_jump,
            detector_imgsz=self.settings.detector_imgsz,
            detect_stride=self.settings.detect_stride,
            preview_every=self.settings.preview_every,
            theme=self.settings.theme,
            infer_trips_from_traces=self.settings.infer_trips_from_traces,
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
        saved = f"  •  {result.export_path}" if result.export_path else ""
        self._append_log(f"{result.video.name}: {status}{extra}{counts}{saved}")

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
        export_folders = [result.export_path.parent for result in ok_results if result.export_path]
        unique_folders = list(dict.fromkeys(export_folders))
        saved = unique_folders[0] if len(unique_folders) == 1 else self.export_dir
        self.report_summary.setText(
            f"{len(ok_results)} video(s) processed  •  {total_trips} trips  •  "
            f"{max(0, skipped)} skipped  •  Spreadsheets saved in {saved}"
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
            self._set_gate_trace(None)
            self._fill_review_issues(None)
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
        self._set_gate_trace(result.trace_image)
        self._fill_review_issues(result)

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

    def _set_gate_trace(self, frame: np.ndarray | None) -> None:
        if frame is None:
            self.gate_trace.clear()
            self.gate_trace.setText("Path traces appear here after processing.")
            return
        rgb = np.ascontiguousarray(frame[:, :, ::-1])
        height, width, channels = rgb.shape
        image = QImage(rgb.data, width, height, channels * width, QImage.Format.Format_RGB888).copy()
        self.gate_trace.setText("")
        self.gate_trace.setPixmap(QPixmap.fromImage(image))
        self.gate_trace.adjustSize()

    def _review_base_frame(self, result: ProcessResult):
        if result.trace_image is not None:
            return result.trace_image
        try:
            frame, _ = grab_frame_at(result.video.path, 0.0)
            return frame
        except Exception:
            return None

    def _show_review_still(self, result: ProcessResult, highlight=None) -> None:
        if highlight is None:
            frame = self._review_base_frame(result)
            if frame is None:
                return
            marked = draw_issue_overlay(frame, result.issues, result.geometry)
            self.review_canvas.set_frame(marked)
            self.review_status.setText(
                f"{len(result.issues)} event(s) on this clip. Dots show where they happened."
            )
            return
        try:
            frame, _ = grab_frame_at(result.video.path, highlight.time_s)
        except Exception:
            frame = self._review_base_frame(result)
        if frame is None:
            return
        marked = draw_issue_overlay(frame, result.issues, result.geometry, highlight=highlight)
        self.review_canvas.set_frame(marked)
        self.preview_canvas.set_frame(marked)
        minutes = int(highlight.time_s) // 60
        seconds = int(highlight.time_s) % 60
        self.review_status.setText(f"{minutes}:{seconds:02d}  {highlight.message}")

    def _fill_review_issues(self, result: ProcessResult | None) -> None:
        self.issue_list.clear()
        if result is None or not result.issues:
            self.issue_list.addItem("No tracking problems listed for this clip.")
            if result is not None and result.trace_image is not None:
                self.review_canvas.set_frame(result.trace_image)
            if hasattr(self, "review_status"):
                self.review_status.setText("No tracking problems listed for this clip.")
            return
        for issue in result.issues:
            minutes = int(issue.time_s) // 60
            seconds = int(issue.time_s) % 60
            self.issue_list.addItem(f"{minutes}:{seconds:02d}  {issue.message}")
            row = self.issue_list.item(self.issue_list.count() - 1)
            if row is not None:
                row.setData(Qt.ItemDataRole.UserRole, issue)
        self._show_review_still(result)

    def _seek_review_issue(self, item) -> None:
        issue = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
        if issue is None:
            return
        ok_results = [result for result in self.results if not result.error]
        if not ok_results:
            return
        index = max(0, min(self.report_clip.currentIndex(), len(ok_results) - 1))
        self._show_review_still(ok_results[index], highlight=issue)

    def _open_export_dir(self) -> None:
        from PyQt6.QtCore import QUrl
        from PyQt6.QtGui import QDesktopServices

        folders = [Path(result.export_path).parent for result in self.results if result.export_path]
        unique = list(dict.fromkeys(folders))
        folder = unique[0] if len(unique) == 1 else self.export_dir
        folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder.resolve())))

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
            self._refresh_visdrone_status()
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
        self._refresh_visdrone_status()

    def _go_settings(self) -> None:
        self.stack.setCurrentIndex(PAGE_SETTINGS)
        self._schedule_frame_check()

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
        iou = self.settings.detector_iou
        parked = self.settings.parked_radius
        link = self.settings.link_speed
        jump = self.settings.path_jump
        imgsz = self.settings.detector_imgsz
        stride = self.settings.detect_stride
        preview_every = self.settings.preview_every
        preset = self.settings.precision_preset
        if hasattr(self, "detector_conf_slider"):
            conf = clamp_detector_conf(self.detector_conf_slider.value() / 100.0)
            iou = clamp_detector_iou(self.detector_iou_slider.value() / 100.0)
            parked = clamp_parked_radius(self.parked_radius_slider.value())
            link = clamp_link_speed(self.link_speed_slider.value())
            jump = clamp_path_jump(self.path_jump_slider.value())
        if hasattr(self, "precision_combo"):
            preset = self.precision_combo.currentData() or preset
        if hasattr(self, "detector_imgsz_combo"):
            imgsz = clamp_detector_imgsz(self.detector_imgsz_combo.currentData())
            stride = clamp_detect_stride(self.detect_stride_combo.currentData())
            preview_every = clamp_preview_every(self.preview_every_combo.currentData())
        infer_trips = self.settings.infer_trips_from_traces
        sender = self.sender()
        if sender is getattr(self, "process_infer_trips_check", None) and sender is not None:
            infer_trips = sender.isChecked()
        elif hasattr(self, "infer_trips_check"):
            infer_trips = self.infer_trips_check.isChecked()
        elif hasattr(self, "process_infer_trips_check"):
            infer_trips = self.process_infer_trips_check.isChecked()
        self.settings = AppSettings(
            export_dir=str(self.export_dir),
            export_pattern=self._current_export_pattern(),
            warn_onedrive=warn,
            detector_mode=mode,
            detector_weights=str(visdrone_weights_path()),
            detector_source="visdrone",
            theme=theme if theme in ("dark", "light") else "dark",
            precision_preset=preset if preset in ("strict", "balanced", "sensitive", "custom") else "balanced",
            detector_conf=conf,
            detector_iou=iou,
            parked_radius=parked,
            link_speed=link,
            path_jump=jump,
            detector_imgsz=imgsz,
            detect_stride=stride,
            preview_every=preview_every,
            infer_trips_from_traces=infer_trips,
        )
        save_settings(self.settings)
        self._refresh_setup_list()

    def _settings_changed(self, *_args) -> None:
        self._persist_settings()
        self._schedule_frame_check()

    def _detection_slider_changed(self, label: QLabel, template: str, raw: int, scale: float) -> None:
        label.setText(template.format(value=raw / scale))
        if self._filling_preset:
            return
        if hasattr(self, "precision_combo") and self.precision_combo.currentData() != "custom":
            self.precision_combo.blockSignals(True)
            self.precision_combo.setCurrentIndex(max(0, self.precision_combo.findData("custom")))
            self.precision_combo.blockSignals(False)
        self._persist_settings()
        self._schedule_frame_check()

    def _precision_preset_changed(self, *_args) -> None:
        name = self.precision_combo.currentData() or "custom"
        if name == "custom":
            self._persist_settings()
            self._schedule_frame_check()
            return
        tuned = apply_precision_preset(self.settings, name)
        self._filling_preset = True
        self.detector_conf_slider.setValue(int(round(tuned.detector_conf * 100)))
        self.detector_iou_slider.setValue(int(round(tuned.detector_iou * 100)))
        self.parked_radius_slider.setValue(int(round(tuned.parked_radius)))
        self.link_speed_slider.setValue(int(round(tuned.link_speed)))
        self.path_jump_slider.setValue(int(round(tuned.path_jump)))
        self._filling_preset = False
        self._persist_settings()
        self._schedule_frame_check()

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
        self._schedule_frame_check()

    def _infer_trips_changed(self, checked: bool) -> None:
        sender = self.sender()
        for box in (
            getattr(self, "infer_trips_check", None),
            getattr(self, "process_infer_trips_check", None),
        ):
            if box is None or box is sender:
                continue
            box.blockSignals(True)
            box.setChecked(bool(checked))
            box.blockSignals(False)
        self._persist_settings()

    def _sync_detector_ui(self) -> None:
        mode = self.settings.detector_mode if self.settings.detector_mode in ("motion", "trained") else "motion"
        for combo in (getattr(self, "detector_combo", None), getattr(self, "process_detector_combo", None)):
            if combo is None:
                continue
            combo.blockSignals(True)
            combo.setCurrentIndex(max(0, combo.findData(mode)))
            combo.blockSignals(False)
        infer = bool(self.settings.infer_trips_from_traces)
        for box in (
            getattr(self, "infer_trips_check", None),
            getattr(self, "process_infer_trips_check", None),
        ):
            if box is None:
                continue
            box.blockSignals(True)
            box.setChecked(infer)
            box.blockSignals(False)
        self._refresh_detector_status()

    def _refresh_detector_status(self) -> None:
        path = visdrone_weights_path()
        if path.is_file():
            text = f"VisDrone weights ready: {path}"
        else:
            text = (
                "VisDrone not trained yet. Open Training and run Train VisDrone "
                f"(writes {path.name}; weights are not shipped)."
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
        if hasattr(self, "header_logo"):
            self.header_logo.setPixmap(load_brand_pixmap(theme, 40))
        if hasattr(self, "home_stage"):
            self.home_stage.set_theme(theme)
        for name in (
            "canvas",
            "ref_canvas",
            "align_canvas",
            "preview_canvas",
            "check_canvas",
            "review_canvas",
        ):
            widget = getattr(self, name, None)
            if widget is not None and hasattr(widget, "set_theme"):
                widget.set_theme(theme)

    def _open_training_folder(self) -> None:
        from PyQt6.QtCore import QUrl
        from PyQt6.QtGui import QDesktopServices

        folder = visdrone_weights_path().parent
        folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder.resolve())))

    def _refresh_settings_page(self) -> None:
        if hasattr(self, "export_pattern_edit"):
            self.export_pattern_edit.blockSignals(True)
            self.export_pattern_edit.setText(self.settings.export_pattern)
            self.export_pattern_edit.blockSignals(False)
            self._sync_export_pattern_combo()
        self._refresh_export_preview()
        self.warn_onedrive_check.blockSignals(True)
        self.warn_onedrive_check.setChecked(self.settings.warn_onedrive)
        self.warn_onedrive_check.blockSignals(False)
        self.theme_combo.blockSignals(True)
        self.theme_combo.setCurrentIndex(max(0, self.theme_combo.findData(self.settings.theme)))
        self.theme_combo.blockSignals(False)
        if hasattr(self, "detector_imgsz_combo"):
            self.detector_imgsz_combo.blockSignals(True)
            self.detector_imgsz_combo.setCurrentIndex(
                max(0, self.detector_imgsz_combo.findData(self.settings.detector_imgsz))
            )
            self.detector_imgsz_combo.blockSignals(False)
            self.detect_stride_combo.blockSignals(True)
            self.detect_stride_combo.setCurrentIndex(
                max(0, self.detect_stride_combo.findData(self.settings.detect_stride))
            )
            self.detect_stride_combo.blockSignals(False)
            self.preview_every_combo.blockSignals(True)
            self.preview_every_combo.setCurrentIndex(
                max(0, self.preview_every_combo.findData(self.settings.preview_every))
            )
            self.preview_every_combo.blockSignals(False)
        self._sync_detector_ui()
        self._schedule_frame_check()

    def _toggle_advanced_sliders(self, checked: bool) -> None:
        self.advanced_panel.setVisible(bool(checked))
        self.advanced_btn.setText("Hide advanced sliders" if checked else "Show advanced sliders")

    def _schedule_frame_check(self) -> None:
        timer = getattr(self, "_frame_check_timer", None)
        if timer is None:
            return
        timer.start()

    def _frame_check_still(self):
        if self._ref_frame is not None:
            return self._ref_frame
        if not self.videos:
            return None
        try:
            frame, _ = grab_frame_at(self.videos[0], 0.0)
            return frame
        except Exception:
            return None

    def _run_frame_check(self) -> None:
        if not hasattr(self, "check_canvas"):
            return
        still = self._frame_check_still()
        if still is None:
            self.frame_check_status.setText(
                "Open a video and draw routes first so this still has a frame to check."
            )
            return
        if self.settings.detector_mode != "trained":
            self.check_canvas.set_frame(still)
            self.frame_check_status.setText(
                "Switch to Trained detector to draw surviving boxes on this still. No full clip."
            )
            return
        weights = visdrone_weights_path()
        if not weights.is_file():
            self.check_canvas.set_frame(still)
            self.frame_check_status.setText(
                "Train VisDrone first. Weights are not bundled with a new install."
            )
            return
        self.frame_check_status.setText("Running detector on this still…")
        self.check_canvas.set_frame(still)
        try:
            from traffic_tracker.core.motion import default_area_limits

            model = load_detector(weights)
            height, width = still.shape[:2]
            limits = self.geometry.vehicle_area_limits()
            if limits is None:
                limits = default_area_limits(width, height)
            detections, _mask = boxes_from_detector(
                model,
                still,
                min_area=limits[0],
                max_area=limits[1],
                conf=self.settings.detector_conf,
                iou=self.settings.detector_iou,
                imgsz=self.settings.detector_imgsz,
            )
            self.check_canvas.set_frame(draw_detection_preview(still, detections))
            self.frame_check_status.setText(
                f"{len(detections)} box(es) on this still  •  "
                f"conf {self.settings.detector_conf:.2f}  •  NMS {self.settings.detector_iou:.2f}"
            )
        except Exception as exc:
            self.check_canvas.set_frame(still)
            self.frame_check_status.setText(f"Could not run detector: {exc}")

    def _refresh_visdrone_status(self) -> None:
        path = visdrone_weights_path()
        report = device_report()
        extra = " — this will take a long time" if report.device == "cpu" else ""
        if path.is_file():
            text = f"VisDrone weights ready: {path}  •  {report.label}{extra}"
        else:
            text = f"Not trained yet. Train VisDrone writes {path}  •  {report.label}{extra}"
        if hasattr(self, "train_status"):
            self.train_status.setText(text)
        self._refresh_detector_status()

    def _set_train_busy(self, busy: bool) -> None:
        if hasattr(self, "train_visdrone_btn"):
            self.train_visdrone_btn.setEnabled(not busy)
        if hasattr(self, "cancel_train_btn"):
            self.cancel_train_btn.setEnabled(busy)

    def _start_visdrone_train(self) -> None:
        if self.train_worker and self.train_worker.isRunning():
            return
        go = QMessageBox.question(
            self,
            APP_NAME,
            "This downloads VisDrone2019-DET (~2 GB) and fine-tunes YOLO11n. "
            "It can take hours. Continue?",
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
        self._launch_train_worker("Starting VisDrone training…")

    def _launch_train_worker(self, start_message: str) -> None:
        self.train_log.clear()
        self.train_progress.setValue(0)
        self.train_log.appendPlainText(start_message)
        self.train_worker = TrainWorker(self)
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
        self.settings.detector_source = "visdrone"
        self.settings.detector_weights = str(visdrone_weights_path())
        for combo in (getattr(self, "detector_combo", None), getattr(self, "process_detector_combo", None)):
            if combo is None:
                continue
            combo.blockSignals(True)
            combo.setCurrentIndex(max(0, combo.findData("trained")))
            combo.blockSignals(False)
        self._persist_settings()
        self._refresh_visdrone_status()
        QMessageBox.information(
            self,
            APP_NAME,
            "VisDrone weights saved. Process will use Trained detector with visdrone.pt.",
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
