"""Prime AE light/dark themes sampled from the official logos.

Dark: navy #221A4A field, white type, orange #F15B2C chevron.
Light: white field, navy #24328A type, orange #F05D2A chevron.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt

BRAND_DIR = Path(__file__).resolve().parent / "brand"

# Logo pixels (rounded).
NAVY = "#221A4A"
NAVY_DEEP = "#181243"
NAVY_CARD = "#2C2460"
NAVY_HOVER = "#3A3272"
NAVY_BORDER = "#4A4278"
NAVY_TEXT = "#24328A"
ORANGE = "#F15B2C"
ORANGE_LIGHT = "#F05D2A"
ORANGE_HOVER = "#F4734A"
WHITE = "#FFFFFF"
MUTED_DARK = "#C5C0DC"
MUTED_LIGHT = "#6B73A1"


def path_color_bgr(theme: str = "dark") -> tuple[int, int, int]:
    """OpenCV BGR for movement traces (primary orange)."""
    hex_color = ORANGE_LIGHT if theme == "light" else ORANGE
    value = hex_color.lstrip("#")
    r = int(value[0:2], 16)
    g = int(value[2:4], 16)
    b = int(value[4:6], 16)
    return (b, g, r)


def brand_logo_path(theme: str) -> Path:
    name = "prime_ae_light.png" if theme == "light" else "prime_ae_dark.png"
    return BRAND_DIR / name


def load_brand_pixmap(theme: str, height: int) -> QPixmap:
    path = brand_logo_path(theme)
    pixmap = QPixmap(str(path))
    if pixmap.isNull() or height <= 0:
        return pixmap
    return pixmap.scaledToHeight(height, Qt.TransformationMode.SmoothTransformation)


DARK_STYLESHEET = """
QWidget {
    background-color: #221A4A;
    color: #FFFFFF;
    font-family: "Segoe UI", "Helvetica Neue", sans-serif;
    font-size: 13px;
}
QMainWindow, QDialog, QMessageBox {
    background-color: #221A4A;
}
QLabel#title {
    font-size: 22px;
    font-weight: 600;
    color: #FFFFFF;
}
QLabel#subtitle {
    color: #C5C0DC;
    font-size: 13px;
}
QLabel#stepTitle {
    font-size: 16px;
    font-weight: 600;
    color: #FFFFFF;
}
QLabel#hint {
    color: #C5C0DC;
}
QLabel#brandMark {
    background: transparent;
}
QFrame#card {
    background-color: #2C2460;
    border: 1px solid #4A4278;
    border-radius: 8px;
}
QWidget#homeStage, QWidget#homeStage QWidget, QWidget#homeStage QLabel {
    background-color: transparent;
    color: #FFFFFF;
}
QWidget#homeRule {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #F15B2C, stop:0.55 #F15B2C, stop:1 transparent);
}
QLineEdit, QDateTimeEdit, QListWidget, QPlainTextEdit, QTableWidget {
    background-color: #181243;
    border: 1px solid #4A4278;
    border-radius: 6px;
    padding: 6px 8px;
    color: #FFFFFF;
    selection-background-color: #F15B2C;
    selection-color: #FFFFFF;
}
QListWidget::item {
    padding: 6px;
}
QListWidget::item:selected {
    background-color: #F15B2C;
    color: #FFFFFF;
}
QPushButton {
    background-color: #3A3272;
    border: 1px solid #4A4278;
    border-radius: 6px;
    padding: 7px 14px;
    color: #FFFFFF;
}
QPushButton:hover {
    background-color: #4A4278;
}
QPushButton:disabled {
    color: #8A84A8;
    background-color: #2A2258;
}
QPushButton#primary {
    background-color: #F15B2C;
    border: 1px solid #F15B2C;
    color: #FFFFFF;
    font-weight: 600;
}
QPushButton#primary:hover {
    background-color: #F4734A;
    border-color: #F4734A;
}
QPushButton#danger {
    background-color: #7A2436;
    border: 1px solid #A33B4C;
    color: #FFFFFF;
}
QPushButton#quiet {
    background-color: transparent;
    border: none;
    color: #C8C4E0;
    padding: 4px 8px;
    font-size: 12px;
}
QPushButton#quiet:hover {
    color: #FFFFFF;
    background-color: #3A3272;
}
QPushButton#rowAction, QPushButton#rowDanger {
    padding: 4px 8px;
    font-size: 12px;
}
QPushButton#rowDanger {
    background-color: #7A2436;
    border: 1px solid #A33B4C;
}
QWidget#itemRow {
    background-color: #241C52;
    border: 1px solid #4A4278;
    border-radius: 6px;
}
QLabel#sectionLabel {
    color: #C5C0DC;
    font-size: 11px;
    font-weight: 600;
}
QComboBox {
    background-color: #181243;
    border: 1px solid #4A4278;
    border-radius: 4px;
    padding: 3px 6px;
    min-width: 56px;
    color: #FFFFFF;
}
QComboBox::drop-down {
    border: none;
    width: 22px;
}
QComboBox QAbstractItemView {
    background-color: #181243;
    color: #FFFFFF;
    selection-background-color: #F15B2C;
    border: 1px solid #4A4278;
}
QProgressBar {
    background-color: #181243;
    border: 1px solid #4A4278;
    border-radius: 6px;
    text-align: center;
    height: 18px;
    color: #FFFFFF;
}
QProgressBar::chunk {
    background-color: #F15B2C;
    border-radius: 5px;
}
QCheckBox {
    spacing: 8px;
    color: #FFFFFF;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #4A4278;
    border-radius: 3px;
    background: #181243;
}
QCheckBox::indicator:checked {
    background: #F15B2C;
    border-color: #F15B2C;
}
QHeaderView::section {
    background-color: #3A3272;
    color: #FFFFFF;
    border: none;
    padding: 6px;
    font-weight: 600;
}
QTableWidget {
    gridline-color: #4A4278;
}
QStatusBar {
    background-color: #181243;
    color: #C5C0DC;
}
QScrollArea {
    border: none;
    background-color: transparent;
}
QScrollBar:vertical {
    background: #181243;
    width: 10px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: #4A4278;
    min-height: 24px;
    border-radius: 4px;
}
QScrollBar::handle:vertical:hover {
    background: #F15B2C;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QScrollBar:horizontal {
    background: #181243;
    height: 10px;
    margin: 0;
}
QScrollBar::handle:horizontal {
    background: #4A4278;
    min-width: 24px;
    border-radius: 4px;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0;
}
QTabWidget::pane {
    border: 1px solid #4A4278;
    background-color: #2C2460;
    top: -1px;
}
QTabBar::tab {
    background-color: #3A3272;
    color: #C5C0DC;
    border: 1px solid #4A4278;
    padding: 8px 14px;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background-color: #F15B2C;
    color: #FFFFFF;
    border-color: #F15B2C;
}
QSlider::groove:horizontal {
    height: 4px;
    background: #4A4278;
    border-radius: 2px;
}
QSlider::sub-page:horizontal {
    background: #F15B2C;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #F15B2C;
    width: 16px;
    height: 16px;
    margin: -6px 0;
    border-radius: 8px;
}
QSlider::handle:horizontal:hover {
    background: #F4734A;
}
QToolTip {
    background-color: #2C2460;
    color: #FFFFFF;
    border: 1px solid #F15B2C;
    padding: 4px 8px;
}
QMenu {
    background-color: #2C2460;
    color: #FFFFFF;
    border: 1px solid #4A4278;
}
QMenu::item:selected {
    background-color: #F15B2C;
}
"""

LIGHT_STYLESHEET = """
QWidget {
    background-color: #F7F8FC;
    color: #24328A;
    font-family: "Segoe UI", "Helvetica Neue", sans-serif;
    font-size: 13px;
}
QMainWindow, QDialog, QMessageBox {
    background-color: #F7F8FC;
}
QLabel#title {
    font-size: 22px;
    font-weight: 600;
    color: #24328A;
}
QLabel#subtitle {
    color: #6B73A1;
    font-size: 13px;
}
QLabel#stepTitle {
    font-size: 16px;
    font-weight: 600;
    color: #24328A;
}
QLabel#hint {
    color: #6B73A1;
}
QLabel#brandMark {
    background: transparent;
}
QFrame#card {
    background-color: #FFFFFF;
    border: 1px solid #D4D8EC;
    border-radius: 8px;
}
QWidget#homeStage, QWidget#homeStage QWidget, QWidget#homeStage QLabel {
    background-color: transparent;
    color: #24328A;
}
QWidget#homeRule {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #F05D2A, stop:0.55 #F05D2A, stop:1 transparent);
}
QLineEdit, QDateTimeEdit, QListWidget, QPlainTextEdit, QTableWidget {
    background-color: #FFFFFF;
    border: 1px solid #D4D8EC;
    border-radius: 6px;
    padding: 6px 8px;
    selection-background-color: #FCD8CC;
    selection-color: #24328A;
    color: #24328A;
}
QListWidget::item {
    padding: 6px;
}
QListWidget::item:selected {
    background-color: #FCD8CC;
    color: #24328A;
}
QPushButton {
    background-color: #EEF0F8;
    border: 1px solid #D4D8EC;
    border-radius: 6px;
    padding: 7px 14px;
    color: #24328A;
}
QPushButton:hover {
    background-color: #E4E7F4;
}
QPushButton:disabled {
    color: #9AA1C4;
    background-color: #F3F4FA;
}
QPushButton#primary {
    background-color: #F05D2A;
    border: 1px solid #F05D2A;
    color: #FFFFFF;
    font-weight: 600;
}
QPushButton#primary:hover {
    background-color: #F4734A;
    border-color: #F4734A;
}
QPushButton#danger {
    background-color: #F8D6D6;
    border: 1px solid #D48A8A;
    color: #7A1F1F;
}
QPushButton#quiet {
    background-color: transparent;
    border: none;
    color: #5C6490;
    padding: 4px 8px;
    font-size: 12px;
}
QPushButton#quiet:hover {
    color: #24328A;
    background-color: #EEF0F8;
}
QPushButton#rowAction, QPushButton#rowDanger {
    padding: 4px 8px;
    font-size: 12px;
}
QPushButton#rowDanger {
    background-color: #F8D6D6;
    border: 1px solid #D48A8A;
}
QWidget#itemRow {
    background-color: #F7F8FC;
    border: 1px solid #D4D8EC;
    border-radius: 6px;
}
QLabel#sectionLabel {
    color: #6B73A1;
    font-size: 11px;
    font-weight: 600;
}
QComboBox {
    background-color: #FFFFFF;
    border: 1px solid #D4D8EC;
    border-radius: 4px;
    padding: 3px 6px;
    min-width: 56px;
    color: #24328A;
}
QComboBox::drop-down {
    border: none;
    width: 22px;
}
QComboBox QAbstractItemView {
    background-color: #FFFFFF;
    color: #24328A;
    selection-background-color: #FCD8CC;
    border: 1px solid #D4D8EC;
}
QProgressBar {
    background-color: #FFFFFF;
    border: 1px solid #D4D8EC;
    border-radius: 6px;
    text-align: center;
    height: 18px;
    color: #24328A;
}
QProgressBar::chunk {
    background-color: #F05D2A;
    border-radius: 5px;
}
QCheckBox {
    spacing: 8px;
    color: #24328A;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #D4D8EC;
    border-radius: 3px;
    background: #FFFFFF;
}
QCheckBox::indicator:checked {
    background: #F05D2A;
    border-color: #F05D2A;
}
QHeaderView::section {
    background-color: #EEF0F8;
    color: #24328A;
    border: none;
    padding: 6px;
    font-weight: 600;
}
QTableWidget {
    gridline-color: #D4D8EC;
}
QStatusBar {
    background-color: #EEF0F8;
    color: #6B73A1;
}
QScrollArea {
    border: none;
    background-color: transparent;
}
QScrollBar:vertical {
    background: #EEF0F8;
    width: 10px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: #D4D8EC;
    min-height: 24px;
    border-radius: 4px;
}
QScrollBar::handle:vertical:hover {
    background: #F05D2A;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QScrollBar:horizontal {
    background: #EEF0F8;
    height: 10px;
    margin: 0;
}
QScrollBar::handle:horizontal {
    background: #D4D8EC;
    min-width: 24px;
    border-radius: 4px;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0;
}
QTabWidget::pane {
    border: 1px solid #D4D8EC;
    background-color: #FFFFFF;
    top: -1px;
}
QTabBar::tab {
    background-color: #EEF0F8;
    color: #24328A;
    border: 1px solid #D4D8EC;
    padding: 8px 14px;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background-color: #F05D2A;
    color: #FFFFFF;
    border-color: #F05D2A;
}
QSlider::groove:horizontal {
    height: 4px;
    background: #D4D8EC;
    border-radius: 2px;
}
QSlider::sub-page:horizontal {
    background: #F05D2A;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #F05D2A;
    width: 16px;
    height: 16px;
    margin: -6px 0;
    border-radius: 8px;
}
QSlider::handle:horizontal:hover {
    background: #F4734A;
}
QToolTip {
    background-color: #FFFFFF;
    color: #24328A;
    border: 1px solid #F05D2A;
    padding: 4px 8px;
}
QMenu {
    background-color: #FFFFFF;
    color: #24328A;
    border: 1px solid #D4D8EC;
}
QMenu::item:selected {
    background-color: #FCD8CC;
}
"""

STYLESHEET = DARK_STYLESHEET


def stylesheet_for(theme: str) -> str:
    return LIGHT_STYLESHEET if theme == "light" else DARK_STYLESHEET
