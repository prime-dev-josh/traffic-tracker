"""Compact dark and light themes for a video-review tool."""

DARK_STYLESHEET = """
QWidget {
    background-color: #1a1d23;
    color: #e8eaed;
    font-family: "Segoe UI", "Helvetica Neue", sans-serif;
    font-size: 13px;
}
QMainWindow, QDialog {
    background-color: #1a1d23;
}
QLabel#title {
    font-size: 22px;
    font-weight: 600;
    color: #ffffff;
}
QLabel#subtitle {
    color: #9aa0a6;
    font-size: 13px;
}
QLabel#stepTitle {
    font-size: 16px;
    font-weight: 600;
}
QLabel#hint {
    color: #9aa0a6;
}
QFrame#card {
    background-color: #242830;
    border: 1px solid #343a46;
    border-radius: 8px;
}
QWidget#homeStage, QWidget#homeStage QWidget, QWidget#homeStage QLabel {
    background-color: transparent;
    color: #e8eaed;
}
QWidget#homeRule {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3d8bfd, stop:0.45 #ffc448, stop:1 transparent);
}
QLineEdit, QDateTimeEdit, QListWidget, QPlainTextEdit, QTableWidget {
    background-color: #12141a;
    border: 1px solid #343a46;
    border-radius: 6px;
    padding: 6px 8px;
    selection-background-color: #3d8bfd;
}
QListWidget::item {
    padding: 6px;
}
QListWidget::item:selected {
    background-color: #2b4a7a;
}
QPushButton {
    background-color: #2b313c;
    border: 1px solid #3a4150;
    border-radius: 6px;
    padding: 7px 14px;
}
QPushButton:hover {
    background-color: #343b49;
}
QPushButton:disabled {
    color: #6b7180;
    background-color: #22252c;
}
QPushButton#primary {
    background-color: #3d8bfd;
    border: 1px solid #3d8bfd;
    color: #ffffff;
    font-weight: 600;
}
QPushButton#primary:hover {
    background-color: #5aa0ff;
}
QPushButton#danger {
    background-color: #5c2b2b;
    border: 1px solid #8a3d3d;
}
QPushButton#rowAction, QPushButton#rowDanger {
    padding: 4px 8px;
    font-size: 12px;
}
QPushButton#rowDanger {
    background-color: #5c2b2b;
    border: 1px solid #8a3d3d;
}
QWidget#itemRow {
    background-color: #1c2028;
    border: 1px solid #343a46;
    border-radius: 6px;
}
QLabel#sectionLabel {
    color: #9aa0a6;
    font-size: 11px;
    font-weight: 600;
}
QComboBox {
    background-color: #12141a;
    border: 1px solid #343a46;
    border-radius: 4px;
    padding: 3px 6px;
    min-width: 56px;
}
QComboBox QAbstractItemView {
    background-color: #12141a;
    color: #e8eaed;
    selection-background-color: #2b4a7a;
}
QProgressBar {
    background-color: #12141a;
    border: 1px solid #343a46;
    border-radius: 6px;
    text-align: center;
    height: 18px;
}
QProgressBar::chunk {
    background-color: #3d8bfd;
    border-radius: 5px;
}
QCheckBox {
    spacing: 8px;
}
QHeaderView::section {
    background-color: #2b313c;
    color: #e8eaed;
    border: none;
    padding: 6px;
    font-weight: 600;
}
QTableWidget {
    gridline-color: #343a46;
}
QStatusBar {
    background-color: #12141a;
    color: #9aa0a6;
}
QScrollArea {
    border: none;
    background-color: transparent;
}
QTabWidget::pane {
    border: 1px solid #343a46;
    background-color: #242830;
    top: -1px;
}
QTabBar::tab {
    background-color: #2b313c;
    color: #e8eaed;
    border: 1px solid #343a46;
    padding: 8px 14px;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background-color: #3d8bfd;
    color: #ffffff;
    border-color: #3d8bfd;
}
"""

LIGHT_STYLESHEET = """
QWidget {
    background-color: #f4f6f8;
    color: #1c2128;
    font-family: "Segoe UI", "Helvetica Neue", sans-serif;
    font-size: 13px;
}
QMainWindow, QDialog {
    background-color: #f4f6f8;
}
QLabel#title {
    font-size: 22px;
    font-weight: 600;
    color: #111418;
}
QLabel#subtitle {
    color: #5b6573;
    font-size: 13px;
}
QLabel#stepTitle {
    font-size: 16px;
    font-weight: 600;
    color: #111418;
}
QLabel#hint {
    color: #5b6573;
}
QFrame#card {
    background-color: #ffffff;
    border: 1px solid #d5dbe3;
    border-radius: 8px;
}
QWidget#homeStage, QWidget#homeStage QWidget, QWidget#homeStage QLabel {
    background-color: transparent;
    color: #1c2128;
}
QWidget#homeRule {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2b6fd6, stop:0.45 #c98400, stop:1 transparent);
}
QLineEdit, QDateTimeEdit, QListWidget, QPlainTextEdit, QTableWidget {
    background-color: #ffffff;
    border: 1px solid #c9d1db;
    border-radius: 6px;
    padding: 6px 8px;
    selection-background-color: #c7dbff;
    color: #1c2128;
}
QListWidget::item {
    padding: 6px;
}
QListWidget::item:selected {
    background-color: #d6e6ff;
    color: #111418;
}
QPushButton {
    background-color: #e8edf3;
    border: 1px solid #c5ced8;
    border-radius: 6px;
    padding: 7px 14px;
    color: #1c2128;
}
QPushButton:hover {
    background-color: #dde4ed;
}
QPushButton:disabled {
    color: #8b94a0;
    background-color: #eef1f5;
}
QPushButton#primary {
    background-color: #2b6fd6;
    border: 1px solid #2b6fd6;
    color: #ffffff;
    font-weight: 600;
}
QPushButton#primary:hover {
    background-color: #4584e6;
}
QPushButton#danger {
    background-color: #f4d6d6;
    border: 1px solid #d48a8a;
    color: #6b1f1f;
}
QPushButton#rowAction, QPushButton#rowDanger {
    padding: 4px 8px;
    font-size: 12px;
}
QPushButton#rowDanger {
    background-color: #f4d6d6;
    border: 1px solid #d48a8a;
}
QWidget#itemRow {
    background-color: #f8fafc;
    border: 1px solid #d5dbe3;
    border-radius: 6px;
}
QLabel#sectionLabel {
    color: #5b6573;
    font-size: 11px;
    font-weight: 600;
}
QComboBox {
    background-color: #ffffff;
    border: 1px solid #c9d1db;
    border-radius: 4px;
    padding: 3px 6px;
    min-width: 56px;
    color: #1c2128;
}
QComboBox QAbstractItemView {
    background-color: #ffffff;
    color: #1c2128;
    selection-background-color: #d6e6ff;
}
QProgressBar {
    background-color: #ffffff;
    border: 1px solid #c9d1db;
    border-radius: 6px;
    text-align: center;
    height: 18px;
}
QProgressBar::chunk {
    background-color: #2b6fd6;
    border-radius: 5px;
}
QCheckBox {
    spacing: 8px;
}
QHeaderView::section {
    background-color: #e8edf3;
    color: #1c2128;
    border: none;
    padding: 6px;
    font-weight: 600;
}
QTableWidget {
    gridline-color: #d5dbe3;
}
QStatusBar {
    background-color: #e8edf3;
    color: #5b6573;
}
QScrollArea {
    border: none;
    background-color: transparent;
}
QTabWidget::pane {
    border: 1px solid #d5dbe3;
    background-color: #ffffff;
    top: -1px;
}
QTabBar::tab {
    background-color: #e8edf3;
    color: #1c2128;
    border: 1px solid #d5dbe3;
    padding: 8px 14px;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background-color: #2b6fd6;
    color: #ffffff;
    border-color: #2b6fd6;
}
"""

STYLESHEET = DARK_STYLESHEET


def stylesheet_for(theme: str) -> str:
    return LIGHT_STYLESHEET if theme == "light" else DARK_STYLESHEET
