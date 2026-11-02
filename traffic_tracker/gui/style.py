"""Compact dark theme for a video-review tool."""

STYLESHEET = """
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
"""
