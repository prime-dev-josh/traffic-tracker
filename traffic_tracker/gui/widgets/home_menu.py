"""Title-screen home menu: numbered rows, HUD chrome, keyboard select."""

from __future__ import annotations

from collections.abc import Callable
import math

from PyQt6.QtCore import QRect, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPen
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from traffic_tracker.config import APP_NAME, APP_VERSION
from traffic_tracker.gui.style import load_brand_pixmap

ACCENT = QColor(241, 91, 44)
AMBER = QColor(241, 91, 44)
INK = QColor(255, 255, 255)
NAVY = QColor(34, 26, 74)
NAVY_TEXT = QColor(36, 50, 138)


def hud_font(size: int, *, bold: bool = True, mono: bool = False, tracking: float = 0.0) -> QFont:
    point = max(1, int(size))
    family = "Cascadia Mono" if mono else "Bahnschrift"
    font = QFont(family)
    font.setPointSize(point)
    if not font.exactMatch():
        font = QFont("Consolas" if mono else "Segoe UI")
        font.setPointSize(point)
    font.setBold(bold)
    if tracking:
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, tracking)
    return font


class HomeMenuRow(QWidget):
    activated = pyqtSignal()
    hovered = pyqtSignal(int)

    def __init__(self, index: int, code: str, title: str, blurb: str, parent=None) -> None:
        super().__init__(parent)
        self.index = index
        self.code = code
        self.title_text = title
        self.blurb = blurb
        self.selected = False
        self._theme = "dark"
        self.setObjectName("homeMenuRow")
        self.setMinimumHeight(72)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(18, 10, 22, 10)
        layout.setSpacing(16)

        self.caret = QLabel("▸")
        self.caret.setFixedWidth(18)
        self.caret.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.caret.setFont(hud_font(18, tracking=0))

        self.code_label = QLabel(code)
        self.code_label.setFixedWidth(36)
        self.code_label.setFont(hud_font(12, mono=True, bold=False, tracking=1))

        self.title_label = QLabel(title)
        self.title_label.setFont(hud_font(20, tracking=3.5))

        layout.addWidget(self.caret)
        layout.addWidget(self.code_label)
        layout.addWidget(self.title_label, 1)
        self.set_selected(False)

    def set_theme(self, theme: str) -> None:
        self._theme = theme
        self.set_selected(self.selected)

    def set_selected(self, selected: bool) -> None:
        self.selected = selected
        self.caret.setVisible(selected)
        light = self._theme == "light"
        color = (NAVY_TEXT if light else INK) if selected else (
            QColor(107, 115, 161) if light else QColor(197, 192, 220)
        )
        accent = ACCENT if selected else (QColor(107, 115, 161) if light else QColor(138, 132, 168))
        self.caret.setStyleSheet(f"color: rgb({ACCENT.red()},{ACCENT.green()},{ACCENT.blue()}); background: transparent;")
        self.code_label.setStyleSheet(f"color: rgb({accent.red()},{accent.green()},{accent.blue()}); background: transparent;")
        self.title_label.setStyleSheet(f"color: rgb({color.red()},{color.green()},{color.blue()}); background: transparent;")
        self.update()

    def enterEvent(self, event) -> None:
        self.hovered.emit(self.index)
        super().enterEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.activated.emit()
        super().mouseReleaseEvent(event)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(2, 4, -2, -4)
        if self.selected:
            fill = QColor(241, 91, 44, 42)
            painter.setPen(QPen(QColor(241, 91, 44, 180), 1))
            painter.setBrush(fill)
            painter.drawRoundedRect(rect, 4, 4)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(ACCENT)
            painter.drawRect(rect.left(), rect.top() + 8, 3, rect.height() - 16)
        else:
            if self._theme == "light":
                painter.setPen(QPen(QColor(212, 216, 236, 180), 1))
                painter.setBrush(QColor(255, 255, 255, 180))
            else:
                painter.setPen(QPen(QColor(74, 66, 120, 120), 1))
                painter.setBrush(QColor(24, 18, 67, 90))
            painter.drawRoundedRect(rect, 4, 4)


class HomeStage(QWidget):
    """Full-bleed title screen with a selectable vertical menu."""

    def __init__(
        self,
        actions: list[tuple[str, str, str, Callable[[], None]]],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("homeStage")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._pulse = 0.0
        self._selected = 0
        self._theme = "dark"
        self._actions = [item[3] for item in actions]

        root = QVBoxLayout(self)
        root.setContentsMargins(72, 40, 72, 36)
        root.setSpacing(0)

        self.logo = QLabel()
        self.logo.setObjectName("brandMark")
        self.logo.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        root.addWidget(self.logo)
        root.addSpacing(8)

        self.eyebrow = QLabel("AERIAL COUNT  //  MOTION MASK")
        self.eyebrow.setFont(hud_font(10, mono=True, bold=False, tracking=3))
        root.addWidget(self.eyebrow)

        self.title_label = QLabel(APP_NAME.upper())
        self.title_label.setFont(hud_font(42, tracking=8))
        root.addWidget(self.title_label)

        rule = QWidget()
        rule.setObjectName("homeRule")
        rule.setFixedHeight(3)
        root.addWidget(rule)
        root.addSpacing(10)

        self.tag = QLabel("SELECT AN OPERATION")
        self.tag.setFont(hud_font(11, mono=True, bold=False, tracking=2.5))
        root.addWidget(self.tag)
        root.addSpacing(28)

        body = QHBoxLayout()
        body.setSpacing(48)
        menu_col = QVBoxLayout()
        menu_col.setSpacing(8)
        self.rows: list[HomeMenuRow] = []
        for index, (code, name, blurb, callback) in enumerate(actions):
            row = HomeMenuRow(index, code, name.upper(), blurb)
            row.hovered.connect(self.select)
            row.activated.connect(lambda cb=callback: cb())
            self.rows.append(row)
            menu_col.addWidget(row)
        menu_col.addStretch(1)
        body.addLayout(menu_col, 3)

        detail = QVBoxLayout()
        detail.setSpacing(8)
        self.detail_code = QLabel("01")
        self.detail_code.setFont(hud_font(14, mono=True, tracking=2))
        self.detail_title = QLabel("")
        self.detail_title.setFont(hud_font(26, tracking=4))
        self.detail_title.setWordWrap(True)
        self.detail_blurb = QLabel("")
        self.detail_blurb.setFont(hud_font(13, bold=False, tracking=0.4))
        self.detail_blurb.setWordWrap(True)
        self.move_hint = QLabel("ENTER  •  CONFIRM\nUP / DOWN  •  MOVE")
        self.move_hint.setFont(hud_font(10, mono=True, bold=False, tracking=1.5))
        detail.addStretch(1)
        detail.addWidget(self.detail_code)
        detail.addWidget(self.detail_title)
        detail.addWidget(self.detail_blurb)
        detail.addSpacing(24)
        detail.addWidget(self.move_hint)
        detail.addStretch(2)
        body.addLayout(detail, 2)
        root.addLayout(body, 1)

        footer = QHBoxLayout()
        self.ver = QLabel(f"BUILD  {APP_VERSION}")
        self.ver.setFont(hud_font(10, mono=True, bold=False, tracking=1.5))
        self.footer_hint = QLabel("PRESS ENTER  OR  CLICK  TO  START")
        self.footer_hint.setFont(hud_font(10, mono=True, bold=False, tracking=1.8))
        footer.addWidget(self.ver)
        footer.addStretch(1)
        footer.addWidget(self.footer_hint)
        root.addLayout(footer)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(40)
        self.set_theme("dark")
        self.select(0)

    def set_theme(self, theme: str) -> None:
        self._theme = "light" if theme == "light" else "dark"
        light = self._theme == "light"
        ink = "#24328A" if light else "#FFFFFF"
        mute = "#6B73A1" if light else "#C5C0DC"
        faint = "#8A90B4" if light else "#8A84A8"
        self.logo.setPixmap(load_brand_pixmap(self._theme, 72 if light else 88))
        self.eyebrow.setStyleSheet("color: #F15B2C; background: transparent;")
        self.title_label.setStyleSheet(f"color: {ink}; background: transparent;")
        self.tag.setStyleSheet(f"color: {mute}; background: transparent;")
        self.detail_code.setStyleSheet("color: #F15B2C; background: transparent;")
        self.detail_title.setStyleSheet(f"color: {ink}; background: transparent;")
        self.detail_blurb.setStyleSheet(f"color: {mute}; background: transparent;")
        self.move_hint.setStyleSheet(f"color: {faint}; background: transparent;")
        self.ver.setStyleSheet(f"color: {faint}; background: transparent;")
        self.footer_hint.setStyleSheet(f"color: {faint}; background: transparent;")
        for row in self.rows:
            row.set_theme(self._theme)
        self.update()

    def focus_menu(self) -> None:
        self.setFocus(Qt.FocusReason.OtherFocusReason)
        self.select(self._selected)

    def select(self, index: int) -> None:
        if not self.rows:
            return
        self._selected = index % len(self.rows)
        for row in self.rows:
            row.set_selected(row.index == self._selected)
        current = self.rows[self._selected]
        self.detail_code.setText(current.code)
        self.detail_title.setText(current.title_text)
        self.detail_blurb.setText(current.blurb)
        self.update()

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key in (Qt.Key.Key_Down, Qt.Key.Key_S):
            self.select(self._selected + 1)
        elif key in (Qt.Key.Key_Up, Qt.Key.Key_W):
            self.select(self._selected - 1)
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            self._actions[self._selected]()
        else:
            super().keyPressEvent(event)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.focus_menu()

    def _tick(self) -> None:
        self._pulse = (self._pulse + 0.07) % (math.pi * 2)
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        width, height = self.width(), self.height()
        light = self._theme == "light"
        painter.fillRect(self.rect(), QColor(247, 248, 252) if light else NAVY)

        glow = QLinearGradient(0, 0, 0, height)
        if light:
            glow.setColorAt(0.0, QColor(252, 216, 204, 90))
            glow.setColorAt(0.4, QColor(247, 248, 252, 0))
            glow.setColorAt(1.0, QColor(228, 231, 244, 180))
        else:
            glow.setColorAt(0.0, QColor(241, 91, 44, 50))
            glow.setColorAt(0.4, QColor(34, 26, 74, 0))
            glow.setColorAt(1.0, QColor(24, 18, 67, 220))
        painter.fillRect(self.rect(), glow)

        pulse = 0.45 + 0.25 * math.sin(self._pulse)
        horizon_y = int(height * 0.42)
        painter.setPen(QPen(QColor(241, 91, 44, int(40 + 50 * pulse)), 2))
        painter.drawLine(0, horizon_y, width, horizon_y)

        vanish_x = int(width * 0.72)
        painter.setPen(QPen(QColor(241, 91, 44, 28), 1))
        for t in range(-8, 9):
            x = vanish_x + t * 70
            painter.drawLine(vanish_x, horizon_y, x + t * 40, height)
        painter.setPen(QPen(AMBER, 1, Qt.PenStyle.DashLine))
        painter.drawLine(vanish_x, horizon_y, int(width * 0.55), height)

        scan = QColor(255, 255, 255, 10) if not light else QColor(36, 50, 138, 8)
        for y in range(0, height, 4):
            painter.setPen(QPen(scan, 1))
            painter.drawLine(0, y, width, y)

        painter.setPen(QPen(QColor(241, 91, 44, 80), 1))
        painter.drawRect(QRect(18, 18, width - 36, height - 36))
        painter.setPen(QPen(QColor(241, 91, 44, 140), 1))
        painter.drawLine(18, 18, 72, 18)
        painter.drawLine(18, 18, 18, 72)
        painter.drawLine(width - 18, height - 18, width - 72, height - 18)
        painter.drawLine(width - 18, height - 18, width - 18, height - 72)
