"""Click-drag line drawing on a video frame."""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np
from PyQt6.QtCore import QLineF, QPointF, Qt
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QWidget

from traffic_tracker.core.geometry import Geometry, Line, Route

START_COLOR = QColor(61, 214, 140)
END_COLOR = QColor(255, 107, 107)
DRAFT_COLOR = QColor(61, 139, 253)
LABEL_COLOR = QColor(232, 234, 237)


class FrameCanvas(QWidget):
    """Shows a BGR OpenCV frame and lets the user draw start/end lines."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(640, 400)
        self._pixmap = QPixmap()
        self._image_size = (1, 1)
        self._routes: list[Route] = []
        self._draft_start: QPointF | None = None
        self._draft_end: QPointF | None = None
        self._drawing = False
        self._enabled = False
        self.on_line_drawn: Callable[[Line], None] | None = None

    def set_frame(self, frame_bgr: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        self._image_size = (width, height)
        image = QImage(rgb.data, width, height, channels * width, QImage.Format.Format_RGB888).copy()
        self._pixmap = QPixmap.fromImage(image)
        self.update()

    def set_routes(self, routes: list[Route]) -> None:
        self._routes = list(routes)
        self.update()

    def set_drawing_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        if not enabled:
            self._draft_start = None
            self._draft_end = None
            self._drawing = False
        self.update()

    def clear_draft(self) -> None:
        self._draft_start = None
        self._draft_end = None
        self._drawing = False
        self.update()

    def has_frame(self) -> bool:
        return not self._pixmap.isNull()

    def image_size(self) -> tuple[int, int]:
        return self._image_size

    def to_geometry(self) -> Geometry:
        width, height = self._image_size
        return Geometry(frame_width=width, frame_height=height, routes=list(self._routes))

    def _display_rect(self):
        if self._pixmap.isNull():
            return self.rect()
        pw, ph = self._pixmap.width(), self._pixmap.height()
        scale = min(self.width() / pw, self.height() / ph)
        w, h = pw * scale, ph * scale
        x = (self.width() - w) / 2
        y = (self.height() - h) / 2
        from PyQt6.QtCore import QRectF

        return QRectF(x, y, w, h)

    def _widget_to_image(self, pos) -> QPointF | None:
        rect = self._display_rect()
        if not rect.contains(pos):
            return None
        x = (pos.x() - rect.x()) / rect.width() * self._image_size[0]
        y = (pos.y() - rect.y()) / rect.height() * self._image_size[1]
        x = min(max(x, 0), self._image_size[0])
        y = min(max(y, 0), self._image_size[1])
        return QPointF(x, y)

    def _image_to_widget(self, x: float, y: float) -> QPointF:
        rect = self._display_rect()
        return QPointF(
            rect.x() + (x / self._image_size[0]) * rect.width(),
            rect.y() + (y / self._image_size[1]) * rect.height(),
        )

    def mousePressEvent(self, event) -> None:
        if not self._enabled or event.button() != Qt.MouseButton.LeftButton:
            return
        image_pos = self._widget_to_image(event.position())
        if image_pos is None:
            return
        self._drawing = True
        self._draft_start = image_pos
        self._draft_end = image_pos
        self.update()

    def mouseMoveEvent(self, event) -> None:
        if not self._drawing:
            return
        image_pos = self._widget_to_image(event.position())
        if image_pos is not None:
            self._draft_end = image_pos
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if not self._drawing or event.button() != Qt.MouseButton.LeftButton:
            return
        self._drawing = False
        if self._draft_start is None or self._draft_end is None:
            return
        line = Line(
            x1=float(self._draft_start.x()),
            y1=float(self._draft_start.y()),
            x2=float(self._draft_end.x()),
            y2=float(self._draft_end.y()),
        )
        self._draft_start = None
        self._draft_end = None
        self.update()
        if self.on_line_drawn:
            self.on_line_drawn(line)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(12, 14, 18))
        if self._pixmap.isNull():
            painter.setPen(LABEL_COLOR)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Load a video to see a frame here")
            return

        rect = self._display_rect()
        painter.drawPixmap(rect.toRect(), self._pixmap)

        def draw_labeled_line(line: Line, color: QColor, label: str) -> None:
            p1 = self._image_to_widget(line.x1, line.y1)
            p2 = self._image_to_widget(line.x2, line.y2)
            pen = QPen(color, 3)
            painter.setPen(pen)
            painter.drawLine(QLineF(p1, p2))
            painter.drawText(p1 + QPointF(6, -6), label)

        for route in self._routes:
            draw_labeled_line(route.start, START_COLOR, f"{route.name} start")
            draw_labeled_line(route.end, END_COLOR, f"{route.name} end")

        if self._draft_start is not None and self._draft_end is not None:
            p1 = self._image_to_widget(self._draft_start.x(), self._draft_start.y())
            p2 = self._image_to_widget(self._draft_end.x(), self._draft_end.y())
            painter.setPen(QPen(DRAFT_COLOR, 2, Qt.PenStyle.DashLine))
            painter.drawLine(QLineF(p1, p2))
