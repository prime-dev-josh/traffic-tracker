"""Click-drag line drawing, overlay nudge, and match-point clicks on a frame."""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np
from PyQt6.QtCore import QLineF, QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QPolygonF
from PyQt6.QtWidgets import QWidget

from traffic_tracker.core.geometry import Box, Geometry, Line

START_COLOR = QColor(0, 230, 120)
END_COLOR = QColor(255, 56, 56)
NORTH_COLOR = QColor(40, 200, 255)
VEHICLE_COLOR = QColor(255, 20, 220)
DRAFT_COLOR = QColor(40, 200, 255)
LABEL_COLOR = QColor(255, 255, 255)
MATCH_COLOR = QColor(255, 20, 220)
HALO_COLOR = QColor(0, 0, 0)

Mode = str  # view | draw | nudge | match


class FrameCanvas(QWidget):
    """Shows a BGR OpenCV frame and supports draw / nudge / match modes."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(640, 400)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.WheelFocus)
        self._pixmap = QPixmap()
        self._image_size = (1, 1)
        self._geometry = Geometry(1, 1)
        self._draft_start: QPointF | None = None
        self._draft_end: QPointF | None = None
        self._drawing = False
        self._mode: Mode = "view"
        self._nudge_last: QPointF | None = None
        self._match_points: list[tuple[float, float]] = []
        self._draft_box = False
        self.on_line_drawn: Callable[[Line], None] | None = None
        self.on_box_drawn: Callable[[Box], None] | None = None
        self.on_nudge: Callable[[float, float], None] | None = None
        self.on_scale: Callable[[float], None] | None = None
        self.on_point: Callable[[float, float], None] | None = None

    def set_mode(self, mode: Mode) -> None:
        self._mode = mode
        if mode != "draw":
            self.clear_draft()
        if mode != "nudge":
            self._nudge_last = None
        self.update()

    def set_match_points(self, points: list[tuple[float, float]]) -> None:
        self._match_points = list(points)
        self.update()

    def set_frame(self, frame_bgr: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        self._image_size = (width, height)
        image = QImage(rgb.data, width, height, channels * width, QImage.Format.Format_RGB888).copy()
        self._pixmap = QPixmap.fromImage(image)
        self.update()

    def set_geometry(self, geometry: Geometry | None) -> None:
        self._geometry = geometry or Geometry(1, 1)
        self.update()

    def set_routes(self, _routes) -> None:
        """Compatibility no-op; use set_geometry."""
        self.update()

    def set_drawing_enabled(self, enabled: bool, *, box: bool = False) -> None:
        self._draft_box = box
        self.set_mode("draw" if enabled else "view")

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
        return self._geometry

    def _display_rect(self) -> QRectF:
        if self._pixmap.isNull() or self.width() <= 0 or self.height() <= 0:
            return QRectF(self.rect())
        pw, ph = max(self._pixmap.width(), 1), max(self._pixmap.height(), 1)
        scale = min(self.width() / pw, self.height() / ph)
        w, h = pw * scale, ph * scale
        x = (self.width() - w) / 2
        y = (self.height() - h) / 2
        return QRectF(x, y, w, h)

    def _widget_to_image(self, pos) -> QPointF | None:
        if self._pixmap.isNull():
            return None
        rect = self._display_rect()
        if rect.width() <= 0 or rect.height() <= 0:
            return None
        x = min(max(float(pos.x()), rect.left()), rect.right())
        y = min(max(float(pos.y()), rect.top()), rect.bottom())
        image_x = (x - rect.x()) / rect.width() * self._image_size[0]
        image_y = (y - rect.y()) / rect.height() * self._image_size[1]
        image_x = min(max(image_x, 0), self._image_size[0])
        image_y = min(max(image_y, 0), self._image_size[1])
        return QPointF(image_x, image_y)

    def _image_to_widget(self, x: float, y: float) -> QPointF:
        rect = self._display_rect()
        return QPointF(
            rect.x() + (x / self._image_size[0]) * rect.width(),
            rect.y() + (y / self._image_size[1]) * rect.height(),
        )

    def wheelEvent(self, event) -> None:
        if self._mode != "nudge" or self.on_scale is None:
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.04 if delta > 0 else 1 / 1.04
        self.on_scale(factor)

    def mousePressEvent(self, event) -> None:
        image_pos = self._widget_to_image(event.position())
        if image_pos is None:
            return
        if self._mode == "match" and event.button() == Qt.MouseButton.LeftButton:
            if self.on_point:
                self.on_point(float(image_pos.x()), float(image_pos.y()))
            return
        if self._mode == "nudge" and event.button() == Qt.MouseButton.LeftButton:
            self._nudge_last = image_pos
            return
        if self._mode != "draw" or event.button() != Qt.MouseButton.LeftButton:
            return
        self._drawing = True
        self._draft_start = image_pos
        self._draft_end = image_pos
        self.update()

    def mouseMoveEvent(self, event) -> None:
        if self._mode == "nudge" and self._nudge_last is not None and self.on_nudge:
            image_pos = self._widget_to_image(event.position())
            if image_pos is None:
                return
            self.on_nudge(float(image_pos.x() - self._nudge_last.x()), float(image_pos.y() - self._nudge_last.y()))
            self._nudge_last = image_pos
            return
        if not self._drawing:
            return
        image_pos = self._widget_to_image(event.position())
        if image_pos is not None:
            self._draft_end = image_pos
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if self._mode == "nudge" and event.button() == Qt.MouseButton.LeftButton:
            self._nudge_last = None
            return
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
        as_box = self._draft_box
        self._draft_start = None
        self._draft_end = None
        self.update()
        if as_box and self.on_box_drawn:
            self.on_box_drawn(Box.from_line(line))
        elif self.on_line_drawn:
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

        def draw_halo_line(p1: QPointF, p2: QPointF, color: QColor, width: int = 3) -> None:
            painter.setPen(QPen(HALO_COLOR, width + 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawLine(QLineF(p1, p2))
            painter.setPen(QPen(color, width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawLine(QLineF(p1, p2))

        def draw_halo_rect(box: QRectF, color: QColor, width: int = 3) -> None:
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(HALO_COLOR, width + 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.SquareCap))
            painter.drawRect(box)
            painter.setPen(QPen(color, width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.SquareCap))
            painter.drawRect(box)

        def draw_halo_text(pos: QPointF, text: str, color: QColor) -> None:
            painter.setPen(HALO_COLOR)
            for dx, dy in ((-1, -1), (-1, 1), (1, -1), (1, 1), (-1, 0), (1, 0), (0, -1), (0, 1)):
                painter.drawText(pos + QPointF(dx, dy), text)
            painter.setPen(color)
            painter.drawText(pos, text)

        def draw_labeled_line(line: Line, color: QColor, label: str) -> None:
            p1 = self._image_to_widget(line.x1, line.y1)
            p2 = self._image_to_widget(line.x2, line.y2)
            draw_halo_line(p1, p2, color)
            draw_halo_text(p1 + QPointF(6, -6), label, color)

        def draw_north(line: Line) -> None:
            p1 = self._image_to_widget(line.x1, line.y1)
            p2 = self._image_to_widget(line.x2, line.y2)
            draw_halo_line(p1, p2, NORTH_COLOR)
            dx = p2.x() - p1.x()
            dy = p2.y() - p1.y()
            length = (dx * dx + dy * dy) ** 0.5 or 1.0
            ux, uy = dx / length, dy / length
            left = QPointF(p2.x() - ux * 14 + uy * 7, p2.y() - uy * 14 - ux * 7)
            right = QPointF(p2.x() - ux * 14 - uy * 7, p2.y() - uy * 14 + ux * 7)
            arrow = QPolygonF([p2, left, right])
            painter.setPen(QPen(HALO_COLOR, 3))
            painter.setBrush(HALO_COLOR)
            painter.drawPolygon(arrow)
            painter.setPen(QPen(NORTH_COLOR, 1))
            painter.setBrush(NORTH_COLOR)
            painter.drawPolygon(arrow)
            draw_halo_text(p2 + QPointF(8, -8), "North", NORTH_COLOR)

        geom = self._geometry
        if geom.north is not None:
            draw_north(geom.north)
        for item in geom.starts:
            draw_labeled_line(item.line, START_COLOR, item.name or "Start")
        for item in geom.ends:
            draw_labeled_line(item.line, END_COLOR, item.name or "End")
        for index, raw in enumerate(geom.vehicle_refs, start=1):
            box = raw.normalized()
            p1 = self._image_to_widget(box.x1, box.y1)
            p2 = self._image_to_widget(box.x2, box.y2)
            draw_halo_rect(QRectF(p1, p2).normalized(), VEHICLE_COLOR)
            w, h = int(box.width()), int(box.height())
            draw_halo_text(p1 + QPointF(6, -6), f"Vehicle {index}  {w}×{h}", VEHICLE_COLOR)

        if self._draft_start is not None and self._draft_end is not None:
            p1 = self._image_to_widget(self._draft_start.x(), self._draft_start.y())
            p2 = self._image_to_widget(self._draft_end.x(), self._draft_end.y())
            if self._draft_box:
                draw_halo_rect(QRectF(p1, p2).normalized(), DRAFT_COLOR)
            else:
                draw_halo_line(p1, p2, DRAFT_COLOR, 2)

        for index, (x, y) in enumerate(self._match_points, start=1):
            pos = self._image_to_widget(x, y)
            painter.setPen(QPen(HALO_COLOR, 4))
            painter.setBrush(HALO_COLOR)
            painter.drawEllipse(pos, 6, 6)
            painter.setPen(QPen(MATCH_COLOR, 2))
            painter.setBrush(MATCH_COLOR)
            painter.drawEllipse(pos, 5, 5)
            draw_halo_text(pos + QPointF(8, -8), str(index), MATCH_COLOR)
