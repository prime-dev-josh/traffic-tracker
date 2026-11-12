"""Layout B: stacked panes, wrapping steps, and dropdowns that ignore the wheel."""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QPoint, QPointF, Qt
from PyQt6.QtGui import QWheelEvent
from PyQt6.QtWidgets import QApplication, QComboBox, QLabel, QScrollArea, QVBoxLayout, QWidget

from traffic_tracker.gui.wheel_guard import ComboWheelGuard
from traffic_tracker.gui.widgets.responsive_split import ResponsiveSplit


_APP: list[QApplication] = []


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    _APP.append(app)
    return app


class ResponsiveSplitTests(unittest.TestCase):
    def test_narrow_puts_picture_on_top(self) -> None:
        _app()
        leading = QLabel("tools")
        trailing = QLabel("picture")
        split = ResponsiveSplit(leading, trailing, breakpoint=1100, narrow_picture_first=True)
        split._apply(1400)
        self.assertEqual(split.orientation(), Qt.Orientation.Horizontal)
        self.assertEqual(split.widget(0).text(), "tools")
        split._apply(800)
        self.assertEqual(split.orientation(), Qt.Orientation.Vertical)
        self.assertEqual(split.widget(0).text(), "picture")
        self.assertEqual(split.widget(1).text(), "tools")

    def test_narrow_keeps_form_first_when_picture_is_not_first(self) -> None:
        _app()
        split = ResponsiveSplit(QLabel("form"), QLabel("check"), breakpoint=1100, narrow_picture_first=False)
        split._apply(700)
        self.assertEqual(split.widget(0).text(), "form")
        self.assertEqual(split.widget(1).text(), "check")

    def test_wide_split_is_user_resizable(self) -> None:
        _app()
        split = ResponsiveSplit(
            QLabel("tools"),
            QLabel("picture"),
            breakpoint=1100,
            leading_max=520,
        )
        split._apply(1400)
        split.resize(1400, 800)
        split._apply_sizes()
        self.assertEqual(split.orientation(), Qt.Orientation.Horizontal)
        self.assertFalse(split.childrenCollapsible())
        self.assertGreaterEqual(split.handleWidth(), 8)
        sizes = split.sizes()
        self.assertEqual(len(sizes), 2)
        self.assertGreater(sizes[0], 0)
        split.setSizes([700, 700])
        self.assertEqual(split.sizes()[0], 700)


class ComboWheelGuardTests(unittest.TestCase):
    def test_closed_combo_ignores_wheel(self) -> None:
        _app()
        host = QWidget()
        layout = QVBoxLayout(host)
        area = QScrollArea()
        combo = QComboBox()
        combo.addItems(["Off", "Every 15 frames"])
        combo.setCurrentIndex(0)
        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.addWidget(combo)
        area.setWidget(inner)
        layout.addWidget(area)
        guard = ComboWheelGuard()
        self.assertFalse(combo.view().isVisible())
        event = QWheelEvent(
            QPointF(4, 4),
            QPointF(4, 4),
            QPoint(0, 0),
            QPoint(0, 120),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.NoScrollPhase,
            False,
        )
        self.assertTrue(guard.eventFilter(combo, event))
        self.assertEqual(combo.currentIndex(), 0)


class StepBarTests(unittest.TestCase):
    def test_process_pages_show_the_five_steps(self) -> None:
        _app()
        from traffic_tracker.gui.main_window import PAGE_ALIGN, PAGE_HOME, PAGE_SETTINGS, MainWindow

        window = MainWindow()
        window.show()
        window.stack.setCurrentIndex(PAGE_ALIGN)
        self.assertTrue(window.step_host.isVisible())
        self.assertTrue(window.steps.isVisible())
        labels = [button.text() for _page, button in window.step_buttons]
        self.assertEqual(labels, ["1. Videos", "2. Routes", "3. Align", "4. Process", "5. Report"])
        for _page, button in window.step_buttons:
            self.assertTrue(button.isVisible())
        window.stack.setCurrentIndex(PAGE_HOME)
        self.assertFalse(window.step_host.isVisible())
        window.stack.setCurrentIndex(PAGE_SETTINGS)
        self.assertFalse(window.step_host.isVisible())
        window.close()


if __name__ == "__main__":
    unittest.main()
