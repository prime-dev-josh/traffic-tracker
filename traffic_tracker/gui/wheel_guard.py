"""Stop a closed dropdown from changing when the page is scrolled."""

from __future__ import annotations

from PyQt6.QtCore import QEvent, QObject
from PyQt6.QtWidgets import QApplication, QComboBox, QScrollArea


class ComboWheelGuard(QObject):
    """Ignore the wheel on a closed combo and let the surrounding page scroll."""

    def eventFilter(self, watched: QObject, event) -> bool:
        if event.type() != QEvent.Type.Wheel or not isinstance(watched, QComboBox):
            return False
        popup = watched.view()
        if popup is not None and popup.isVisible():
            return False
        parent = watched.parentWidget()
        while parent is not None and not isinstance(parent, QScrollArea):
            parent = parent.parentWidget()
        if isinstance(parent, QScrollArea):
            viewport = parent.viewport()
            if viewport is not None:
                QApplication.sendEvent(viewport, event)
        return True
