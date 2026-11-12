"""Side-by-side panes that stack when the window is narrow.

When wide, the divider is a QSplitter handle so the user can drag the tool
column wider. The same handle stays when stacked.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QSizePolicy, QSplitter, QWidget


class ResponsiveSplit(QSplitter):
    """Tools beside the picture when wide. Stacked when narrower than `breakpoint`.

    `narrow_picture_first` puts the trailing pane (usually the picture) on top
    when stacked. Otherwise the leading pane stays first.

    `leading_max` is the starting width of the leading pane, not a hard cap —
    the user can drag past it.
    """

    def __init__(
        self,
        leading: QWidget,
        trailing: QWidget,
        *,
        breakpoint: int = 1100,
        narrow_picture_first: bool = False,
        leading_stretch: int = 0,
        trailing_stretch: int = 1,
        leading_max: int | None = None,
        spacing: int = 16,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(Qt.Orientation.Horizontal, parent)
        self._leading = leading
        self._trailing = trailing
        self._breakpoint = breakpoint
        self._picture_first = narrow_picture_first
        self._leading_stretch = leading_stretch
        self._trailing_stretch = trailing_stretch
        self._leading_max = leading_max
        self._narrow: bool | None = None
        self._sized = False
        self.setChildrenCollapsible(False)
        self.setHandleWidth(max(10, min(spacing, 12)))
        self.setOpaqueResize(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.addWidget(leading)
        self.addWidget(trailing)
        self._apply(breakpoint + 1)

    def resizeEvent(self, event) -> None:
        self._apply(self.width())
        super().resizeEvent(event)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._apply(max(self.width(), 1))
        if not self._sized:
            self._apply_sizes()
            self._sized = True

    def _apply(self, width: int) -> None:
        if width <= 0:
            return
        narrow = width < self._breakpoint
        if narrow == self._narrow and self.count() == 2:
            return
        self._narrow = narrow
        if narrow:
            self.setOrientation(Qt.Orientation.Vertical)
        else:
            self.setOrientation(Qt.Orientation.Horizontal)
        self._sync_order()
        self._apply_stretch()
        self._apply_sizes()

    def _sync_order(self) -> None:
        picture_on_top = bool(self._narrow and self._picture_first)
        first = self._trailing if picture_on_top else self._leading
        second = self._leading if picture_on_top else self._trailing
        if self.widget(0) is not first:
            self.insertWidget(0, first)
        if self.widget(1) is not second:
            self.insertWidget(1, second)

    def _apply_stretch(self) -> None:
        if self._narrow:
            if self._picture_first:
                self.setStretchFactor(0, max(self._trailing_stretch, 2))
                self.setStretchFactor(1, 1)
            else:
                self.setStretchFactor(0, max(self._leading_stretch, 1))
                self.setStretchFactor(1, max(self._trailing_stretch, 1))
            return
        self.setStretchFactor(0, self._leading_stretch)
        self.setStretchFactor(1, self._trailing_stretch)

    def _apply_sizes(self) -> None:
        total = self.width() if self.orientation() == Qt.Orientation.Horizontal else self.height()
        if total <= 0:
            return
        if self._narrow:
            picture = max(180, int(total * 0.58))
            tools = max(120, total - picture)
            if self._picture_first:
                self.setSizes([picture, tools])
            else:
                self.setSizes([tools, picture])
            return
        lead = self._leading_max or 420
        lead = min(lead, max(240, total // 2))
        self.setSizes([lead, max(200, total - lead)])
