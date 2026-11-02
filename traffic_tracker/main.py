"""Application entry point."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _prepare_qt() -> None:
    """Point Qt at its bundled plugins so the Windows platform loads from a venv."""
    import PyQt6

    plugin_root = Path(PyQt6.__file__).resolve().parent / "Qt6" / "plugins"
    platforms = plugin_root / "platforms"
    if platforms.exists():
        os.environ["QT_PLUGIN_PATH"] = str(plugin_root)
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(platforms)
    if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        os.environ.pop("QT_QPA_PLATFORM", None)


def main() -> int:
    _prepare_qt()
    from PyQt6.QtWidgets import QApplication

    from traffic_tracker.config import APP_NAME
    from traffic_tracker.gui.main_window import MainWindow
    from traffic_tracker.gui.style import STYLESHEET

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setStyleSheet(STYLESHEET)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
