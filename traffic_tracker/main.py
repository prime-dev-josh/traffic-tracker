"""Application entry point."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _qt_plugin_candidates() -> list[Path]:
    import PyQt6

    candidates: list[Path] = []
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            candidates.append(Path(meipass) / "PyQt6" / "Qt6" / "plugins")
        exe_dir = Path(sys.executable).resolve().parent
        candidates.append(exe_dir / "_internal" / "PyQt6" / "Qt6" / "plugins")
        candidates.append(exe_dir / "PyQt6" / "Qt6" / "plugins")
    candidates.append(Path(PyQt6.__file__).resolve().parent / "Qt6" / "plugins")
    return candidates


def _prepare_qt() -> None:
    """Point Qt at bundled plugins (venv, PyInstaller one-folder, or one-file)."""
    for plugin_root in _qt_plugin_candidates():
        platforms = plugin_root / "platforms"
        if platforms.exists():
            os.environ["QT_PLUGIN_PATH"] = str(plugin_root)
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(platforms)
            break
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
