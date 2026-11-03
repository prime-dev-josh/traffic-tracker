# UI toolkit license evaluation (PyQt6 vs PySide6)

This is a practical comparison for shipping Traffic Tracker. **It is not legal advice.** Have counsel review before you distribute a closed-source installer to clients.

## Current choice

The app uses **PyQt6** (`PyQt6.QtWidgets`). The GUI is a thin wrapper around the processing core, so a later swap is mostly import rewrites.

## Comparison

| | **PyQt6** | **PySide6** |
|--|-----------|-------------|
| API | Qt 6 widgets (what we use) | Official Qt for Python; nearly 1:1 (`from PySide6.QtWidgets import …`) |
| Open-source license | **GPL v3** | **LGPL v3** (also GPL options) |
| Closed-source .exe | Need a **Riverbank commercial** license, or license the whole app GPL | Allowed if you meet **LGPL** duties (dynamic link, notices, ability to relink the library) |
| Qt Company commercial | Separate Qt commercial if you need it | Same Qt commercial option |

PyQt6 is **not** LGPL. Shipping a proprietary desktop build with the default PyPI PyQt6 wheels is a GPL problem unless the product itself is GPL-compatible.

## Recommendation

| How you ship | Toolkit |
|--------------|---------|
| Internal use, or you are fine GPL-licensing the app | **Keep PyQt6** |
| Closed-source client installers | **Switch to PySide6** (dynamic link) **or** buy PyQt commercial |

No algorithm or TMC rewrite is required for a PySide6 swap. Touch:

- `traffic_tracker/gui/*.py` imports
- [`traffic_tracker/main.py`](../traffic_tracker/main.py) (`QApplication`)
- [`TrafficTracker.spec`](../TrafficTracker.spec) (`collect_all("PySide6")` instead of PyQt6)
- [`requirements.txt`](../requirements.txt)

Stay on one toolkit per build. Do not mix PyQt6 and PySide6 in the same venv.
