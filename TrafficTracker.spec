# -*- mode: python ; coding: utf-8 -*-
# Build: .\build_exe.ps1   or   pyinstaller TrafficTracker.spec
# Output: dist\Traffic Tracker\Traffic Tracker.exe

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

block_cipher = None
project_root = Path(SPECPATH)

datas: list = []
binaries: list = []
hiddenimports: list = collect_submodules("traffic_tracker")
datas.append(
    (
        str(project_root / "traffic_tracker" / "core" / "bytetrack.yaml"),
        "traffic_tracker/core",
    )
)

for pkg in (
    "PyQt6",
    "supervision",
    "ultralytics",
    "torch",
    "torchvision",
    "cv2",
    "lap",
    "yaml",
    "openpyxl",
    "pandas",
):
    try:
        tmp = collect_all(pkg)
        datas += tmp[0]
        binaries += tmp[1]
        hiddenimports += tmp[2]
    except Exception as exc:
        print(f"Warning: collect_all({pkg!r}) failed: {exc}", file=sys.stderr)

# Optional: place weights in the project root before building to ship offline.
for name in ("yolo11n.pt", "yolo11n.onnx"):
    bundled = project_root / name
    if bundled.is_file():
        datas.append((str(bundled), "."))
openvino_dir = project_root / "yolo11n_openvino_model"
if openvino_dir.is_dir():
    datas.append((str(openvino_dir), "yolo11n_openvino_model"))

a = Analysis(
    [str(project_root / "run.py")],
    pathex=[str(project_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter"],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Traffic Tracker",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Traffic Tracker",
)
