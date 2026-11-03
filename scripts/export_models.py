"""Export YOLO11 weights for Ultralytics AutoBackend.

Default: yolo11n.onnx at imgsz=640.
Optional: --openvino for Intel CPU/iGPU field machines.

Usage:
    python scripts/export_models.py
    python scripts/export_models.py --openvino
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from traffic_tracker.config import EXPORT_IMGSZ, OPENVINO_DIR_NAME, YOLO_ONNX_NAME, YOLO_PT_NAME


def export(*, openvino: bool = False) -> None:
    from ultralytics import YOLO

    pt_path = ROOT / YOLO_PT_NAME
    print(f"Loading {YOLO_PT_NAME} (downloads on first use)…")
    model = YOLO(str(pt_path) if pt_path.is_file() else YOLO_PT_NAME)

    onnx_path = ROOT / YOLO_ONNX_NAME
    print(f"Exporting ONNX imgsz={EXPORT_IMGSZ} → {onnx_path}")
    model.export(format="onnx", imgsz=EXPORT_IMGSZ, dynamic=False)
    produced = Path.cwd() / YOLO_ONNX_NAME
    if produced.is_file() and produced.resolve() != onnx_path.resolve():
        onnx_path.write_bytes(produced.read_bytes())
    if not onnx_path.is_file() and produced.is_file():
        produced.replace(onnx_path)
    print(f"ONNX ready: {onnx_path if onnx_path.is_file() else produced}")

    if openvino:
        print(f"Exporting OpenVINO → {ROOT / OPENVINO_DIR_NAME}")
        model.export(format="openvino", imgsz=EXPORT_IMGSZ)
        print("OpenVINO export finished.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export YOLO11 for Traffic Tracker")
    parser.add_argument(
        "--openvino",
        action="store_true",
        help="Also export an OpenVINO folder for Intel CPU/iGPU laptops",
    )
    args = parser.parse_args()
    export(openvino=args.openvino)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
