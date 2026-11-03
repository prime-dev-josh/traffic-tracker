"""Resolve YOLO11 weights for Ultralytics AutoBackend (pt / onnx / OpenVINO).

Detection and ByteTrack stay in Ultralytics. This module only chooses which
exported file to load.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from traffic_tracker.config import (
    BACKEND_ENV,
    MODEL_ENV,
    OPENVINO_DIR_NAME,
    YOLO_ONNX_NAME,
    YOLO_PT_NAME,
)

VALID_BACKENDS = ("auto", "pt", "onnx", "openvino")


@dataclass(frozen=True)
class ModelSource:
    path: Path
    backend: str
    label: str


def candidate_roots() -> list[Path]:
    roots: list[Path] = []
    if getattr(sys, "frozen", False):
        roots.append(Path(sys.executable).resolve().parent)
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            roots.append(Path(meipass))
    roots.append(Path.cwd())
    roots.append(Path(__file__).resolve().parents[2])
    seen: set[Path] = set()
    unique: list[Path] = []
    for root in roots:
        resolved = root.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def _first_existing(name: str, *, directory: bool = False) -> Path | None:
    for root in candidate_roots():
        path = root / name
        if directory and path.is_dir():
            return path
        if not directory and path.is_file():
            return path
    return None


def resolve_backend(explicit: str | None = None) -> str:
    raw = (explicit or os.environ.get(BACKEND_ENV) or "auto").strip().lower()
    if raw not in VALID_BACKENDS:
        raise ValueError(f"Unknown backend {raw!r}. Use one of: {', '.join(VALID_BACKENDS)}")
    return raw


def resolve_model_source(backend: str | None = None) -> ModelSource:
    """Pick weights: explicit path, then onnx, OpenVINO folder, else .pt."""
    override = os.environ.get(MODEL_ENV, "").strip()
    if override:
        path = Path(override)
        kind = _backend_from_path(path)
        return ModelSource(path=path, backend=kind, label=f"{kind} ({path.name})")

    chosen = resolve_backend(backend)
    onnx_path = _first_existing(YOLO_ONNX_NAME)
    openvino_path = _first_existing(OPENVINO_DIR_NAME, directory=True)
    pt_path = _first_existing(YOLO_PT_NAME)

    if chosen == "onnx":
        if onnx_path is None:
            raise FileNotFoundError(
                f"{YOLO_ONNX_NAME} not found. Run: python scripts/export_models.py"
            )
        return ModelSource(path=onnx_path, backend="onnx", label=f"onnx ({onnx_path.name})")
    if chosen == "openvino":
        if openvino_path is None:
            raise FileNotFoundError(
                f"{OPENVINO_DIR_NAME} not found. Run: python scripts/export_models.py --openvino"
            )
        return ModelSource(path=openvino_path, backend="openvino", label=f"openvino ({openvino_path.name})")
    if chosen == "pt":
        return ModelSource(
            path=pt_path or Path(YOLO_PT_NAME),
            backend="pt",
            label=f"pt ({YOLO_PT_NAME})",
        )

    if onnx_path is not None:
        return ModelSource(path=onnx_path, backend="onnx", label=f"onnx ({onnx_path.name})")
    if openvino_path is not None:
        return ModelSource(
            path=openvino_path,
            backend="openvino",
            label=f"openvino ({openvino_path.name})",
        )
    return ModelSource(
        path=pt_path or Path(YOLO_PT_NAME),
        backend="pt",
        label=f"pt ({YOLO_PT_NAME})",
    )


def _backend_from_path(path: Path) -> str:
    if path.is_dir() or path.name.endswith("_openvino_model"):
        return "openvino"
    if path.suffix.lower() == ".onnx":
        return "onnx"
    return "pt"
