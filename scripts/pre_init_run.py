"""Preflight checks before launching Traffic Tracker or processing with YOLO."""

from __future__ import annotations

import argparse
import re
import sys
import sysconfig
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAINED_WEIGHTS = ROOT / "training" / "weights" / "best.pt"
REQUIREMENTS_CUDA = ROOT / "requirements-cuda.txt"


class CheckFailure(Exception):
    pass


def _site_packages() -> Path:
    return Path(sysconfig.get_paths()["purelib"])


def _parse_pinned_cuda_packages() -> dict[str, str]:
    if not REQUIREMENTS_CUDA.is_file():
        return {}
    pins: dict[str, str] = {}
    for line in REQUIREMENTS_CUDA.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^(torch|torchvision)==(.+)$", line)
        if match:
            pins[match.group(1)] = match.group(2)
    return pins


def check_python_version() -> None:
    if sys.version_info[:2] != (3, 11):
        raise CheckFailure(
            f"Python 3.11 required (got {sys.version_info.major}.{sys.version_info.minor})."
        )


def check_duplicate_torch_installs(site: Path) -> None:
    torch_infos = sorted(site.glob("torch-*.dist-info"))
    vision_infos = sorted(site.glob("torchvision-*.dist-info"))
    if len(torch_infos) > 1:
        names = ", ".join(p.name.replace(".dist-info", "") for p in torch_infos)
        raise CheckFailure(
            "Multiple torch wheels are installed (mixed PyTorch). "
            f"Found: {names}. This causes import errors such as missing is_fake_tensor."
        )
    if len(vision_infos) > 1:
        names = ", ".join(p.name.replace(".dist-info", "") for p in vision_infos)
        raise CheckFailure(f"Multiple torchvision wheels are installed. Found: {names}.")


def check_torch_import(pins: dict[str, str]) -> str:
    try:
        import torch
    except Exception as exc:
        raise CheckFailure(f"Cannot import torch: {exc}") from exc

    version = getattr(torch, "__version__", "unknown")
    cuda = getattr(torch.version, "cuda", None)
    label = f"torch {version}" + (f", CUDA build {cuda}" if cuda else ", CPU-only build")
    expected = pins.get("torch")
    if expected and version != expected:
        raise CheckFailure(
            f"torch version is {version}, but requirements-cuda.txt pins {expected}. "
            "Reinstall with: .\\run.ps1"
        )
    if not cuda:
        raise CheckFailure(
            f"{label}. Windows needs the CUDA wheels from requirements-cuda.txt, not PyPI CPU torch."
        )
    return label


def check_torchvision_import(pins: dict[str, str]) -> str:
    try:
        import torchvision
    except Exception as exc:
        raise CheckFailure(f"Cannot import torchvision: {exc}") from exc
    version = getattr(torchvision, "__version__", "unknown")
    expected = pins.get("torchvision")
    if expected and version != expected:
        raise CheckFailure(
            f"torchvision version is {version}, but requirements-cuda.txt pins {expected}."
        )
    return f"torchvision {version}"


def check_ultralytics() -> str:
    try:
        import ultralytics
    except Exception as exc:
        raise CheckFailure(f"Cannot import ultralytics: {exc}") from exc
    return f"ultralytics {ultralytics.__version__}"


def check_pyqt6() -> None:
    try:
        from PyQt6.QtCore import QT_VERSION_STR  # noqa: F401
    except Exception as exc:
        raise CheckFailure(f"Cannot import PyQt6 (GUI will not start): {exc}") from exc


def check_yolo_weights(weights: Path) -> None:
    if not weights.is_file():
        return
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise CheckFailure(f"Cannot import ultralytics.YOLO: {exc}") from exc
    try:
        YOLO(str(weights))
    except Exception as exc:
        raise CheckFailure(f"Cannot load detector weights {weights}: {exc}") from exc


def run_checks(*, load_weights: bool, weights: Path) -> int:
    pins = _parse_pinned_cuda_packages()
    site = _site_packages()
    lines: list[str] = []

    try:
        check_python_version()
        lines.append(f"OK  Python {sys.version_info.major}.{sys.version_info.minor}")

        check_duplicate_torch_installs(site)
        lines.append("OK  Single torch / torchvision install")

        lines.append("OK  " + check_torch_import(pins))
        lines.append("OK  " + check_torchvision_import(pins))
        lines.append("OK  " + check_ultralytics())
        check_pyqt6()
        lines.append("OK  PyQt6")

        if load_weights and weights.is_file():
            check_yolo_weights(weights)
            lines.append(f"OK  Loaded {weights.name}")
        elif load_weights:
            lines.append(f"SKIP  No weights at {weights} (motion-only is fine)")
        else:
            lines.append("SKIP  YOLO weight load (--quick)")

    except CheckFailure as exc:
        print("Pre-init check FAILED", file=sys.stderr)
        print(f"  {exc}", file=sys.stderr)
        print("", file=sys.stderr)
        print("Fix: close the app, delete the venv, reinstall, then re-run this script:", file=sys.stderr)
        print("  Remove-Item -Recurse -Force .venv", file=sys.stderr)
        print("  .\\run.ps1", file=sys.stderr)
        print("  .\\pre-init-run.ps1", file=sys.stderr)
        if "OneDrive" in str(site):
            print(
                "  If the repo is on OneDrive, use %LOCALAPPDATA%\\traffic-tracker\\venv (see README).",
                file=sys.stderr,
            )
        return 1

    print("Pre-init check passed")
    for line in lines:
        print(f"  {line}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Traffic Tracker pre-init environment checks.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip loading training/weights/best.pt (faster; still validates torch and ultralytics).",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=TRAINED_WEIGHTS,
        help="Path to a .pt file to load when not using --quick.",
    )
    args = parser.parse_args(argv)
    load_weights = not args.quick
    weights = args.weights if args.weights.is_absolute() else (ROOT / args.weights)
    return run_checks(load_weights=load_weights, weights=weights)


if __name__ == "__main__":
    raise SystemExit(main())
