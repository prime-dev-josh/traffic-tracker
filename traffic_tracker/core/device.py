"""Pick a YOLO device and explain why GPU training is or is not available."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Thread
import subprocess

_cached_device: str | None = None


@dataclass(frozen=True)
class DeviceReport:
    device: str
    label: str
    warning: str | None = None


def reset_device_cache() -> None:
    global _cached_device
    _cached_device = None


def nvidia_gpu_names() -> list[str]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return []
    if completed.returncode != 0:
        return []
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def torch_cuda_build() -> bool:
    try:
        import torch

        return bool(getattr(torch.version, "cuda", None))
    except Exception:
        return False


def detect_device(timeout_s: float = 30.0) -> str:
    """Prefer CUDA, then Apple MPS, otherwise CPU.

    The probe runs in a short-lived thread so a stuck driver check
    cannot freeze the GUI. First CUDA init can take well over 4 seconds.
    """
    global _cached_device
    if _cached_device is not None:
        return _cached_device

    result = ["cpu"]

    def _probe() -> None:
        try:
            import torch

            if torch.cuda.is_available():
                result[0] = "cuda"
                return
            mps = getattr(torch.backends, "mps", None)
            if mps is not None and mps.is_available():
                result[0] = "mps"
        except Exception:
            pass

    thread = Thread(target=_probe, daemon=True)
    thread.start()
    thread.join(timeout_s)
    if thread.is_alive():
        return "cpu"
    _cached_device = result[0]
    return _cached_device


def describe_device(device: str) -> str:
    if device == "cuda":
        try:
            import torch

            name = torch.cuda.get_device_name(0)
            return f"CUDA ({name})"
        except Exception:
            return "CUDA"
    if device == "mps":
        return "MPS (Apple Silicon)"
    if device == "cpu":
        return "CPU"
    return device


def device_report(timeout_s: float = 30.0) -> DeviceReport:
    device = detect_device(timeout_s)
    gpus = nvidia_gpu_names()
    if device == "cuda":
        return DeviceReport(device, describe_device(device))
    if device == "mps":
        return DeviceReport(device, describe_device(device))

    names = ", ".join(gpus)
    if gpus and not torch_cuda_build():
        return DeviceReport(
            "cpu",
            f"CPU (PyTorch is CPU-only; {names} unused)",
            f"This app's PyTorch is the CPU-only build, so it cannot use {names}. "
            "Install the CUDA wheels (see README: pip install -r requirements-cuda.txt), "
            "then restart. Continue on CPU only if you want a much slower train.",
        )
    if gpus:
        return DeviceReport(
            "cpu",
            f"CPU ({names} present but CUDA did not initialize)",
            f"An NVIDIA GPU was found ({names}) but PyTorch CUDA is not ready. "
            "Restart the app. Continue on CPU only if you want a much slower train.",
        )
    return DeviceReport(
        "cpu",
        "CPU",
        "No GPU was found. Fine-tuning YOLO on CPU can take hours. Continue?",
    )
