"""Pick a YOLO inference device without forcing a GPU install."""

from __future__ import annotations

from threading import Thread

_cached_device: str | None = None


def detect_device(timeout_s: float = 4.0) -> str:
    """Return 'cuda' when an NVIDIA GPU is usable, otherwise 'cpu'.

    The CUDA probe runs in a short-lived thread so a stuck driver check
    cannot freeze the GUI. macOS MPS can be added later.
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
    return "CPU"
