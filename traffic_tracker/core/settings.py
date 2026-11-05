"""Local app preferences (export folder, warnings, detector mode)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

from traffic_tracker.config import (
    DETECTOR_CONF,
    DETECTOR_CONF_MAX,
    DETECTOR_CONF_MIN,
    SETTINGS_FILE_NAME,
)


def clamp_detector_conf(value: object) -> float:
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return DETECTOR_CONF
    return max(DETECTOR_CONF_MIN, min(DETECTOR_CONF_MAX, conf))


@dataclass
class AppSettings:
    export_dir: str = ""
    warn_onedrive: bool = True
    detector_mode: str = "motion"
    detector_weights: str = ""
    theme: str = "dark"
    detector_conf: float = DETECTOR_CONF


def settings_path(base: Path | None = None) -> Path:
    return (base or Path.cwd()) / SETTINGS_FILE_NAME


def load_settings(base: Path | None = None) -> AppSettings:
    path = settings_path(base)
    if not path.is_file():
        return AppSettings(export_dir=str((base or Path.cwd()) / "outputs"))
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return AppSettings(export_dir=str((base or Path.cwd()) / "outputs"))
    mode = str(data.get("detector_mode", "motion"))
    theme = str(data.get("theme", "dark"))
    return AppSettings(
        export_dir=str(data.get("export_dir") or (base or Path.cwd()) / "outputs"),
        warn_onedrive=bool(data.get("warn_onedrive", True)),
        detector_mode=mode if mode in ("motion", "trained") else "motion",
        detector_weights=str(data.get("detector_weights") or ""),
        theme=theme if theme in ("dark", "light") else "dark",
        detector_conf=clamp_detector_conf(data.get("detector_conf", DETECTOR_CONF)),
    )


def save_settings(settings: AppSettings, base: Path | None = None) -> Path:
    path = settings_path(base)
    path.write_text(json.dumps(asdict(settings), indent=2), encoding="utf-8")
    return path
