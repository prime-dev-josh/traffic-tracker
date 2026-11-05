"""Local app preferences (export folder, warnings, detector mode)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import json

from traffic_tracker.config import (
    DETECTOR_CONF,
    DETECTOR_CONF_MAX,
    DETECTOR_CONF_MIN,
    DETECTOR_IOU,
    DETECTOR_IOU_MAX,
    DETECTOR_IOU_MIN,
    DETECTOR_LINK_SPEED_MAX,
    DETECTOR_LINK_SPEED_MIN,
    DETECTOR_LINK_SPEED_PX_S,
    DETECTOR_PARKED_RADIUS,
    DETECTOR_PARKED_RADIUS_MAX,
    DETECTOR_PARKED_RADIUS_MIN,
    DETECTOR_PATH_JUMP,
    DETECTOR_PATH_JUMP_MAX,
    DETECTOR_PATH_JUMP_MIN,
    PRECISION_PRESETS,
    SETTINGS_FILE_NAME,
)


def _clamp(value: object, default: float, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


def clamp_detector_conf(value: object) -> float:
    return _clamp(value, DETECTOR_CONF, DETECTOR_CONF_MIN, DETECTOR_CONF_MAX)


def clamp_detector_iou(value: object) -> float:
    return _clamp(value, DETECTOR_IOU, DETECTOR_IOU_MIN, DETECTOR_IOU_MAX)


def clamp_parked_radius(value: object) -> float:
    return _clamp(value, DETECTOR_PARKED_RADIUS, DETECTOR_PARKED_RADIUS_MIN, DETECTOR_PARKED_RADIUS_MAX)


def clamp_link_speed(value: object) -> float:
    return _clamp(value, DETECTOR_LINK_SPEED_PX_S, DETECTOR_LINK_SPEED_MIN, DETECTOR_LINK_SPEED_MAX)


def clamp_path_jump(value: object) -> float:
    return _clamp(value, DETECTOR_PATH_JUMP, DETECTOR_PATH_JUMP_MIN, DETECTOR_PATH_JUMP_MAX)


@dataclass
class AppSettings:
    export_dir: str = ""
    warn_onedrive: bool = True
    detector_mode: str = "motion"
    detector_weights: str = ""
    detector_source: str = "site"
    theme: str = "dark"
    precision_preset: str = "balanced"
    detector_conf: float = DETECTOR_CONF
    detector_iou: float = DETECTOR_IOU
    parked_radius: float = DETECTOR_PARKED_RADIUS
    link_speed: float = DETECTOR_LINK_SPEED_PX_S
    path_jump: float = DETECTOR_PATH_JUMP


def apply_precision_preset(settings: AppSettings, name: str) -> AppSettings:
    """Copy a named preset onto the detection sliders. Unknown names become custom."""
    preset = PRECISION_PRESETS.get(name)
    if preset is None:
        return replace(settings, precision_preset="custom")
    return replace(
        settings,
        precision_preset=name,
        detector_conf=float(preset["detector_conf"]),
        detector_iou=float(preset["detector_iou"]),
        parked_radius=float(preset["parked_radius"]),
        link_speed=float(preset["link_speed"]),
        path_jump=float(preset["path_jump"]),
    )


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
    source = str(data.get("detector_source", "site"))
    preset = str(data.get("precision_preset", "balanced"))
    if preset not in ("strict", "balanced", "sensitive", "custom"):
        preset = "balanced"
    if "precision_preset" not in data:
        preset = "custom"
    return AppSettings(
        export_dir=str(data.get("export_dir") or (base or Path.cwd()) / "outputs"),
        warn_onedrive=bool(data.get("warn_onedrive", True)),
        detector_mode=mode if mode in ("motion", "trained") else "motion",
        detector_weights=str(data.get("detector_weights") or ""),
        detector_source=source if source in ("site", "visdrone", "custom") else "site",
        theme=theme if theme in ("dark", "light") else "dark",
        precision_preset=preset,
        detector_conf=clamp_detector_conf(data.get("detector_conf", DETECTOR_CONF)),
        detector_iou=clamp_detector_iou(data.get("detector_iou", DETECTOR_IOU)),
        parked_radius=clamp_parked_radius(data.get("parked_radius", DETECTOR_PARKED_RADIUS)),
        link_speed=clamp_link_speed(data.get("link_speed", DETECTOR_LINK_SPEED_PX_S)),
        path_jump=clamp_path_jump(data.get("path_jump", DETECTOR_PATH_JUMP)),
    )


def save_settings(settings: AppSettings, base: Path | None = None) -> Path:
    path = settings_path(base)
    path.write_text(json.dumps(asdict(settings), indent=2), encoding="utf-8")
    return path
