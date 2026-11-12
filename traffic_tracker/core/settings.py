"""Local app preferences (export folder, warnings, detector mode)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
import json
import re

from traffic_tracker.config import (
    DETECTOR_CONF,
    DETECTOR_CONF_MAX,
    DETECTOR_CONF_MIN,
    DETECTOR_IMGSZ,
    DETECTOR_IMGSZ_CHOICES,
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
    DETECTOR_STRIDE,
    DETECTOR_STRIDE_MAX,
    EXPORT_SUBFOLDER_PATTERN,
    PRECISION_PRESETS,
    PREVIEW_EVERY,
    PREVIEW_EVERY_CHOICES,
    SETTINGS_FILE_NAME,
)

DETECTOR_SOURCES = ("visdrone",)


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


def clamp_detector_imgsz(value: object) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return DETECTOR_IMGSZ
    return min(DETECTOR_IMGSZ_CHOICES, key=lambda choice: abs(choice - number))


def clamp_detect_stride(value: object) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return DETECTOR_STRIDE
    return max(1, min(DETECTOR_STRIDE_MAX, number))


def clamp_preview_every(value: object) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return PREVIEW_EVERY
    return min(PREVIEW_EVERY_CHOICES, key=lambda choice: abs(choice - number))


_EXPORT_TOKEN_RE = re.compile(r"\{([a-z]+)\}")


def sanitize_export_part(value: object) -> str:
    token = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(value or "").strip())
    return token.strip("._") or "export"


def clamp_export_pattern(value: object) -> str:
    raw = str(value if value is not None else EXPORT_SUBFOLDER_PATTERN).strip().replace("\\", "/")
    parts: list[str] = []
    for part in raw.split("/"):
        piece = part.strip()
        if not piece or piece in {".", ".."}:
            continue
        if Path(piece).is_absolute() or ":" in piece:
            continue
        parts.append(piece[:80])
    return "/".join(parts)[:160]


def resolve_export_folder(
    base: Path,
    pattern: str,
    *,
    video_stem: str,
    when: datetime | None = None,
) -> Path:
    """Build the per-run folder under the export directory from a token pattern."""
    stamp = when or datetime.now()
    values = {
        "video": sanitize_export_part(video_stem),
        "date": stamp.strftime("%Y-%m-%d"),
        "time": stamp.strftime("%H%M%S"),
        "datetime": stamp.strftime("%Y-%m-%d_%H%M%S"),
    }
    filled = _EXPORT_TOKEN_RE.sub(
        lambda match: values.get(match.group(1), ""),
        clamp_export_pattern(pattern),
    )
    parts = [sanitize_export_part(part) for part in filled.split("/") if part.strip()]
    folder = Path(base)
    for part in parts:
        folder = folder / part
    return folder


def export_workbook_path(
    base: Path,
    pattern: str,
    video_stem: str,
    *,
    when: datetime | None = None,
) -> Path:
    stem = sanitize_export_part(video_stem)
    return resolve_export_folder(base, pattern, video_stem=video_stem, when=when) / f"{stem}_tmc.xlsx"


@dataclass
class AppSettings:
    export_dir: str = ""
    export_pattern: str = EXPORT_SUBFOLDER_PATTERN
    warn_onedrive: bool = True
    detector_mode: str = "motion"
    detector_weights: str = ""
    detector_source: str = "visdrone"
    theme: str = "light"
    precision_preset: str = "balanced"
    detector_conf: float = DETECTOR_CONF
    detector_iou: float = DETECTOR_IOU
    parked_radius: float = DETECTOR_PARKED_RADIUS
    link_speed: float = DETECTOR_LINK_SPEED_PX_S
    path_jump: float = DETECTOR_PATH_JUMP
    detector_imgsz: int = DETECTOR_IMGSZ
    detect_stride: int = DETECTOR_STRIDE
    preview_every: int = PREVIEW_EVERY
    infer_trips_from_traces: bool = True


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
    theme = str(data.get("theme", "light"))
    source = "visdrone"
    preset = str(data.get("precision_preset", "balanced"))
    if preset not in ("strict", "balanced", "sensitive", "custom"):
        preset = "balanced"
    if "precision_preset" not in data:
        preset = "custom"
    if "infer_trips_from_traces" in data:
        infer_trips = bool(data.get("infer_trips_from_traces"))
    else:
        infer_trips = mode != "trained"
    return AppSettings(
        export_dir=str(data.get("export_dir") or (base or Path.cwd()) / "outputs"),
        export_pattern=clamp_export_pattern(data.get("export_pattern", EXPORT_SUBFOLDER_PATTERN)),
        warn_onedrive=bool(data.get("warn_onedrive", True)),
        detector_mode=mode if mode in ("motion", "trained") else "motion",
        detector_weights=str(data.get("detector_weights") or ""),
        detector_source=source,
        theme=theme if theme in ("dark", "light") else "light",
        precision_preset=preset,
        detector_conf=clamp_detector_conf(data.get("detector_conf", DETECTOR_CONF)),
        detector_iou=clamp_detector_iou(data.get("detector_iou", DETECTOR_IOU)),
        parked_radius=clamp_parked_radius(data.get("parked_radius", DETECTOR_PARKED_RADIUS)),
        link_speed=clamp_link_speed(data.get("link_speed", DETECTOR_LINK_SPEED_PX_S)),
        path_jump=clamp_path_jump(data.get("path_jump", DETECTOR_PATH_JUMP)),
        detector_imgsz=clamp_detector_imgsz(data.get("detector_imgsz", DETECTOR_IMGSZ)),
        detect_stride=clamp_detect_stride(data.get("detect_stride", DETECTOR_STRIDE)),
        preview_every=clamp_preview_every(data.get("preview_every", PREVIEW_EVERY)),
        infer_trips_from_traces=infer_trips,
    )


def save_settings(settings: AppSettings, base: Path | None = None) -> Path:
    path = settings_path(base)
    path.write_text(json.dumps(asdict(settings), indent=2), encoding="utf-8")
    return path
