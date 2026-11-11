"""Labeled training frames: size prior plus a YOLO dataset for site fine-tunes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import json
import random
import shutil
import tempfile
import zipfile

import cv2
import numpy as np

from traffic_tracker.config import (
    DRONE_VEHICLES_WEIGHTS_REL,
    SIZE_PRIOR_NAME,
    TRAIN_DATASET_REL,
    TRAIN_MIN_BOXES,
    TRAIN_MIN_FRAMES,
    TRAIN_SPLIT_SEED,
    TRAIN_VAL_FRACTION,
    SITE_WEIGHTS_REL,
    TRAINED_WEIGHTS_REL,
    VISDRONE_WEIGHTS_REL,
    TRAINING_DIR_NAME,
    VEHICLE_REF_MAX_FACTOR,
    VEHICLE_REF_MIN_FACTOR,
    drone_vehicle_class_id,
    drone_vehicles_data_yaml,
)
from traffic_tracker.core.geometry import Box, area_limits_from_areas


@dataclass
class LabeledFrame:
    frame_index: int
    image_name: str
    width: int
    height: int
    boxes: list[Box] = field(default_factory=list)
    time_s: float = 0.0


@dataclass
class TrainingSession:
    video_path: str
    video_stem: str
    frames: list[LabeledFrame] = field(default_factory=list)
    saved_at: str = ""

    def box_count(self) -> int:
        return sum(len(frame.boxes) for frame in self.frames)


def training_root(base: Path | None = None) -> Path:
    return (base or Path.cwd()) / TRAINING_DIR_NAME


def session_dir(video_stem: str, base: Path | None = None) -> Path:
    return training_root(base) / "sessions" / video_stem


def trained_weights_path(base: Path | None = None) -> Path:
    return training_root(base) / TRAINED_WEIGHTS_REL


def visdrone_weights_path(base: Path | None = None) -> Path:
    return training_root(base) / VISDRONE_WEIGHTS_REL


def drone_vehicles_weights_path(base: Path | None = None) -> Path:
    return training_root(base) / DRONE_VEHICLES_WEIGHTS_REL


def site_weights_path(base: Path | None = None) -> Path:
    return training_root(base) / SITE_WEIGHTS_REL


def dataset_dir(base: Path | None = None) -> Path:
    return training_root(base) / TRAIN_DATASET_REL


def has_trained_weights(base: Path | None = None) -> bool:
    return trained_weights_path(base).is_file()


def list_sessions(base: Path | None = None) -> list[TrainingSession]:
    root = training_root(base) / "sessions"
    if not root.is_dir():
        return []
    sessions = []
    for folder in sorted(root.iterdir()):
        if folder.is_dir() and (folder / "session.json").is_file():
            try:
                sessions.append(load_session(folder.name, base))
            except (OSError, ValueError, KeyError, TypeError):
                continue
    return sessions


def load_session(video_stem: str, base: Path | None = None) -> TrainingSession:
    path = session_dir(video_stem, base) / "session.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    frames = [
        LabeledFrame(
            frame_index=int(item["frame_index"]),
            image_name=str(item["image_name"]),
            width=int(item["width"]),
            height=int(item["height"]),
            boxes=[_box_from_payload(box).normalized() for box in item.get("boxes", [])],
            time_s=float(item.get("time_s", 0.0)),
        )
        for item in data.get("frames", [])
    ]
    return TrainingSession(
        video_path=str(data.get("video_path", "")),
        video_stem=str(data.get("video_stem", video_stem)),
        frames=frames,
        saved_at=str(data.get("saved_at", "")),
    )


def save_session(
    session: TrainingSession,
    frames_bgr: dict[int, np.ndarray] | None = None,
    base: Path | None = None,
) -> Path:
    folder = session_dir(session.video_stem, base)
    images = folder / "images"
    labels = folder / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    session.saved_at = datetime.now(timezone.utc).isoformat()

    for labeled in session.frames:
        if frames_bgr and labeled.frame_index in frames_bgr:
            image_path = images / labeled.image_name
            cv2.imwrite(str(image_path), frames_bgr[labeled.frame_index])
        _write_yolo_label(labels / f"{Path(labeled.image_name).stem}.txt", labeled)

    payload = {
        "video_path": session.video_path,
        "video_stem": session.video_stem,
        "saved_at": session.saved_at,
        "frames": [
            {
                "frame_index": frame.frame_index,
                "image_name": frame.image_name,
                "width": frame.width,
                "height": frame.height,
                "time_s": frame.time_s,
                "boxes": [asdict(box.normalized()) for box in frame.boxes],
            }
            for frame in session.frames
        ],
    }
    (folder / "session.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    rebuild_size_prior(base)
    return folder


def delete_session(video_stem: str, base: Path | None = None) -> None:
    folder = session_dir(video_stem, base)
    if folder.exists():
        shutil.rmtree(folder)
    rebuild_size_prior(base)


def collect_box_areas(base: Path | None = None) -> list[float]:
    areas: list[float] = []
    for session in list_sessions(base):
        for frame in session.frames:
            areas.extend(box.area() for box in frame.boxes if box.area() >= 1)
    return areas


def load_size_prior(base: Path | None = None) -> dict | None:
    path = training_root(base) / SIZE_PRIOR_NAME
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def rebuild_size_prior(base: Path | None = None) -> dict | None:
    areas = collect_box_areas(base)
    limits = area_limits_from_areas(areas)
    root = training_root(base)
    root.mkdir(parents=True, exist_ok=True)
    path = root / SIZE_PRIOR_NAME
    if not areas or limits is None:
        if path.exists():
            path.unlink()
        return None
    payload = {
        "areas": areas,
        "min_area": limits[0],
        "max_area": limits[1],
        "min_factor": VEHICLE_REF_MIN_FACTOR,
        "max_factor": VEHICLE_REF_MAX_FACTOR,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def prior_areas(base: Path | None = None) -> list[float]:
    data = load_size_prior(base)
    if not data:
        return []
    raw = data.get("areas") or []
    return [float(area) for area in raw if float(area) >= 1]


def training_inventory(base: Path | None = None) -> dict:
    sessions = list_sessions(base)
    return {
        "root": training_root(base),
        "sessions": sessions,
        "session_count": len(sessions),
        "frame_count": sum(len(session.frames) for session in sessions),
        "box_count": sum(session.box_count() for session in sessions),
        "has_prior": (training_root(base) / SIZE_PRIOR_NAME).is_file(),
        "has_weights": has_trained_weights(base),
    }


def collect_dataset_items(base: Path | None = None) -> list[dict]:
    """Frames that have an image on disk, with YOLO label text ready to copy."""
    items: list[dict] = []
    for session in list_sessions(base):
        folder = session_dir(session.video_stem, base)
        images = folder / "images"
        labels = folder / "labels"
        for frame in session.frames:
            image_path = images / frame.image_name
            if not image_path.is_file():
                continue
            stem = Path(frame.image_name).stem
            label_path = labels / f"{stem}.txt"
            label_text = label_path.read_text(encoding="utf-8") if label_path.is_file() else _yolo_label_text(frame)
            items.append(
                {
                    "image_path": image_path,
                    "label_text": label_text,
                    "dest_stem": f"{session.video_stem}__{stem}",
                    "box_count": len(frame.boxes),
                }
            )
    return items


def build_yolo_dataset(base: Path | None = None) -> dict:
    """Flatten sessions into training/dataset with an 80/20 frame split."""
    items = collect_dataset_items(base)
    frame_count = len(items)
    box_count = sum(int(item["box_count"]) for item in items)
    if frame_count < TRAIN_MIN_FRAMES or box_count < TRAIN_MIN_BOXES:
        raise ValueError(
            f"Need at least {TRAIN_MIN_FRAMES} labeled frames and {TRAIN_MIN_BOXES} boxes "
            f"to train. Found {frame_count} frame(s) and {box_count} box(es). "
            "Box every vehicle on more frames, then try again."
        )

    train_items, val_items = _split_items(items)
    root = dataset_dir(base)
    # #region agent log
    def _dbg(message: str, data: dict, hid: str) -> None:
        import json, time
        try:
            with open(r"c:\Users\Josh.Marrero\OneDrive - PRIME AE Group, Inc\Desktop\Project Process AI Work\traffic-tracker\.cursor\debug-da0e48.log", "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"sessionId": "da0e48", "timestamp": int(time.time() * 1000), "location": "training.py:build_yolo_dataset", "message": message, "hypothesisId": hid, "data": data}) + "\n")
        except Exception:
            pass
    train_dir = root / "images" / "train"
    _dbg("before rebuild", {
        "root": str(root),
        "root_exists": root.exists(),
        "train_exists": train_dir.exists(),
        "onedrive": "OneDrive" in str(root),
        "train_items": len(train_items),
        "val_items": len(val_items),
    }, "A")
    # #endregion
    if root.exists():
        try:
            shutil.rmtree(root)
            # #region agent log
            _dbg("rmtree ok", {"root": str(root), "still_exists": root.exists()}, "A")
            # #endregion
        except Exception as exc:
            # #region agent log
            _dbg("rmtree failed", {"root": str(root), "error": f"{type(exc).__name__}: {exc}", "winerror": getattr(exc, "winerror", None)}, "A")
            # #endregion
            raise
    for split, split_items in (("train", train_items), ("val", val_items)):
        images = root / "images" / split
        labels = root / "labels" / split
        try:
            images.mkdir(parents=True, exist_ok=True)
            labels.mkdir(parents=True, exist_ok=True)
            # #region agent log
            _dbg("mkdir ok", {"images": str(images), "labels": str(labels)}, "B")
            # #endregion
        except Exception as exc:
            # #region agent log
            _dbg("mkdir failed", {"images": str(images), "error": f"{type(exc).__name__}: {exc}", "winerror": getattr(exc, "winerror", None)}, "B")
            # #endregion
            raise
        for item in split_items:
            dest_image = images / f"{item['dest_stem']}{item['image_path'].suffix}"
            dest_label = labels / f"{item['dest_stem']}.txt"
            shutil.copy2(item["image_path"], dest_image)
            dest_label.write_text(item["label_text"], encoding="utf-8")

    yaml_path = root / "data.yaml"
    yaml_path.write_text(drone_vehicles_data_yaml(root), encoding="utf-8")
    return {
        "root": root,
        "yaml_path": yaml_path,
        "frame_count": frame_count,
        "box_count": box_count,
        "train_count": len(train_items),
        "val_count": len(val_items),
    }


def _split_items(items: list[dict]) -> tuple[list[dict], list[dict]]:
    shuffled = list(items)
    random.Random(TRAIN_SPLIT_SEED).shuffle(shuffled)
    if len(shuffled) < 2:
        return shuffled, []
    n_val = max(1, int(round(len(shuffled) * TRAIN_VAL_FRACTION)))
    if n_val >= len(shuffled):
        n_val = len(shuffled) - 1
    return shuffled[n_val:], shuffled[:n_val]


def export_training_archive(dest: Path, base: Path | None = None) -> Path:
    root = training_root(base)
    if not root.exists() or not any(root.rglob("*")):
        raise FileNotFoundError("No training data to export.")
    dest = Path(dest)
    if dest.suffix.lower() != ".zip":
        dest = dest.with_suffix(".zip")
    dest.parent.mkdir(parents=True, exist_ok=True)
    parent = root.parent
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(parent).as_posix())
    return dest


def import_training_archive(src: Path, base: Path | None = None, *, replace: bool = False) -> dict:
    src = Path(src)
    if not src.exists():
        raise FileNotFoundError(f"Nothing to import: {src}")
    dest_root = training_root(base)
    dest_sessions = dest_root / "sessions"
    dest_sessions.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        extracted = Path(tmp) / "incoming"
        extracted.mkdir()
        if src.is_file() and src.suffix.lower() == ".zip":
            _safe_extract_zip(src, extracted)
        elif src.is_dir():
            shutil.copytree(src, extracted / src.name, dirs_exist_ok=True)
        else:
            raise ValueError("Import a training folder or a .zip archive.")
        source_root = _find_training_root(extracted)
        incoming = _session_folders(source_root)
        if not incoming:
            raise ValueError("No training sessions found in that file.")

        imported = 0
        replaced = 0
        skipped = 0
        for folder in incoming:
            target = dest_sessions / folder.name
            if target.exists() and not replace:
                skipped += 1
                continue
            if target.exists():
                shutil.rmtree(target)
                replaced += 1
            else:
                imported += 1
            shutil.copytree(folder, target)

        incoming_prior = source_root / SIZE_PRIOR_NAME
        if incoming_prior.is_file() and (replace or not (dest_root / SIZE_PRIOR_NAME).is_file()):
            shutil.copy2(incoming_prior, dest_root / SIZE_PRIOR_NAME)
        incoming_weights = source_root / TRAINED_WEIGHTS_REL
        if incoming_weights.is_file() and (replace or not trained_weights_path(base).is_file()):
            trained_weights_path(base).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(incoming_weights, trained_weights_path(base))

    rebuild_size_prior(base)
    return {"imported": imported, "replaced": replaced, "skipped": skipped}


def _safe_extract_zip(archive_path: Path, dest: Path) -> None:
    dest = dest.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            name = Path(info.filename)
            if name.is_absolute() or ".." in name.parts:
                raise ValueError(f"Unsafe path in archive: {info.filename}")
            target = (dest / name).resolve()
            if dest not in target.parents and target != dest:
                raise ValueError(f"Unsafe path in archive: {info.filename}")
        archive.extractall(dest)


def _find_training_root(path: Path) -> Path:
    if (path / "sessions").is_dir():
        return path
    matches = [candidate for candidate in path.rglob("sessions") if candidate.is_dir()]
    if matches:
        return matches[0].parent
    return path


def _session_folders(root: Path) -> list[Path]:
    sessions = root / "sessions"
    search = sessions if sessions.is_dir() else root
    folders = []
    for folder in sorted(search.iterdir()) if search.is_dir() else []:
        if folder.is_dir() and (folder / "session.json").is_file():
            folders.append(folder)
    if (root / "session.json").is_file():
        folders.append(root)
    return folders


def _yolo_label_text(frame: LabeledFrame) -> str:
    width = max(frame.width, 1)
    height = max(frame.height, 1)
    lines: list[str] = []
    for box in frame.boxes:
        norm = box.normalized()
        cx = ((norm.x1 + norm.x2) / 2.0) / width
        cy = ((norm.y1 + norm.y2) / 2.0) / height
        bw = norm.width() / width
        bh = norm.height() / height
        class_id = drone_vehicle_class_id(getattr(norm, "label", "car"))
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return "\n".join(lines) + ("\n" if lines else "")


def _box_from_payload(raw: object) -> Box:
    data = raw if isinstance(raw, dict) else {}
    return Box(
        x1=float(data.get("x1", 0.0)),
        y1=float(data.get("y1", 0.0)),
        x2=float(data.get("x2", 0.0)),
        y2=float(data.get("y2", 0.0)),
        label=str(data.get("label") or data.get("cls") or "car"),
    )


def _write_yolo_label(path: Path, frame: LabeledFrame) -> None:
    path.write_text(_yolo_label_text(frame), encoding="utf-8")
