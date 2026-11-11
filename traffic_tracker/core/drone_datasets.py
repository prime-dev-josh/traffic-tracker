"""Build a 5-class YOLO set from VisDrone and UAVDT for drone-base training."""

from __future__ import annotations

from pathlib import Path
import json
import os
import re
import shutil

from traffic_tracker.config import (
    DRONE_VEHICLES_DATASET_REL,
    TRAINING_DIR_NAME,
    UAVDT_ENV,
    UAVDT_FRAME_STRIDE,
    UAVDT_RAW_REL,
    drone_vehicles_data_yaml,
)

# Native VisDrone DET categories (1-based in annotation files).
# 1 pedestrian, 2 people, 3 bicycle, 4 car, 5 van, 6 truck,
# 7 tricycle, 8 awning-tricycle, 9 bus, 10 motor.
VISDRONE_NATIVE_TO_DRONE = {4: 0, 5: 1, 6: 2, 9: 3, 10: 4}
# After Ultralytics converts VisDrone to YOLO, classes are 0-based:
# 0 pedestrian … 3 car, 4 van, 5 truck, 8 bus, 9 motor.
VISDRONE_YOLO_TO_DRONE = {3: 0, 4: 1, 5: 2, 8: 3, 9: 4}
# UAVDT MOT categories are 1=car, 2=truck, 3=bus.
UAVDT_MOT_TO_DRONE = {1: 0, 2: 2, 3: 3}
# Already-YOLO UAVDT is usually 0=car, 1=truck, 2=bus.
UAVDT_YOLO_TO_DRONE = {0: 0, 1: 2, 2: 3}
# Dataset Ninja / Supervisely titles. Skip the extra "vehicle" SOT class.
UAVDT_SLY_TO_DRONE = {"car": 0, "truck": 2, "bus": 3}
_UAVDT_SLY_STEM = re.compile(r"^(M\d+)_img(\d+)$", re.IGNORECASE)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def datasets_root(base: Path | None = None) -> Path:
    return (base or Path.cwd()) / TRAINING_DIR_NAME / "datasets"


def drone_vehicles_dataset_dir(base: Path | None = None) -> Path:
    return (base or Path.cwd()) / TRAINING_DIR_NAME / DRONE_VEHICLES_DATASET_REL


def uavdt_root(base: Path | None = None, explicit: Path | None = None) -> Path:
    if explicit is not None:
        return Path(explicit)
    env = os.environ.get(UAVDT_ENV)
    if env:
        return Path(env)
    return (base or Path.cwd()) / TRAINING_DIR_NAME / UAVDT_RAW_REL


def visdrone_search_roots(base: Path | None = None) -> list[Path]:
    roots = [
        datasets_root(base) / "visdrone",
        datasets_root(base) / "VisDrone",
        (base or Path.cwd()) / "datasets" / "VisDrone",
        Path.cwd() / "datasets" / "VisDrone",
        Path.home() / "datasets" / "VisDrone",
    ]
    try:
        from ultralytics.utils import SETTINGS

        datasets_dir = SETTINGS.get("datasets_dir")
        if datasets_dir:
            roots.append(Path(datasets_dir) / "VisDrone")
    except Exception:
        pass
    seen: list[Path] = []
    for path in roots:
        resolved = path
        if resolved not in seen:
            seen.append(resolved)
    return seen


def find_visdrone_root(base: Path | None = None) -> Path | None:
    for root in visdrone_search_roots(base):
        if _visdrone_split_dir(root, "train") is not None:
            return root
    return None


def ensure_visdrone_downloaded(base: Path | None = None, *, download: bool = False) -> Path | None:
    found = find_visdrone_root(base)
    if found is not None:
        return found
    if not download:
        return None
    try:
        from ultralytics.data.utils import check_det_dataset

        data = check_det_dataset("VisDrone.yaml")
        path = data.get("path") if isinstance(data, dict) else None
        if path:
            candidate = Path(path)
            if candidate.exists():
                return candidate
    except Exception:
        pass
    return find_visdrone_root(base)


def build_drone_vehicles_dataset(
    base: Path | None = None,
    *,
    skip_uavdt: bool = False,
    visdrone_src: Path | None = None,
    uavdt_src: Path | None = None,
    download_visdrone: bool = False,
    frame_stride: int = UAVDT_FRAME_STRIDE,
    log_cb=None,
) -> dict:
    """Flatten VisDrone (+ optional UAVDT) into training/datasets/drone_vehicles."""

    def log(message: str) -> None:
        if log_cb:
            log_cb(message)

    dest = drone_vehicles_dataset_dir(base)
    if dest.exists():
        shutil.rmtree(dest)
    for split in ("train", "val"):
        (dest / "images" / split).mkdir(parents=True, exist_ok=True)
        (dest / "labels" / split).mkdir(parents=True, exist_ok=True)

    visdrone = visdrone_src if visdrone_src is not None else ensure_visdrone_downloaded(
        base, download=download_visdrone
    )
    visdrone_stats = {"images": 0, "boxes": 0, "train": 0, "val": 0}
    if visdrone is not None and visdrone.exists():
        visdrone_stats = convert_visdrone_vehicle_labels(visdrone, dest)
        log(
            f"VisDrone: {visdrone_stats['images']} images, {visdrone_stats['boxes']} vehicle boxes "
            f"({visdrone_stats['train']} train / {visdrone_stats['val']} val)."
        )
    else:
        log("VisDrone not found. Place it under training/datasets/visdrone or train VisDrone once.")

    uavdt_stats = {"images": 0, "boxes": 0, "train": 0, "val": 0}
    uavdt = None if skip_uavdt else uavdt_root(base, uavdt_src)
    if skip_uavdt:
        log("Skipping UAVDT.")
    elif uavdt is not None and uavdt.exists():
        uavdt_stats = convert_uavdt_labels(uavdt, dest, frame_stride=frame_stride)
        log(
            f"UAVDT: {uavdt_stats['images']} images, {uavdt_stats['boxes']} vehicle boxes "
            f"({uavdt_stats['train']} train / {uavdt_stats['val']} val)."
        )
    else:
        log(
            "UAVDT not found. Put UAV-benchmark-M + MOTD, or a Dataset Ninja extract, into "
            f"{uavdt_root(base)} or set {UAVDT_ENV}."
        )

    train_count = visdrone_stats["train"] + uavdt_stats["train"]
    val_count = visdrone_stats["val"] + uavdt_stats["val"]
    if train_count == 0:
        raise FileNotFoundError(
            "No VisDrone or UAVDT images to merge. Train on VisDrone first, or copy UAVDT into "
            f"{uavdt_root(base)}."
        )
    if val_count == 0:
        # Ultralytics needs a val split; reuse train as val.
        _copy_split(dest, "train", "val")
        val_count = train_count
        log("No validation images; copied the train split to val.")

    yaml_path = dest / "data.yaml"
    yaml_path.write_text(drone_vehicles_data_yaml(dest), encoding="utf-8")
    return {
        "root": dest,
        "yaml_path": yaml_path,
        "frame_count": train_count + val_count,
        "box_count": visdrone_stats["boxes"] + uavdt_stats["boxes"],
        "train_count": train_count,
        "val_count": val_count,
        "visdrone": visdrone_stats,
        "uavdt": uavdt_stats,
    }


def _ensure_dest_splits(dest_root: Path) -> None:
    for split in ("train", "val"):
        (dest_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (dest_root / "labels" / split).mkdir(parents=True, exist_ok=True)


def convert_visdrone_vehicle_labels(src_root: Path, dest_root: Path) -> dict:
    _ensure_dest_splits(dest_root)
    stats = {"images": 0, "boxes": 0, "train": 0, "val": 0}
    for split, names in (("train", ("train", "VisDrone2019-DET-train")), ("val", ("val", "VisDrone2019-DET-val"))):
        split_dir = _first_existing(src_root, names) or _visdrone_split_dir(src_root, split)
        if split_dir is None:
            continue
        images, boxes = _convert_visdrone_split(split_dir, dest_root, split, prefix=f"visdrone_{split}")
        stats["images"] += images
        stats["boxes"] += boxes
        stats[split] += images
    return stats


def convert_uavdt_labels(src_root: Path, dest_root: Path, *, frame_stride: int = UAVDT_FRAME_STRIDE) -> dict:
    _ensure_dest_splits(dest_root)
    yolo = _uavdt_yolo_layout(src_root)
    if yolo is not None:
        return _convert_uavdt_yolo(yolo, dest_root, frame_stride=frame_stride)
    sly_splits = _uavdt_supervisely_splits(src_root)
    if sly_splits:
        return _convert_uavdt_supervisely(sly_splits, dest_root, frame_stride=max(1, int(frame_stride)))
    return _convert_uavdt_mot(src_root, dest_root, frame_stride=max(1, int(frame_stride)))


def _visdrone_split_dir(root: Path, split: str) -> Path | None:
    names = ("train", "VisDrone2019-DET-train") if split == "train" else ("val", "VisDrone2019-DET-val")
    direct = _first_existing(root, names)
    if direct is not None:
        return direct
    for candidate in root.rglob("VisDrone2019-DET-train" if split == "train" else "VisDrone2019-DET-val"):
        if candidate.is_dir():
            return candidate
    return None


def _first_existing(root: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        path = root / name
        if path.is_dir():
            return path
    return None


def _convert_visdrone_split(split_dir: Path, dest_root: Path, split: str, prefix: str) -> tuple[int, int]:
    images_dir = split_dir / "images" if (split_dir / "images").is_dir() else split_dir
    native_ann = split_dir / "annotations"
    yolo_labels = split_dir / "labels"
    dest_images = dest_root / "images" / split
    dest_labels = dest_root / "labels" / split
    image_count = 0
    box_count = 0
    for image_path in sorted(_iter_images(images_dir)):
        stem = image_path.stem
        native = native_ann / f"{stem}.txt"
        yolo = yolo_labels / f"{stem}.txt"
        if native.is_file():
            lines, boxes = _remap_visdrone_native(native)
        elif yolo.is_file():
            lines, boxes = _remap_yolo_file(yolo, VISDRONE_YOLO_TO_DRONE)
        else:
            continue
        if not lines:
            continue
        dest_stem = f"{prefix}__{stem}"
        _copy_file(image_path, dest_images / f"{dest_stem}{image_path.suffix.lower()}")
        (dest_labels / f"{dest_stem}.txt").write_text("".join(lines), encoding="utf-8")
        image_count += 1
        box_count += boxes
    return image_count, box_count


def _remap_visdrone_native(path: Path) -> tuple[list[str], int]:
    image_w, image_h = _image_size_for_annotation(path)
    lines: list[str] = []
    boxes = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        parts = [part.strip() for part in raw.replace(" ", ",").split(",") if part.strip()]
        if len(parts) < 6:
            continue
        try:
            left, top, width, height = (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))
            category = int(float(parts[5]))
        except ValueError:
            continue
        class_id = VISDRONE_NATIVE_TO_DRONE.get(category)
        if class_id is None or width <= 0 or height <= 0:
            continue
        if image_w <= 0 or image_h <= 0:
            continue
        xc = (left + width / 2.0) / image_w
        yc = (top + height / 2.0) / image_h
        bw = width / image_w
        bh = height / image_h
        if bw <= 0 or bh <= 0:
            continue
        lines.append(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
        boxes += 1
    return lines, boxes


def _image_size_for_annotation(ann_path: Path) -> tuple[int, int]:
    stem = ann_path.stem
    for folder in (ann_path.parent.parent / "images", ann_path.parent):
        for suffix in IMAGE_SUFFIXES:
            image = folder / f"{stem}{suffix}"
            if image.is_file():
                try:
                    import cv2

                    frame = cv2.imread(str(image))
                    if frame is not None:
                        height, width = frame.shape[:2]
                        return int(width), int(height)
                except Exception:
                    return 0, 0
    return 0, 0


def _remap_yolo_file(path: Path, mapping: dict[int, int]) -> tuple[list[str], int]:
    lines: list[str] = []
    boxes = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        parts = raw.split()
        if len(parts) < 5:
            continue
        try:
            src_cls = int(float(parts[0]))
        except ValueError:
            continue
        class_id = mapping.get(src_cls)
        if class_id is None:
            continue
        lines.append(f"{class_id} {' '.join(parts[1:5])}\n")
        boxes += 1
    return lines, boxes


def _uavdt_supervisely_splits(root: Path) -> list[tuple[str, Path, Path]]:
    found: list[tuple[str, Path, Path]] = []
    seen: set[Path] = set()
    for ann_dir in root.rglob("ann"):
        if not ann_dir.is_dir():
            continue
        split_dir = ann_dir.parent
        images_dir = split_dir / "img"
        if not images_dir.is_dir() or split_dir in seen:
            continue
        if "_downloads" in split_dir.parts:
            continue
        seen.add(split_dir)
        names = {part.lower() for part in split_dir.parts}
        split = "val" if names & {"val", "test", "valid"} else "train"
        found.append((split, images_dir, ann_dir))
    return found


def _convert_uavdt_supervisely(
    splits: list[tuple[str, Path, Path]], dest_root: Path, *, frame_stride: int
) -> dict:
    stats = {"images": 0, "boxes": 0, "train": 0, "val": 0}
    stride = max(1, int(frame_stride))
    for split, images_dir, ann_dir in splits:
        grouped: dict[str, list[tuple[int, Path, Path]]] = {}
        ungrouped: list[tuple[Path, Path]] = []
        for image_path in sorted(_iter_images(images_dir)):
            ann_path = _supervisely_ann_path(ann_dir, image_path)
            if ann_path is None:
                continue
            parsed = _UAVDT_SLY_STEM.match(image_path.stem)
            if parsed is None:
                ungrouped.append((image_path, ann_path))
                continue
            seq, frame_idx = parsed.group(1), int(parsed.group(2))
            grouped.setdefault(seq, []).append((frame_idx, image_path, ann_path))
        kept = 0
        boxes = 0
        for seq, frames in grouped.items():
            for frame_idx, image_path, ann_path in sorted(frames, key=lambda item: item[0]):
                if (max(frame_idx, 1) - 1) % stride != 0:
                    continue
                image_n, box_n = _write_supervisely_frame(
                    image_path, ann_path, dest_root, split, f"uavdt_{seq}__{image_path.stem}"
                )
                kept += image_n
                boxes += box_n
        for index, (image_path, ann_path) in enumerate(ungrouped):
            if stride > 1 and index % stride != 0:
                continue
            image_n, box_n = _write_supervisely_frame(
                image_path, ann_path, dest_root, split, f"uavdt_{split}__{image_path.stem}"
            )
            kept += image_n
            boxes += box_n
        stats["images"] += kept
        stats["boxes"] += boxes
        stats[split] += kept
    return stats


def _write_supervisely_frame(
    image_path: Path, ann_path: Path, dest_root: Path, split: str, dest_stem: str
) -> tuple[int, int]:
    lines = _parse_supervisely_ann(ann_path)
    if not lines:
        return 0, 0
    _copy_file(image_path, dest_root / "images" / split / f"{dest_stem}{image_path.suffix.lower()}")
    (dest_root / "labels" / split / f"{dest_stem}.txt").write_text("".join(lines), encoding="utf-8")
    return 1, len(lines)


def _supervisely_ann_path(ann_dir: Path, image_path: Path) -> Path | None:
    direct = ann_dir / f"{image_path.name}.json"
    if direct.is_file():
        return direct
    stem = ann_dir / f"{image_path.stem}.json"
    if stem.is_file():
        return stem
    return None


def _parse_supervisely_ann(ann_path: Path) -> list[str]:
    try:
        data = json.loads(ann_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    size = data.get("size") or {}
    try:
        width = float(size.get("width") or 0)
        height = float(size.get("height") or 0)
    except (TypeError, ValueError):
        return []
    if width <= 0 or height <= 0:
        return []
    lines: list[str] = []
    for obj in data.get("objects") or []:
        if not isinstance(obj, dict):
            continue
        title = str(obj.get("classTitle") or "").strip().lower()
        class_id = UAVDT_SLY_TO_DRONE.get(title)
        box = _supervisely_xywh(obj)
        if class_id is None or box is None:
            continue
        xc, yc, bw, bh = box
        xc /= width
        yc /= height
        bw /= width
        bh /= height
        if bw <= 0 or bh <= 0:
            continue
        lines.append(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
    return lines


def _supervisely_xywh(obj: dict) -> tuple[float, float, float, float] | None:
    points = ((obj.get("points") or {}).get("exterior")) if isinstance(obj.get("points"), dict) else None
    if not isinstance(points, list) or len(points) < 2:
        return None
    try:
        xs = [float(pt[0]) for pt in points[:2]]
        ys = [float(pt[1]) for pt in points[:2]]
    except (TypeError, ValueError, IndexError):
        return None
    left, right = min(xs), max(xs)
    top, bottom = min(ys), max(ys)
    bw = right - left
    bh = bottom - top
    if bw <= 0 or bh <= 0:
        return None
    return left + bw / 2.0, top + bh / 2.0, bw, bh


def _uavdt_yolo_layout(root: Path) -> Path | None:
    if (root / "train" / "images").is_dir() and (root / "train" / "labels").is_dir():
        return root
    if (root / "images" / "train").is_dir() and (root / "labels" / "train").is_dir():
        return root
    return None


def _convert_uavdt_yolo(root: Path, dest_root: Path, *, frame_stride: int) -> dict:
    stats = {"images": 0, "boxes": 0, "train": 0, "val": 0}
    layouts = (
        ("train", root / "train" / "images", root / "train" / "labels"),
        ("val", root / "val" / "images", root / "val" / "labels"),
        ("train", root / "images" / "train", root / "labels" / "train"),
        ("val", root / "images" / "val", root / "labels" / "val"),
    )
    seen: set[tuple[str, Path]] = set()
    for split, images_dir, labels_dir in layouts:
        if not images_dir.is_dir() or not labels_dir.is_dir():
            continue
        key = (split, images_dir.resolve())
        if key in seen:
            continue
        seen.add(key)
        kept = 0
        boxes = 0
        for index, image_path in enumerate(sorted(_iter_images(images_dir))):
            if frame_stride > 1 and index % frame_stride != 0:
                continue
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                continue
            lines, box_n = _remap_yolo_file(label_path, UAVDT_YOLO_TO_DRONE)
            if not lines:
                continue
            dest_stem = f"uavdt_{split}__{image_path.stem}"
            _copy_file(image_path, dest_root / "images" / split / f"{dest_stem}{image_path.suffix.lower()}")
            (dest_root / "labels" / split / f"{dest_stem}.txt").write_text("".join(lines), encoding="utf-8")
            kept += 1
            boxes += box_n
        stats["images"] += kept
        stats["boxes"] += boxes
        stats[split] += kept
    return stats


def _convert_uavdt_mot(root: Path, dest_root: Path, *, frame_stride: int) -> dict:
    stats = {"images": 0, "boxes": 0, "train": 0, "val": 0}
    sequences = _uavdt_sequences(root)
    if not sequences:
        return stats
    val_cut = max(1, len(sequences) // 5)
    for index, (seq_name, images_dir, gt_path) in enumerate(sequences):
        split = "val" if index < val_cut else "train"
        image_n, box_n = _convert_uavdt_sequence(
            seq_name, images_dir, gt_path, dest_root, split, frame_stride
        )
        stats["images"] += image_n
        stats["boxes"] += box_n
        stats[split] += image_n
    return stats


def _uavdt_sequences(root: Path) -> list[tuple[str, Path, Path]]:
    found: list[tuple[str, Path, Path]] = []
    image_dirs = [path for path in root.rglob("*") if path.is_dir() and _looks_like_uavdt_seq(path)]
    for images_dir in sorted(image_dirs):
        seq = images_dir.name
        gt = _find_uavdt_gt(root, seq, images_dir)
        if gt is None:
            continue
        found.append((seq, images_dir, gt))
    return found


def _looks_like_uavdt_seq(folder: Path) -> bool:
    try:
        return any(path.suffix.lower() in IMAGE_SUFFIXES for path in folder.iterdir() if path.is_file())
    except OSError:
        return False


def _find_uavdt_gt(root: Path, seq: str, images_dir: Path) -> Path | None:
    names = (
        f"{seq}_gt_det.txt",
        f"{seq}_gt.txt",
        "gt_det.txt",
        "gt.txt",
    )
    search = [
        images_dir,
        images_dir.parent,
        root / "GT",
        root / "UAV-benchmark-MOTD_v1.0" / "GT",
        *root.rglob("GT"),
    ]
    seen: set[Path] = set()
    for folder in search:
        if not folder.is_dir() or folder in seen:
            continue
        seen.add(folder)
        for name in names:
            path = folder / name
            if path.is_file():
                return path
    return None


def _convert_uavdt_sequence(
    seq: str,
    images_dir: Path,
    gt_path: Path,
    dest_root: Path,
    split: str,
    frame_stride: int,
) -> tuple[int, int]:
    by_frame: dict[int, list[str]] = {}
    for frame_idx, class_id, xc, yc, bw, bh in _parse_uavdt_gt(gt_path, images_dir):
        if (max(frame_idx, 1) - 1) % max(1, frame_stride) != 0:
            continue
        by_frame.setdefault(frame_idx, []).append(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
    image_count = 0
    box_count = 0
    for frame_idx, lines in sorted(by_frame.items()):
        image_path = _uavdt_frame_image(images_dir, frame_idx)
        if image_path is None:
            continue
        dest_stem = f"uavdt_{seq}__{image_path.stem}"
        _copy_file(image_path, dest_root / "images" / split / f"{dest_stem}{image_path.suffix.lower()}")
        (dest_root / "labels" / split / f"{dest_stem}.txt").write_text("".join(lines), encoding="utf-8")
        image_count += 1
        box_count += len(lines)
    return image_count, box_count


def _parse_uavdt_gt(gt_path: Path, images_dir: Path) -> list[tuple[int, int, float, float, float, float]]:
    width, height = _uavdt_image_size(images_dir)
    rows: list[tuple[int, int, float, float, float, float]] = []
    for raw in gt_path.read_text(encoding="utf-8").splitlines():
        parts = [part.strip() for part in raw.replace(" ", ",").split(",") if part.strip()]
        if len(parts) < 8:
            continue
        try:
            frame_idx = int(float(parts[0]))
            left, top, box_w, box_h = (float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5]))
            category = int(float(parts[7]))
        except ValueError:
            continue
        class_id = UAVDT_MOT_TO_DRONE.get(category)
        if class_id is None or box_w <= 0 or box_h <= 0 or width <= 0 or height <= 0:
            continue
        xc = (left + box_w / 2.0) / width
        yc = (top + box_h / 2.0) / height
        bw = box_w / width
        bh = box_h / height
        rows.append((frame_idx, class_id, xc, yc, bw, bh))
    return rows


def _uavdt_image_size(images_dir: Path) -> tuple[int, int]:
    for image in _iter_images(images_dir):
        try:
            import cv2

            frame = cv2.imread(str(image))
            if frame is not None:
                height, width = frame.shape[:2]
                return int(width), int(height)
        except Exception:
            break
    return 1024, 540


def _uavdt_frame_image(images_dir: Path, frame_idx: int) -> Path | None:
    candidates = (
        f"img{frame_idx:06d}.jpg",
        f"img{frame_idx:05d}.jpg",
        f"{frame_idx:06d}.jpg",
        f"{frame_idx:05d}.jpg",
    )
    for name in candidates:
        path = images_dir / name
        if path.is_file():
            return path
    return None


def _iter_images(folder: Path):
    if not folder.is_dir():
        return
    for path in folder.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def _copy_file(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)


def _copy_split(root: Path, src_split: str, dest_split: str) -> None:
    for kind in ("images", "labels"):
        src = root / kind / src_split
        dest = root / kind / dest_split
        dest.mkdir(parents=True, exist_ok=True)
        if not src.is_dir():
            continue
        for path in src.iterdir():
            if path.is_file():
                shutil.copy2(path, dest / path.name)
