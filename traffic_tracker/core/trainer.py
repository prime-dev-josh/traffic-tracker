"""Fine-tune YOLO11 on boxed Training sessions and write site weights."""

from __future__ import annotations

from pathlib import Path
import shutil

from traffic_tracker.config import (
    TRAIN_BATCH,
    TRAIN_EPOCHS,
    TRAIN_IMGSZ,
    TRAIN_PATIENCE,
    TRAIN_RUNS_REL,
    YOLO_PT_NAME,
)
from traffic_tracker.core.device import describe_device, detect_device
from traffic_tracker.core.models import resolve_model_source
from traffic_tracker.core.training import (
    build_yolo_dataset,
    trained_weights_path,
    training_root,
    visdrone_weights_path,
)


def base_train_weights() -> Path:
    try:
        source = resolve_model_source("pt")
        if source.path.is_file():
            return source.path
    except Exception:
        pass
    return Path(YOLO_PT_NAME)


def yolo_train_device(device: str | None = None) -> str | int:
    chosen = device or detect_device()
    if chosen == "cuda":
        return 0
    return chosen


def _copy_best_weights(results, dest: Path, fallback_dir: Path) -> Path:
    save_dir = Path(getattr(results, "save_dir", fallback_dir))
    best = save_dir / "weights" / "best.pt"
    if not best.is_file():
        last = save_dir / "weights" / "last.pt"
        if last.is_file():
            best = last
    if not best.is_file():
        raise FileNotFoundError("Training finished but best.pt was not written.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, dest)
    return dest


def _fine_tune_yolo(
    *,
    data: str,
    run_name: str,
    dest: Path,
    base: Path | None,
    epochs: int,
    imgsz: int,
    progress_cb,
    cancel_flag,
    start_message: str,
    start_weights: Path | None = None,
    batch: int = TRAIN_BATCH,
) -> Path:
    device = detect_device()
    start = start_weights if start_weights and start_weights.is_file() else base_train_weights()
    if progress_cb:
        progress_cb(
            0,
            epochs,
            f"{start_message} {start.name} on {describe_device(device)} at imgsz {imgsz}, batch {batch}…",
        )

    from ultralytics import YOLO

    model = YOLO(str(start))

    def on_train_epoch_end(trainer) -> None:
        epoch = int(getattr(trainer, "epoch", 0)) + 1
        total = int(getattr(trainer, "epochs", epochs) or epochs)
        if progress_cb:
            progress_cb(epoch, total, f"Epoch {epoch}/{total}")
        if cancel_flag and cancel_flag():
            trainer.stop = True
            if hasattr(trainer, "stop_training"):
                trainer.stop_training = True

    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    results = model.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=TRAIN_PATIENCE,
        plots=False,
        exist_ok=True,
        project=str(training_root(base) / TRAIN_RUNS_REL),
        name=run_name,
        device=yolo_train_device(device),
        workers=2,
        verbose=True,
    )
    if cancel_flag and cancel_flag():
        raise RuntimeError("Training cancelled.")
    fallback = training_root(base) / TRAIN_RUNS_REL / run_name
    _copy_best_weights(results, dest, fallback)
    if dest != trained_weights_path(base):
        site = trained_weights_path(base)
        site.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dest, site)
    if progress_cb:
        progress_cb(epochs, epochs, f"Saved weights to {dest}")
    return dest


def train_site_detector(
    base: Path | None = None,
    *,
    epochs: int = TRAIN_EPOCHS,
    imgsz: int = TRAIN_IMGSZ,
    progress_cb=None,
    log_cb=None,
    cancel_flag=None,
) -> Path:
    """Build the flattened dataset, fine-tune yolo11n, copy best.pt into training/weights."""

    def log(message: str) -> None:
        if log_cb:
            log_cb(message)

    def progress(current: int, total: int, message: str) -> None:
        if progress_cb:
            progress_cb(current, total, message)

    if cancel_flag and cancel_flag():
        raise RuntimeError("Training cancelled.")

    progress(0, epochs, "Building YOLO dataset from boxed frames…")
    dataset = build_yolo_dataset(base)
    log(
        f"{dataset['frame_count']} frames ({dataset['train_count']} train / "
        f"{dataset['val_count']} val), {dataset['box_count']} boxes."
    )
    if cancel_flag and cancel_flag():
        raise RuntimeError("Training cancelled.")

    return _fine_tune_yolo(
        data=str(dataset["yaml_path"]),
        run_name="site",
        dest=trained_weights_path(base),
        base=base,
        epochs=epochs,
        imgsz=imgsz,
        progress_cb=progress,
        cancel_flag=cancel_flag,
        start_message="Fine-tuning",
    )


def train_visdrone_detector(
    base: Path | None = None,
    *,
    epochs: int = TRAIN_EPOCHS,
    imgsz: int = TRAIN_IMGSZ,
    progress_cb=None,
    log_cb=None,
    cancel_flag=None,
) -> Path:
    """Fine-tune yolo11n on VisDrone2019-DET. Ultralytics downloads the dataset (~2 GB)."""

    def progress(current: int, total: int, message: str) -> None:
        if progress_cb:
            progress_cb(current, total, message)
        if log_cb:
            log_cb(message)

    if cancel_flag and cancel_flag():
        raise RuntimeError("Training cancelled.")

    progress(0, epochs, "Downloading and converting VisDrone if needed (~2 GB)…")
    dest = visdrone_weights_path(base)
    last = training_root(base) / TRAIN_RUNS_REL / "visdrone" / "weights" / "last.pt"
    path = _fine_tune_yolo(
        data="VisDrone.yaml",
        run_name="visdrone",
        dest=dest,
        base=base,
        epochs=epochs,
        imgsz=imgsz,
        progress_cb=progress,
        cancel_flag=cancel_flag,
        start_message="Fine-tuning on VisDrone with",
        start_weights=last if last.is_file() else None,
        batch=TRAIN_BATCH,
    )
    if log_cb:
        log_cb(f"VisDrone weights: {path}")
        log_cb(f"Also copied to {trained_weights_path(base)} for Process → Trained detector.")
    return path
