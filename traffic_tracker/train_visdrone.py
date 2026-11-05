"""Download VisDrone and fine-tune the bundled YOLO11n weights.

Run from the repo root:

    .\\.venv\\Scripts\\python.exe -m traffic_tracker.train_visdrone
"""

from __future__ import annotations

from traffic_tracker.core.trainer import train_visdrone_detector


def main() -> int:
    path = train_visdrone_detector(
        progress_cb=lambda current, total, message: print(f"[{current}/{total}] {message}", flush=True),
        log_cb=lambda message: print(message, flush=True),
    )
    print(f"Done: {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
