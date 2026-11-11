"""Drone-base (VisDrone + UAVDT) training was removed.

Use VisDrone only:

    .\\.venv\\Scripts\\python.exe -m traffic_tracker.train_visdrone
"""


def main() -> int:
    raise SystemExit(
        "Drone-base training was removed. Train VisDrone instead:\n"
        "  python -m traffic_tracker.train_visdrone"
    )


if __name__ == "__main__":
    raise SystemExit(main())
