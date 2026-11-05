"""Foreground motion blobs. Size, not class, decides what counts as a vehicle."""

from __future__ import annotations

import cv2
import numpy as np

from traffic_tracker.config import (
    MOTION_DIFF_THRESHOLD,
    MOTION_HISTORY,
    MOTION_MAX_AREA_FRAC,
    MOTION_MAX_ASPECT,
    MOTION_MIN_AREA_FRAC,
    MOTION_PROCESS_MAX_SIDE,
    MOTION_VAR_THRESHOLD,
)


def default_area_limits(width: int, height: int) -> tuple[int, int]:
    area = max(width * height, 1)
    return max(80, int(area * MOTION_MIN_AREA_FRAC)), max(2_000, int(area * MOTION_MAX_AREA_FRAC))


def process_scale(width: int, height: int) -> float:
    longest = max(width, height)
    if longest <= MOTION_PROCESS_MAX_SIDE:
        return 1.0
    return MOTION_PROCESS_MAX_SIDE / longest


def create_subtractor():
    return cv2.createBackgroundSubtractorMOG2(
        history=MOTION_HISTORY,
        varThreshold=MOTION_VAR_THRESHOLD,
        detectShadows=True,
    )


def blobs_from_frame(
    subtractor,
    frame: np.ndarray,
    *,
    min_area: float,
    max_area: float,
    scale: float = 1.0,
    prev_gray: np.ndarray | None = None,
    freeze_boxes: list[tuple[float, float, float, float]] | None = None,
) -> tuple[list[tuple[float, float, float, float]], np.ndarray, np.ndarray]:
    """Return original-frame boxes, a full-size binary mask, and the working gray frame."""
    work = frame
    if scale < 1.0:
        work = cv2.resize(frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    learn = work
    if freeze_boxes:
        background = subtractor.getBackgroundImage()
        if background is not None and background.shape[:2] == work.shape[:2]:
            learn = work.copy()
            for box in freeze_boxes:
                x1, y1, x2, y2 = _scaled_int_box(box, scale, work.shape[1], work.shape[0])
                if x2 <= x1 or y2 <= y1:
                    continue
                learn[y1:y2, x1:x2] = background[y1:y2, x1:x2]

    fg = subtractor.apply(learn)
    _, mask = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    if prev_gray is not None and prev_gray.shape == gray.shape:
        diff = cv2.absdiff(gray, prev_gray)
        _, delta = cv2.threshold(diff, MOTION_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_or(mask, delta)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    area_scale = scale * scale
    min_s = min_area * area_scale
    max_s = max_area * area_scale
    inv = 1.0 / scale if scale > 0 else 1.0

    boxes: list[tuple[float, float, float, float]] = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_s or area > max_s:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if min(w, h) < 1:
            continue
        if max(w, h) / min(w, h) > MOTION_MAX_ASPECT:
            continue
        boxes.append((x * inv, y * inv, (x + w) * inv, (y + h) * inv))

    if scale < 1.0:
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    return boxes, mask, gray


def _scaled_int_box(
    box: tuple[float, float, float, float],
    scale: float,
    width: int,
    height: int,
    pad: int = 4,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        max(0, int(min(x1, x2) * scale) - pad),
        max(0, int(min(y1, y2) * scale) - pad),
        min(width, int(max(x1, x2) * scale) + pad),
        min(height, int(max(y1, y2) * scale) + pad),
    )
