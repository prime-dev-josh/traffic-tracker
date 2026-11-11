"""Class-agnostic NMS extras: containment merge and frame-check drawing."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import unittest

import numpy as np

from traffic_tracker.core.detector import (
    boxes_from_detector,
    draw_detection_preview,
    suppress_duplicate_boxes,
)


class DuplicateBoxTests(unittest.TestCase):
    def test_overlapping_car_and_truck_become_one(self) -> None:
        car = ((10.0, 10.0, 50.0, 40.0), "car", 0.91)
        truck = ((12.0, 11.0, 48.0, 39.0), "truck", 0.62)
        kept = suppress_duplicate_boxes([car, truck])
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0][1], "car")
        self.assertAlmostEqual(kept[0][2], 0.91)

    def test_inner_box_is_dropped(self) -> None:
        outer = ((0.0, 0.0, 100.0, 80.0), "car", 0.4)
        inner = ((20.0, 20.0, 40.0, 40.0), "van", 0.9)
        kept = suppress_duplicate_boxes([outer, inner])
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0][1], "van")

    def test_separate_cars_stay(self) -> None:
        left = ((0.0, 0.0, 20.0, 20.0), "car", 0.8)
        right = ((80.0, 0.0, 100.0, 20.0), "car", 0.8)
        kept = suppress_duplicate_boxes([left, right])
        self.assertEqual(len(kept), 2)

    def test_specific_class_wins_tied_confidence(self) -> None:
        car = ((10.0, 10.0, 50.0, 40.0), "car", 0.70)
        bus = ((12.0, 12.0, 48.0, 38.0), "bus", 0.70)
        kept = suppress_duplicate_boxes([car, bus])
        self.assertEqual(kept[0][1], "bus")

    def test_predict_uses_agnostic_nms(self) -> None:
        model = MagicMock()
        model.names = {2: "car"}
        result = MagicMock()
        result.boxes = None
        model.predict.return_value = [result]
        frame = np.zeros((80, 80, 3), dtype=np.uint8)
        with patch("traffic_tracker.core.detector.yolo_runtime", return_value=(0, 16)):
            boxes_from_detector(model, frame, min_area=1, max_area=1e9, imgsz=640, device="cuda")
        self.assertTrue(model.predict.call_args.kwargs["agnostic_nms"])

    def test_draw_preview_keeps_frame_size(self) -> None:
        frame = np.zeros((40, 60, 3), dtype=np.uint8)
        out = draw_detection_preview(frame, [((5.0, 5.0, 20.0, 15.0), "car", 0.8)])
        self.assertEqual(out.shape, frame.shape)
        self.assertGreater(int(out.sum()), 0)
