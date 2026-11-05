import unittest

import numpy as np

from traffic_tracker.core.motion import blobs_from_frame, create_subtractor, default_area_limits, process_scale


class MotionLimitTests(unittest.TestCase):
    def test_defaults_scale_with_frame(self) -> None:
        small_min, small_max = default_area_limits(640, 480)
        big_min, big_max = default_area_limits(3840, 2160)
        self.assertLess(small_min, big_min)
        self.assertLess(small_max, big_max)
        self.assertGreater(small_max, small_min)

    def test_4k_is_downscaled(self) -> None:
        self.assertLess(process_scale(3840, 2160), 1.0)
        self.assertEqual(process_scale(1280, 720), 1.0)


class MotionBlobTests(unittest.TestCase):
    def test_moving_square_is_kept_tiny_noise_is_dropped(self) -> None:
        subtractor = create_subtractor()
        bg = np.full((240, 320, 3), 40, dtype=np.uint8)
        kept: list[tuple[float, float, float, float]] = []
        for i in range(90):
            frame = bg.copy()
            x = 30 + i
            frame[90:130, x : x + 36] = 210
            frame[10:14, 10:14] = 255
            boxes, *_ = blobs_from_frame(subtractor, frame, min_area=400, max_area=8_000)
            if i > 40:
                kept.extend(boxes)
        self.assertGreater(len(kept), 0)
        for x1, y1, x2, y2 in kept:
            self.assertGreaterEqual((x2 - x1) * (y2 - y1), 400)
            self.assertGreater(y1, 20)

    def test_oversized_blob_is_rejected(self) -> None:
        subtractor = create_subtractor()
        bg = np.full((200, 200, 3), 20, dtype=np.uint8)
        found = []
        for i in range(60):
            frame = bg.copy()
            if i > 15:
                frame[10:190, 10:190] = 220
            boxes, *_ = blobs_from_frame(subtractor, frame, min_area=80, max_area=2_000)
            found.extend(boxes)
        self.assertEqual(found, [])
