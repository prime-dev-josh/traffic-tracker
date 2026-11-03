"""Compare TMC tables across backends. Live video compare is opt-in."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

from traffic_tracker.core.geometry import Geometry
from traffic_tracker.core.report import Trip, build_tmc_table, tmc_tables_close


class TmcToleranceTests(unittest.TestCase):
    def test_tables_close_within_one(self) -> None:
        left = build_tmc_table(
            [Trip("a.mp4", 1, "N", 30, "SB", 30, "car")],
            ["N-SB", "S-NB"],
            900,
        )
        right = build_tmc_table(
            [Trip("a.mp4", 1, "N", 40, "SB", 40, "car"), Trip("a.mp4", 2, "N", 50, "SB", 50, "car")],
            ["N-SB", "S-NB"],
            900,
        )
        self.assertTrue(tmc_tables_close(left, right, max_abs_diff=1))

    def test_tables_reject_large_drift(self) -> None:
        left = build_tmc_table([Trip("a.mp4", 1, "N", 30, "SB", 30, "car")], ["N-SB"], 900)
        right = build_tmc_table(
            [Trip("a.mp4", i, "N", 30, "SB", 30, "car") for i in range(5)],
            ["N-SB"],
            900,
        )
        self.assertFalse(tmc_tables_close(left, right, max_abs_diff=1))


class LiveParityTests(unittest.TestCase):
    """Set TRAFFIC_TRACKER_GOLDEN_VIDEO and TRAFFIC_TRACKER_GOLDEN_ROUTES to run."""

    def test_pt_vs_onnx_on_golden_video(self) -> None:
        video = os.environ.get("TRAFFIC_TRACKER_GOLDEN_VIDEO", "").strip()
        routes = os.environ.get("TRAFFIC_TRACKER_GOLDEN_ROUTES", "").strip()
        if not video or not Path(video).is_file() or not routes or not Path(routes).is_file():
            self.skipTest("Set TRAFFIC_TRACKER_GOLDEN_VIDEO and TRAFFIC_TRACKER_GOLDEN_ROUTES")

        from traffic_tracker.core.processor import process_video, reset_model_cache
        from traffic_tracker.core.report import build_tmc_table

        geometry = Geometry.load(routes)
        names = geometry.movement_names()

        os.environ["TRAFFIC_TRACKER_BACKEND"] = "pt"
        reset_model_cache()
        pt_result = process_video(video, geometry)
        os.environ["TRAFFIC_TRACKER_BACKEND"] = "onnx"
        reset_model_cache()
        onnx_result = process_video(video, geometry)

        self.assertFalse(pt_result.error)
        self.assertFalse(onnx_result.error)
        pt_table = build_tmc_table(pt_result.trips, names, pt_result.video.duration_s)
        onnx_table = build_tmc_table(onnx_result.trips, names, onnx_result.video.duration_s)
        self.assertTrue(
            tmc_tables_close(pt_table, onnx_table, max_abs_diff=1),
            msg=f"PT trips={len(pt_result.trips)} ONNX trips={len(onnx_result.trips)}",
        )


if __name__ == "__main__":
    unittest.main()
