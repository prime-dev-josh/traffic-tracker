import unittest
from datetime import datetime

from traffic_tracker.core.geometry import Geometry, Line, Route
from traffic_tracker.core.report import Trip, build_tmc_table, interval_label


class ReportTests(unittest.TestCase):
    def test_elapsed_bins(self) -> None:
        trips = [
            Trip("a.mp4", 1, "NB Left", 60, "car"),
            Trip("a.mp4", 2, "NB Left", 900, "car"),
            Trip("a.mp4", 3, "SB Through", 901, "truck"),
        ]
        table = build_tmc_table(trips, ["NB Left", "SB Through"], duration_s=1800)
        # 0–15 min and 15–30 min. A crossing at exactly 15:00 belongs in the second bin.
        self.assertEqual(list(table["NB Left"]), [1, 1])
        self.assertEqual(list(table["SB Through"]), [0, 1])
        self.assertEqual(list(table["Total"]), [1, 2])

    def test_clock_labels(self) -> None:
        start, end = interval_label(0, datetime(2026, 11, 2, 8, 0))
        self.assertEqual(start, "2026-11-02 08:00")
        self.assertEqual(end, "2026-11-02 08:15")

    def test_geometry_roundtrip(self) -> None:
        geom = Geometry(
            1920,
            1080,
            [Route("NB Left", Line(10, 20, 30, 40), Line(50, 60, 70, 80))],
        )
        data = geom.to_dict()
        loaded = Geometry.from_dict(data)
        self.assertEqual(loaded.routes[0].name, "NB Left")
        scaled = loaded.scaled_to(960, 540)
        self.assertEqual(scaled.routes[0].start.x1, 5)


if __name__ == "__main__":
    unittest.main()
