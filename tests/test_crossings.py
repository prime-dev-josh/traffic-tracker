import unittest

from traffic_tracker.core.crossings import bbox_center, path_crosses_line, segments_intersect
from traffic_tracker.core.geometry import Geometry, Line, NamedLine
from traffic_tracker.core.processor import (
    _Pending,
    _class_name,
    _expire_tracks,
    _record_crossings,
    _track_imgsz,
)
from traffic_tracker.core.report import Trip


class CrossingGeometryTests(unittest.TestCase):
    def test_bbox_center(self) -> None:
        self.assertEqual(bbox_center((10, 20, 30, 40)), (20.0, 30.0))

    def test_segments_cross(self) -> None:
        self.assertTrue(segments_intersect((0, 50), (100, 50), (50, 0), (50, 100)))
        self.assertFalse(segments_intersect((0, 0), (10, 0), (0, 10), (10, 10)))

    def test_path_crosses_drawn_gate(self) -> None:
        gate = Line(0, 50, 100, 50)
        self.assertTrue(path_crosses_line((50, 10), (50, 90), gate))
        self.assertFalse(path_crosses_line((10, 10), (20, 10), gate))

    def test_near_miss_at_endpoint_still_counts(self) -> None:
        gate = Line(20, 50, 80, 50)
        self.assertTrue(path_crosses_line((10, 10), (10, 90), gate, pad_px=24))


class RecordCrossingTests(unittest.TestCase):
    def _geom(self, starts, ends) -> Geometry:
        return Geometry(100, 100, north=Line(50, 55, 50, 10), starts=starts, ends=ends)

    def _step(self, y: float):
        return type("D", (), {"xyxy": [[40, y - 10, 60, y + 10]], "tracker_id": [7], "class_id": [2]})()

    def test_start_then_end_completes_trip(self) -> None:
        geom = self._geom([NamedLine("N", Line(0, 40, 100, 40))], [NamedLine("SB", Line(0, 70, 100, 70))])
        pending = _Pending()
        trips: list[Trip] = []
        _record_crossings(self._step(10), geom, pending, trips, "a.mp4", 1.0)
        starts_hit, ends_hit = _record_crossings(self._step(90), geom, pending, trips, "a.mp4", 2.0)
        self.assertEqual(starts_hit, 1)
        self.assertEqual(ends_hit, 1)
        self.assertEqual(len(trips), 1)
        self.assertEqual(trips[0].start_destination, "N")
        self.assertEqual(trips[0].end_destination, "SB")

    def test_swapped_start_end_colors_still_counts_n_sb(self) -> None:
        """Green start drawn on the south, red end on the north — still N-SB."""
        geom = self._geom([NamedLine("N", Line(0, 70, 100, 70))], [NamedLine("SB", Line(0, 40, 100, 40))])
        pending = _Pending()
        trips: list[Trip] = []
        _record_crossings(self._step(10), geom, pending, trips, "a.mp4", 1.0)
        _record_crossings(self._step(90), geom, pending, trips, "a.mp4", 2.0)
        self.assertEqual(len(trips), 1)
        self.assertEqual(trips[0].start_destination, "N")
        self.assertEqual(trips[0].end_destination, "SB")

    def _step_id(self, y: float, tid: int):
        return type("D", (), {"xyxy": [[40, y - 10, 60, y + 10]], "tracker_id": [tid], "class_id": [2]})()

    def test_id_switch_mid_intersection_still_counts(self) -> None:
        """ByteTrack often issues a new ID on crowded nadir cars."""
        geom = self._geom([NamedLine("N", Line(0, 40, 100, 40))], [NamedLine("SB", Line(0, 70, 100, 70))])
        pending = _Pending()
        trips: list[Trip] = []
        _record_crossings(self._step_id(10, 7), geom, pending, trips, "a.mp4", 1.0)
        starts_hit, ends_hit = _record_crossings(self._step_id(90, 99), geom, pending, trips, "a.mp4", 2.0)
        self.assertEqual(starts_hit, 1)
        self.assertEqual(ends_hit, 1)
        self.assertEqual(len(trips), 1)
        self.assertEqual(trips[0].start_destination, "N")
        self.assertEqual(trips[0].end_destination, "SB")

    def test_path_sides_count_when_gates_are_missed(self) -> None:
        geom = self._geom([NamedLine("N", Line(0, 2, 100, 2))], [NamedLine("SB", Line(0, 98, 100, 98))])
        pending = _Pending()
        trips: list[Trip] = []
        _record_crossings(self._step_id(22, 3), geom, pending, trips, "a.mp4", 1.0)
        _record_crossings(self._step_id(82, 3), geom, pending, trips, "a.mp4", 2.0)
        self.assertEqual(trips, [])
        inferred = _expire_tracks(pending, geom, trips, "a.mp4", 10.0, coast_s=0.0)
        self.assertEqual(inferred, 1)
        self.assertEqual(len(trips), 1)
        self.assertEqual(trips[0].start_destination, "N")
        self.assertEqual(trips[0].end_destination, "SB")


class OverheadDetectionTests(unittest.TestCase):
    def test_nadir_mislabels_map_to_car(self) -> None:
        self.assertEqual(_class_name(2), "car")
        self.assertEqual(_class_name(4), "car")  # airplane
        self.assertEqual(_class_name(6), "car")  # train
        self.assertEqual(_class_name(8), "car")  # boat
        self.assertEqual(_class_name(3), "motorcycle")

    def test_drone_uses_larger_imgsz(self) -> None:
        self.assertEqual(_track_imgsz(3840, 2160, overhead_drone=True), 1920)
        self.assertEqual(_track_imgsz(1920, 1080, overhead_drone=False), 1280)
        self.assertEqual(_track_imgsz(640, 480, overhead_drone=False), 640)


if __name__ == "__main__":
    unittest.main()
