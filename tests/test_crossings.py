import unittest

from traffic_tracker.core.crossings import bbox_center, box_hits_line, path_crosses_line, segments_intersect
from traffic_tracker.core.geometry import Geometry, Line, NamedLine
from traffic_tracker.core.processor import _Pending, _expire_tracks, _record_crossings
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

    def test_box_sitting_on_gate_counts(self) -> None:
        gate = Line(0, 50, 100, 50)
        self.assertTrue(box_hits_line((40, 30, 60, 70), gate))
        self.assertFalse(box_hits_line((40, 10, 60, 25), gate, pad_px=8))


class RecordCrossingTests(unittest.TestCase):
    def _geom(self, starts, ends) -> Geometry:
        return Geometry(100, 100, north=Line(50, 55, 50, 10), starts=starts, ends=ends)

    def _box(self, y: float) -> list[tuple[float, float, float, float]]:
        return [(40, y - 10, 60, y + 10)]

    def test_start_then_end_completes_trip(self) -> None:
        geom = self._geom([NamedLine("N", Line(0, 40, 100, 40))], [NamedLine("SB", Line(0, 70, 100, 70))])
        pending = _Pending()
        trips: list[Trip] = []
        _record_crossings(self._box(10), geom, pending, trips, "a.mp4", 1.0)
        _record_crossings(self._box(90), geom, pending, trips, "a.mp4", 2.0)
        self.assertEqual(len(trips), 1)
        self.assertEqual(trips[0].start_destination, "N")
        self.assertEqual(trips[0].end_destination, "SB")
        self.assertEqual(trips[0].vehicle_class, "vehicle")

    def test_swapped_start_end_colors_still_counts_n_sb(self) -> None:
        """Green start drawn on the south, red end on the north — still N-SB."""
        geom = self._geom([NamedLine("N", Line(0, 70, 100, 70))], [NamedLine("SB", Line(0, 40, 100, 40))])
        pending = _Pending()
        trips: list[Trip] = []
        _record_crossings(self._box(10), geom, pending, trips, "a.mp4", 1.0)
        _record_crossings(self._box(90), geom, pending, trips, "a.mp4", 2.0)
        self.assertEqual(len(trips), 1)
        self.assertEqual(trips[0].start_destination, "N")
        self.assertEqual(trips[0].end_destination, "SB")

    def test_nearby_blob_continues_same_vehicle(self) -> None:
        geom = self._geom([NamedLine("N", Line(0, 40, 100, 40))], [NamedLine("SB", Line(0, 70, 100, 70))])
        pending = _Pending()
        trips: list[Trip] = []
        _record_crossings(self._box(10), geom, pending, trips, "a.mp4", 1.0)
        _record_crossings(self._box(90), geom, pending, trips, "a.mp4", 2.0)
        self.assertEqual(len(trips), 1)

    def test_box_on_gate_counts_without_prior_point(self) -> None:
        geom = self._geom([NamedLine("N", Line(0, 40, 100, 40))], [NamedLine("SB", Line(0, 70, 100, 70))])
        pending = _Pending()
        trips: list[Trip] = []
        starts_hit, _ = _record_crossings([(30, 20, 70, 60)], geom, pending, trips, "a.mp4", 1.0)
        self.assertGreaterEqual(starts_hit, 1)
        _record_crossings([(30, 50, 70, 90)], geom, pending, trips, "a.mp4", 2.0)
        self.assertGreaterEqual(len(trips), 1)
        self.assertEqual(trips[0].start_destination, "N")
        self.assertEqual(trips[0].end_destination, "SB")

    def test_path_sides_count_when_gates_are_missed(self) -> None:
        geom = Geometry(
            400,
            400,
            north=Line(200, 220, 200, 40),
            starts=[NamedLine("N", Line(0, 8, 400, 8))],
            ends=[NamedLine("SB", Line(0, 392, 400, 392))],
        )
        pending = _Pending()
        trips: list[Trip] = []
        _record_crossings([(180, 100, 220, 140)], geom, pending, trips, "a.mp4", 1.0)
        _record_crossings([(180, 250, 220, 290)], geom, pending, trips, "a.mp4", 2.0)
        self.assertEqual(trips, [])
        inferred = _expire_tracks(pending, geom, trips, "a.mp4", 10.0, coast_s=0.0)
        self.assertEqual(inferred, 1)
        self.assertEqual(len(trips), 1)
        self.assertEqual(trips[0].start_destination, "N")
        self.assertEqual(trips[0].end_destination, "SB")

    def test_stopped_track_is_held_then_resumes(self) -> None:
        geom = self._geom([NamedLine("N", Line(0, 40, 100, 40))], [NamedLine("SB", Line(0, 70, 100, 70))])
        pending = _Pending()
        trips: list[Trip] = []
        _record_crossings(self._box(10), geom, pending, trips, "a.mp4", 1.0)
        _record_crossings(self._box(12), geom, pending, trips, "a.mp4", 1.2)
        tid = next(iter(pending.live))
        _record_crossings([], geom, pending, trips, "a.mp4", 20.0)
        self.assertIn(tid, pending.live)
        self.assertTrue(pending.live[tid].holding)
        _expire_tracks(pending, geom, trips, "a.mp4", 20.0)
        self.assertIn(tid, pending.live)
        _record_crossings(self._box(90), geom, pending, trips, "a.mp4", 21.0)
        self.assertEqual(len(trips), 1)
        self.assertEqual(trips[0].track_id, tid)

    def test_trip_completion_releases_hold(self) -> None:
        geom = self._geom([NamedLine("N", Line(0, 40, 100, 40))], [NamedLine("SB", Line(0, 70, 100, 70))])
        pending = _Pending()
        trips: list[Trip] = []
        _record_crossings(self._box(10), geom, pending, trips, "a.mp4", 1.0)
        _record_crossings(self._box(12), geom, pending, trips, "a.mp4", 1.2)
        tid = next(iter(pending.live))
        _record_crossings([], geom, pending, trips, "a.mp4", 20.0)
        self.assertTrue(pending.live[tid].holding)
        _record_crossings(self._box(90), geom, pending, trips, "a.mp4", 21.0)
        self.assertEqual(len(trips), 1)
        self.assertNotIn(tid, pending.live)

    def test_fast_vanish_expires_on_coast(self) -> None:
        geom = self._geom([NamedLine("N", Line(0, 40, 100, 40))], [NamedLine("SB", Line(0, 70, 100, 70))])
        pending = _Pending()
        trips: list[Trip] = []
        _record_crossings(self._box(8), geom, pending, trips, "a.mp4", 1.0)
        _record_crossings(self._box(18), geom, pending, trips, "a.mp4", 1.05)
        tid = next(iter(pending.live))
        self.assertGreater(pending.live[tid].speed, 40.0)
        _record_crossings([], geom, pending, trips, "a.mp4", 8.0)
        _expire_tracks(pending, geom, trips, "a.mp4", 8.0)
        self.assertNotIn(tid, pending.live)


if __name__ == "__main__":
    unittest.main()
