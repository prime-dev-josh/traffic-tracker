import unittest

from traffic_tracker.core.crossings import bbox_center, bbox_corners, box_hits_line, box_iou, path_crosses_line, segments_intersect
from traffic_tracker.core.geometry import Box, Geometry, Line, NamedLine
from traffic_tracker.core.processor import (
    _Pending,
    _draw_paths,
    _expire_tracks,
    _path_chunks,
    _record_crossings,
)
from traffic_tracker.core.report import Trip


class CrossingGeometryTests(unittest.TestCase):
    def test_bbox_center(self) -> None:
        self.assertEqual(bbox_center((10, 20, 30, 40)), (20.0, 30.0))

    def test_bbox_corners(self) -> None:
        self.assertEqual(
            bbox_corners((0, 10, 40, 30)),
            ((0.0, 10.0), (40.0, 10.0), (40.0, 30.0), (0.0, 30.0)),
        )

    def test_box_iou(self) -> None:
        self.assertAlmostEqual(box_iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertAlmostEqual(box_iou((0, 0, 10, 10), (20, 0, 30, 10)), 0.0)
        self.assertGreater(box_iou((0, 0, 10, 10), (5, 0, 15, 10)), 0.3)

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

    def test_missed_frames_keep_the_same_moving_track(self) -> None:
        geom = Geometry(
            400,
            400,
            north=Line(200, 220, 200, 40),
            starts=[NamedLine("N", Line(0, 8, 400, 8))],
            ends=[NamedLine("SB", Line(0, 392, 400, 392))],
        )
        pending = _Pending(link_speed=180.0, path_jump=90.0, parked_radius=40.0)
        trips: list[Trip] = []
        boxes = [
            (1.00, (20, 180, 50, 210)),
            (1.12, (36, 180, 66, 210)),
            (1.24, (52, 180, 82, 210)),
            (1.36, (68, 180, 98, 210)),
        ]
        for when, box in boxes:
            _record_crossings([box], geom, pending, trips, "a.mp4", when, moving_only=True, classes=["car"])
        tid = next(iter(pending.live))
        self.assertTrue(pending.live[tid].active)
        _record_crossings([], geom, pending, trips, "a.mp4", 1.48, moving_only=True)
        _record_crossings([], geom, pending, trips, "a.mp4", 1.60, moving_only=True)
        _record_crossings([(100, 180, 130, 210)], geom, pending, trips, "a.mp4", 1.72, moving_only=True, classes=["car"])
        self.assertEqual(list(pending.live), [tid])
        self.assertGreaterEqual(len(pending.live[tid].path), 2)

    def test_drawn_shadow_keeps_track_across_the_gap(self) -> None:
        kwargs = dict(
            north=Line(200, 220, 200, 40),
            starts=[NamedLine("N", Line(0, 8, 400, 8))],
            ends=[NamedLine("SB", Line(0, 392, 400, 392))],
        )
        bare = Geometry(400, 400, **kwargs)
        marked = Geometry(400, 400, **kwargs, occlusions=[Box(90, 160, 110, 230)])
        trips: list[Trip] = []

        def run(geom: Geometry) -> list[int]:
            pending = _Pending(link_speed=180.0)
            _record_crossings([(20, 180, 50, 210)], geom, pending, trips, "a.mp4", 1.00, moving_only=True, classes=["car"])
            _record_crossings([(36, 180, 66, 210)], geom, pending, trips, "a.mp4", 1.12, moving_only=True, classes=["car"])
            first = next(iter(pending.live))
            _record_crossings([(120, 180, 150, 210)], geom, pending, trips, "a.mp4", 1.36, moving_only=True, classes=["car"])
            return first, list(pending.live)

        bare_first, bare_ids = run(bare)
        self.assertNotEqual(bare_ids, [bare_first])
        marked_first, marked_ids = run(marked)
        self.assertEqual(marked_ids, [marked_first])

    def test_close_moving_cars_keep_separate_tracks(self) -> None:
        geom = self._geom([NamedLine("N", Line(0, 40, 100, 40))], [NamedLine("SB", Line(0, 70, 100, 70))])
        pending = _Pending()
        trips: list[Trip] = []
        for step in range(6):
            y = 8.0 + step * 18.0
            _record_crossings(
                [(20, y, 50, y + 30), (120, y, 150, y + 30)],
                geom,
                pending,
                trips,
                "a.mp4",
                1.0 + step * 0.12,
                moving_only=True,
                classes=["car", "car"],
            )
        self.assertEqual(len(pending.live), 2)
        for state in pending.live.values():
            self.assertTrue(state.active)
            xs = [point[0] for point in state.path]
            self.assertLess(max(xs) - min(xs), 20.0)

    def test_parked_neighbors_do_not_share_a_path(self) -> None:
        geom = Geometry(
            400,
            400,
            north=Line(200, 220, 200, 40),
            starts=[NamedLine("N", Line(0, 8, 400, 8))],
            ends=[NamedLine("SB", Line(0, 392, 400, 392))],
        )
        pending = _Pending()
        trips: list[Trip] = []
        left = (40, 80, 80, 120)
        right = (180, 80, 220, 120)
        _record_crossings([left, right], geom, pending, trips, "a.mp4", 1.00, moving_only=True, classes=["car", "car"])
        _record_crossings([left, right], geom, pending, trips, "a.mp4", 1.04, moving_only=True, classes=["car", "car"])
        self.assertEqual(len(pending.live), 2)
        self.assertEqual(trips, [])
        for state in pending.live.values():
            self.assertFalse(state.active)
            self.assertEqual(state.path, [])
        _expire_tracks(pending, geom, trips, "a.mp4", 8.0, coast_s=0.0, infer_path=False)
        self.assertEqual(pending.finished_paths, [])

    def test_parked_boxes_stay_inactive_and_untraced(self) -> None:
        geom = self._geom([NamedLine("N", Line(0, 40, 100, 40))], [NamedLine("SB", Line(0, 70, 100, 70))])
        pending = _Pending()
        trips: list[Trip] = []
        parked = [(40, 10, 60, 30)]
        _record_crossings(parked, geom, pending, trips, "a.mp4", 1.0, moving_only=True, classes=["car"])
        _record_crossings(parked, geom, pending, trips, "a.mp4", 1.5, moving_only=True, classes=["car"])
        self.assertEqual(trips, [])
        self.assertTrue(pending.live)
        state = next(iter(pending.live.values()))
        self.assertFalse(state.active)
        self.assertEqual(state.path, [])

    def test_hover_drift_does_not_activate_parked(self) -> None:
        geom = Geometry(
            400,
            400,
            north=Line(200, 220, 200, 40),
            starts=[NamedLine("N", Line(0, 8, 400, 8))],
            ends=[NamedLine("SB", Line(0, 392, 400, 392))],
        )
        pending = _Pending()
        trips: list[Trip] = []
        for step in range(12):
            y = 120.0 + step * 6.0
            _record_crossings(
                [(180, y, 220, y + 40)],
                geom,
                pending,
                trips,
                "a.mp4",
                1.0 + step * 1.6,
                moving_only=True,
                classes=["car"],
            )
        self.assertEqual(trips, [])
        if pending.live:
            self.assertFalse(next(iter(pending.live.values())).active)

    def test_trained_mode_skips_parked_and_inferred_trips(self) -> None:
        geom = Geometry(
            400,
            400,
            north=Line(200, 220, 200, 40),
            starts=[NamedLine("N", Line(0, 8, 400, 8))],
            ends=[NamedLine("SB", Line(0, 392, 400, 392))],
        )
        pending = _Pending()
        trips: list[Trip] = []
        parked = [(180, 100, 220, 140)]
        _record_crossings(parked, geom, pending, trips, "a.mp4", 1.0, moving_only=True)
        _record_crossings(parked, geom, pending, trips, "a.mp4", 2.0, moving_only=True)
        self.assertEqual(trips, [])
        inferred = _expire_tracks(pending, geom, trips, "a.mp4", 10.0, coast_s=0.0, infer_path=False)
        self.assertEqual(inferred, 0)
        self.assertEqual(trips, [])

        moving = _Pending()
        _record_crossings([(180, 100, 220, 140)], geom, moving, trips, "a.mp4", 1.0, moving_only=True)
        _record_crossings([(180, 250, 220, 290)], geom, moving, trips, "a.mp4", 2.0, moving_only=True)
        inferred = _expire_tracks(moving, geom, trips, "a.mp4", 10.0, coast_s=0.0, infer_path=False)
        self.assertEqual(inferred, 0)
        self.assertEqual(trips, [])

    def test_large_gap_splits_the_line(self) -> None:
        points = [(10.0, 40.0), (40.0, 40.0), (70.0, 40.0), (200.0, 40.0)]
        chunks = _path_chunks(points, max_jump=90.0)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0][-1], (70.0, 40.0))

    def test_sideways_hop_still_splits(self) -> None:
        points = [(10.0, 40.0), (40.0, 40.0), (70.0, 40.0), (70.0, 180.0)]
        chunks = _path_chunks(points, max_jump=90.0)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0][-1], (70.0, 40.0))

    def test_path_draws_as_continuous_line(self) -> None:
        import numpy as np

        frame = np.zeros((80, 120, 3), dtype=np.uint8)
        pending = _Pending()
        pending.finished_paths.append(("car", [(10.0, 40.0), (40.0, 40.0), (70.0, 40.0), (110.0, 40.0)]))
        _draw_paths(frame, pending)
        lit = int((frame[38:43] > 0).any(axis=2).sum())
        self.assertGreater(lit, 90)

    def test_expired_path_is_kept_without_a_trip(self) -> None:
        geom = self._geom([NamedLine("N", Line(0, 40, 100, 40))], [NamedLine("SB", Line(0, 70, 100, 70))])
        pending = _Pending()
        trips: list[Trip] = []
        _record_crossings([(40, 8, 60, 12)], geom, pending, trips, "a.mp4", 1.0)
        _record_crossings([(40, 18, 60, 22)], geom, pending, trips, "a.mp4", 1.2)
        self.assertTrue(next(iter(pending.live.values())).path)
        inferred = _expire_tracks(pending, geom, trips, "a.mp4", 20.0, coast_s=0.0, infer_path=False)
        self.assertEqual(inferred, 0)
        self.assertEqual(trips, [])
        self.assertEqual(pending.live, {})
        self.assertTrue(pending.finished_paths)
        self.assertGreaterEqual(len(pending.finished_paths[0][1]), 2)

    def test_class_and_path_are_recorded(self) -> None:
        geom = self._geom([NamedLine("N", Line(0, 40, 100, 40))], [NamedLine("SB", Line(0, 70, 100, 70))])
        pending = _Pending()
        trips: list[Trip] = []
        _record_crossings(self._box(10), geom, pending, trips, "a.mp4", 1.0, classes=["car"])
        _record_crossings(self._box(90), geom, pending, trips, "a.mp4", 2.0, classes=["car"])
        self.assertEqual(len(trips), 1)
        self.assertEqual(trips[0].vehicle_class, "car")
        self.assertGreaterEqual(len(trips[0].path), 2)

    def test_trained_mode_counts_moving_gate_pair(self) -> None:
        geom = self._geom([NamedLine("N", Line(0, 40, 100, 40))], [NamedLine("SB", Line(0, 70, 100, 70))])
        pending = _Pending()
        trips: list[Trip] = []
        _record_crossings(self._box(10), geom, pending, trips, "a.mp4", 1.0, moving_only=True)
        self.assertEqual(trips, [])
        _record_crossings(self._box(90), geom, pending, trips, "a.mp4", 2.0, moving_only=True)
        self.assertEqual(len(trips), 1)
        self.assertEqual(trips[0].start_destination, "N")
        self.assertEqual(trips[0].end_destination, "SB")


if __name__ == "__main__":
    unittest.main()
