"""Identity, evidence, drop-zone, and roadway rules. Synthetic boxes only."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from traffic_tracker.config import DETECTOR_STATIC_EXPIRE_S, MOTION_COAST_S
from traffic_tracker.core.geometry import (
    Box,
    Geometry,
    Line,
    NamedLine,
    Polygon,
    Transform,
    polygon_is_valid,
    roadway_approach_warnings,
)
from traffic_tracker.core.processor import _Pending, _expire_tracks, _record_crossings
from traffic_tracker.core.project import BundleProject, ClipRecord
from traffic_tracker.core.report import Trip


def _geom(**kwargs) -> Geometry:
    base = dict(
        north=Line(200, 220, 200, 40),
        starts=[NamedLine("N", Line(0, 40, 400, 40))],
        ends=[NamedLine("SB", Line(0, 300, 400, 300))],
    )
    base.update(kwargs)
    return Geometry(400, 400, **base)


def _box(cx: float, cy: float, w: float = 30.0, h: float = 20.0):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def _feed(pending, geom, samples, trips, *, moving_only=True, expire=True):
    for when, boxes, classes in samples:
        _record_crossings(
            boxes,
            geom,
            pending,
            trips,
            "a.mp4",
            when,
            moving_only=moving_only,
            classes=classes,
            confidences=[0.8] * len(boxes),
        )
        if expire:
            _expire_tracks(pending, geom, trips, "a.mp4", when, infer_path=not moving_only)


class TrackingRuleTests(unittest.TestCase):
    def test_stationary_vehicle_keeps_one_id_and_does_not_count(self) -> None:
        geom = _geom()
        pending = _Pending(parked_radius=40.0, diagnostics_enabled=True)
        trips: list[Trip] = []
        box = _box(200, 120)
        samples = [(1.0 + step * 0.5, [box], ["car"]) for step in range(24)]
        _feed(pending, geom, samples, trips)
        self.assertEqual(len(pending.live), 1)
        state = next(iter(pending.live.values()))
        self.assertEqual(state.tid, 1)
        self.assertFalse(state.active)
        self.assertEqual(trips, [])
        self.assertGreater(state.last_observed_s, 12.0)

    def test_queued_vehicle_keeps_id_when_it_starts_moving(self) -> None:
        geom = _geom()
        pending = _Pending(parked_radius=40.0, link_speed=180.0)
        trips: list[Trip] = []
        samples = [(1.0 + step * 0.2, [_box(200, 80)], ["car"]) for step in range(6)]
        samples += [
            (2.4 + step * 0.2, [_box(200, 80 + (step + 1) * 18)], ["truck"])
            for step in range(14)
        ]
        _feed(pending, geom, samples, trips)
        self.assertEqual(list(pending.live), [1])
        self.assertTrue(pending.live[1].active)
        self.assertEqual(pending.live[1].cls, "car")
        self.assertTrue(any(trip.start_destination == "N" for trip in trips) or pending.live[1].start)
        self.assertTrue(all(trip.evidence in {"observed", "gap"} for trip in trips))

    def test_slow_creep_counts_and_jitter_does_not(self) -> None:
        geom = _geom(starts=[NamedLine("N", Line(0, 80, 400, 80))], ends=[NamedLine("SB", Line(0, 240, 400, 240))])
        creep = _Pending(parked_radius=30.0, link_speed=180.0)
        trips: list[Trip] = []
        samples = [(1.0 + step * 0.25, [_box(180, 40 + step * 6)], ["car"]) for step in range(40)]
        _feed(creep, geom, samples, trips)
        self.assertEqual(len(trips), 1)
        self.assertEqual(trips[0].track_id, 1)

        jitter = _Pending(parked_radius=30.0)
        quiet: list[Trip] = []
        samples = [(1.0 + step * 0.25, [_box(180, 40 + (4 if step % 2 else -4))], ["car"]) for step in range(40)]
        _feed(jitter, geom, samples, quiet)
        self.assertEqual(quiet, [])
        self.assertFalse(next(iter(jitter.live.values())).active)

    def test_missing_vehicle_expires_without_prediction_refresh(self) -> None:
        geom = _geom()
        pending = _Pending(link_speed=180.0, diagnostics_enabled=True)
        trips: list[Trip] = []
        _feed(pending, geom, [(1.0, [_box(80, 200)], ["car"]), (1.2, [_box(110, 200)], ["car"])], trips)
        state = pending.live[1]
        observed = state.last_observed_s
        _record_crossings([], geom, pending, trips, "a.mp4", 1.5, moving_only=True)
        self.assertEqual(state.last_observed_s, observed)
        self.assertNotEqual(state.estimate, "predicted")
        _expire_tracks(pending, geom, trips, "a.mp4", observed + MOTION_COAST_S + 0.1, infer_path=False)
        self.assertEqual(pending.live, {})
        self.assertTrue(any(event["action"] == "expired" for event in pending.diagnostics))

    def test_visible_detection_inside_drop_zone_is_kept(self) -> None:
        geom = _geom(occlusions=[Box(70, 160, 220, 240)])
        pending = _Pending(link_speed=180.0, parked_radius=40.0)
        trips: list[Trip] = []
        samples = [
            (1.00, [_box(90, 200)], ["car"]),
            (1.20, [_box(112, 200)], ["car"]),
            (1.40, [_box(134, 200)], ["car"]),
            (1.60, [_box(156, 200)], ["car"]),
            (1.80, [_box(190, 200)], ["car"]),
        ]
        _feed(pending, geom, samples, trips, expire=False)
        self.assertEqual(list(pending.live), [1])
        self.assertGreater(pending.live[1].last_observed_xy[0], 170)

    def test_brief_gap_reacquires_once_and_counts_one_crossing(self) -> None:
        geom = _geom()
        pending = _Pending(link_speed=180.0, parked_radius=40.0)
        trips: list[Trip] = []
        samples = [(1.00, [_box(200, 10)], ["car"]), (1.20, [_box(200, 28)], ["car"])]
        when = 1.80
        y = 100.0
        samples.append((when, [_box(200, y)], ["car"]))
        while y < 320:
            when += 0.2
            y += 22
            samples.append((when, [_box(200, y)], ["car"]))
        _feed(pending, geom, samples, trips, expire=False)
        self.assertEqual(len(trips), 1)
        self.assertEqual(trips[0].track_id, 1)
        self.assertEqual(trips[0].start_destination, "N")
        self.assertEqual(trips[0].end_destination, "SB")
        self.assertEqual(pending.live, {})

    def test_prediction_across_a_gate_does_not_count(self) -> None:
        geom = _geom()
        pending = _Pending(link_speed=220.0, parked_radius=24.0)
        trips: list[Trip] = []
        _feed(
            pending,
            geom,
            [(1.0, [_box(200, 10)], ["car"]), (1.15, [_box(200, 28)], ["car"])],
            trips,
            expire=False,
        )
        _record_crossings([], geom, pending, trips, "a.mp4", 1.4, moving_only=True)
        self.assertEqual(trips, [])
        _expire_tracks(pending, geom, trips, "a.mp4", 30.0, coast_s=0.0, infer_path=True)
        self.assertEqual(trips, [])

    def test_reacquire_on_the_same_side_is_not_a_crossing(self) -> None:
        geom = _geom()
        pending = _Pending(link_speed=180.0, parked_radius=30.0)
        trips: list[Trip] = []
        _feed(
            pending,
            geom,
            [
                (1.00, [_box(200, 10)], ["car"]),
                (1.20, [_box(200, 24)], ["car"]),
                (1.70, [_box(200, 18)], ["car"]),
            ],
            trips,
            expire=False,
        )
        self.assertEqual(trips, [])
        self.assertEqual(list(pending.live), [1])

    def test_stationary_jitter_on_a_gate_does_not_repeat(self) -> None:
        geom = _geom(starts=[NamedLine("N", Line(0, 100, 400, 100))])
        pending = _Pending(parked_radius=40.0)
        trips: list[Trip] = []
        samples = [(1.0 + step * 0.2, [_box(200, 100 + (2 if step % 2 else -2))], ["car"]) for step in range(20)]
        _feed(pending, geom, samples, trips)
        self.assertEqual(trips, [])
        self.assertEqual(len(pending.live), 1)

    def test_neighbors_and_a_turn_stay_one_to_one(self) -> None:
        geom = _geom(occlusions=[Box(140, 150, 220, 230)])
        pending = _Pending(link_speed=180.0, parked_radius=28.0)
        trips: list[Trip] = []
        for step in range(5):
            y = 180 + step * 8
            _record_crossings(
                [_box(80, y), _box(200, 60)],
                geom,
                pending,
                trips,
                "a.mp4",
                1.0 + step * 0.15,
                moving_only=True,
                classes=["car", "car"],
            )
        ids = set(pending.live)
        self.assertEqual(len(ids), 2)
        _record_crossings([_box(200, 60)], geom, pending, trips, "a.mp4", 1.9, moving_only=True, classes=["car"])
        _record_crossings(
            [_box(150, 230), _box(200, 60)],
            geom,
            pending,
            trips,
            "a.mp4",
            2.3,
            moving_only=True,
            classes=["van", "car"],
        )
        self.assertEqual(set(pending.live), ids)
        mover = max(pending.live.values(), key=lambda state: state.xy[0] if state.xy[1] > 100 else -1)
        parked = [state for state in pending.live.values() if state is not mover][0]
        self.assertGreater(mover.xy[1], 180)
        self.assertLess(parked.xy[1], 90)
        self.assertEqual(len(ids), 2)

    def test_roadway_rejects_boats_and_keeps_the_full_frame_without_one(self) -> None:
        road = Polygon([(100, 0), (300, 0), (300, 400), (100, 400)])
        marked = _geom(roadways=[road])
        pending = _Pending(diagnostics_enabled=True)
        trips: list[Trip] = []
        _record_crossings(
            [_box(40, 200), _box(200, 200)],
            marked,
            pending,
            trips,
            "a.mp4",
            1.0,
            moving_only=True,
            classes=["car", "car"],
        )
        self.assertEqual(len(pending.live), 1)
        self.assertAlmostEqual(next(iter(pending.live.values())).xy[0], 200, delta=1)
        self.assertTrue(any(event["reason"] == "outside roadway" for event in pending.diagnostics))

        open_frame = _geom()
        both = _Pending()
        _record_crossings(
            [_box(40, 200), _box(200, 200)],
            open_frame,
            both,
            trips,
            "a.mp4",
            1.0,
            moving_only=True,
            classes=["car", "car"],
        )
        self.assertEqual(len(both.live), 2)

        edge = _Pending()
        _record_crossings([_box(108, 200)], marked, edge, [], "a.mp4", 1.0, moving_only=True, classes=["car"])
        self.assertEqual(len(edge.live), 1)

    def test_roadway_persists_through_scale_transform_and_projects(self) -> None:
        points = [(10.0, 10.0), (80.0, 12.0), (70.0, 90.0), (12.0, 80.0)]
        self.assertTrue(polygon_is_valid(points))
        self.assertFalse(polygon_is_valid([(0, 0), (10, 0), (20, 0)]))
        geom = Geometry(200, 100, roadways=[Polygon(points)], occlusions=[Box(1, 1, 4, 4)])
        loaded = Geometry.from_dict(geom.to_dict())
        self.assertEqual(loaded.roadways[0].points[0], (10.0, 10.0))
        legacy = Geometry.from_dict({"frame_width": 20, "frame_height": 20, "starts": [], "ends": []})
        self.assertEqual(legacy.roadways, [])
        scaled = loaded.scaled_to(100, 50)
        self.assertAlmostEqual(scaled.roadways[0].points[0][0], 5.0)
        shifted = Transform(tx=5, ty=-3).apply_geometry(geom)
        self.assertAlmostEqual(shifted.roadways[0].points[0][0], 15.0)
        self.assertAlmostEqual(shifted.roadways[0].points[0][1], 7.0)
        homo = Transform(homography=[1, 0, 4, 0, 1, -2, 0, 0, 1]).apply_geometry(geom)
        self.assertAlmostEqual(homo.roadways[0].points[1][0], 84.0)
        self.assertAlmostEqual(homo.roadways[0].points[1][1], 10.0)
        clipped = Geometry(
            200,
            200,
            north=Line(100, 150, 100, 40),
            starts=[NamedLine("N", Line(20, 40, 180, 40))],
            ends=[NamedLine("SB", Line(20, 170, 180, 170))],
            roadways=[Polygon([(60, 60), (140, 60), (140, 140), (60, 140)])],
        )
        self.assertTrue(roadway_approach_warnings(clipped))
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "bundle.ttproj.json"
            project = BundleProject(geometry=geom)
            project.clips = [ClipRecord(Path("a.mp4"), status="reference", geometry_override=geom)]
            project.save(path)
            restored = BundleProject.load(path)
            self.assertEqual(restored.geometry.roadways[0].points[2][0], 70.0)
            self.assertEqual(restored.clips[0].geometry_override.roadways[0].points[3][1], 80.0)

    def test_motion_mode_hold_does_not_invent_a_trip(self) -> None:
        geom = _geom()
        pending = _Pending()
        trips: list[Trip] = []
        _record_crossings([_box(200, 20)], geom, pending, trips, "a.mp4", 1.0, moving_only=False)
        _record_crossings([_box(200, 28)], geom, pending, trips, "a.mp4", 1.3, moving_only=False)
        _record_crossings([], geom, pending, trips, "a.mp4", 5.0, moving_only=False)
        self.assertTrue(pending.live[1].holding or pending.live[1].estimate in {"held", "predicted", "observed"})
        self.assertEqual(trips, [])
        _expire_tracks(pending, geom, trips, "a.mp4", 20.0, coast_s=0.0, infer_path=False)
        self.assertEqual(trips, [])
        self.assertLess(DETECTOR_STATIC_EXPIRE_S, MOTION_COAST_S)


if __name__ == "__main__":
    unittest.main()
