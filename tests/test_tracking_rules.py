"""Identity, evidence, drop-zone, and roadway rules. Synthetic boxes only."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from traffic_tracker.config import DETECTOR_STATIC_EXPIRE_S, MOTION_COAST_S, PATH_SMOOTH_ALPHA
from traffic_tracker.core.geometry import (
    Box,
    Geometry,
    Line,
    NamedLine,
    Polygon,
    Transform,
    drop_zone_too_wide,
    polygon_is_valid,
    roadway_approach_warnings,
)
from traffic_tracker.core.processor import (
    ID_COUNTED_BGR,
    ID_LIVE_BGR,
    TrackIssue,
    _Pending,
    _TrackState,
    _expire_tracks,
    _id_label_color,
    _record_crossings,
    _route_label,
    draw_issue_overlay,
    should_emit_preview,
    should_run_detector,
)
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

    def test_three_stacked_boxes_yield_one_id(self) -> None:
        geom = _geom()
        pending = _Pending(link_speed=180.0)
        trips: list[Trip] = []
        boxes = [
            _box(200, 120, 40, 24),
            _box(202, 121, 38, 22),
            _box(198, 119, 44, 28),
        ]
        _record_crossings(
            boxes,
            geom,
            pending,
            trips,
            "a.mp4",
            1.0,
            moving_only=False,
            classes=["car", "truck", "van"],
            confidences=[0.91, 0.62, 0.55],
        )
        self.assertEqual(len(pending.live), 1)
        self.assertEqual(next(iter(pending.live.values())).tid, 1)
        stack = next(issue for issue in pending.issues if issue.kind == "duplicate_stack")
        self.assertIsNotNone(stack.xy)
        self.assertIsNotNone(stack.xyxy)

    def test_issue_overlay_marks_location(self) -> None:
        import numpy as np

        frame = np.zeros((80, 120, 3), dtype=np.uint8)
        issue = TrackIssue(
            kind="drop_zone_attach",
            time_s=1.5,
            tid=3,
            message="#3 reattached",
            xy=(40.0, 40.0),
            from_xy=(10.0, 40.0),
            xyxy=(30.0, 30.0, 50.0, 50.0),
        )
        marked = draw_issue_overlay(frame, [issue], highlight=issue)
        self.assertGreater(int(marked.sum()), 0)
        self.assertGreater(int(marked[40, 40].sum()), 0)

    def test_new_track_refused_when_box_overlaps_live(self) -> None:
        geom = _geom()
        pending = _Pending(link_speed=80.0, parked_radius=40.0)
        trips: list[Trip] = []
        _record_crossings(
            [_box(200, 200, 40, 24)],
            geom,
            pending,
            trips,
            "a.mp4",
            1.0,
            moving_only=False,
            classes=["car"],
        )
        _record_crossings(
            [_box(201, 201, 36, 22)],
            geom,
            pending,
            trips,
            "a.mp4",
            8.0,
            moving_only=False,
            classes=["truck"],
        )
        self.assertEqual(len(pending.live), 1)
        self.assertEqual(next(iter(pending.live.values())).tid, 1)

    def test_path_uses_smoothed_center(self) -> None:
        geom = _geom()
        pending = _Pending(link_speed=400.0)
        trips: list[Trip] = []
        _record_crossings([_box(200, 200)], geom, pending, trips, "a.mp4", 1.0, moving_only=False, classes=["car"])
        _record_crossings([_box(206, 200)], geom, pending, trips, "a.mp4", 1.1, moving_only=False, classes=["car"])
        path = next(iter(pending.live.values())).path
        self.assertGreaterEqual(len(path), 2)
        expected = PATH_SMOOTH_ALPHA * 206.0 + (1.0 - PATH_SMOOTH_ALPHA) * 200.0
        self.assertAlmostEqual(path[-1][0], expected, places=1)
        self.assertLess(path[-1][0], 206.0)

    def test_drop_zone_too_wide_needs_vehicle_size(self) -> None:
        geom = _geom(vehicle_refs=[Box(0, 0, 20, 12)])
        self.assertFalse(drop_zone_too_wide(Box(0, 0, 40, 12), geom))
        self.assertTrue(drop_zone_too_wide(Box(0, 0, 80, 12), geom))
        self.assertFalse(drop_zone_too_wide(Box(0, 0, 80, 12), _geom()))

    def test_roadway_standstill_keeps_id_after_one_detection(self) -> None:
        road = Polygon([(0.0, 0.0), (400.0, 0.0), (400.0, 400.0), (0.0, 400.0)])
        geom = _geom(roadways=[road])
        pending = _Pending(parked_radius=40.0, link_speed=180.0)
        trips: list[Trip] = []
        _record_crossings([_box(200, 200)], geom, pending, trips, "a.mp4", 1.0, moving_only=True, classes=["car"])
        tid = next(iter(pending.live))
        self.assertFalse(pending.live[tid].active)
        for when in (1.5, 2.5, 4.0, 6.0):
            _record_crossings([], geom, pending, trips, "a.mp4", when, moving_only=True)
            _expire_tracks(pending, geom, trips, "a.mp4", when, infer_path=False)
        self.assertIn(tid, pending.live)
        _record_crossings([_box(202, 198)], geom, pending, trips, "a.mp4", 6.4, moving_only=True, classes=["car"])
        self.assertEqual(list(pending.live), [tid])
        self.assertFalse(pending.live[tid].active)
        self.assertEqual(trips, [])

    def test_queued_vehicle_keeps_id_when_it_starts_moving(self) -> None:
        geom = _geom(starts=[NamedLine("N", Line(0, 90, 400, 90))], ends=[NamedLine("SB", Line(0, 300, 400, 300))])
        pending = _Pending(parked_radius=40.0, link_speed=180.0)
        trips: list[Trip] = []
        samples = [(1.0 + step * 0.2, [_box(200, 20)], ["car"]) for step in range(6)]
        samples += [
            (2.4 + step * 0.2, [_box(200, 20 + (step + 1) * 18)], ["truck"])
            for step in range(18)
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

    def test_drop_zone_lead_pickup_stitches_id_and_path(self) -> None:
        geom = _geom(
            occlusions=[Box(120, 170, 300, 230)],
            prediction_reach_px=64.0,
        )
        pending = _Pending(link_speed=60.0, parked_radius=40.0)
        trips: list[Trip] = []
        samples = []
        x = 40.0
        when = 1.0
        for _step in range(8):
            samples.append((when, [_box(x, 200)], ["car"]))
            x += 16.0
            when += 0.2
        _feed(pending, geom, samples, trips, expire=False)
        self.assertEqual(list(pending.live), [1])
        self.assertTrue(pending.live[1].active)
        last_x = pending.live[1].last_observed_xy[0]
        _record_crossings([], geom, pending, trips, "a.mp4", when, moving_only=True)
        self.assertIsNotNone(pending.live[1].pickup_xy)
        pickup_x = pending.live[1].pickup_xy[0]
        self.assertGreater(pickup_x, last_x + 20)
        _record_crossings(
            [_box(pickup_x - 4.0, 200)],
            geom,
            pending,
            trips,
            "a.mp4",
            when + 0.2,
            moving_only=True,
            classes=["car"],
        )
        self.assertEqual(list(pending.live), [1])
        self.assertIsNone(pending.live[1].pickup_xy)
        path = pending.live[1].path
        self.assertGreater(len(path), 8)
        steps = [
            ((path[i][0] - path[i - 1][0]) ** 2 + (path[i][1] - path[i - 1][1]) ** 2) ** 0.5
            for i in range(1, len(path))
        ]
        self.assertLess(max(steps), 40.0)
        self.assertGreater(path[-1][0], last_x)
        self.assertEqual(trips, [])
        attach = next(issue for issue in pending.issues if issue.kind == "drop_zone_attach")
        self.assertIsNotNone(attach.xy)
        self.assertIsNotNone(attach.from_xy)

    def test_trace_continues_through_exit_gate(self) -> None:
        geom = _geom()
        pending = _Pending(link_speed=180.0, parked_radius=24.0)
        trips: list[Trip] = []
        samples = []
        y = 20.0
        when = 1.0
        while y < 360:
            samples.append((when, [_box(200, y)], ["car"]))
            y += 24.0
            when += 0.2
        _feed(pending, geom, samples, trips, expire=False)
        self.assertEqual(len(trips), 1)
        self.assertEqual(list(pending.live), [1])
        self.assertGreater(pending.live[1].path[-1][1], 300.0)
        self.assertGreater(pending.live[1].xy[1], 300.0)
        self.assertEqual(trips[0].track_id, 1)

    def test_drop_zone_flicker_does_not_spike_through_lead(self) -> None:
        geom = _geom(
            occlusions=[Box(120, 170, 300, 230)],
            prediction_reach_px=80.0,
        )
        pending = _Pending(link_speed=60.0, parked_radius=40.0)
        trips: list[Trip] = []
        samples = []
        x = 40.0
        when = 1.0
        for _step in range(8):
            samples.append((when, [_box(x, 200)], ["car"]))
            x += 16.0
            when += 0.2
        _feed(pending, geom, samples, trips, expire=False)
        last_x = pending.live[1].last_observed_xy[0]
        _record_crossings([], geom, pending, trips, "a.mp4", when, moving_only=True)
        pickup_x = pending.live[1].pickup_xy[0]
        self.assertGreater(pickup_x, last_x + 30)
        _record_crossings(
            [_box(last_x + 6.0, 200)],
            geom,
            pending,
            trips,
            "a.mp4",
            when + 0.2,
            moving_only=True,
            classes=["car"],
        )
        path_xs = [point[0] for point in pending.live[1].path]
        self.assertLess(max(path_xs), pickup_x - 10)

    def test_lead_pickup_across_a_gate_does_not_count(self) -> None:
        geom = _geom(
            occlusions=[Box(150, 250, 250, 360)],
            prediction_reach_px=80.0,
            starts=[NamedLine("N", Line(0, 40, 400, 40))],
            ends=[NamedLine("SB", Line(0, 300, 400, 300))],
        )
        pending = _Pending(link_speed=80.0, parked_radius=24.0)
        trips: list[Trip] = []
        samples = [(1.0 + step * 0.15, [_box(200, 40 + step * 18)], ["car"]) for step in range(10)]
        _feed(pending, geom, samples, trips, expire=False)
        self.assertTrue(pending.live)
        last_t = samples[-1][0]
        _record_crossings([], geom, pending, trips, "a.mp4", last_t + 0.2, moving_only=True)
        _record_crossings([], geom, pending, trips, "a.mp4", last_t + 0.4, moving_only=True)
        self.assertEqual(trips, [])
        _expire_tracks(pending, geom, trips, "a.mp4", last_t + 8.0, coast_s=0.0, infer_path=True)
        self.assertEqual(trips, [])

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
        self.assertEqual(list(pending.live), [1])
        self.assertGreater(pending.live[1].path[-1][1], 300.0)

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


class IdLabelTests(unittest.TestCase):
    def test_id_is_red_until_counted_then_green(self) -> None:
        live = _TrackState(tid=1, xy=(0.0, 0.0), first_xy=(0.0, 0.0), first_time_s=0.0, last_time_s=0.0)
        self.assertEqual(_id_label_color(None), ID_LIVE_BGR)
        self.assertEqual(_id_label_color(live), ID_LIVE_BGR)
        live.had_gate_trip = True
        self.assertEqual(_id_label_color(live), ID_COUNTED_BGR)

    def test_id_labels_are_drawn_after_traces(self) -> None:
        import inspect

        from traffic_tracker.core.processor import _draw_motion_preview

        source = inspect.getsource(_draw_motion_preview)
        self.assertGreater(source.rfind("_draw_id_label"), source.rfind("_draw_paths"))

    def test_counted_word_is_not_drawn(self) -> None:
        import inspect

        from traffic_tracker.core.processor import _draw_id_label

        self.assertNotIn('"counted"', inspect.getsource(_draw_id_label))

    def test_route_label_shows_entry_then_exit(self) -> None:
        live = _TrackState(tid=1, xy=(0.0, 0.0), first_xy=(0.0, 0.0), first_time_s=0.0, last_time_s=0.0)
        self.assertEqual(_route_label(live), "")
        live.entry_origin = "N"
        self.assertEqual(_route_label(live), "N→")
        live.exit_bound = "SB"
        self.assertEqual(_route_label(live), "N→SB")

    def test_live_id_keeps_route_after_trip(self) -> None:
        geom = Geometry(
            100,
            100,
            north=Line(50, 55, 50, 10),
            starts=[NamedLine("N", Line(0, 40, 100, 40))],
            ends=[NamedLine("SB", Line(0, 70, 100, 70))],
        )
        pending = _Pending()
        trips: list[Trip] = []
        _record_crossings([_box(50, 10, 20, 20)], geom, pending, trips, "a.mp4", 1.0, moving_only=False)
        _record_crossings([_box(50, 55, 20, 20)], geom, pending, trips, "a.mp4", 1.5, moving_only=False)
        state = pending.live[1]
        self.assertEqual(_route_label(state), "N→")
        self.assertEqual(trips, [])
        _record_crossings([_box(50, 90, 20, 20)], geom, pending, trips, "a.mp4", 2.0, moving_only=False)
        self.assertEqual(len(trips), 1)
        state = pending.live[1]
        self.assertEqual(_route_label(state), "N→SB")
        self.assertTrue(state.had_gate_trip)


class ProcessSpeedHelpersTests(unittest.TestCase):
    def test_should_run_detector_stride(self) -> None:
        self.assertTrue(should_run_detector(1, 1))
        self.assertTrue(should_run_detector(2, 1))
        self.assertTrue(should_run_detector(1, 2))
        self.assertFalse(should_run_detector(2, 2))
        self.assertTrue(should_run_detector(3, 2))
        self.assertTrue(should_run_detector(1, 3))
        self.assertFalse(should_run_detector(2, 3))
        self.assertFalse(should_run_detector(3, 3))
        self.assertTrue(should_run_detector(4, 3))
        self.assertTrue(should_run_detector(1, 0))

    def test_should_emit_preview(self) -> None:
        self.assertFalse(should_emit_preview(1, 0))
        self.assertFalse(should_emit_preview(15, 0))
        self.assertTrue(should_emit_preview(1, 15))
        self.assertFalse(should_emit_preview(14, 15))
        self.assertTrue(should_emit_preview(15, 15))
        self.assertTrue(should_emit_preview(30, 30))
        self.assertFalse(should_emit_preview(29, 30))


if __name__ == "__main__":
    unittest.main()
