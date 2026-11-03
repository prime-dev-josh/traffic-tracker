import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from traffic_tracker.core.geometry import Geometry, Line, NamedLine, Transform, transform_from_matches
from traffic_tracker.core.project import BundleProject, ClipRecord
from traffic_tracker.core.video import format_duration, is_onedrive_path


def _geom(width: int = 200, height: int = 100) -> Geometry:
    return Geometry(
        width,
        height,
        north=Line(width / 2, height * 0.9, width / 2, height * 0.1),
        starts=[NamedLine("N", Line(10, 20, 30, 40))],
        ends=[NamedLine("SB", Line(50, 60, 70, 80))],
    )


class TransformTests(unittest.TestCase):
    def test_identity_leaves_points(self) -> None:
        geom = Transform().apply_geometry(_geom())
        self.assertEqual(geom.starts[0].line.x1, 10)
        self.assertEqual(geom.starts[0].line.y1, 20)

    def test_translate(self) -> None:
        geom = Transform(tx=5, ty=-3).apply_geometry(_geom())
        self.assertAlmostEqual(geom.starts[0].line.x1, 15)
        self.assertAlmostEqual(geom.starts[0].line.y1, 17)

    def test_scale_about_center(self) -> None:
        # Center is (100, 50). Point (10, 20): (10-100)*2 + 100 = -80; (20-50)*2+50 = -10
        geom = Transform(scale=2).apply_geometry(_geom())
        self.assertAlmostEqual(geom.starts[0].line.x1, -80)
        self.assertAlmostEqual(geom.starts[0].line.y1, -10)

    def test_homography_identity(self) -> None:
        t = Transform(homography=[1, 0, 0, 0, 1, 0, 0, 0, 1])
        x, y = t.apply_point(12, 8, 0, 0)
        self.assertAlmostEqual(x, 12)
        self.assertAlmostEqual(y, 8)

    def test_matches_translation(self) -> None:
        src = [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)]
        dst = [(5.0, 7.0), (15.0, 7.0), (5.0, 17.0)]
        t = transform_from_matches(src, dst)
        x, y = t.apply_point(0, 0, 0, 0)
        self.assertAlmostEqual(x, 5, places=4)
        self.assertAlmostEqual(y, 7, places=4)

    def test_roundtrip_transform_dict(self) -> None:
        t = Transform(tx=1, ty=2, scale=1.5, rotation_deg=15)
        loaded = Transform.from_dict(t.to_dict())
        self.assertEqual(loaded.tx, 1)
        self.assertEqual(loaded.scale, 1.5)


class ProjectTests(unittest.TestCase):
    def test_ensure_clips_marks_reference(self) -> None:
        project = BundleProject()
        project.ensure_clips([Path("a.mp4"), Path("b.mp4")])
        self.assertEqual(project.clips[0].status, "reference")
        self.assertEqual(project.clips[1].status, "needs_align")

    def test_ready_jobs_skips_unaligned(self) -> None:
        geom = Geometry(100, 100, starts=[NamedLine("N", Line(0, 0, 10, 0))], ends=[NamedLine("SB", Line(0, 10, 10, 10))])
        project = BundleProject(geometry=geom)
        project.clips = [
            ClipRecord(Path("a.mp4"), status="reference"),
            ClipRecord(Path("b.mp4"), status="needs_align"),
        ]
        jobs = project.ready_jobs(geom)
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0][0], Path("a.mp4"))

    def test_project_file_roundtrip(self) -> None:
        geom = Geometry(
            10,
            10,
            north=Line(5, 9, 5, 1),
            starts=[NamedLine("N", Line(1, 2, 3, 4))],
            ends=[NamedLine("SB", Line(5, 6, 7, 8))],
        )
        project = BundleProject(reference=Path("a.mp4"), geometry=geom)
        project.clips = [ClipRecord(Path("a.mp4"), status="reference")]
        with TemporaryDirectory() as raw:
            path = Path(raw) / "bundle.ttproj.json"
            project.save(path)
            loaded = BundleProject.load(path)
        self.assertEqual(loaded.clips[0].status, "reference")
        self.assertEqual(loaded.geometry.starts[0].name, "N")
        self.assertEqual(loaded.geometry.ends[0].name, "SB")


class VideoHelperTests(unittest.TestCase):
    def test_format_duration(self) -> None:
        self.assertEqual(format_duration(90), "1m 30s")
        self.assertTrue(format_duration(3700).startswith("1h"))

    def test_onedrive_detect(self) -> None:
        self.assertTrue(is_onedrive_path(Path("C:/Users/x/OneDrive - Firm/clip.mp4")))
        self.assertFalse(is_onedrive_path(Path("D:/media/clip.mp4")))


if __name__ == "__main__":
    unittest.main()
