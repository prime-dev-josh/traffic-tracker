"""Training session round-trip and size-prior merge."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile
import unittest

import numpy as np

from traffic_tracker.config import DETECTOR_CONF, DETECTOR_IMGSZ, TRAIN_IMGSZ, TRAIN_MIN_BOXES, TRAIN_MIN_FRAMES
from traffic_tracker.core.detector import boxes_from_detector, keep_detector_class, resolve_detector_weights, weights_for_source
from traffic_tracker.core.settings import (
    AppSettings,
    apply_precision_preset,
    clamp_detect_stride,
    clamp_detector_conf,
    clamp_detector_imgsz,
    clamp_detector_iou,
    clamp_preview_every,
    load_settings,
    save_settings,
)
from traffic_tracker.core.geometry import Box, Geometry, area_limits_from_areas, merge_area_limits
from traffic_tracker.core.trainer import train_visdrone_detector, yolo_train_device
from traffic_tracker.core.training import (
    LabeledFrame,
    TrainingSession,
    build_yolo_dataset,
    collect_dataset_items,
    delete_session,
    export_training_archive,
    has_trained_weights,
    import_training_archive,
    list_sessions,
    load_session,
    load_size_prior,
    prior_areas,
    rebuild_size_prior,
    save_session,
    trained_weights_path,
    training_inventory,
    visdrone_weights_path,
)


class TrainingSessionTests(unittest.TestCase):
    def test_save_load_and_yolo_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            frame = np.zeros((100, 200, 3), dtype=np.uint8)
            session = TrainingSession(
                video_path="clip.mp4",
                video_stem="clip",
                frames=[
                    LabeledFrame(
                        frame_index=12,
                        image_name="frame_000012.jpg",
                        width=200,
                        height=100,
                        boxes=[Box(20, 10, 60, 50)],
                        time_s=0.4,
                    )
                ],
            )
            folder = save_session(session, {12: frame}, base=base)
            self.assertTrue((folder / "images" / "frame_000012.jpg").is_file())
            label = (folder / "labels" / "frame_000012.txt").read_text(encoding="utf-8").strip()
            cls, cx, cy, bw, bh = label.split()
            self.assertEqual(cls, "0")
            self.assertAlmostEqual(float(cx), 0.2, places=5)
            self.assertAlmostEqual(float(cy), 0.3, places=5)
            self.assertAlmostEqual(float(bw), 0.2, places=5)
            self.assertAlmostEqual(float(bh), 0.4, places=5)

            loaded = load_session("clip", base=base)
            self.assertEqual(loaded.video_stem, "clip")
            self.assertEqual(len(loaded.frames), 1)
            self.assertEqual(loaded.frames[0].boxes[0].area(), 1600)
            self.assertEqual(len(list_sessions(base)), 1)

            prior = load_size_prior(base)
            self.assertIsNotNone(prior)
            self.assertEqual(prior["areas"], [1600])
            expected = area_limits_from_areas([1600])
            self.assertEqual((prior["min_area"], prior["max_area"]), expected)

    def test_delete_session_clears_prior(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            session = TrainingSession(
                video_path="a.mp4",
                video_stem="a",
                frames=[
                    LabeledFrame(1, "frame_000001.jpg", 80, 80, [Box(0, 0, 20, 20)]),
                ],
            )
            save_session(session, {1: np.zeros((80, 80, 3), dtype=np.uint8)}, base=base)
            self.assertTrue(prior_areas(base))
            delete_session("a", base=base)
            self.assertEqual(list_sessions(base), [])
            self.assertIsNone(load_size_prior(base))
            self.assertFalse(has_trained_weights(base))

    def test_empty_rebuild_removes_prior(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self.assertIsNone(rebuild_size_prior(base))
            self.assertIsNone(load_size_prior(base))


class SizePriorMergeTests(unittest.TestCase):
    def test_training_areas_union_with_masks(self) -> None:
        geom = Geometry(200, 200, vehicle_refs=[Box(0, 0, 20, 20)])
        mask_only = geom.vehicle_area_limits()
        merged = geom.vehicle_area_limits(extra_areas=[10000])
        self.assertIsNotNone(mask_only)
        self.assertIsNotNone(merged)
        self.assertEqual(merged[0], mask_only[0])
        self.assertGreater(merged[1], mask_only[1])

    def test_training_only_sets_band(self) -> None:
        geom = Geometry(200, 200)
        self.assertIsNone(geom.vehicle_area_limits())
        limits = geom.vehicle_area_limits(extra_areas=[400, 1600])
        self.assertEqual(limits, area_limits_from_areas([400, 1600]))

    def test_merge_area_limits(self) -> None:
        self.assertIsNone(merge_area_limits(None, None))
        self.assertEqual(merge_area_limits((100, 500), (80, 900)), (80, 900))


class SettingsTests(unittest.TestCase):
    def test_round_trip_and_trained_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            save_settings(
                AppSettings(
                    export_dir=str(base / "out"),
                    warn_onedrive=False,
                    detector_mode="trained",
                    detector_source="visdrone",
                    theme="light",
                    detector_conf=0.4,
                    detector_imgsz=640,
                    detect_stride=2,
                    preview_every=0,
                    infer_trips_from_traces=False,
                ),
                base=base,
            )
            loaded = load_settings(base)
            self.assertEqual(loaded.export_dir, str(base / "out"))
            self.assertFalse(loaded.warn_onedrive)
            self.assertEqual(loaded.detector_mode, "trained")
            self.assertEqual(loaded.detector_source, "visdrone")
            self.assertEqual(loaded.theme, "light")
            self.assertAlmostEqual(loaded.detector_conf, 0.4)
            self.assertEqual(loaded.detector_imgsz, 640)
            self.assertEqual(loaded.detect_stride, 2)
            self.assertEqual(loaded.preview_every, 0)
            self.assertFalse(loaded.infer_trips_from_traces)

    def test_legacy_infer_trips_follows_detector_mode(self) -> None:
        from traffic_tracker.config import SETTINGS_FILE_NAME

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            path = base / SETTINGS_FILE_NAME
            path.write_text('{"detector_mode": "trained", "export_dir": "out"}', encoding="utf-8")
            self.assertFalse(load_settings(base).infer_trips_from_traces)
            path.write_text('{"detector_mode": "motion", "export_dir": "out"}', encoding="utf-8")
            self.assertTrue(load_settings(base).infer_trips_from_traces)

    def test_detector_conf_clamps(self) -> None:
        self.assertEqual(clamp_detector_conf(0.4), 0.4)
        self.assertEqual(clamp_detector_conf(-1), 0.15)
        self.assertEqual(clamp_detector_conf(2), 0.60)
        self.assertEqual(clamp_detector_conf("nope"), DETECTOR_CONF)
        self.assertEqual(clamp_detector_iou(0.2), 0.2)
        self.assertEqual(clamp_detector_iou(2), 0.70)
        self.assertEqual(clamp_detector_imgsz(640), 640)
        self.assertEqual(clamp_detector_imgsz(700), 640)
        self.assertEqual(clamp_detector_imgsz(2000), 1280)
        self.assertEqual(clamp_detector_imgsz("nope"), DETECTOR_IMGSZ)
        self.assertEqual(clamp_detect_stride(2), 2)
        self.assertEqual(clamp_detect_stride(0), 1)
        self.assertEqual(clamp_detect_stride(9), 3)
        self.assertEqual(clamp_preview_every(0), 0)
        self.assertEqual(clamp_preview_every(18), 15)
        self.assertEqual(clamp_preview_every(90), 60)

    def test_precision_preset_fills_sliders(self) -> None:
        tuned = apply_precision_preset(AppSettings(), "strict")
        self.assertEqual(tuned.precision_preset, "strict")
        self.assertAlmostEqual(tuned.detector_conf, 0.45)
        self.assertAlmostEqual(tuned.detector_iou, 0.28)
        self.assertGreater(tuned.parked_radius, AppSettings().parked_radius)
        self.assertLess(tuned.link_speed, AppSettings().link_speed)
        custom = apply_precision_preset(tuned, "nope")
        self.assertEqual(custom.precision_preset, "custom")

    def test_precision_preset_leaves_speed_knobs(self) -> None:
        settings = AppSettings(detector_imgsz=640, detect_stride=3, preview_every=0)
        tuned = apply_precision_preset(settings, "strict")
        self.assertEqual(tuned.detector_imgsz, 640)
        self.assertEqual(tuned.detect_stride, 3)
        self.assertEqual(tuned.preview_every, 0)


class DetectorWeightsTests(unittest.TestCase):
    def test_explicit_weights_win(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            weights = Path(tmp) / "custom.pt"
            weights.write_bytes(b"pt")
            self.assertEqual(resolve_detector_weights(str(weights), Path(tmp)), weights)

    def test_trained_mode_uses_visdrone_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            vis = visdrone_weights_path(base)
            best = trained_weights_path(base)
            vis.parent.mkdir(parents=True, exist_ok=True)
            vis.write_bytes(b"vis")
            best.write_bytes(b"best")
            self.assertEqual(weights_for_source("visdrone", None, base), vis)
            self.assertEqual(weights_for_source("site", None, base), vis)
            self.assertEqual(weights_for_source("drone", None, base), vis)
            self.assertEqual(resolve_detector_weights(None, base), vis)

    def test_legacy_sources_are_loaded_as_visdrone(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            save_settings(
                AppSettings(detector_mode="trained", detector_source="site"),
                base=base,
            )
            loaded = load_settings(base)
            self.assertEqual(loaded.detector_source, "visdrone")

    def test_trainer_exposes_visdrone_only(self) -> None:
        import inspect
        import traffic_tracker.core.trainer as trainer

        self.assertTrue(hasattr(trainer, "train_visdrone_detector"))
        self.assertFalse(hasattr(trainer, "train_site_detector"))
        self.assertFalse(hasattr(trainer, "train_drone_vehicles_detector"))
        self.assertIn("VisDrone.yaml", inspect.getsource(train_visdrone_detector))

    def test_predict_uses_detector_image_size(self) -> None:
        self.assertEqual(TRAIN_IMGSZ, 1280)
        self.assertEqual(DETECTOR_IMGSZ, 960)
        self.assertIn("DETECTOR_IMGSZ", boxes_from_detector.__code__.co_names)
        self.assertNotIn("TRAIN_IMGSZ", boxes_from_detector.__code__.co_names)

    def test_predict_passes_device_quantize_and_imgsz(self) -> None:
        model = MagicMock()
        model.names = {2: "car"}
        result = MagicMock()
        result.boxes = None
        model.predict.return_value = [result]
        frame = np.zeros((80, 80, 3), dtype=np.uint8)
        with patch("traffic_tracker.core.detector.yolo_runtime", return_value=(0, 16)) as runtime:
            boxes_from_detector(
                model,
                frame,
                min_area=1,
                max_area=1e9,
                imgsz=640,
                device="cuda",
            )
        runtime.assert_called_once_with("cuda")
        kwargs = model.predict.call_args.kwargs
        self.assertEqual(kwargs["imgsz"], 640)
        self.assertEqual(kwargs["device"], 0)
        self.assertEqual(kwargs["quantize"], 16)
        self.assertNotIn("half", kwargs)

    def test_visdrone_keeps_vehicles_drops_people(self) -> None:
        names = {
            0: "pedestrian",
            1: "people",
            2: "bicycle",
            3: "car",
            4: "van",
            5: "truck",
            6: "tricycle",
            7: "awning-tricycle",
            8: "bus",
            9: "motor",
        }
        self.assertTrue(keep_detector_class(names, 3))
        self.assertTrue(keep_detector_class(names, 4))
        self.assertTrue(keep_detector_class(names, 5))
        self.assertTrue(keep_detector_class(names, 8))
        self.assertTrue(keep_detector_class(names, 9))
        self.assertFalse(keep_detector_class(names, 0))
        self.assertFalse(keep_detector_class(names, 1))
        self.assertFalse(keep_detector_class(names, 2))
        self.assertFalse(keep_detector_class(names, 6))
        self.assertFalse(keep_detector_class(names, 7))
        self.assertFalse(keep_detector_class({8: "boat", 2: "car"}, 8))
        self.assertFalse(keep_detector_class({8: "ship", 2: "car"}, 8))
        self.assertFalse(keep_detector_class({8: "speedboat", 2: "car"}, 8))
        self.assertFalse(keep_detector_class({8: "yacht", 2: "car"}, 8))

    def test_site_single_class_keeps_all(self) -> None:
        self.assertTrue(keep_detector_class({0: "vehicle"}, 0))


class YoloDatasetTests(unittest.TestCase):
    def _session(
        self,
        stem: str,
        frame_count: int,
        boxes_per_frame: int,
    ) -> tuple[TrainingSession, dict[int, np.ndarray]]:
        frames = []
        images: dict[int, np.ndarray] = {}
        boxes = [Box(2, 2, 12, 12) for _ in range(boxes_per_frame)]
        for index in range(frame_count):
            frames.append(
                LabeledFrame(
                    frame_index=index,
                    image_name=f"frame_{index:06d}.jpg",
                    width=40,
                    height=40,
                    boxes=list(boxes),
                )
            )
            images[index] = np.zeros((40, 40, 3), dtype=np.uint8)
        return TrainingSession(f"{stem}.mp4", stem, frames), images

    def test_build_dataset_splits_and_keeps_class_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            session, images = self._session("alpha", TRAIN_MIN_FRAMES, 2)
            save_session(session, images, base=base)
            other, other_images = self._session("beta", 4, 2)
            save_session(other, other_images, base=base)
            result = build_yolo_dataset(base)
            self.assertTrue(result["yaml_path"].is_file())
            yaml_text = result["yaml_path"].read_text(encoding="utf-8")
            self.assertIn("0: car", yaml_text)
            self.assertIn("4: motorcycle", yaml_text)
            self.assertNotIn("0: vehicle", yaml_text)
            self.assertIn("images/train", yaml_text)
            self.assertEqual(result["frame_count"], TRAIN_MIN_FRAMES + 4)
            self.assertEqual(result["box_count"], (TRAIN_MIN_FRAMES + 4) * 2)
            self.assertGreater(result["train_count"], 0)
            self.assertGreater(result["val_count"], 0)
            self.assertEqual(result["train_count"] + result["val_count"], result["frame_count"])
            train_images = list((result["root"] / "images" / "train").glob("*.jpg"))
            val_images = list((result["root"] / "images" / "val").glob("*.jpg"))
            self.assertEqual(len(train_images) + len(val_images), result["frame_count"])
            names = {path.stem for path in train_images + val_images}
            self.assertTrue(any(name.startswith("alpha__") for name in names))
            self.assertTrue(any(name.startswith("beta__") for name in names))
            sample = next((result["root"] / "labels").rglob("*.txt"))
            line = sample.read_text(encoding="utf-8").strip().splitlines()[0]
            self.assertTrue(line.startswith("0 "))

    def test_too_few_frames_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            session, images = self._session("tiny", TRAIN_MIN_FRAMES - 1, 4)
            save_session(session, images, base=base)
            with self.assertRaises(ValueError) as raised:
                build_yolo_dataset(base)
            self.assertIn(str(TRAIN_MIN_FRAMES), str(raised.exception))

    def test_too_few_boxes_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            session, images = self._session("sparse", TRAIN_MIN_FRAMES, 1)
            save_session(session, images, base=base)
            self.assertLess(TRAIN_MIN_FRAMES, TRAIN_MIN_BOXES)
            with self.assertRaises(ValueError) as raised:
                build_yolo_dataset(base)
            self.assertIn(str(TRAIN_MIN_BOXES), str(raised.exception))

    def test_collect_skips_missing_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            session = TrainingSession(
                "ghost.mp4",
                "ghost",
                [LabeledFrame(1, "frame_000001.jpg", 40, 40, [Box(0, 0, 10, 10)])],
            )
            save_session(session, {}, base=base)
            self.assertEqual(collect_dataset_items(base), [])

    def test_yolo_train_device_maps_cuda(self) -> None:
        self.assertEqual(yolo_train_device("cuda"), 0)
        self.assertEqual(yolo_train_device("cpu"), "cpu")
        self.assertEqual(yolo_train_device("mps"), "mps")


class TrainingArchiveTests(unittest.TestCase):
    def _session(self, stem: str) -> TrainingSession:
        return TrainingSession(
            video_path=f"{stem}.mp4",
            video_stem=stem,
            frames=[LabeledFrame(3, "frame_000003.jpg", 80, 80, [Box(0, 0, 20, 20)])],
        )

    def test_export_import_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src"
            dest = Path(tmp) / "dest"
            save_session(self._session("clip"), {3: np.zeros((80, 80, 3), dtype=np.uint8)}, base=src)
            archive = export_training_archive(Path(tmp) / "pack.zip", base=src)
            self.assertTrue(archive.is_file())
            result = import_training_archive(archive, base=dest)
            self.assertEqual(result["imported"], 1)
            inventory = training_inventory(dest)
            self.assertEqual(inventory["session_count"], 1)
            self.assertEqual(inventory["box_count"], 1)
            loaded = load_session("clip", base=dest)
            self.assertEqual(loaded.frames[0].boxes[0].area(), 400)

    def test_import_skips_existing_unless_replace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src"
            dest = Path(tmp) / "dest"
            frame = {3: np.zeros((80, 80, 3), dtype=np.uint8)}
            save_session(self._session("clip"), frame, base=src)
            save_session(self._session("clip"), frame, base=dest)
            archive = export_training_archive(Path(tmp) / "pack.zip", base=src)
            skipped = import_training_archive(archive, base=dest, replace=False)
            self.assertEqual(skipped["skipped"], 1)
            replaced = import_training_archive(archive, base=dest, replace=True)
            self.assertEqual(replaced["replaced"], 1)


if __name__ == "__main__":
    unittest.main()
