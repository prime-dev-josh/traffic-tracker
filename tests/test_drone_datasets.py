"""VisDrone / UAVDT merge into the 5-class drone-vehicles YOLO set."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from traffic_tracker.core.drone_datasets import (
    build_drone_vehicles_dataset,
    convert_uavdt_labels,
    convert_visdrone_vehicle_labels,
)
from traffic_tracker.config import drone_vehicles_data_yaml


def _write_jpg(path: Path, width: int = 40, height: int = 40) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.zeros((height, width, 3), dtype=np.uint8))


class VisDroneConvertTests(unittest.TestCase):
    def test_native_visdrone_keeps_vehicles_drops_people(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "VisDrone2019-DET-train"
            images = root / "images"
            anns = root / "annotations"
            anns.mkdir(parents=True, exist_ok=True)
            _write_jpg(images / "000001.jpg", 100, 80)
            # x,y,w,h,score,category,trunc,occ — 1 people, 4 car, 10 motor
            (anns / "000001.txt").write_text(
                "1,1,10,10,1,1,0,0\n"
                "20,10,20,10,1,4,0,0\n"
                "50,20,10,8,1,10,0,0\n",
                encoding="utf-8",
            )
            dest = Path(tmp) / "out"
            stats = convert_visdrone_vehicle_labels(Path(tmp), dest)
            self.assertEqual(stats["images"], 1)
            self.assertEqual(stats["boxes"], 2)
            label = next((dest / "labels").rglob("*.txt")).read_text(encoding="utf-8").splitlines()
            classes = {int(line.split()[0]) for line in label}
            self.assertEqual(classes, {0, 4})

    def test_yolo_visdrone_remaps_classes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            split = Path(tmp) / "train"
            _write_jpg(split / "images" / "a.jpg")
            (split / "labels").mkdir(parents=True, exist_ok=True)
            (split / "labels" / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n3 0.2 0.2 0.1 0.1\n", encoding="utf-8")
            dest = Path(tmp) / "out"
            stats = convert_visdrone_vehicle_labels(Path(tmp), dest)
            self.assertEqual(stats["train"], 1)
            self.assertEqual(stats["boxes"], 1)
            text = next((dest / "labels").rglob("*.txt")).read_text(encoding="utf-8")
            self.assertTrue(text.startswith("0 "))


class UavdtConvertTests(unittest.TestCase):
    def test_mot_gt_maps_car_truck_bus_and_subsamples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            seq = Path(tmp) / "M0101"
            for frame in range(1, 5):
                _write_jpg(seq / f"img{frame:06d}.jpg", 100, 80)
            gt = Path(tmp) / "GT" / "M0101_gt_det.txt"
            gt.parent.mkdir(parents=True, exist_ok=True)
            # frame,id,x,y,w,h,conf,class  (1 car, 2 truck, 3 bus)
            gt.write_text(
                "1,1,10,10,20,10,1,1,0,0\n"
                "2,1,12,10,20,10,1,1,0,0\n"
                "3,2,30,20,24,12,1,2,0,0\n"
                "4,3,40,20,30,14,1,3,0,0\n",
                encoding="utf-8",
            )
            dest = Path(tmp) / "out"
            stats = convert_uavdt_labels(Path(tmp), dest, frame_stride=2)
            self.assertGreaterEqual(stats["images"], 1)
            labels = list((dest / "labels").rglob("*.txt"))
            self.assertTrue(labels)
            classes = set()
            for path in labels:
                for line in path.read_text(encoding="utf-8").splitlines():
                    classes.add(int(line.split()[0]))
            self.assertTrue(classes <= {0, 2, 3})
            self.assertIn(0, classes)

    def test_dataset_ninja_supervisely_maps_and_skips_sot_vehicle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            split = Path(tmp) / "UAVDT" / "train"
            img = split / "img"
            ann = split / "ann"
            img.mkdir(parents=True)
            ann.mkdir(parents=True)
            _write_jpg(img / "M0101_img000001.jpg", 100, 80)
            _write_jpg(img / "M0101_img000002.jpg", 100, 80)
            (ann / "M0101_img000001.jpg.json").write_text(
                '{"size":{"height":80,"width":100},"objects":['
                '{"classTitle":"car","points":{"exterior":[[10,10],[30,20]]}},'
                '{"classTitle":"truck","points":{"exterior":[[40,20],[64,32]]}},'
                '{"classTitle":"vehicle","points":{"exterior":[[1,1],[9,9]]}}'
                "]}",
                encoding="utf-8",
            )
            (ann / "M0101_img000002.jpg.json").write_text(
                '{"size":{"height":80,"width":100},"objects":['
                '{"classTitle":"bus","points":{"exterior":[[40,20],[70,34]]}}'
                "]}",
                encoding="utf-8",
            )
            dest = Path(tmp) / "out"
            stats = convert_uavdt_labels(Path(tmp), dest, frame_stride=2)
            self.assertEqual(stats["train"], 1)
            text = next((dest / "labels").rglob("*.txt")).read_text(encoding="utf-8")
            classes = {int(line.split()[0]) for line in text.splitlines()}
            self.assertEqual(classes, {0, 2})

    def test_yolo_layout_remaps_truck_and_bus(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_jpg(root / "train" / "images" / "a.jpg")
            labels = root / "train" / "labels"
            labels.mkdir(parents=True, exist_ok=True)
            (labels / "a.txt").write_text("1 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            dest = Path(tmp) / "out"
            stats = convert_uavdt_labels(root, dest, frame_stride=1)
            self.assertEqual(stats["train"], 1)
            text = next((dest / "labels").rglob("*.txt")).read_text(encoding="utf-8")
            self.assertTrue(text.startswith("2 "))


class MergeDatasetTests(unittest.TestCase):
    def test_build_merges_visdrone_skips_missing_uavdt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            vis = base / "training" / "datasets" / "visdrone" / "VisDrone2019-DET-train"
            _write_jpg(vis / "images" / "000001.jpg", 80, 60)
            (vis / "annotations").mkdir(parents=True, exist_ok=True)
            (vis / "annotations" / "000001.txt").write_text("10,10,20,10,1,4,0,0\n", encoding="utf-8")
            result = build_drone_vehicles_dataset(base, skip_uavdt=True, visdrone_src=vis.parent, download_visdrone=False)
            self.assertTrue(result["yaml_path"].is_file())
            yaml_text = result["yaml_path"].read_text(encoding="utf-8")
            self.assertEqual(yaml_text, drone_vehicles_data_yaml(result["root"]))
            self.assertGreaterEqual(result["train_count"], 1)
            self.assertGreaterEqual(result["val_count"], 1)
            self.assertGreaterEqual(result["box_count"], 1)

    def test_build_raises_when_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                build_drone_vehicles_dataset(Path(tmp), skip_uavdt=True, download_visdrone=False)


if __name__ == "__main__":
    unittest.main()
