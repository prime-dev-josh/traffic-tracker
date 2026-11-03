import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from traffic_tracker.config import OPENVINO_DIR_NAME, YOLO_ONNX_NAME, YOLO_PT_NAME
from traffic_tracker.core.models import resolve_backend, resolve_model_source


class ModelResolveTests(unittest.TestCase):
    def setUp(self) -> None:
        os.environ.pop("TRAFFIC_TRACKER_BACKEND", None)
        os.environ.pop("TRAFFIC_TRACKER_MODEL", None)

    def test_backend_default_auto(self) -> None:
        self.assertEqual(resolve_backend(), "auto")

    def test_backend_env(self) -> None:
        os.environ["TRAFFIC_TRACKER_BACKEND"] = "onnx"
        self.assertEqual(resolve_backend(), "onnx")

    def test_explicit_model_path(self) -> None:
        os.environ["TRAFFIC_TRACKER_MODEL"] = r"C:\weights\custom.onnx"
        source = resolve_model_source()
        self.assertEqual(source.backend, "onnx")
        self.assertEqual(source.path, Path(r"C:\weights\custom.onnx"))

    def test_auto_prefers_onnx_when_present(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            (root / YOLO_ONNX_NAME).write_bytes(b"onnx")
            (root / YOLO_PT_NAME).write_bytes(b"pt")
            with patch("traffic_tracker.core.models.candidate_roots", return_value=[root]):
                source = resolve_model_source("auto")
            self.assertEqual(source.backend, "onnx")
            self.assertEqual(source.path, root / YOLO_ONNX_NAME)

    def test_openvino_forced(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            (root / OPENVINO_DIR_NAME).mkdir()
            with patch("traffic_tracker.core.models.candidate_roots", return_value=[root]):
                source = resolve_model_source("openvino")
            self.assertEqual(source.backend, "openvino")

    def test_pt_forced_even_if_onnx_exists(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            (root / YOLO_ONNX_NAME).write_bytes(b"onnx")
            with patch("traffic_tracker.core.models.candidate_roots", return_value=[root]):
                source = resolve_model_source("pt")
            self.assertEqual(source.backend, "pt")


if __name__ == "__main__":
    unittest.main()
