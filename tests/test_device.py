import unittest
from unittest.mock import patch

from traffic_tracker.core.device import (
    device_report,
    inference_device_label,
    nvidia_gpu_names,
    reset_device_cache,
    torch_cuda_build,
    yolo_runtime,
)


class DeviceReportTests(unittest.TestCase):
    def tearDown(self) -> None:
        reset_device_cache()

    def test_nvidia_gpu_names_parses_smi(self) -> None:
        completed = type("R", (), {"returncode": 0, "stdout": "NVIDIA GeForce RTX 4060\n"})()
        with patch("traffic_tracker.core.device.subprocess.run", return_value=completed):
            self.assertEqual(nvidia_gpu_names(), ["NVIDIA GeForce RTX 4060"])

    def test_cpu_torch_explains_unused_gpu(self) -> None:
        reset_device_cache()
        with (
            patch("traffic_tracker.core.device.detect_device", return_value="cpu"),
            patch("traffic_tracker.core.device.nvidia_gpu_names", return_value=["NVIDIA GeForce RTX 4060"]),
            patch("traffic_tracker.core.device.torch_cuda_build", return_value=False),
        ):
            report = device_report()
        self.assertEqual(report.device, "cpu")
        self.assertIn("CPU-only", report.label)
        self.assertIn("RTX 4060", report.warning or "")

    def test_cuda_device_has_no_warning(self) -> None:
        reset_device_cache()
        with (
            patch("traffic_tracker.core.device.detect_device", return_value="cuda"),
            patch("traffic_tracker.core.device.describe_device", return_value="CUDA (NVIDIA GeForce RTX 4060)"),
        ):
            report = device_report()
        self.assertEqual(report.device, "cuda")
        self.assertIsNone(report.warning)

    def test_torch_cuda_build_reads_version(self) -> None:
        self.assertIsInstance(torch_cuda_build(), bool)

    def test_inference_device_label_marks_fp16(self) -> None:
        with patch(
            "traffic_tracker.core.device.describe_device",
            return_value="CUDA (NVIDIA GeForce RTX 4060)",
        ):
            self.assertEqual(
                inference_device_label("cuda"),
                "CUDA (NVIDIA GeForce RTX 4060) • FP16",
            )
        self.assertEqual(inference_device_label("cpu"), "CPU")
        self.assertEqual(inference_device_label("mps"), "MPS (Apple Silicon)")

    def test_yolo_runtime_uses_cuda_index_and_fp16(self) -> None:
        self.assertEqual(yolo_runtime("cuda"), (0, 16))
        self.assertEqual(yolo_runtime("cpu"), ("cpu", None))
        self.assertEqual(yolo_runtime("mps"), ("mps", None))
        with patch("traffic_tracker.core.device.detect_device", return_value="cuda"):
            self.assertEqual(yolo_runtime(), (0, 16))


if __name__ == "__main__":
    unittest.main()
