import unittest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from openpyxl import load_workbook

from traffic_tracker.core.geometry import Box, Geometry, Line, NamedLine
from traffic_tracker.core.report import (
    Trip,
    build_gate_statistics_table,
    build_overview_rows,
    build_overview_table,
    build_tmc_summary_table,
    build_tmc_table,
    build_trip_table,
    export_tmc_workbook,
    interval_label,
)


class ReportTests(unittest.TestCase):
    def test_elapsed_bins(self) -> None:
        trips = [
            Trip("a.mp4", 1, "N", 60, "SB", 60, "car"),
            Trip("a.mp4", 2, "N", 900, "SB", 900, "car"),
            Trip("a.mp4", 3, "S", 901, "NB", 901, "truck"),
        ]
        table = build_tmc_table(trips, ["N-SB", "S-NB"], duration_s=1800)
        self.assertEqual(list(table["N-SB"]), [1, 1])
        self.assertEqual(list(table["S-NB"]), [0, 1])
        self.assertEqual(list(table["Total"]), [1, 2])

    def test_trip_table_fields(self) -> None:
        trips = [Trip("a.mp4", 1, "N", 60, "SB", 75, "car")]
        table = build_trip_table(trips)
        self.assertEqual(list(table.columns), ["Start Destination", "Start Time", "End Destination", "End Time"])
        self.assertEqual(table.iloc[0]["Start Destination"], "N")
        self.assertEqual(table.iloc[0]["End Destination"], "SB")
        self.assertEqual(table.iloc[0]["Start Time"], "01:00")
        self.assertEqual(table.iloc[0]["End Time"], "01:15")

    def test_clock_labels(self) -> None:
        start, end = interval_label(0, datetime(2026, 11, 2, 8, 0))
        self.assertEqual(start, "2026-11-02 08:00")
        self.assertEqual(end, "2026-11-02 08:15")

    def test_overview_and_gate_statistics(self) -> None:
        geom = Geometry(
            100,
            100,
            north=Line(50, 90, 50, 10),
            starts=[NamedLine("N", Line(10, 10, 20, 10)), NamedLine("S", Line(10, 90, 20, 90))],
            ends=[NamedLine("SB", Line(80, 90, 90, 90)), NamedLine("NB", Line(80, 10, 90, 10))],
        )
        trips = [
            Trip("a.mp4", 1, "N", 10, "SB", 20, "car"),
            Trip("a.mp4", 2, "N", 30, "NB", 40, "car"),
            Trip("a.mp4", 3, "S", 50, "NB", 60, "truck"),
        ]
        overview = dict(build_overview_rows(trips))
        self.assertEqual(overview["Number of all vehicles"], 3)
        self.assertEqual(overview["Car count"], 2)
        self.assertEqual(overview["Truck count"], 1)
        sheet = build_overview_table(
            trips,
            video_name="a.mp4",
            duration_s=120,
            clock_start=None,
            device_label="motion mask (CPU)",
        )
        self.assertEqual(list(sheet.columns), ["Field", "Value"])
        self.assertIn("Number of all vehicles", list(sheet["Field"]))
        self.assertEqual(int(sheet.loc[sheet["Field"] == "Number of all vehicles", "Value"].iloc[0]), 3)
        self.assertIn("Car count", overview)
        gates = build_gate_statistics_table(trips, geom)
        self.assertEqual(list(gates["Category"]), [
            "Car count",
            "Van count",
            "Truck count",
            "Motorcycle count",
            "Bus count",
            "Number of all vehicles",
        ])
        total_row = gates.loc[gates["Category"] == "Number of all vehicles"].iloc[0]
        self.assertEqual(int(total_row["N (Entry)"]), 2)
        self.assertEqual(int(total_row["NB (Exit)"]), 2)
        car_row = gates.loc[gates["Category"] == "Car count"].iloc[0]
        self.assertEqual(int(car_row["N (Entry)"]), 2)

    def test_tmc_summary_groups_by_entry(self) -> None:
        geom = Geometry(
            100,
            100,
            starts=[NamedLine("N", Line(1, 1, 2, 1))],
            ends=[NamedLine("SB", Line(3, 3, 4, 4)), NamedLine("NB", Line(5, 5, 6, 6))],
        )
        trips = [
            Trip("a.mp4", 1, "N", 10, "SB", 20, "car"),
            Trip("a.mp4", 2, "N", 910, "NB", 920, "car"),
        ]
        table = build_tmc_summary_table(trips, geom, duration_s=1800)
        self.assertEqual(list(table.columns), ["Period", "N-NB", "N-SB", "N Total"])
        self.assertEqual(int(table.iloc[0]["N-SB"]), 1)
        self.assertEqual(int(table.iloc[1]["N-NB"]), 1)
        self.assertEqual(table.iloc[-1]["Period"], "Total (whole sequence)")
        self.assertEqual(int(table.iloc[-1]["N Total"]), 2)

    def test_export_has_las_olas_sheets(self) -> None:
        geom = Geometry(
            100,
            100,
            north=Line(50, 90, 50, 10),
            starts=[NamedLine("N", Line(10, 10, 20, 10))],
            ends=[NamedLine("SB", Line(80, 90, 90, 90))],
        )
        trips = [Trip("site.mp4", 1, "N", 10, "SB", 20, "car")]
        with TemporaryDirectory() as raw:
            path = Path(raw) / "out.xlsx"
            export_tmc_workbook(
                path,
                trips,
                geom,
                video_name="site.mp4",
                duration_s=900,
                clock_start=datetime(2026, 8, 8, 14, 0),
                device_label="CPU",
            )
            wb = load_workbook(path)
            self.assertEqual(wb.sheetnames, ["Overview", "Gate Statistics", "TMC Summary"])
            overview_labels = [wb["Overview"].cell(row, 2).value for row in range(8, 16)]
            self.assertIn("Car count", overview_labels)
            self.assertIn("Number of all vehicles", overview_labels)
            self.assertEqual(wb["Gate Statistics"]["B5"].value, "Car count")
            self.assertEqual(wb["Gate Statistics"]["B10"].value, "Number of all vehicles")
            self.assertEqual(wb["Gate Statistics"]["B2"].value, "Survey report - Gate Statistics")
            self.assertEqual(wb["TMC Summary"]["A1"].value, "TMC Summary")
            self.assertEqual(wb["TMC Summary"]["C3"].value, "ENTRY N")

    def test_export_embeds_path_image(self) -> None:
        import numpy as np

        geom = Geometry(
            100,
            100,
            north=Line(50, 90, 50, 10),
            starts=[NamedLine("N", Line(10, 10, 20, 10))],
            ends=[NamedLine("SB", Line(80, 90, 90, 90))],
        )
        trips = [Trip("site.mp4", 1, "N", 10, "SB", 20, "car", path=[(10.0, 10.0), (80.0, 80.0)])]
        image = np.full((60, 90, 3), 40, dtype=np.uint8)
        with TemporaryDirectory() as raw:
            path = Path(raw) / "out.xlsx"
            export_tmc_workbook(
                path,
                trips,
                geom,
                video_name="site.mp4",
                duration_s=900,
                clock_start=None,
                device_label="CPU",
                path_image=image,
            )
            png = path.with_name("out_paths.png")
            self.assertTrue(png.exists())
            wb = load_workbook(path)
            self.assertTrue(wb["Gate Statistics"]._images)

    def test_geometry_roundtrip(self) -> None:
        geom = Geometry(
            1920,
            1080,
            north=Line(960, 1000, 960, 80),
            starts=[NamedLine("N", Line(10, 20, 30, 40))],
            ends=[NamedLine("SB", Line(50, 60, 70, 80))],
            vehicle_refs=[Box(100, 200, 140, 240)],
            occlusions=[Box(400, 100, 430, 500)],
        )
        data = geom.to_dict()
        loaded = Geometry.from_dict(data)
        self.assertEqual(loaded.starts[0].name, "N")
        self.assertEqual(loaded.ends[0].name, "SB")
        self.assertEqual(loaded.vehicle_refs[0].area(), 1600)
        self.assertEqual(loaded.occlusions[0].width(), 30)
        scaled = loaded.scaled_to(960, 540)
        self.assertEqual(scaled.starts[0].line.x1, 5)


if __name__ == "__main__":
    unittest.main()
