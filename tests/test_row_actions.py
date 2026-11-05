import unittest

from traffic_tracker.core.geometry import Box, Geometry, Line, NamedLine
from traffic_tracker.gui.main_window import bind_row_action


class RowActionTests(unittest.TestCase):
    def test_bind_row_action_ignores_clicked_checked_flag(self) -> None:
        seen: list[object] = []
        item = NamedLine("N", Line(0, 0, 10, 0))
        bound = bind_row_action(lambda gate=item: seen.append(gate))
        bound(False)
        self.assertEqual(seen, [item])

    def test_live_gate_matches_equal_copy(self) -> None:
        original = NamedLine("NB", Line(1, 2, 3, 4))
        geom = Geometry(100, 100, north=Line(50, 80, 50, 10), ends=[original])

        class Holder:
            geometry = geom

            def _live_gate(self, item):
                from traffic_tracker.gui.main_window import MainWindow

                return MainWindow._live_gate(self, item)

        holder = Holder()
        copy = NamedLine("NB", Line(9, 9, 8, 8))
        found = holder._live_gate(copy)
        self.assertIsNotNone(found)
        self.assertEqual(found[0], "end")
        self.assertIs(found[1], original)

    def test_live_vehicle_matches_equal_box(self) -> None:
        original = Box(10, 10, 40, 30)
        geom = Geometry(100, 100, vehicle_refs=[original])

        class Holder:
            geometry = geom

            def _live_vehicle(self, box):
                from traffic_tracker.gui.main_window import MainWindow

                return MainWindow._live_vehicle(self, box)

        holder = Holder()
        self.assertIs(holder._live_vehicle(Box(10, 10, 40, 30)), original)
        self.assertIsNone(holder._live_vehicle(False))


if __name__ == "__main__":
    unittest.main()
