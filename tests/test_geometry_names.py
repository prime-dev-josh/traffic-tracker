import unittest

from traffic_tracker.core.geometry import Geometry, Line, NamedLine, all_movement_names, movement_name


class NamingTests(unittest.TestCase):
    def test_movement_name_format(self) -> None:
        self.assertEqual(movement_name("N", "SB"), "N-SB")
        self.assertIn("E-WB", all_movement_names())
        self.assertEqual(len(all_movement_names()), 16)

    def test_assign_names_from_north_up(self) -> None:
        geom = Geometry(
            200,
            200,
            north=Line(100, 150, 100, 20),
            starts=[
                NamedLine("", Line(80, 20, 120, 20)),
                NamedLine("", Line(180, 80, 180, 120)),
                NamedLine("", Line(80, 180, 120, 180)),
                NamedLine("", Line(20, 80, 20, 120)),
            ],
            ends=[
                NamedLine("", Line(70, 30, 130, 30)),
                NamedLine("", Line(170, 70, 170, 130)),
                NamedLine("", Line(70, 170, 130, 170)),
                NamedLine("", Line(30, 70, 30, 130)),
            ],
        )
        geom.assign_names()
        self.assertEqual({item.name for item in geom.starts}, {"N", "E", "S", "W"})
        self.assertEqual({item.name for item in geom.ends}, {"NB", "EB", "SB", "WB"})
        by_start = {item.name: item.line.midpoint() for item in geom.starts}
        self.assertLess(by_start["N"][1], by_start["S"][1])
        self.assertGreater(by_start["E"][0], by_start["W"][0])
        by_end = {item.name: item.line.midpoint() for item in geom.ends}
        self.assertLess(by_end["NB"][1], by_end["SB"][1])
        self.assertGreater(by_end["EB"][0], by_end["WB"][0])

    def test_is_ready_allows_one_way(self) -> None:
        geom = Geometry(10, 10)
        self.assertFalse(geom.is_ready())
        geom.north = Line(5, 9, 5, 1)
        geom.starts = [NamedLine("N", Line(0, 0, 1, 0))]
        geom.ends = [NamedLine("SB", Line(0, 1, 1, 1))]
        self.assertTrue(geom.is_ready())
        self.assertEqual(geom.movement_names(), ["N-SB"])

    def test_assign_two_opposite_starts(self) -> None:
        geom = Geometry(
            200,
            200,
            north=Line(100, 150, 100, 20),
            starts=[
                NamedLine("", Line(80, 20, 120, 20)),
                NamedLine("", Line(80, 180, 120, 180)),
            ],
        )
        geom.assign_names()
        self.assertEqual({item.name for item in geom.starts}, {"N", "S"})


if __name__ == "__main__":
    unittest.main()
