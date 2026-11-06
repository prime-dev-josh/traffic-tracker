import unittest

from traffic_tracker.gui.style import brand_logo_path, stylesheet_for


class BrandThemeTests(unittest.TestCase):
    def test_dark_and_light_use_logo_palette(self) -> None:
        dark = stylesheet_for("dark")
        light = stylesheet_for("light")
        self.assertIn("#221A4A", dark)
        self.assertIn("#F15B2C", dark)
        self.assertIn("#24328A", light)
        self.assertIn("#F05D2A", light)
        self.assertNotEqual(dark, light)

    def test_brand_logo_files_exist(self) -> None:
        self.assertTrue(brand_logo_path("dark").is_file())
        self.assertTrue(brand_logo_path("light").is_file())


if __name__ == "__main__":
    unittest.main()
