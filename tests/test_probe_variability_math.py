"""Pure-logic unit tests for probe_variability.py's vector/stats helpers.

Run via tests/run_tests.py inside Blender's bundled Python. Unlike
test_addon_smoke.py, these don't register the add-on or touch a live scene -
they call plain Python/NumPy math functions with hand-computed expected
values, so they catch numeric regressions the smoke test can't see.
"""

import unittest

import numpy as np

from _addon_helpers import import_addon


class VectorHelperTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pv = import_addon().probe_variability

    def test_sub_add_scale(self):
        self.assertEqual(self.pv._sub([5, 5, 5], [1, 2, 3]), [4, 3, 2])
        self.assertEqual(self.pv._add([1, 2, 3], [4, 5, 6]), [5, 7, 9])
        self.assertEqual(self.pv._scale([1, 2, 3], 2), [2, 4, 6])

    def test_dot(self):
        self.assertEqual(self.pv._dot([1, 0, 0], [0, 1, 0]), 0)
        self.assertEqual(self.pv._dot([1, 2, 3], [4, 5, 6]), 32)

    def test_cross(self):
        self.assertEqual(self.pv._cross([1, 0, 0], [0, 1, 0]), [0, 0, 1])

    def test_norm(self):
        result = self.pv._norm([3, 4, 0])
        for actual, expected in zip(result, [0.6, 0.8, 0.0]):
            self.assertAlmostEqual(actual, expected)
        # zero vector must not raise a division-by-zero error
        self.assertEqual(self.pv._norm([0, 0, 0]), [0.0, 0.0, 0.0])

    def test_dist(self):
        self.assertAlmostEqual(self.pv._dist([0, 0, 0], [3, 4, 0]), 5.0)


class HeadFrameTest(unittest.TestCase):
    """Synthetic fiducials on the coordinate axes, chosen so every
    origin/axis/scale value in build_head_frame can be hand-verified:
    Lpa/Rpa straddle the origin on X, Cz sits straight up on Z, Nz/Iz
    straddle the origin on Y.
    """

    @classmethod
    def setUpClass(cls):
        cls.pv = import_addon().probe_variability
        cls.landmarks = {
            "Cz": [0, 0, 1],
            "Nz": [0, 1, 0],
            "Iz": [0, -1, 0],
            "Lpa": [-1, 0, 0],
            "Rpa": [1, 0, 0],
        }

    def test_missing_landmark_returns_none(self):
        incomplete = dict(self.landmarks)
        del incomplete["Iz"]
        self.assertIsNone(self.pv.build_head_frame(incomplete))

    def test_frame_axes_and_scales(self):
        frame = self.pv.build_head_frame(self.landmarks)
        self.assertIsNotNone(frame)
        self.assertEqual(frame["origin"], [0.0, 0.0, 0.0])
        for actual, expected in zip(frame["z"], [0.0, 0.0, 1.0]):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(frame["x"], [0.0, 1.0, 0.0]):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(frame["y"], [-1.0, 0.0, 0.0]):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(frame["scale_x"], 2.0)
        self.assertAlmostEqual(frame["scale_y"], 2.0)
        self.assertAlmostEqual(frame["scale_z"], 1.0)

    def test_project_into_head_frame(self):
        frame = self.pv.build_head_frame(self.landmarks)
        fx, fy, fz, fx_sc, fy_sc, fz_sc = self.pv.project_into_head_frame(
            [0, 0, 0.5], frame
        )
        self.assertAlmostEqual(fx, 0.0)
        self.assertAlmostEqual(fy, 0.0)
        self.assertAlmostEqual(fz, 0.5)
        self.assertAlmostEqual(fx_sc, 0.0)
        self.assertAlmostEqual(fy_sc, 0.0)
        self.assertAlmostEqual(fz_sc, 0.5)  # scaled by scale_z == 1.0


class ShortSepFlagsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pv = import_addon().probe_variability

    def test_close_source_detector_pair_flags_only_the_detector(self):
        optodes = [("Source1", [0, 0, 0]), ("Detector1", [0, 0, 5])]
        flags = self.pv.compute_short_sep_flags(optodes, threshold_mm=10.0)
        self.assertEqual(flags, {"Source1": False, "Detector1": True})

    def test_far_apart_pair_is_not_flagged(self):
        optodes = [("Source1", [0, 0, 0]), ("Detector1", [0, 0, 5])]
        flags = self.pv.compute_short_sep_flags(optodes, threshold_mm=3.0)
        self.assertEqual(flags, {"Source1": False, "Detector1": False})

    def test_two_detectors_are_never_flagged(self):
        # Only source-detector pairs are checked, per compute_short_sep_flags'
        # docstring - two close detectors must not trip the flag.
        optodes = [("Detector1", [0, 0, 0]), ("Detector2", [0, 0, 1])]
        flags = self.pv.compute_short_sep_flags(optodes, threshold_mm=10.0)
        self.assertEqual(flags, {"Detector1": False, "Detector2": False})


class HsvToRgbTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pv = import_addon().probe_variability

    def test_known_colors(self):
        self.assertEqual(self.pv._hsv_to_rgb(0, 0, 0), (0.0, 0.0, 0.0))  # black
        self.assertEqual(self.pv._hsv_to_rgb(0, 0, 1), (1.0, 1.0, 1.0))  # white
        self.assertEqual(self.pv._hsv_to_rgb(0, 1, 1), (1.0, 0.0, 0.0))  # red


class IntersubjectStatsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pv = import_addon().probe_variability

    def test_two_subject_variance_along_one_axis(self):
        # 1 optode, 2 subjects at [0,0,0] and [2,0,0]: mean is [1,0,0] and
        # each subject is exactly 1 unit away from it.
        positions = np.array([[[0.0, 2.0], [0.0, 0.0], [0.0, 0.0]]])
        stats = self.pv.compute_intersubject_stats(positions)
        np.testing.assert_allclose(stats["mean_pos"], [[1.0, 0.0, 0.0]])
        np.testing.assert_allclose(stats["sd"], [1.0])
        np.testing.assert_allclose(stats["errors"], [[1.0, 1.0]])
