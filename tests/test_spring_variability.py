"""Tests for spring-length variability in probe_variability.py:
the per-subject "springs" export and the group stiff-vs-flexible analysis."""

import json
import os
import tempfile
import unittest

import bpy

from _addon_helpers import import_addon, register_addon


def _subject(lengths, kinds=None, rest=30.0):
    """Subject dict with one spring per entry in `lengths` (optode pairs S_i-D_i)."""
    kinds = kinds or ["stiff"] * len(lengths)
    optodes, springs = [], []
    for i, (length, kind) in enumerate(zip(lengths, kinds), start=1):
        optodes += [{"name": f"Source_{i}", "type": "source", "world_pos": [0, 100 * i, 0]},
                    {"name": f"Detector_{i}", "type": "detector", "world_pos": [length, 100 * i, 0]}]
        springs.append({"optode1": f"Source_{i}", "optode2": f"Detector_{i}", "kind": kind,
                        "touches_anchor": i == 1, "rest_length_mm": rest,
                        "design_rest_length_mm": rest, "length_mm": length,
                        "strain": (length - rest) / rest})
    return {"optodes": optodes, "springs": springs}


class SpringVariabilityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addon = import_addon()
        cls.pv = cls.addon.probe_variability

    def _names(self, n):
        return [f"{k}_{i}" for i in range(1, n + 1) for k in ("Source", "Detector")]

    def test_spring_record_strain(self):
        r = self.pv.spring_record("Source_1", "Detector_1", [0, 0, 0], [33, 0, 0], 30.0, False, True)
        self.assertEqual(r["kind"], "stiff")
        self.assertAlmostEqual(r["length_mm"], 33.0)
        self.assertAlmostEqual(r["strain"], 0.1)
        flex = self.pv.spring_record("Source_1", "Detector_1", [0, 0, 0], [5, 0, 0], None, True, False)
        self.assertEqual(flex["kind"], "flexible")
        self.assertIsNone(flex["strain"])

    def test_stiff_vs_flexible_from_subject_springs(self):
        # spring 1, 2 stiff and constant; spring 3 flexible and varying
        subjects = [_subject([30, 30, 20], ["stiff", "stiff", "flexible"]),
                    _subject([30, 30, 40], ["stiff", "stiff", "flexible"])]
        edges = self.pv.compute_edge_distances(subjects, self._names(3))
        by_pair = {(e["optode1"], e["optode2"]): e for e in edges}
        self.assertEqual(len(edges), 3)
        self.assertAlmostEqual(by_pair[("Source_1", "Detector_1")]["sd_dist"], 0.0)
        flex = by_pair[("Source_3", "Detector_3")]
        self.assertEqual(flex["edge_type"], "flexible")
        self.assertAlmostEqual(flex["sd_dist"], 14.1421, places=3)
        self.assertAlmostEqual(flex["mean_abs_strain"], 1 / 3, places=4)

        summary = dict(self.pv.summarize_spring_variability(edges))
        self.assertAlmostEqual(summary["stiff"]["mean_sd_mm"], 0.0)
        self.assertEqual(summary["stiff"]["n_springs"], 2)
        self.assertEqual(summary["stiff (free)"]["n_springs"], 1)   # spring 1 touches an anchor
        self.assertGreater(summary["flexible"]["mean_sd_mm"], 10.0)

    def test_falls_back_to_probe_json_for_old_exports(self):
        subjects = [_subject([30, 31]), _subject([32, 29])]
        for s in subjects:
            del s["springs"]
        cfg = {"optodes": [{"name": "Source_1", "is_anchor": True}],
               "connections": [{"optode1": "Source_1", "optode2": "Detector_1", "rest_length": 30.0,
                                "is_flexible": False},
                               {"optode1": "Source_2", "optode2": "Detector_2", "rest_length": 30.0,
                                "is_flexible": True}]}
        path = os.path.join(tempfile.mkdtemp(), "probe_config.json")
        with open(path, "w") as f:
            json.dump(cfg, f)
        edges = self.pv.compute_edge_distances(subjects, self._names(2), probe_json_path=path)
        by_pair = {(e["optode1"], e["optode2"]): e for e in edges}
        self.assertTrue(by_pair[("Source_1", "Detector_1")]["touches_anchor"])
        self.assertAlmostEqual(by_pair[("Source_1", "Detector_1")]["mean_dist"], 31.0)
        self.assertEqual(by_pair[("Source_2", "Detector_2")]["edge_type"], "flexible")


class SpringExportSceneTest(unittest.TestCase):
    """get_springs() reads Optode_Connections spring_states from a live scene."""

    @classmethod
    def setUpClass(cls):
        cls.addon = import_addon()
        register_addon(cls.addon)
        cls.pv = cls.addon.probe_variability

    @classmethod
    def tearDownClass(cls):
        cls.addon.unregister()

    def test_get_springs_reports_scaled_and_design_rest(self):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        mesh = bpy.data.meshes.new("conn")
        conn = bpy.data.objects.new("Optode_Connections", mesh)
        bpy.context.scene.collection.objects.link(conn)
        conn["spring_states"] = {
            "Detector_1_Source_1": {"optode1": "Source_1", "optode2": "Detector_1",
                                    "rest_length": 16.0, "is_flexible": False},
            "Detector_2_Source_1": {"optode1": "Source_1", "optode2": "Detector_2",
                                    "rest_length": 60.0, "is_flexible": True},
        }
        conn["rest_length_scale"] = 2.0
        positions = {"Source_1": [0, 0, 0], "Detector_1": [16, 0, 0], "Detector_2": [0, 66, 0]}
        springs = {s["optode2"]: s for s in self.pv.get_springs(positions, {"Source_1"})}
        self.assertEqual(springs["Detector_1"]["kind"], "stiff")
        self.assertAlmostEqual(springs["Detector_1"]["design_rest_length_mm"], 8.0)
        self.assertAlmostEqual(springs["Detector_1"]["strain"], 0.0)
        self.assertEqual(springs["Detector_2"]["kind"], "flexible")
        self.assertAlmostEqual(springs["Detector_2"]["strain"], 0.1)
        self.assertTrue(springs["Detector_2"]["touches_anchor"])
