"""Headless test: AtlasViewer dummy optodes survive SD import -> JSON export ->
JSON import on a different-sized head -> cloth re-relaxation.

Dummies are created as Dummy_N objects that register and pin like any other
anchor optode, and are hidden by default.  Hidden dummies keep their springs
in spring_states (registration) but have no edge on the connection mesh
(display).  The sources/detectors sprung to a dummy are NOT anchors.
Uses the synthetic probe from test_spring_encoding.py: dummies 6-9 anchored
to C3, C4, Fz, Pz, with springs Dummy_1-Source_1, Dummy_2-Detector_1 and
Dummy_3-Detector_2.
"""

import importlib
import json
import os
import tempfile
import unittest

import bpy

from _addon_helpers import import_addon, register_addon
from test_spring_encoding import ANCHOR_LABELS, _build_scene, _key, _write_sd

DUMMIES = [f"Dummy_{k + 1}" for k in range(len(ANCHOR_LABELS))]
DUMMY_SPRINGS = {_key("Dummy_1", "Source_1"): 40.0,
                 _key("Dummy_2", "Detector_1"): 10.0,
                 _key("Dummy_3", "Detector_2"): 10.0}


def _landmark(label):
    lm = bpy.data.objects["LandmarkMesh"]
    return lm.matrix_world @ lm.data.vertices[list(lm["landmark_labels"]).index(label)].co


def _drawn_pairs():
    """Optode-name pairs that have an edge on the connection mesh."""
    conn = bpy.data.objects["Optode_Connections"]
    names = list(conn["optode_names"])
    return {_key(names[e.vertices[0]], names[e.vertices[1]]) for e in conn.data.edges}


class SDDummyOptodesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addon = import_addon()
        cls.tmp = tempfile.mkdtemp()
        cls.sd_path = os.path.join(cls.tmp, "probe.SD")
        _write_sd(cls.sd_path)

    def _import_sd(self, **kwargs):
        _build_scene(1.0)
        register_addon(self.addon)
        self.assertEqual(
            bpy.ops.neurocaptain.import_sd_probe(filepath=self.sd_path, **kwargs), {"FINISHED"})

    def _assert_dummies(self, shown, tol_mm, step):
        for name, label in zip(DUMMIES, ANCHOR_LABELS):
            with self.subTest(step=step, dummy=name):
                d = bpy.data.objects[name]
                self.assertTrue(d.get("is_dummy"))
                self.assertTrue(d.get("is_anchor"))
                self.assertEqual(d.get("anchor_landmark"), label)
                self.assertEqual(d.hide_get(), not shown)
                self.assertEqual(d.hide_render, not shown)
                self.assertLess((d.location - _landmark(label)).length, tol_mm,
                                "dummy not pinned at its landmark")
        real_anchors = {o.name for o in bpy.data.objects
                        if o.name.startswith(("Source_", "Detector_")) and o.get("is_anchor")}
        self.assertEqual(real_anchors, set(), f"{step}: dummy neighbours promoted to anchors")
        states = bpy.data.objects["Optode_Connections"]["spring_states"].to_dict()
        for key, design in DUMMY_SPRINGS.items():
            with self.subTest(step=step, spring=key):
                self.assertIn(key, states, "dummy spring dropped from spring_states")
                self.assertFalse(states[key]["is_flexible"])
                self.assertAlmostEqual(states[key]["design_rest_length"], design, places=3)
        drawn = _drawn_pairs() & set(DUMMY_SPRINGS)
        self.assertEqual(drawn, set(DUMMY_SPRINGS) if shown else set(),
                         f"{step}: dummy springs drawn={sorted(drawn)}")

    def test_hidden_by_default(self):
        self._import_sd()
        self._assert_dummies(shown=False, tol_mm=1.0, step="SD import, hidden")

    def test_include_dummies(self):
        self._import_sd(include_dummies=True)
        self._assert_dummies(shown=True, tol_mm=1.0, step="SD import, shown")

    def test_options_popup_operator_runs_import(self):
        _build_scene(1.0)
        register_addon(self.addon)
        self.assertEqual(bpy.ops.neurocaptain.import_sd_probe_options(
            "EXEC_DEFAULT", filepath=self.sd_path, include_dummies=True), {"FINISHED"})
        self._assert_dummies(shown=True, tol_mm=1.0, step="options operator")

    def test_json_round_trip_registers_and_pins_dummies(self):
        self._import_sd()
        json_path = os.path.join(self.tmp, "probe_dummies.json")
        self.assertEqual(bpy.ops.neurocaptain.export_optode_json(filepath=json_path), {"FINISHED"})
        with open(json_path) as f:
            config = json.load(f)
        exported = {o["name"]: o for o in config["optodes"] if o["type"] == "dummy"}
        self.assertEqual(set(exported), set(DUMMIES))
        self.assertEqual(config["metadata"]["num_dummies"], len(DUMMIES))
        for name in DUMMIES:
            self.assertTrue(exported[name]["is_anchor"])
            self.assertEqual(exported[name]["registration"]["method"], "barycentric")
            self.assertEqual(exported[name]["custom_properties"]["show_dummy"], 0)
        conns = {_key(c["optode1"], c["optode2"]) for c in config["connections"]}
        self.assertTrue(set(DUMMY_SPRINGS) <= conns, "dummy springs not exported")

        # Registered on a head twice the size by the normal barycentric + cloth path
        _build_scene(2.0)
        register_addon(self.addon)
        self.assertEqual(
            bpy.ops.neurocaptain.import_optode_json_blender_goal(
                filepath=json_path, scale_rest_lengths=True), {"FINISHED"})
        self._assert_dummies(shown=False, tol_mm=2.0, step="JSON import on 2x head")

        # Variability export only sees the real optodes
        pv = importlib.import_module(f"{self.addon.__name__}.probe_variability")
        self.assertEqual({o.name for o, _, _ in pv.get_optodes()},
                         {o["name"] for o in config["optodes"] if o["type"] != "dummy"})

        # Re-running the cloth relaxation keeps the hidden dummy springs and pins
        self.assertEqual(bpy.ops.neurocaptain.relax_probe_manual("EXEC_DEFAULT"), {"FINISHED"})
        self._assert_dummies(shown=False, tol_mm=2.0, step="Relax Probe (Cloth)")


if __name__ == "__main__":
    unittest.main()
