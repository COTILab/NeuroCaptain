"""Headless test: spring stiffness survives SD import -> JSON export ->
JSON import on a different-sized head -> JSON export.

A synthetic SD probe (stiff long springs, a stiff 8 mm short-separation
spring, a flexible spring and springs to anchor dummies) is imported, then
round-tripped through the probe JSON.  At every step the stiff/flexible
flags in spring_states, the edge crease/bevel marking and the designed rest
lengths must be preserved.  Also checks the Make Springs Stiff / Flexible
operators, whose edge-layer code changed for Blender 4.0+.
"""

import json
import math
import os
import tempfile
import unittest

import bmesh
import bpy
import numpy as np
from mathutils import Vector
from scipy.io import savemat

from _addon_helpers import import_addon, register_addon

R0 = 90.0
# 2D flat layout (mm): sources 1-2, detectors 3-5, dummies 6-9
SRC_2D = [(0, 0), (30, 30)]
DET_2D = [(30, 0), (0, 30), (0, -8)]
DUMMY_2D = [(-40, 0), (40, 0), (0, 40), (0, -40)]
ANCHOR_LABELS = ["C3", "C4", "Fz", "Pz"]
SPRINGS = [  # (i1, i2, length) 1-based SD indices; -1 = flexible
    (1, 3, 30.0), (1, 4, 30.0), (2, 3, 42.4264), (1, 5, 8.0),   # stiff (1-5 = short-sep)
    (2, 4, -1.0),                                             # flexible
    (7, 3, 10.0), (8, 4, 10.0), (6, 1, 40.0),                 # to dummies
]
EXPECTED = {  # spring key -> (is_flexible, design rest length or None), real optodes only
    "Detector_1_Source_1": (False, 30.0),
    "Detector_2_Source_1": (False, 30.0),
    "Detector_1_Source_2": (False, 42.4264),
    "Detector_3_Source_1": (False, 8.0),
    "Detector_2_Source_2": (True, None),
}


def _on_sphere(xy, r):
    """Map a flat-layout point onto the top of a sphere of radius r."""
    return (Vector((xy[0], xy[1], R0)).normalized() * r)


def _write_sd(path, dummy_2d=DUMMY_2D, anchor_list=None, springs=SPRINGS):
    """anchor_list: [(1-based SD index, landmark label)]; default = one per dummy."""
    if anchor_list is None:
        n_real = len(SRC_2D) + len(DET_2D)
        anchor_list = [(n_real + k + 1, lbl) for k, lbl in enumerate(ANCHOR_LABELS)]
    anchors = np.empty((len(anchor_list), 2), dtype=object)
    for k, (idx, label) in enumerate(anchor_list):
        anchors[k, 0] = np.array([[idx]], dtype=float)
        anchors[k, 1] = np.array([label])
    xyz = lambda pts: np.array([[x, y, 0.0] for x, y in pts])
    savemat(path, {"SD": {
        "SrcPos": xyz(SRC_2D), "DetPos": xyz(DET_2D), "DummyPos": xyz(dummy_2d),
        "nSrcs": float(len(SRC_2D)), "nDets": float(len(DET_2D)),
        "AnchorList": anchors, "SpringList": np.array(springs, dtype=float),
        "SpatialUnit": "mm",
    }})


def _build_scene(scale):
    """headmesh sphere + icosphere LandmarkMesh with the anchor labels, both scaled."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    r = R0 * scale
    bpy.ops.mesh.primitive_uv_sphere_add(radius=r, segments=96, ring_count=48)
    bpy.context.active_object.name = "headmesh"
    bpy.ops.mesh.primitive_ico_sphere_add(radius=r, subdivisions=3)
    lm = bpy.context.active_object
    lm.name = "LandmarkMesh"
    labels = [""] * len(lm.data.vertices)
    for label, xy in [("Cz", (0, 0))] + list(zip(ANCHOR_LABELS, DUMMY_2D)):
        target = _on_sphere(xy, r)
        nearest = min(range(len(labels)), key=lambda i: (lm.data.vertices[i].co - target).length)
        labels[nearest] = label
    lm["landmark_labels"] = labels


def _key(n1, n2):
    a, b = sorted((n1, n2))
    return f"{a}_{b}"


def _real(*names):
    """True if every name is a source/detector (not a dummy)."""
    return all(n.startswith(("Source_", "Detector_")) for n in names)


def _edge_marks(bm):
    """(crease, bevel) per edge, read independently of the add-on's helpers."""
    layers = bm.edges.layers
    if hasattr(layers, "crease"):                         # Blender < 4.0
        crease, bevel = layers.crease.active, layers.bevel_weight.active
    else:
        crease, bevel = layers.float.get("crease_edge"), layers.float.get("bevel_weight_edge")
    return lambda e: (e[crease] if crease else 0.0, e[bevel] if bevel else 0.0)


def _springs_on_mesh():
    """{spring key: (is_flexible, crease, bevel, rest_length, design_rest_length)}"""
    conn = bpy.data.objects["Optode_Connections"]
    names = list(conn["optode_names"])
    bm = bmesh.new()
    bm.from_mesh(conn.data)
    mark = _edge_marks(bm)
    marks = {_key(names[e.verts[0].index], names[e.verts[1].index]): mark(e) for e in bm.edges}
    bm.free()
    out = {}
    for key, ss in conn["spring_states"].to_dict().items():
        if not _real(ss["optode1"], ss["optode2"]):
            continue    # dummy springs: test_sd_dummy_optodes.py
        out[key] = (bool(ss["is_flexible"]), *marks[key], ss["rest_length"], ss.get("design_rest_length"))
    return out


class SpringEncodingRoundTripTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addon = import_addon()
        cls.tmp = tempfile.mkdtemp()
        cls.sd_path = os.path.join(cls.tmp, "probe.SD")
        _write_sd(cls.sd_path)

    def setUp(self):
        register_addon(self.addon)

    def _assert_encoding(self, springs, rest_scale, step):
        self.assertEqual(set(springs), set(EXPECTED), f"{step}: spring set")
        for key, (flexible, design) in EXPECTED.items():
            is_flex, crease, bevel, rest, design_stored = springs[key]
            with self.subTest(step=step, spring=key):
                self.assertEqual(is_flex, flexible)
                self.assertEqual((crease, bevel), (1.0, 1.0) if flexible else (0.0, 0.0))
                if not flexible:
                    self.assertAlmostEqual(design_stored, design, places=3)
                    self.assertAlmostEqual(rest / design, rest_scale, delta=0.05 * rest_scale)
        self.assertIn("Bevel_Springs", bpy.data.objects["Optode_Connections"].modifiers)

    def _assert_json(self, path, step):
        with open(path) as f:
            conns = {_key(c["optode1"], c["optode2"]): c for c in json.load(f)["connections"]
                     if _real(c["optode1"], c["optode2"])}
        self.assertEqual(set(conns), set(EXPECTED), f"{step}: exported springs")
        for key, (flexible, design) in EXPECTED.items():
            with self.subTest(step=step, spring=key):
                self.assertEqual(conns[key]["is_flexible"], flexible)
                if not flexible:   # designed length, not the registered distance
                    self.assertAlmostEqual(conns[key]["rest_length"], design, places=3)
                    self.assertEqual(conns[key]["pull"], 0.9)
                else:
                    self.assertEqual(conns[key]["pull"], 0.3)

    def test_sd_to_json_to_bigger_head_round_trip(self):
        # 1. SD import
        _build_scene(1.0)
        register_addon(self.addon)
        self.assertEqual(bpy.ops.neurocaptain.import_sd_probe(filepath=self.sd_path), {"FINISHED"})
        self._assert_encoding(_springs_on_mesh(), 1.0, "SD import")

        # 2. JSON export
        json1 = os.path.join(self.tmp, "probe_from_sd.json")
        self.assertEqual(bpy.ops.neurocaptain.export_optode_json(filepath=json1), {"FINISHED"})
        self._assert_json(json1, "export after SD import")

        # 3. JSON import on a head twice the size
        _build_scene(2.0)
        register_addon(self.addon)
        self.assertEqual(
            bpy.ops.neurocaptain.import_optode_json_blender_goal(
                filepath=json1, scale_rest_lengths=True), {"FINISHED"})
        self._assert_encoding(_springs_on_mesh(), 2.0, "JSON import on 2x head")

        # 4. Re-export: designed lengths must not drift
        json2 = os.path.join(self.tmp, "probe_from_2x.json")
        self.assertEqual(bpy.ops.neurocaptain.export_optode_json(filepath=json2), {"FINISHED"})
        self._assert_json(json2, "export from 2x head")

    def test_direct_optode_anchor_is_preserved(self):
        # Dummies 6-8 anchored to C3, Fz, Pz; Source_2 (index 2) anchored directly to C4
        sd_path = os.path.join(self.tmp, "probe_direct_anchor.SD")
        _write_sd(sd_path, dummy_2d=[(-40, 0), (0, 40), (0, -40)],
                  anchor_list=[(6, "C3"), (7, "Fz"), (8, "Pz"), (2, "C4")],
                  springs=[s for s in SPRINGS if max(s[0], s[1]) <= 5] + [(6, 1, 40.0)])
        # Only Source_2 (anchored directly) - Source_1 is sprung to the C3
        # dummy, which is the anchor itself, so Source_1 stays free
        expected = {"Source_2"}

        def anchors_in_scene():
            return {o.name for o in bpy.data.objects
                    if _real(o.name) and o.get("is_anchor")}

        def landmark(label):
            lm = bpy.data.objects["LandmarkMesh"]
            return lm.matrix_world @ lm.data.vertices[list(lm["landmark_labels"]).index(label)].co

        _build_scene(1.0)
        register_addon(self.addon)
        self.assertEqual(bpy.ops.neurocaptain.import_sd_probe(filepath=sd_path), {"FINISHED"})
        self.assertEqual(anchors_in_scene(), expected)   # Source_2's neighbours not promoted
        s2 = bpy.data.objects["Source_2"]
        self.assertEqual(s2["anchor_landmark"], "C4")
        self.assertLess((s2.location - landmark("C4")).length, 1.0,
                        "direct anchor not held at its landmark")

        json_path = os.path.join(self.tmp, "probe_direct_anchor.json")
        bpy.ops.neurocaptain.export_optode_json(filepath=json_path)
        with open(json_path) as f:
            optodes = {o["name"]: o for o in json.load(f)["optodes"]}
        self.assertEqual({n for n, o in optodes.items() if o["is_anchor"] and _real(n)}, expected)
        self.assertEqual(optodes["Source_2"]["custom_properties"]["anchor_landmark"], "C4")

        _build_scene(2.0)
        register_addon(self.addon)
        bpy.ops.neurocaptain.import_optode_json_blender_goal(filepath=json_path)
        self.assertEqual(anchors_in_scene(), expected)
        self.assertLess((bpy.data.objects["Source_2"].location - landmark("C4")).length, 2.0,
                        "direct anchor not pinned at its landmark on the 2x head")

    def test_make_stiff_and_flexible_operators(self):
        _build_scene(1.0)
        register_addon(self.addon)
        bpy.ops.neurocaptain.import_sd_probe(filepath=self.sd_path)
        conn = bpy.data.objects["Optode_Connections"]
        names = list(conn["optode_names"])
        key = "Detector_1_Source_1"

        def select_edge(n1, n2):
            bpy.ops.object.select_all(action="DESELECT")
            conn.select_set(True)
            bpy.context.view_layer.objects.active = conn
            bpy.ops.object.mode_set(mode="EDIT")
            bm = bmesh.from_edit_mesh(conn.data)
            for e in bm.edges:
                ends = {names[e.verts[0].index], names[e.verts[1].index]}
                e.select_set(ends == {n1, n2})
            bmesh.update_edit_mesh(conn.data)

        select_edge("Source_1", "Detector_1")
        self.assertEqual(bpy.ops.neurocaptain.modify_spring_properties("EXEC_DEFAULT"), {"FINISHED"})
        bpy.ops.object.mode_set(mode="OBJECT")
        is_flex, crease, bevel, _, design = _springs_on_mesh()[key]
        self.assertTrue(is_flex)
        self.assertEqual((crease, bevel), (1.0, 1.0))
        self.assertIsNone(design)

        select_edge("Source_1", "Detector_1")
        self.assertEqual(bpy.ops.neurocaptain.make_spring_stiff("EXEC_DEFAULT"), {"FINISHED"})
        bpy.ops.object.mode_set(mode="OBJECT")
        is_flex, crease, bevel, _, design = _springs_on_mesh()[key]
        self.assertFalse(is_flex)
        self.assertEqual((crease, bevel), (0.0, 0.0))
        self.assertIsNone(design)   # length taken from the scene, not a defined value
