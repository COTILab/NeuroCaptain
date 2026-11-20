"""Headless test: with "Scale Springs to Head" on, JSON probe import keeps the
probe proportional on a different-sized head (short-separation springs
included); by default the springs keep the probe file's designed lengths.

A small synthetic probe is "designed" on a sphere of radius R0 with a
LandmarkMesh on that sphere, and written as a NeuroCaptain probe JSON
(barycentric registration + absolute rest lengths, as the exporter does).
It is then imported with neurocaptain.import_optode_json_blender_goal onto
a head/LandmarkMesh scaled by SCALE.  Barycentric placement scales every
optode with the head, so after relaxation every connection - long and
short-separation alike - should be SCALE x its designed length, and no
optodes should collapse onto each other.
"""

import json
import math
import os
import tempfile
import unittest

import bpy
from mathutils import Vector

from _addon_helpers import import_addon, register_addon

R0 = 90.0           # design head radius (mm)
SS_MM = 8.0         # short-separation distance (mm)
LONG_MM = 30.0      # long-channel spacing (mm)


def _sphere_point(theta_deg, phi_deg, r=R0):
    t, p = math.radians(theta_deg), math.radians(phi_deg)
    return Vector((r * math.sin(t) * math.cos(p), r * math.sin(t) * math.sin(p), r * math.cos(t)))


def _landmarks():
    """Rings of landmark points on the upper hemisphere (plus the vertex)."""
    pts = [_sphere_point(0, 0)]
    for theta in (20, 40, 60, 80):
        for phi in range(0, 360, 30):
            pts.append(_sphere_point(theta, phi))
    return pts


def _barycentric(p, a, b, c):
    v0, v1, v2 = b - a, c - a, p - a
    d00, d01, d11 = v0.dot(v0), v0.dot(v1), v1.dot(v1)
    d20, d21 = v2.dot(v0), v2.dot(v1)
    den = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    return [1.0 - v - w, v, w]


def _tangent_offset(p, direction, dist):
    """Point `dist` mm from p along the sphere, heading `direction`."""
    n = p.normalized()
    t = (direction - n * direction.dot(n)).normalized()
    return (p + t * dist).normalized() * R0


def _design_probe():
    """2x2 sources / 2x2 detectors grid near the vertex + 2 short-sep detectors."""
    origin = _sphere_point(25, 0)
    ex, ey = Vector((0, 1, 0)), Vector((-1, 0, 1)).normalized()
    pos = {
        "Source_1": origin,
        "Source_2": _tangent_offset(origin, ex, 2 * LONG_MM),
    }
    pos["Detector_1"] = _tangent_offset(origin, ex, LONG_MM)
    pos["Detector_2"] = _tangent_offset(pos["Source_2"], ex, LONG_MM)
    pos["Source_3"] = _tangent_offset(pos["Detector_1"], ey, LONG_MM)
    pos["Detector_3"] = _tangent_offset(origin, ey, LONG_MM)
    pos["Detector_4"] = _tangent_offset(pos["Detector_2"], ey, LONG_MM)
    # short-separation detectors: one on an anchored source, one on a free one
    pos["Detector_5"] = _tangent_offset(pos["Source_1"], -ey, SS_MM)
    pos["Detector_6"] = _tangent_offset(pos["Source_3"], ey, SS_MM)
    anchors = {"Source_1", "Source_2", "Detector_4"}
    springs = [("Source_1", "Detector_1"), ("Detector_1", "Source_2"), ("Source_2", "Detector_2"),
               ("Source_1", "Detector_3"), ("Detector_3", "Source_3"), ("Source_3", "Detector_1"),
               ("Source_3", "Detector_4"), ("Detector_2", "Detector_4"),
               ("Source_1", "Source_3"), ("Source_2", "Detector_4"),   # diagonals -> triangles for the cloth
               ("Source_1", "Detector_5"), ("Source_3", "Detector_6")]
    return pos, anchors, springs


def _write_probe_json(path, landmarks):
    pos, anchors, springs = _design_probe()
    optodes = []
    for name, p in list(pos.items()):
        idx = sorted(range(len(landmarks)), key=lambda i: (landmarks[i] - p).length)[:3]
        w = _barycentric(p, *(landmarks[i] for i in idx))
        # Design position = exactly what barycentric reconstruction + surface
        # snap gives on the design head, so rest lengths are self-consistent.
        p = sum((wi * landmarks[i] for wi, i in zip(w, idx)), Vector()).normalized() * R0
        pos[name] = p
        optodes.append({
            "name": name,
            "type": "source" if name.startswith("Source") else "detector",
            "position": list(p),
            "is_anchor": name in anchors,
            "registration": {
                "method": "barycentric",
                "vertex_indices": idx,
                "barycentric_coords": w,
            },
        })
    connections = [{"optode1": a, "optode2": b, "rest_length": (pos[a] - pos[b]).length,
                    "pull": 0.9, "push": 0.9, "is_flexible": False} for a, b in springs]
    with open(path, "w") as f:
        json.dump({"format": "NeuroCaptain Probe Configuration", "version": "2.0",
                   "metadata": {"num_landmarks": len(landmarks)},
                   "optodes": optodes, "connections": connections}, f)
    return {(a, b): c["rest_length"] for (a, b), c in zip(springs, connections)}


class JsonImportScalingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addon = import_addon()
        register_addon(cls.addon)
        cls.landmarks = _landmarks()
        cls.tmpdir = tempfile.mkdtemp()
        cls.json_path = os.path.join(cls.tmpdir, "probe.json")
        cls.design = _write_probe_json(cls.json_path, cls.landmarks)

    @classmethod
    def tearDownClass(cls):
        cls.addon.unregister()

    def setUp(self):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        register_addon(self.addon)

    def tearDown(self):
        # Don't leave the cloth-simulated probe in the scene: the next test
        # file's depsgraph evaluation would re-run the cloth (hangs on 5.2,
        # crashes 3.6 in CI).
        bpy.ops.wm.read_factory_settings(use_empty=True)
        register_addon(self.addon)

    def _build_scene(self, scale):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=R0 * scale, segments=128, ring_count=64)
        bpy.context.active_object.name = "headmesh"
        mesh = bpy.data.meshes.new("LandmarkMesh")
        mesh.from_pydata([v * scale for v in self.landmarks], [], [])
        lm = bpy.data.objects.new("LandmarkMesh", mesh)
        lm["landmark_labels"] = [f"L{i}" for i in range(len(self.landmarks))]
        bpy.context.collection.objects.link(lm)

    def _import(self, scale, **options):
        self._build_scene(scale)
        result = bpy.ops.neurocaptain.import_optode_json_blender_goal(
            filepath=self.json_path, **options)
        self.assertEqual(result, {"FINISHED"})
        return {n: bpy.data.objects[n].matrix_world.translation.copy()
                for n in ("Source_1", "Source_2", "Source_3", "Detector_1", "Detector_2",
                          "Detector_3", "Detector_4", "Detector_5", "Detector_6")}

    def _assert_proportional(self, scale):
        pos = self._import(scale, scale_rest_lengths=True)
        for (a, b), rest in self.design.items():
            ratio = (pos[a] - pos[b]).length / rest
            kind = "short-sep" if rest < 15 else "long"
            with self.subTest(pair=f"{a}-{b}", kind=kind):
                self.assertAlmostEqual(ratio, scale, delta=0.15 * scale,
                                       msg=f"{a}-{b} ({kind}) length ratio {ratio:.2f}, expected {scale}")
        names = list(pos)
        closest = min((pos[a] - pos[b]).length for i, a in enumerate(names) for b in names[i + 1:])
        self.assertGreater(closest, 0.5 * SS_MM * scale, "optodes collapsed onto each other")

        # Stored springs (used by "Relax Probe") must carry the scaled lengths too
        states = bpy.data.objects["Optode_Connections"]["spring_states"]
        for (a, b), rest in self.design.items():
            stored = float(states[f"{min(a, b)}_{max(a, b)}"]["rest_length"])
            self.assertAlmostEqual(stored / rest, scale, delta=0.1 * scale,
                                   msg=f"{a}-{b} stored rest length not scaled")

    def test_same_size_head(self):
        self._assert_proportional(1.0)

    def test_head_twice_the_size(self):
        self._assert_proportional(2.0)

    def test_default_keeps_design_lengths(self):
        # Default: rest lengths are the probe file's, not scaled to the head
        self._import(1.1)
        conn = bpy.data.objects["Optode_Connections"]
        self.assertEqual(conn["rest_length_scale"], 1.0)
        states = conn["spring_states"]
        for (a, b), rest in self.design.items():
            stored = float(states[f"{min(a, b)}_{max(a, b)}"]["rest_length"])
            self.assertAlmostEqual(stored, rest, places=4, msg=f"{a}-{b} rest length was scaled")

    def test_smaller_head(self):
        self._assert_proportional(0.7)
