"""Headless test: Import Fiducials (.txt) reads Nz/Iz/LPA/RPA/Cz from a text
file and feeds them, with the chosen arc steps, to the same 10-20 landmark
mesh generation as the manual NZ/LPA/RPA/IZ/CZ buttons.

Mesh generation needs iso2mesh, so the main tests replace that call with a
recorder; the end-to-end test only runs where iso2mesh is installed.
"""

import importlib
import os
import tempfile
import unittest

import bmesh
import bpy
import numpy as np

from _addon_helpers import import_addon, register_addon

R = 90.0
FIDS = {"nz": (0.0, 88.0, 5.0), "iz": (0.0, -88.0, 5.0), "lpa": (-88.0, 0.0, 5.0),
        "rpa": (88.0, 0.0, 5.0), "cz": (0.0, 3.0, 89.0)}   # deliberately ~2 mm off the sphere
FILE_TEXT = """# fiducials in head coordinates (mm)
Nasion 0.0 88.0 5.0
Iz:    0.0, -88.0, 5.0
lpa\t-88.0\t0.0\t5.0
RPA, 88.0, 0.0, 5.0
Cz 0.0 3.0 89.0
T7 -90 0 0
"""


def _write(text):
    fd, path = tempfile.mkstemp(suffix=".txt")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    return path


class ImportFiducialsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addon = import_addon()
        cls.mod = importlib.import_module(f"{cls.addon.__name__}.brain1020mesh")
        cls.path = _write(FILE_TEXT)

    def setUp(self):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        register_addon(self.addon)
        bpy.ops.mesh.primitive_uv_sphere_add(radius=R, segments=64, ring_count=32)
        head = bpy.context.active_object
        head.name = "headmesh"
        bm = bmesh.new()                    # head meshes are triangle meshes
        bm.from_mesh(head.data)
        bmesh.ops.triangulate(bm, faces=bm.faces[:])
        bm.to_mesh(head.data)
        bm.free()
        self.calls = []
        self._orig = self.mod.brain1020mesh.brain1020_mesh

        def record(context, save_file=False, save_filepath=""):
            m = self.mod
            self.calls.append({k: np.array(getattr(m, k))[0] for k in ("nz", "iz", "lpa", "rpa", "cz")}
                              | {"p1": m.p1, "p2": m.p2, "active": context.view_layer.objects.active.name})
        self.mod.brain1020mesh.brain1020_mesh = staticmethod(record)

    def tearDown(self):
        self.mod.brain1020mesh.brain1020_mesh = staticmethod(self._orig)

    def test_read_fiducials_formats_and_aliases(self):
        pts = self.mod.read_fiducials(self.path)
        self.assertEqual(set(pts), set(FIDS))
        for k, xyz in FIDS.items():
            np.testing.assert_allclose(pts[k], xyz)

    def test_al_ar_labels(self):
        # digpts.txt style: ear points named AL / AR
        path = _write("Nz 0 88 5\nIz 0 -88 5\nCz 0 3 89\nAR 88 0 5\nAL -88 0 5\n")
        pts = self.mod.read_fiducials(path)
        np.testing.assert_allclose(pts["lpa"], FIDS["lpa"])
        np.testing.assert_allclose(pts["rpa"], FIDS["rpa"])

    def test_unlabelled_points_with_refpts_labels_file(self):
        # refpts.txt (bare x y z) + refpts_labels.txt in the same folder
        folder = tempfile.mkdtemp()
        order = ["nz", "iz", "rpa", "lpa", "cz"]
        with open(os.path.join(folder, "refpts.txt"), "w") as f:
            f.writelines("   %.7e   %.7e   %.7e\n" % FIDS[k] for k in order)
            f.write("   1.0   2.0   3.0\n")                          # an extra 10-20 point
        with open(os.path.join(folder, "refpts_labels.txt"), "w") as f:
            f.write("Nz\nIz\nRPA\nLPA\nCz\nNFpz\n")
        pts = self.mod.read_fiducials(os.path.join(folder, "refpts.txt"))
        for k in order:
            np.testing.assert_allclose(pts[k], FIDS[k])

    def test_points_off_the_head_cancel(self):
        cases = {"shifted": "Nz 0 288 5\nIz 0 112 5\nLPA -88 200 5\nRPA 88 200 5\nCz 0 203 89\n",
                 "metres":  "Nz 0 0.088 0.005\nIz 0 -0.088 0.005\nLPA -0.088 0 0.005\n"
                            "RPA 0.088 0 0.005\nCz 0 0.003 0.089\n"}
        for name, text in cases.items():
            with self.subTest(case=name):
                with self.assertRaisesRegex(RuntimeError, "same space/scale"):
                    bpy.ops.braincapgen.import_fiducials_options(
                        "EXEC_DEFAULT", filepath=_write(text), snap_to_head=True)
                self.assertEqual(self.calls, [], "mesh generation ran despite the failure")

    def test_missing_fiducial_is_reported(self):
        path = _write("Nz 0 88 5\nIz 0 -88 5\nLPA -88 0 5\nCz 0 3 89\n")
        with self.assertRaisesRegex(ValueError, "RPA"):
            self.mod.read_fiducials(path)
        with self.assertRaisesRegex(RuntimeError, "Missing fiducial.*RPA"):   # reported as an error
            bpy.ops.braincapgen.import_fiducials_options("EXEC_DEFAULT", filepath=path)
        self.assertEqual(self.calls, [])

    def test_used_exactly_as_given(self):
        self.assertEqual(bpy.ops.braincapgen.import_fiducials_options(
            "EXEC_DEFAULT", filepath=self.path, point1=5.0, point2=10.0), {"FINISHED"})
        self.assertEqual(len(self.calls), 1)
        call = self.calls[0]
        for k, xyz in FIDS.items():
            np.testing.assert_allclose(call[k], xyz)
            self.assertTrue(getattr(bpy.context.scene, f"{k}_assigned"))
        self.assertEqual((call["p1"], call["p2"]), (5.0, 10.0))
        self.assertEqual(call["active"], "headmesh")

    def test_optional_snap_to_head(self):
        bpy.ops.braincapgen.import_fiducials_options("EXEC_DEFAULT", filepath=self.path, snap_to_head=True)
        head = bpy.data.objects["headmesh"]
        verts = np.array([list(head.matrix_world @ v.co) for v in head.data.vertices])
        for k, xyz in FIDS.items():
            p = self.calls[0][k]
            self.assertLess(np.min(np.linalg.norm(verts - p, axis=1)), 1e-4, f"{k} not on a head vertex")
            self.assertGreater(np.linalg.norm(p - np.array(xyz)), 0.5, f"{k} was not moved")

    def test_generates_landmark_mesh(self):
        if not hasattr(self.mod.safe_import("iso2mesh"), "brain1020"):
            self.skipTest("iso2mesh with brain1020 not available in this Python environment")
        self.mod.brain1020mesh.brain1020_mesh = staticmethod(self._orig)
        # The landmark mesh is added to the default "Collection" of a normal startup scene
        bpy.context.scene.collection.children.link(bpy.data.collections.new("Collection"))
        # Its "completed" message box is a popup, which crashes Blender in background mode
        orig_box, self.mod.ShowMessageBox = self.mod.ShowMessageBox, lambda *a, **k: None
        self.addCleanup(setattr, self.mod, "ShowMessageBox", orig_box)
        self.assertEqual(bpy.ops.braincapgen.import_fiducials_options(
            "EXEC_DEFAULT", filepath=self.path, snap_to_head=True), {"FINISHED"})
        lm = bpy.data.objects.get("LandmarkMesh")
        self.assertIsNotNone(lm)
        self.assertIn("Cz", list(lm["landmark_labels"]))


if __name__ == "__main__":
    unittest.main()
