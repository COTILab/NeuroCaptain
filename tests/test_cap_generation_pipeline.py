"""Headless integration test for the cap-generation pipeline.

Drives the real operator chain a user follows to turn a head model into a
3D-printable cap, using the bundled Colin27 sample data instead of
interactive vertex picking:

    select_model(ADD_HEADMESH)      -> headmesh, headmesh.001
    select_model(ADD_BRAIN1020MESH) -> LandmarkMesh (precomputed landmarks)
    insert_shape(ADD_CYLINDER)      -> cutout
    geo_nodes()                     -> cuts landmark holes into headmesh
    decimate_mesh()                 -> reduces face count
    cap_generation(PLACE_CUTOUTS, BOOLEAN_CUT) -> wireframe cap shape
    dual_mesh_NC()                  -> polygonal dual mesh
    export_mesh()                   -> writes a .jmsh file

brain1020mesh.py's interactive 5-point (Nz/Iz/Lpa/Rpa/Cz) picking has no
scriptable equivalent and needs iso2mesh, so it's deliberately skipped by
loading the precomputed Colin27_Atlas_landmarks.jmsh landmark set instead -
this is the only realistic headless path through this pipeline.

select_model's ADD_HEADMESH path internally calls
bpy.ops.view3d.snap_selected_to_cursor(), which fails in plain
`blender --background` mode with "context is incorrect" unless wrapped in
a temp_override supplying a VIEW_3D area/region (verified empirically -
the default factory-startup screen does have one even headless).

add_headmesh() used to call bpy.ops.object.duplicate_move(...) (a macro
combining OBJECT_OT_duplicate + TRANSFORM_OT_translate) to create the
hidden headmesh.001 duplicate. That crashed the whole process with a native
EXCEPTION_ACCESS_VIOLATION/SIGSEGV inside Blender's transform/gizmo
machinery (ED_region_draw_cb_activate expects a real, GPU-initialized
viewport region that plain `--background` mode doesn't have) - reproduced
directly, isolated to this exact call, and confirmed it is NOT a Python
exception (temp_override can't fix a native crash). CI confirmed it on both
3.4 and 4.2. add_headmesh() now duplicates via bpy.data (obj.copy()) and
sets .location directly instead, which has no viewport dependency and no
longer crashes on any version, so this test no longer needs a version skip.
"""

import os
import unittest

import bpy

from _addon_helpers import REPO_ROOT, import_addon, register_addon, get_view3d_area_and_region

HEAD_MODEL_PATH = os.path.join(REPO_ROOT, "HeadModels", "Colin27_Atlas_scalp.bmsh")
LANDMARK_PATH = os.path.join(REPO_ROOT, "BrainLandmarks", "Colin27_Atlas_landmarks.jmsh")

CREATED_OBJECT_NAMES = [
    "headmesh",
    "headmesh.001",
    "LandmarkMesh",
    "cutout",
    "face_cutout",
    "bottom_cutout",
    "ear_cutout",
    "importedmodel",
]
CREATED_NODE_GROUP_NAMES = ["Geometry Nodes", "DualMeshNodeTree"]


class CapGenerationPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addon = import_addon()
        register_addon(cls.addon)
        cls.view3d_area, cls.view3d_region = get_view3d_area_and_region()

    @classmethod
    def tearDownClass(cls):
        cls.addon.unregister()

    def setUp(self):
        self.export_filename = "test_cap_export.jmsh"

    def tearDown(self):
        for name in CREATED_OBJECT_NAMES:
            obj = bpy.data.objects.get(name)
            if obj is None:
                continue
            mesh = obj.data if obj.type == "MESH" else None
            bpy.data.objects.remove(obj, do_unlink=True)
            if mesh is not None and mesh.users == 0:
                bpy.data.meshes.remove(mesh)

        for name in CREATED_NODE_GROUP_NAMES:
            node_group = bpy.data.node_groups.get(name)
            if node_group is not None:
                bpy.data.node_groups.remove(node_group)

        export_path = os.path.join(self._export_dir(), self.export_filename)
        if os.path.exists(export_path):
            os.remove(export_path)

        if "saved_nz" in bpy.context.scene:
            del bpy.context.scene["saved_nz"]

    def _export_dir(self):
        return self.addon.utils.GetBPWorkFolder()

    def test_full_pipeline_produces_a_valid_cap_mesh(self):
        self.assertIsNotNone(
            self.view3d_area, "no VIEW_3D area found in factory-startup screen"
        )

        with bpy.context.temp_override(area=self.view3d_area, region=self.view3d_region):
            result = bpy.ops.braincapgen.select_model(
                action="ADD_HEADMESH",
                filepath=HEAD_MODEL_PATH,
                files=[{"name": os.path.basename(HEAD_MODEL_PATH)}],
            )
        self.assertEqual(result, {"FINISHED"})
        self.assertIsNotNone(bpy.data.objects.get("headmesh"))
        self.assertIsNotNone(bpy.data.objects.get("headmesh.001"))

        result = bpy.ops.braincapgen.select_model(
            action="ADD_BRAIN1020MESH",
            filepath=LANDMARK_PATH,
            files=[{"name": os.path.basename(LANDMARK_PATH)}],
        )
        self.assertEqual(result, {"FINISHED"})
        landmark_obj = bpy.data.objects.get("LandmarkMesh")
        self.assertIsNotNone(landmark_obj)

        labels = list(landmark_obj.get("landmark_labels") or [])
        self.assertIn("Nz", labels, "Colin27_Atlas_landmarks.jmsh should embed an Nz label")
        nz_index = labels.index("Nz")
        nz_local = landmark_obj.data.vertices[nz_index].co
        nz_world = landmark_obj.matrix_world @ nz_local
        bpy.context.scene["saved_nz"] = list(nz_world)

        result = bpy.ops.braincapgen.insert_shape(action="ADD_CYLINDER")
        self.assertEqual(result, {"FINISHED"})
        self.assertIsNotNone(bpy.data.objects.get("cutout"))

        result = bpy.ops.braincapgen.geo_nodes()
        self.assertEqual(result, {"FINISHED"})
        head = bpy.data.objects["headmesh"]
        self.assertGreater(len(head.data.vertices), 0)

        result = bpy.ops.braincapgen.decimate_mesh(number=0.5)
        self.assertEqual(result, {"FINISHED"})

        result = bpy.ops.braincapgen.cap_generation(action="PLACE_CUTOUTS", add_cylinder=True)
        self.assertEqual(result, {"FINISHED"})
        for name in ("face_cutout", "bottom_cutout", "ear_cutout"):
            self.assertIsNotNone(bpy.data.objects.get(name), f"{name} was not created")

        result = bpy.ops.braincapgen.cap_generation(action="BOOLEAN_CUT", thick=2, voxel=0.5)
        self.assertEqual(result, {"FINISHED"})
        head = bpy.data.objects["headmesh"]
        self.assertGreater(len(head.data.vertices), 0)
        self.assertGreater(len(head.data.polygons), 0)

        result = bpy.ops.object.dual_mesh()
        self.assertEqual(result, {"FINISHED"})
        head = bpy.data.objects["headmesh"]
        self.assertGreater(len(head.data.vertices), 0)

        bpy.context.view_layer.objects.active = head
        result = bpy.ops.braincapgen.export_mesh(filename=self.export_filename)
        self.assertEqual(result, {"FINISHED"})

        export_path = os.path.join(self._export_dir(), self.export_filename)
        self.assertTrue(os.path.exists(export_path), f"export file not found at {export_path}")

        import jdata as jd

        exported = jd.load(export_path)
        self.assertIn("MeshVertex3", exported)
        self.assertIn("MeshTri3", exported)
        self.assertGreater(len(exported["MeshVertex3"]), 0)
        self.assertGreater(len(exported["MeshTri3"]), 0)
